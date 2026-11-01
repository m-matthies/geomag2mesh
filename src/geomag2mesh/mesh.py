"""Triangle meshes of the reconstructed framework and their export.

Spheres are icospheres; rods are capped cylinders whose ends are embedded
``rod_inset_mm`` from the sphere centres (the real flat magnetic tips touch the
sphere surface at a point, which would make a boolean union non-manifold).

Output frame: millimetres, z up, the support plane (mat or table) at z = 0.
With a marker mat the mat frame is used directly; otherwise the convex-hull
face the framework rests on is found with the cameras' "up" direction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .parts import RENDER_RGB, STEEL_RGB, PartCatalog
from .solve import Solution


def table_frame(sol: Solution, sphere_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (rotation matrix, translation) mapping solution coordinates to the output frame."""
    X = sol.X
    if sol.stats.get("frame") == "marker":
        return np.eye(3), np.zeros(3)
    up = sol.up_hint()
    if len(X) >= 4:
        hull = ConvexHull(X)
        # resting face: outward normal most opposed to "up"
        n = max(hull.equations, key=lambda e: float(np.dot(-e[:3], up)))[:3]
        R = Rotation.align_vectors([[0, 0, -1]], [n])[0].as_matrix()
    else:
        R = Rotation.align_vectors([[0, 0, 1]], [up])[0].as_matrix()
    Y = X @ R.T
    t = np.array([-Y[:, 0].mean(), -Y[:, 1].mean(), sphere_radius - Y[:, 2].min()])
    return R, t


def build_meshes(sol: Solution, parts: PartCatalog, rod_inset_mm: float = 5.0,
                 sphere_subdiv: int = 3, rod_sections: int = 40):
    """Return (centres in output frame, list of (kind, name, trimesh.Trimesh))."""
    Rs = parts.sphere_radius_mm
    R, t = table_frame(sol, Rs)
    Y = sol.X @ R.T + t
    out = []
    for i, k in enumerate(sol.labels):
        m = trimesh.creation.icosphere(subdivisions=sphere_subdiv, radius=Rs)
        m.apply_translation(Y[i])
        m.visual.face_colors = (*STEEL_RGB, 255)
        out.append(("sphere", k, m))
    for r in sol.rods:
        a, b = Y[sol.index(r.a)], Y[sol.index(r.b)]
        u = (b - a) / np.linalg.norm(b - a)
        rt = parts.rod_types.get(r.type or parts.default_rod, parts.rod_types[parts.default_rod])
        seg = np.array([a + u * rod_inset_mm, b - u * rod_inset_mm])
        m = trimesh.creation.cylinder(radius=rt.diameter_mm / 2, segment=seg, sections=rod_sections)
        m.visual.face_colors = (*RENDER_RGB.get(r.color, RENDER_RGB["unknown"]), 255)
        out.append(("rod", f"{r.a}-{r.b}", m))
    return Y, out


def _components(sol: Solution) -> list[set[str]]:
    """Rigid bodies: spheres connected by rods (contacts do not join bodies)."""
    parent = {k: k for k in sol.labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in sol.rods:
        parent[find(r.a)] = find(r.b)
    groups: dict[str, set[str]] = {}
    for k in sol.labels:
        groups.setdefault(find(k), set()).add(k)
    return list(groups.values())


def export(sol: Solution, parts: PartCatalog, out_dir: str | Path, stem: str = "model",
           formats=("glb", "stl", "obj"), log=print) -> dict[str, Path]:
    """Write meshes and a centres JSON. STL is a boolean union per rigid body."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Y, parts_m = build_meshes(sol, parts)
    paths: dict[str, Path] = {}
    if "glb" in formats:
        scene = trimesh.Scene()
        mats: dict[tuple, trimesh.visual.material.PBRMaterial] = {}
        for kind, name, m in parts_m:
            key = (kind, tuple(m.visual.face_colors[0][:3]))
            if key not in mats:
                mats[key] = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[*(np.array(key[1]) / 255), 1.0],
                    metallicFactor=1.0 if kind == "sphere" else 0.0,
                    roughnessFactor=0.15 if kind == "sphere" else 0.45)
            mm = m.copy()
            mm.visual = trimesh.visual.TextureVisuals(material=mats[key])
            scene.add_geometry(mm, node_name=f"{kind}_{name}", geom_name=f"{kind}_{name}")
        paths["glb"] = out_dir / f"{stem}.glb"
        scene.export(paths["glb"])
    if "obj" in formats:
        paths["obj"] = out_dir / f"{stem}.obj"
        trimesh.Scene([m for _, _, m in parts_m]).export(paths["obj"])
    if "stl" in formats:
        by_name = {(k, n): m for k, n, m in parts_m}
        shells = []
        for comp in _components(sol):
            ms = [by_name[("sphere", k)] for k in comp] + \
                 [by_name[("rod", f"{r.a}-{r.b}")] for r in sol.rods if r.a in comp]
            try:
                shells.append(trimesh.boolean.union(ms, engine="manifold"))
            except Exception as e:  # manifold3d missing or failed: keep separate parts
                log(f"boolean union unavailable ({e}); STL contains overlapping parts")
                shells.extend(ms)
        paths["stl"] = out_dir / f"{stem}.stl"
        trimesh.util.concatenate(shells).export(paths["stl"])
        watertight = all(s.is_watertight for s in shells)
        log(f"STL: {len(shells)} body/bodies, watertight={watertight}")
    paths["json"] = out_dir / f"{stem}_centres.json"
    import json
    paths["json"].write_text(json.dumps({
        "units": "mm", "frame": "z up, support plane at z = 0",
        "sphere_diameter_mm": parts.sphere_diameter_mm,
        "centres": {k: Y[i].round(3).tolist() for i, k in enumerate(sol.labels)},
        "rods": [{"a": r.a, "b": r.b, "color": r.color, "type": r.type} for r in sol.rods],
        "contacts": [list(c) for c in sol.contacts],
    }, indent=1))
    return paths
