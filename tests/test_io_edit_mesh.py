"""Annotation I/O and validation, edit operations, mesh export, marker poses."""
import json

import numpy as np
import pytest
import trimesh

from geomag2mesh import edit
from geomag2mesh import synthetic as syn
from geomag2mesh.annotations import Obs, Project, Rod, View
from geomag2mesh.mesh import export
from geomag2mesh.solve import SolveOptions, solve


def _small_project():
    return Project(
        views=[View("a", None, 100, 80, {"A": Obs(10, 10, 5, 1), "B": Obs(50, 10, None, 0.5)}),
               View("b", None, 100, 80, {"A": Obs(12, 11, 5, 1), "C": Obs(70, 40, 4, 1)})],
        rods=[Rod("A", "B", "orange"), Rod("B", "C", "magenta", "long")],
        contacts=[("A", "C")])


def test_roundtrip(tmp_path):
    p = _small_project()
    p.save(tmp_path / "ann.json")
    q = Project.load(tmp_path / "ann.json")
    assert q.labels == p.labels
    assert q.views[0].spheres["B"].r is None and q.views[0].spheres["B"].w == 0.5
    assert [(r.a, r.b, r.color, r.type) for r in q.rods] == [("A", "B", "orange", None),
                                                               ("B", "C", "magenta", "long")]
    assert q.contacts == [("A", "C")]


def test_compact_rod_form_and_format_check():
    d = _small_project().to_dict()
    d["rods"] = [["A", "B", "orange"]]
    assert Project.from_dict(d).rods[0].color == "orange"
    d["format"] = "something@9"
    with pytest.raises(ValueError):
        Project.from_dict(d)


def test_validation():
    p = _small_project()
    p.rods.append(Rod("B", "A"))
    with pytest.raises(ValueError, match="duplicate"):
        p.validate()
    p = _small_project()
    p.rods.append(Rod("A", "C"))
    with pytest.raises(ValueError, match="both as rod and as contact"):
        p.validate()
    p = _small_project()
    p.rods.append(Rod("C", "Z"))
    assert any("Z" in w and "undetermined" in w for w in p.validate())


def test_edit_merge_and_delete():
    p = _small_project()
    edit.rename(p, "C", "B")             # merge C into B
    assert "C" not in p.labels
    assert p.views[1].spheres["B"].x == 70
    assert all(r.a != r.b for r in p.rods)  # B-C collapsed away
    edit.toggle_rod(p, "A", "B")         # removes existing A-B
    assert not p.rods
    edit.toggle_rod(p, "A", "B", "orange")
    assert p.rods[0].color == "orange"
    edit.delete_sphere(p, "B")
    assert "B" not in p.labels and not p.rods
    assert edit.next_label(p) == "S01"


def test_nearest_sphere():
    p = _small_project()
    assert edit.nearest_sphere(p, "a", 12, 11) == "A"
    assert edit.nearest_sphere(p, "a", 90, 70) is None


def test_mesh_export(tmp_path, twin_cube, catalog):
    pts, edges, _ = twin_cube
    proj, priors, _ = syn.make_project(pts, edges, n_views=4, seed=3)
    sol = solve(proj, priors, SolveOptions(restarts=1), log=lambda *_: None)
    paths = export(sol, catalog, tmp_path, log=lambda *_: None)
    stl = trimesh.load(paths["stl"])
    bodies = stl.split(only_watertight=False)
    assert len(bodies) == 2 and all(b.is_watertight for b in bodies)
    centres = json.loads(paths["json"].read_text())["centres"]
    z = np.array(list(centres.values()))[:, 2]
    assert abs(z.min() - catalog.sphere_radius_mm) < 1e-6   # resting on z = 0
    scene = trimesh.load(paths["glb"])
    assert len(scene.geometry) == 16 + 24


def test_marker_pose_recovery():
    import cv2

    from geomag2mesh.markers import BoardSpec, board_pose, detect_board, render_board
    spec = BoardSpec()
    dpi = 300
    img = render_board(spec, dpi, margin_mm=10)
    pxmm = dpi / 25.4
    A = np.array([[1 / pxmm, 0, -img.shape[1] / 2 / pxmm], [0, -1 / pxmm, img.shape[0] / 2 / pxmm],
                  [0, 0, 1]])
    f, W, H = 3000.0, 3024, 4032
    eye = 420 * np.array([np.cos(0.7) * np.cos(0.8), np.sin(0.7) * np.cos(0.8), np.sin(0.8)])
    R, t = syn.look_at(eye, np.zeros(3))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    cam = cv2.warpPerspective(img, K @ np.c_[R[:, 0], R[:, 1], t] @ A, (W, H), borderValue=128)
    c = K @ t
    cv2.circle(cam, (int(c[0] / c[2]), int(c[1] / c[2])), 450, 90, -1)  # object hides the centre
    ids, corners = detect_board(cv2.cvtColor(cam, cv2.COLOR_GRAY2RGB), spec)
    rv, tv, rms = board_pose(ids, corners, spec, f, (W / 2, H / 2))
    ang = np.degrees(np.arccos(np.clip((np.trace(cv2.Rodrigues(rv)[0].T @ R) - 1) / 2, -1, 1)))
    assert ang < 0.1 and np.linalg.norm(tv - t) < 1.0 and rms < 1.0
