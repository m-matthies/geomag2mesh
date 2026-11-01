"""Synthetic frameworks and photo observations for testing and benchmarking.

Occlusion model: a sphere is unobserved in a view when more than
``occlusion_frac`` of its image disc is covered by the discs of nearer spheres
(rods are ignored as occluders). Observation noise is isotropic Gaussian in
pixels for centres and relative Gaussian for radii.
"""
from __future__ import annotations

import itertools

import numpy as np

from .annotations import Obs, Project, Rod, View
from .parts import PartCatalog

COLORS = ["dark_blue", "light_blue", "magenta", "orange"]


def cube(pitch: float, shear: float = 0.0, rng: np.random.Generator | None = None,
         prefix: str = "") -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    """A (optionally sheared) cube: 8 spheres, 12 rods; edge vectors are unit-length
    directions scaled by ``pitch``."""
    rng = rng or np.random.default_rng(0)
    e = np.eye(3)
    if shear:
        e = e + rng.normal(0, shear, (3, 3))
        e /= np.linalg.norm(e, axis=1, keepdims=True)
    pts = {}
    for bits in itertools.product((0, 1), repeat=3):
        pts[prefix + "".join(map(str, bits))] = pitch * (np.array(bits) @ e)
    edges = [(a, b) for a, b in itertools.combinations(pts, 2)
             if sum(x != y for x, y in zip(a[len(prefix):], b[len(prefix):])) == 1]
    return pts, edges


def twin_cube(pitch: float, D: float, rng: np.random.Generator | None = None):
    """Two cubes joined face to face by four sphere contacts (as in the reference photos)."""
    rng = rng or np.random.default_rng(1)
    p1, e1 = cube(pitch, 0.0, rng, "a")
    p2, e2 = cube(pitch, 0.0, rng, "b")
    shift = np.array([0.0, pitch + D, 0.0])
    p2 = {k: v + shift for k, v in p2.items()}
    contacts = [("a" + k[1:], "b" + k[1:]) for k in p1 if k[2] == "1"
                for _ in [0]]
    contacts = [(a, "b" + a[1] + "0" + a[3]) for a, _ in contacts]
    return {**p1, **p2}, e1 + e2, contacts


def octahedron(pitch: float):
    s = pitch / np.sqrt(2)
    pts = {"px": [s, 0, 0], "nx": [-s, 0, 0], "py": [0, s, 0], "ny": [0, -s, 0],
           "pz": [0, 0, s], "nz": [0, 0, -s]}
    pts = {k: np.array(v, float) for k, v in pts.items()}
    edges = [(a, b) for a, b in itertools.combinations(pts, 2) if a[1] != b[1]]
    return pts, edges


def look_at(eye: np.ndarray, target: np.ndarray, up=np.array([0, 0, 1.0])):
    """World-to-camera rotation and translation (camera z forward, y down)."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def make_project(points: dict[str, np.ndarray], edges, contacts=(), *, n_views: int = 4,
                 focal_px: float = 3000.0, size=(3024, 4032), distance: float = 300.0,
                 elevation_deg: float = 35.0, noise_px: float = 4.0, radius_noise: float = 0.04,
                 occlusion_frac: float = 0.6, crop_views: tuple[int, ...] = (), seed: int = 0,
                 parts: PartCatalog | None = None):
    """Render a framework from ``n_views`` cameras on a ring; returns (project, priors, truth)."""
    from .images import CameraPrior
    rng = np.random.default_rng(seed)
    parts = parts or PartCatalog()
    R_mm = parts.sphere_radius_mm
    labels = list(points)
    X = np.array([points[k] for k in labels])
    X = X - X.mean(0)
    W, H = size
    views, priors = [], {}
    for v in range(n_views):
        az = 2 * np.pi * v / n_views + rng.uniform(-0.3, 0.3)
        el = np.radians(elevation_deg + rng.uniform(-10, 10))
        eye = distance * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        R, t = look_at(eye, rng.normal(0, 5, 3))
        Xc = X @ R.T + t
        uv = focal_px * Xc[:, :2] / Xc[:, 2:3] + np.array([W / 2, H / 2])
        r = focal_px * R_mm / Xc[:, 2]
        visible = _visibility(uv, r, Xc[:, 2], occlusion_frac)
        w, h, off = W, H, np.zeros(2)
        if v in crop_views:  # unscaled crop of the central region, offset randomly
            w, h = int(W * 0.6), int(H * 0.55)
            off = np.array([rng.integers(0, W - w), rng.integers(0, H - h)], float)
        sph = {}
        for i, k in enumerate(labels):
            p = uv[i] - off
            if visible[i] and 0 <= p[0] < w and 0 <= p[1] < h:
                p = p + rng.normal(0, noise_px, 2)
                sph[k] = Obs(float(p[0]), float(p[1]), float(r[i] * (1 + rng.normal(0, radius_noise))), 1.0)
        name = f"view{v}"
        views.append(View(name, None, w, h, sph))
        priors[name] = CameraPrior(focal_px * (1 + rng.normal(0, 0.03)), 0.1,
                                   v not in crop_views, "synthetic")
    rods = [Rod(a, b, COLORS[i % len(COLORS)]) for i, (a, b) in enumerate(edges)]
    proj = Project(views=views, rods=rods, contacts=[], auto_contacts=True, parts=parts)
    truth = {k: X[i] for i, k in enumerate(labels)}
    return proj, priors, truth


def _visibility(uv, r, z, frac):
    n = len(r)
    vis = np.ones(n, bool)
    # Monte-Carlo coverage estimate per disc: O(n^2 * s) with s samples per disc
    rng = np.random.default_rng(123)
    ang = rng.uniform(0, 2 * np.pi, 200)
    rad = np.sqrt(rng.uniform(0, 1, 200))
    for i in range(n):
        s = uv[i] + (r[i] * rad)[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
        covered = np.zeros(len(s), bool)
        for j in range(n):
            if j != i and z[j] < z[i]:
                covered |= np.linalg.norm(s - uv[j], axis=1) < r[j]
        vis[i] = covered.mean() < frac
    return vis


def aligned_errors(truth: dict[str, np.ndarray], labels, X: np.ndarray) -> np.ndarray:
    """Per-sphere error (mm) after rigid alignment of a solution onto ground truth."""
    from .solve import _umeyama
    A = np.array([truth[k] for k in labels])
    s, R, t = _umeyama(X, A, with_scale=False)
    return np.linalg.norm(X @ R.T + t - A, axis=1)


# =========================================================================== photo rendering
def render_scene(points: dict[str, np.ndarray], rods: list[tuple[str, str, str]], *,
                 board=None, n_views: int = 6, focal_px: float = 2400.0, size=(2016, 1512),
                 distance: float = 330.0, elevation_deg: float = 40.0, seed: int = 0,
                 parts: PartCatalog | None = None):
    """Render simple photographs (painter's algorithm): grey table, optional marker mat,
    coloured rods as shaded strips, steel spheres as shaded discs with a highlight.

    ``points`` are in the mat frame (mm, z up, mat at z = 0). Returns
    (list of RGB images, list of (R, t) world->camera).
    Not photorealistic; intended to exercise the pipeline end to end.
    """
    import cv2

    from .markers import render_board
    from .parts import RENDER_RGB
    rng = np.random.default_rng(seed)
    parts = parts or PartCatalog()
    Rs = parts.sphere_radius_mm
    rod_r = min(rt.diameter_mm for rt in parts.rod_types.values()) / 2
    labels = list(points)
    X = np.array([points[k] for k in labels], float)
    W, H = size
    K = np.array([[focal_px, 0, W / 2], [0, focal_px, H / 2], [0, 0, 1.0]])
    if board is not None:
        dpi = 200
        bimg = render_board(board, dpi, margin_mm=10)
        pxmm = dpi / 25.4
        B = np.array([[1 / pxmm, 0, -bimg.shape[1] / 2 / pxmm], [0, -1 / pxmm, bimg.shape[0] / 2 / pxmm],
                      [0, 0, 1]])
    imgs, poses = [], []
    centre = X.mean(0)
    for v in range(n_views):
        az = 2 * np.pi * v / n_views + rng.uniform(-0.2, 0.2)
        el = np.radians(elevation_deg + rng.uniform(-8, 8))
        eye = centre + distance * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        R, t = look_at(eye, centre + rng.normal(0, 4, 3))
        img = np.full((H, W, 3), 95, np.uint8)
        img = (img.astype(np.int16) + rng.normal(0, 4, img.shape)).clip(0, 255).astype(np.uint8)
        if board is not None:
            Hm = K @ np.c_[R[:, 0], R[:, 1], t] @ B
            warped = cv2.warpPerspective(cv2.cvtColor(bimg, cv2.COLOR_GRAY2RGB), Hm, (W, H),
                                         borderValue=(0, 0, 0))
            mask = cv2.warpPerspective(np.full(bimg.shape, 255, np.uint8), Hm, (W, H)) > 0
            img[mask] = (0.85 * warped[mask] + 20).astype(np.uint8)
        Xc = X @ R.T + t
        uv = Xc @ K.T
        uv = uv[:, :2] / uv[:, 2:3]
        items = []
        for a, b, col in rods:
            i, j = labels.index(a), labels.index(b)
            items.append((0.5 * (Xc[i, 2] + Xc[j, 2]), "rod", (i, j, col)))
        for i in range(len(labels)):
            items.append((Xc[i, 2], "sphere", i))
        for _, kind, data in sorted(items, key=lambda it: -it[0]):
            if kind == "rod":
                i, j, col = data
                # the rod spans sphere surface to sphere surface (tips touch the spheres)
                u3 = (Xc[j] - Xc[i]) / np.linalg.norm(Xc[j] - Xc[i])
                Ai, Aj = Xc[i] + u3 * Rs, Xc[j] - u3 * Rs
                pi_ = K @ Ai
                pj_ = K @ Aj
                ui, uj = pi_[:2] / pi_[2], pj_[:2] / pj_[2]
                d = uj - ui
                n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
                wi, wj = focal_px * rod_r / Ai[2], focal_px * rod_r / Aj[2]
                base = np.array(RENDER_RGB.get(col, RENDER_RGB["unknown"]), float)
                for k, shade in ((1.0, 0.75), (0.6, 0.95), (0.25, 1.15)):  # simple cylinder shading
                    poly = np.array([ui + n * wi * k, uj + n * wj * k,
                                     uj - n * wj * k, ui - n * wi * k]).astype(np.int32)
                    cv2.fillConvexPoly(img, poly, tuple(int(c) for c in np.clip(base * shade, 0, 255)),
                                       lineType=cv2.LINE_AA)
            else:
                i = data
                r = focal_px * Rs / Xc[i, 2]
                c = uv[i]
                for k in np.linspace(1.0, 0.1, 10):  # darker rim, brighter centre
                    g = int(70 + 110 * (1 - k))
                    cv2.circle(img, tuple(np.round(c).astype(int)), max(1, int(r * k)), (g, g, g + 6), -1,
                               lineType=cv2.LINE_AA)
                for _ in range(6):  # environment reflections
                    o = rng.uniform(-0.5, 0.5, 2) * r
                    cv2.circle(img, tuple(np.round(c + o).astype(int)), max(1, int(r * rng.uniform(0.08, 0.2))),
                               tuple(int(x) for x in rng.integers(40, 160, 3)), -1)
                cv2.circle(img, tuple(np.round(c + np.array([-0.35, -0.35]) * r).astype(int)),
                           max(1, int(0.18 * r)), (250, 250, 250), -1, lineType=cv2.LINE_AA)
        imgs.append(cv2.GaussianBlur(img, (0, 0), 0.8))
        poses.append((R, t))
    return imgs, poses
