"""Two-view relative pose and sphere correspondence without a marker mat.

Why not align radius-based point clouds? The depth of a sphere from its apparent
radius, z = f R / r, scatters by ~10 % on real photos (measured on the example:
specular rims bias and blur the silhouette), i.e. +-35-45 mm at 40 cm, about one
rod pitch. In a lattice-like framework neighbouring spheres are one pitch apart,
so cloud alignment cannot tell them apart.

What is precise: sphere *centres* in the image (0.1-0.25 sphere radii) and rod
lengths (known to < 1 %). This module therefore

1. generates hypotheses from rod paths i-j-k with equal colours in both photos
   (3 correspondences + 2 rods),
2. refines the relative pose (6 dof) by minimising reprojection error of
   midpoint-triangulated points, rod-length residuals (scale) and a loose
   apparent-radius depth prior,
3. grows the correspondence set by guided matching: all detection pairs are
   triangulated under the pose; feasible pairs reproject within
   ``reproj_tol`` radii in both photos and agree with both apparent radii; a
   linear assignment picks a one-to-one set; steps 2-3 are iterated
   (locally optimised RANSAC),
4. scores the result by inliers plus 3-D rod checks: every rod detected in
   either photo between two matched spheres must be one pitch long and, if
   detected in both, have one colour. Wrong hypotheses produce wrong lengths.

Complexity: H hypotheses (rod-path pairs with equal colours, typically < 500
per photo pair) x a few small least-squares solves (6 parameters) x O(n_a n_b)
guided matching.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation

from .detect import ViewDetections


@dataclass
class TwoViewOptions:
    reproj_tol: float = 0.4        # x apparent radius, both photos
    depth_tol: float = 0.30        # |log(z / z_radius)| (radius-depth scatter ~0.1)
    radius_sigma: float = 0.10     # relative, for the depth prior in pose refinement
    centre_sigma: float = 0.15     # x apparent radius, for reprojection residuals
    rod_sigma: float = 0.02        # relative rod-length tolerance in refinement
    rod_ok_tol: float = 0.08       # relative tolerance for a rod to count as consistent
    min_parallax_deg: float = 3.0
    lo_iters: int = 3
    rod_weight: float = 1.0
    rod_bad_weight: float = 2.0
    max_hypotheses: int = 4000


@dataclass
class TwoViewResult:
    a: int
    b: int
    R: np.ndarray                  # pose of photo b relative to photo a (x_b = R x_a + t)
    t: np.ndarray
    pairs: list[tuple[int, int]]
    score: float
    n_rods_ok: int = 0
    n_rods_bad: int = 0
    X: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # points in photo-a frame


class ViewGeom:
    """Detections of one photo as bearing vectors (camera frame) and radius depths."""

    def __init__(self, det: ViewDetections, focal_px: float, principal, sphere_radius_mm: float):
        self.det = det
        self.f = float(focal_px)
        c = np.asarray(principal, float)
        xy = np.array([[s.x, s.y] for s in det.spheres], float).reshape(-1, 2)
        self.xy = xy
        self.c = c
        self.r = np.array([s.r for s in det.spheres], float)
        b = np.c_[(xy - c) / self.f, np.ones(len(xy))]
        self.u = b / np.linalg.norm(b, axis=1, keepdims=True)
        self.z = self.f * sphere_radius_mm / np.maximum(self.r, 1e-6)   # radius depth
        self.P = (xy - c) / self.f * self.z[:, None]
        self.P = np.c_[self.P, self.z]
        self.rods = {frozenset((r.i, r.j)): r.color for r in det.rods}
        nb: dict[int, list[tuple[int, str]]] = {}
        for r in det.rods:
            nb.setdefault(r.i, []).append((r.j, r.color))
            nb.setdefault(r.j, []).append((r.i, r.color))
        self.paths = [(i, j, k, ci, ck) for j, lst in nb.items()
                      for (i, ci), (k, ck) in itertools.permutations(lst, 2)]

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.f * X[:, :2] / X[:, 2:3] + self.c, X[:, 2]


def _midpoints(ua: np.ndarray, vb_world: np.ndarray, Cb: np.ndarray):
    """Closest-point triangulation of rays a (from origin) and b (from Cb), vectorised.
    Returns points (n, 3) and the parallax angle (rad)."""
    w0 = -Cb
    a = np.einsum("ij,ij->i", ua, ua)
    b = np.einsum("ij,ij->i", ua, vb_world)
    c = np.einsum("ij,ij->i", vb_world, vb_world)
    d = ua @ w0
    e = vb_world @ w0
    den = a * c - b * b
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    pa = ua * s[:, None]
    pb = Cb + vb_world * t[:, None]
    ang = np.arccos(np.clip(b / np.sqrt(a * c), -1, 1))
    return 0.5 * (pa + pb), ang


class TwoView:
    def __init__(self, A: ViewGeom, B: ViewGeom, pitch: float, D: float, opts: TwoViewOptions):
        self.A, self.B, self.L, self.D, self.o = A, B, pitch, D, opts

    # ------------------------------------------------------------ geometry
    def triangulate(self, R, t, ia, ib):
        Cb = -R.T @ t
        vb = self.B.u[ib] @ R               # R^T v  (row form)
        X, ang = _midpoints(self.A.u[ia], vb, Cb)
        return X, ang

    def reproj(self, R, t, X, ia, ib):
        pa, za = self.A.project(X)
        Xb = X @ R.T + t
        pb, zb = self.B.project(Xb)
        ea = np.linalg.norm(pa - self.A.xy[ia], axis=1) / self.A.r[ia]
        eb = np.linalg.norm(pb - self.B.xy[ib], axis=1) / self.B.r[ib]
        return ea, eb, za, zb

    # ------------------------------------------------------------ refinement
    def refine(self, R, t, pairs, rods_ab):
        ia = np.array([p[0] for p in pairs])
        ib = np.array([p[1] for p in pairs])
        idx = {p: n for n, p in enumerate(pairs)}
        rod_idx = [(idx[p], idx[q]) for p, q in rods_ab if p in idx and q in idx]
        o = self.o

        def res(x):
            Rm = Rotation.from_rotvec(x[:3]).as_matrix()
            tt = x[3:]
            X, _ = self.triangulate(Rm, tt, ia, ib)
            ea, eb, za, zb = self.reproj(Rm, tt, X, ia, ib)
            out = [ea / o.centre_sigma, eb / o.centre_sigma,
                   np.log(np.maximum(za, 1e-3) / self.A.z[ia]) / o.radius_sigma,
                   np.log(np.maximum(zb, 1e-3) / self.B.z[ib]) / o.radius_sigma]
            if rod_idx:
                ri = np.array(rod_idx)
                d = np.linalg.norm(X[ri[:, 0]] - X[ri[:, 1]], axis=1)
                out.append((d - self.L) / (o.rod_sigma * self.L))
            return np.concatenate(out)

        x0 = np.r_[Rotation.from_matrix(R).as_rotvec(), t]
        try:
            sol = least_squares(res, x0, loss="soft_l1", f_scale=2.0, max_nfev=60)
        except ValueError:
            return R, t
        return Rotation.from_rotvec(sol.x[:3]).as_matrix(), sol.x[3:]

    # ------------------------------------------------------------ matching
    def guided(self, R, t):
        """One-to-one correspondences consistent with the pose."""
        nA, nB = len(self.A.r), len(self.B.r)
        if not nA or not nB:
            return []
        ia, ib = np.meshgrid(np.arange(nA), np.arange(nB), indexing="ij")
        ia, ib = ia.ravel(), ib.ravel()
        X, ang = self.triangulate(R, t, ia, ib)
        ea, eb, za, zb = self.reproj(R, t, X, ia, ib)
        la = np.abs(np.log(np.maximum(za, 1e-3) / self.A.z[ia]))
        lb = np.abs(np.log(np.maximum(zb, 1e-3) / self.B.z[ib]))
        o = self.o
        ok = (ea < o.reproj_tol) & (eb < o.reproj_tol) & (la < o.depth_tol) & (lb < o.depth_tol) \
            & (za > 0) & (zb > 0)
        cost = np.full(nA * nB, 1e6)
        cost[ok] = (ea[ok] ** 2 + eb[ok] ** 2) / o.centre_sigma ** 2 + \
            (la[ok] ** 2 + lb[ok] ** 2) / o.radius_sigma ** 2
        C = cost.reshape(nA, nB)
        r_, c_ = linear_sum_assignment(C)
        return [(int(i), int(j)) for i, j in zip(r_, c_) if C[i, j] < 1e5]

    def rod_pairs(self, pairs):
        """Rods detected in either photo between matched spheres, as pairs of (ia, ib) keys,
        with colour conflict flag."""
        ma = dict(pairs)
        mb = {j: i for i, j in pairs}
        out = {}
        for key, col in self.A.rods.items():
            i, k = tuple(key)
            if i in ma and k in ma:
                p, q = (i, ma[i]), (k, ma[k])
                colb = self.B.rods.get(frozenset((ma[i], ma[k])))
                out[frozenset((p, q))] = (p, q, colb is not None and colb != col)
        for key, col in self.B.rods.items():
            j, l = tuple(key)
            if j in mb and l in mb:
                p, q = (mb[j], j), (mb[l], l)
                k_ = frozenset((p, q))
                if k_ not in out:
                    out[k_] = (p, q, False)
        return list(out.values())

    def score(self, R, t, pairs):
        if len(pairs) < 3:
            return -1e9, 0, 0, np.zeros((0, 3))
        ia = np.array([p[0] for p in pairs])
        ib = np.array([p[1] for p in pairs])
        X, ang = self.triangulate(R, t, ia, ib)
        idx = {p: n for n, p in enumerate(pairs)}
        ok = bad = 0
        for p, q, conflict in self.rod_pairs(pairs):
            d = np.linalg.norm(X[idx[p]] - X[idx[q]])
            if conflict or abs(d - self.L) > self.o.rod_ok_tol * self.L:
                bad += 1
            else:
                ok += 1
        # non-penetration between matched spheres
        dd = np.linalg.norm(X[:, None] - X[None], axis=2)[np.triu_indices(len(X), 1)]
        clash = int((dd < 0.85 * self.D).sum())
        s = len(pairs) + self.o.rod_weight * ok - self.o.rod_bad_weight * (bad + clash)
        return s, ok, bad + clash, X

    def run_hypothesis(self, R, t, seed_pairs):
        pairs = list(seed_pairs)
        best = None
        for _ in range(self.o.lo_iters):
            rods = [(p, q) for p, q, c in self.rod_pairs(pairs) if not c]
            R, t = self.refine(R, t, pairs, rods)
            new = self.guided(R, t)
            # keep seed if guided matching lost it entirely
            if len(new) < 3:
                new = pairs
            s, ok, bad, X = self.score(R, t, new)
            if best is None or s > best[0]:
                best = (s, R, t, new, ok, bad, X)
            if set(new) == set(pairs):
                break
            pairs = new
        return best


def _init_pose(A: ViewGeom, B: ViewGeom, ia, ib):
    """Rigid alignment of radius-based clouds (photo a -> photo b), then rescale so the
    mean depth is preserved."""
    from .solve import _umeyama
    s, R, t = _umeyama(A.P[ia], B.P[ib], with_scale=False)
    return R, t


def match_two_views(da: ViewDetections, db: ViewDetections, fa, ca, fb, cb, pitch, D,
                    a: int, b: int, opts: TwoViewOptions | None = None,
                    sphere_radius_mm: float = 6.35, top_k: int = 5) -> list[TwoViewResult]:
    """Top-k distinct relative-pose hypotheses for photos a and b, best first."""
    opts = opts or TwoViewOptions()
    A = ViewGeom(da, fa, ca, sphere_radius_mm)
    B = ViewGeom(db, fb, cb, sphere_radius_mm)
    tv = TwoView(A, B, pitch, D, opts)
    by_col: dict[tuple[str, str], list] = {}
    for p in B.paths:
        by_col.setdefault((p[3], p[4]), []).append(p)
    results: dict[frozenset, TwoViewResult] = {}
    n = 0
    for (i, j, k, ci, ck) in A.paths:
        for (i2, j2, k2, _, _) in by_col.get((ci, ck), []):
            n += 1
            if n > opts.max_hypotheses:
                break
            seed = [(i, i2), (j, j2), (k, k2)]
            R, t = _init_pose(A, B, [i, j, k], [i2, j2, k2])
            out = tv.run_hypothesis(R, t, seed)
            if out is None:
                continue
            s, R, t, pairs, ok, bad, X = out
            key = frozenset(pairs)
            if key not in results or s > results[key].score:
                results[key] = TwoViewResult(a, b, R, t, pairs, s, ok, bad, X)
    ranked = sorted(results.values(), key=lambda r: -r.score)
    return ranked[:top_k]
