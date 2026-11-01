"""Multi-view reconstruction of sphere centres under rigid-part constraints.

Model
-----
Unknowns: sphere centres X (N x 3, mm); per view a pose (rotation vector,
translation); one focal length per camera group; a principal point for every
view whose principal point is not known (cropped images).

Residuals (all dimensionless, i.e. divided by their standard deviation):

* reprojection of observed sphere centres (sigma = ``sigma_angle`` x focal length,
  i.e. an angular error, invariant to cropping and resizing),
* apparent sphere radius ``f R / z`` versus measured radius (depth cue),
* rod lengths: |Xa - Xb| = pitch(rod type)  (sigma = ``rod_tol_rel`` x pitch),
* sphere contacts: |Xa - Xb| = sphere diameter,
* non-penetration: |Xa - Xb| >= 0.97 D for every other pair (one-sided),
* support plane (when the project declares the mat frame): z >= R per sphere (one-sided),
* focal-length prior, weak centroid gauge prior.

Initialisation needs no template: in every view the depth of each sphere
follows from its apparent radius (z = f R / r), giving a metric point cloud in
camera coordinates; clouds are registered across views by a similarity
(Umeyama) fit on shared labels, rescaled so median rod length equals the
pitch, and camera poses are re-estimated by PnP.

Complexity: one Gauss-Newton/trust-region step costs O(m p_eff^2) with m
residuals and the Jacobian's sparse column groups; for N spheres the
non-penetration term contributes O(N^2) residuals (pairs are restricted to a
neighbourhood once N > 150).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .annotations import Project, Rod
from .images import CameraPrior


# --------------------------------------------------------------------------- options/results
@dataclass
class SolveOptions:
    sigma_angle: float = 0.003        # sphere-centre localisation std / focal length (rad, ~0.17 deg; 9 px at f = 3000 px)
    radius_sigma_rel: float = 0.10    # apparent-radius std / radius
    radius_weight: float = 1.0        # set 0 to ignore radii in the optimisation
    rod_tol_rel: float = 0.01         # rod length std / pitch
    contact_tol_mm: float = 0.4
    contact_detect_rel: float = 0.10  # auto contact if |d| < D (1 + this)
    nonpenetration: bool = True
    loss: str = "soft_l1"
    loss_scale: float = 3.0
    restarts: int = 3
    outlier_sigma: float = 6.0        # drop observations above this after a first pass
    leave_one_out: bool = False
    seed: int = 0
    max_nfev: int = 400  # function evaluations per run; well-posed problems converge in < 100


@dataclass
class ViewCamera:
    name: str
    rvec: np.ndarray
    tvec: np.ndarray
    focal_px: float
    principal: np.ndarray
    width: int
    height: int

    @property
    def R(self) -> np.ndarray:
        return Rotation.from_rotvec(self.rvec).as_matrix()

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xc = X @ self.R.T + self.tvec
        return self.focal_px * Xc[:, :2] / Xc[:, 2:3] + self.principal, Xc[:, 2]


@dataclass
class Solution:
    labels: list[str]
    X: np.ndarray                        # (N, 3) mm
    cameras: list[ViewCamera]
    rods: list[Rod]
    contacts: list[tuple[str, str]]
    sphere_diameter_mm: float
    pitch_by_type: dict[str, float]
    stats: dict = field(default_factory=dict)

    def index(self, label: str) -> int:
        return self.labels.index(label)

    def up_hint(self) -> np.ndarray:
        """World direction that the cameras saw as image-up (mean over views)."""
        ups = [c.R.T @ np.array([0.0, -1.0, 0.0]) for c in self.cameras]
        u = np.mean(ups, axis=0)
        return u / np.linalg.norm(u)

    def to_dict(self) -> dict:
        return {
            "format": "geomag2mesh/solution@1",
            "units": "mm",
            "sphere_diameter_mm": self.sphere_diameter_mm,
            "pitch_by_type": self.pitch_by_type,
            "centres": {k: self.X[i].round(4).tolist() for i, k in enumerate(self.labels)},
            "rods": [{"a": r.a, "b": r.b, "color": r.color, "type": r.type} for r in self.rods],
            "contacts": [list(c) for c in self.contacts],
            "cameras": [{"name": c.name, "rvec": c.rvec.tolist(), "tvec": c.tvec.tolist(),
                         "focal_px": c.focal_px, "principal": c.principal.tolist(),
                         "width": c.width, "height": c.height} for c in self.cameras],
            "stats": self.stats,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Solution":
        d = json.loads(Path(path).read_text())
        labels = list(d["centres"])
        cams = [ViewCamera(c["name"], np.array(c["rvec"]), np.array(c["tvec"]), c["focal_px"],
                           np.array(c["principal"]), c["width"], c["height"]) for c in d["cameras"]]
        return cls(labels, np.array([d["centres"][k] for k in labels]), cams,
                   [Rod(r["a"], r["b"], r.get("color", "unknown"), r.get("type")) for r in d["rods"]],
                   [tuple(c) for c in d["contacts"]], d["sphere_diameter_mm"],
                   d["pitch_by_type"], d.get("stats", {}))


# --------------------------------------------------------------------------- problem assembly
class _Problem:
    """Index bookkeeping and the residual function for one configuration."""

    def __init__(self, project: Project, priors: dict[str, CameraPrior], opts: SolveOptions,
                 rod_targets: np.ndarray, contacts: list[tuple[str, str]],
                 active_views: list[int] | None = None, rod_sigma: np.ndarray | None = None):
        self.p, self.opts = project, opts
        self.labels = project.labels
        self.lid = {k: i for i, k in enumerate(self.labels)}
        self.N = len(self.labels)
        self.V = len(project.views)
        self.active = set(range(self.V)) if active_views is None else set(active_views)
        D = project.parts.sphere_diameter_mm
        self.R_mm = D / 2

        # camera groups (shared focal length) and per-view unknown principal points
        self.priors = [priors[v.name] for v in project.views]
        gids: dict[str, int] = {}
        self.group = np.array([gids.setdefault(pr.camera_id, len(gids)) for pr in self.priors])
        self.G = len(gids)
        self.f0 = np.array([np.mean([pr.focal_px for pr, g in zip(self.priors, self.group) if g == k])
                            for k in range(self.G)])
        self.fsig = np.array([max(pr.focal_sigma_rel for pr, g in zip(self.priors, self.group) if g == k)
                              for k in range(self.G)])
        self.pp_view = [i for i, pr in enumerate(self.priors) if not pr.principal_known]

        # observations
        rows = []
        for vi, v in enumerate(project.views):
            if vi not in self.active:
                continue
            sig = opts.sigma_angle * self.priors[vi].focal_px
            for k, o in v.spheres.items():
                rows.append((vi, self.lid[k], o.x, o.y, np.nan if o.r is None else o.r, o.w, sig))
        a = np.array(rows, dtype=float).reshape(-1, 7)
        self.ov, self.op = a[:, 0].astype(int), a[:, 1].astype(int)
        self.ouv, self.orad, self.ow, self.osig = a[:, 2:4], a[:, 4], a[:, 5], a[:, 6]
        self.has_r = ~np.isnan(self.orad) & (opts.radius_weight > 0)

        self.rod_ij = np.array([(self.lid[r.a], self.lid[r.b]) for r in project.rods], int).reshape(-1, 2)
        self.rod_len = rod_targets
        # rods of undetermined type ("auto") are only weakly tied to a length in pass 1
        self.rod_sig = opts.rod_tol_rel * rod_targets if rod_sigma is None else rod_sigma
        self.con_ij = np.array([(self.lid[a], self.lid[b]) for a, b in contacts], int).reshape(-1, 2)
        self.D = D
        bonded = {frozenset(map(int, e)) for e in self.rod_ij} | {frozenset(map(int, e)) for e in self.con_ij}
        self.np_ij = np.array([(i, j) for i in range(self.N) for j in range(i + 1, self.N)
                               if frozenset((i, j)) not in bonded], int).reshape(-1, 2) \
            if opts.nonpenetration else np.zeros((0, 2), int)

        # parameter layout
        self.n_x = 3 * self.N
        self.n_cam = 6 * self.V
        self.off_cam = self.n_x
        self.off_f = self.off_cam + self.n_cam
        self.off_pp = self.off_f + self.G
        self.n = self.off_pp + 2 * len(self.pp_view)
        self.pp_slot = {vi: k for k, vi in enumerate(self.pp_view)}
        # fixed principal points: explicit (known crop offset) or the image centre
        self._centres = np.array([pr.principal if pr.principal is not None else (v.width / 2.0, v.height / 2.0)
                                  for pr, v in zip(self.priors, project.views)], float)
        boxed = [(k, vi) for k, vi in enumerate(self.pp_view) if self.priors[vi].principal_box is not None]
        self.pp_box_slots = np.array([k for k, _ in boxed], int)
        self.pp_box_views = np.array([vi for _, vi in boxed], int)
        self.pp_box_lo = np.array([self.priors[vi].principal_box[0] for _, vi in boxed], float).reshape(-1, 2)
        self.pp_box_hi = np.array([self.priors[vi].principal_box[1] for _, vi in boxed], float).reshape(-1, 2)
        for vi in self.pp_view:  # unknown principal points start from their hint (or centre)
            if self.priors[vi].principal_init is not None:
                self._centres[vi] = self.priors[vi].principal_init
        self.pose_views = np.array([i for i, pr in enumerate(self.priors)
                                    if pr.has_pose and i in self.active], int)
        self.pose_r0 = np.array([self.priors[i].rvec for i in self.pose_views], float).reshape(-1, 3)
        self.pose_t0 = np.array([self.priors[i].tvec for i in self.pose_views], float).reshape(-1, 3)
        self.pose_sig_deg = np.array([self.priors[i].pose_sigma_deg for i in self.pose_views], float)
        self.pose_sig_mm = np.array([self.priors[i].pose_sigma_mm for i in self.pose_views], float)
        # with mat poses the world frame is the mat (z up); spheres rest on or above it
        self.support_plane = bool(len(self.pose_views)) and project.support_plane

    # ------------------------------------------------------------------ helpers
    def unpack(self, q: np.ndarray):
        X = q[:self.n_x].reshape(self.N, 3)
        cams = q[self.off_cam:self.off_f].reshape(self.V, 6)
        f = q[self.off_f:self.off_pp]
        pp = q[self.off_pp:].reshape(-1, 2)
        return X, cams, f, pp

    def principal(self, vi: int, pp: np.ndarray) -> np.ndarray:
        if vi in self.pp_slot:
            return pp[self.pp_slot[vi]]
        return self._centres[vi].copy()

    def pp_init(self, vi: int) -> tuple[float, float]:
        """Starting value of an unknown principal point (prior hint or image centre)."""
        pr = self.priors[vi]
        if pr.principal_init is not None:
            return tuple(pr.principal_init)
        v = self.p.views[vi]
        return (v.width / 2.0, v.height / 2.0)

    def principal_all(self, pp: np.ndarray) -> np.ndarray:
        """(V, 2) principal points, vectorised for the residual function."""
        c = self._centres.copy()
        if len(self.pp_view):
            c[self.pp_view] = pp
        return c

    def pack(self, X, cams, f, pp) -> np.ndarray:
        return np.concatenate([X.ravel(), cams.ravel(), f, pp.ravel()])

    # ---------------------------------------------------------------- residuals
    def residuals(self, q: np.ndarray) -> np.ndarray:
        X, cams, f, pp = self.unpack(q)
        R = Rotation.from_rotvec(cams[:, :3]).as_matrix()          # (V,3,3)
        Xc = np.einsum("nij,nj->ni", R[self.ov], X[self.op]) + cams[self.ov, 3:]
        z = Xc[:, 2]
        fo = f[self.group[self.ov]]
        c = self.principal_all(pp)[self.ov]
        zs = np.where(z > 1e-6, z, 1e-6)
        uv = fo[:, None] * Xc[:, :2] / zs[:, None] + c
        out = [((uv - self.ouv) * (self.ow / self.osig)[:, None]).ravel(),
               np.minimum(z - 1.0, 0.0) * 10.0]
        if self.has_r.any():
            m = self.has_r
            rpred = fo[m] * self.R_mm / zs[m]
            out.append(self.opts.radius_weight * self.ow[m] * (rpred - self.orad[m])
                       / (self.opts.radius_sigma_rel * self.orad[m]))
        if len(self.rod_ij):
            d = np.linalg.norm(X[self.rod_ij[:, 0]] - X[self.rod_ij[:, 1]], axis=1)
            out.append((d - self.rod_len) / self.rod_sig)
        if len(self.con_ij):
            d = np.linalg.norm(X[self.con_ij[:, 0]] - X[self.con_ij[:, 1]], axis=1)
            out.append((d - self.D) / self.opts.contact_tol_mm)
        if len(self.np_ij):
            d = np.linalg.norm(X[self.np_ij[:, 0]] - X[self.np_ij[:, 1]], axis=1)
            out.append(np.minimum(d - 0.97 * self.D, 0.0) / self.opts.contact_tol_mm)
        if self.support_plane:  # mat frame: nothing passes through the mat (z = 0, z up)
            out.append(np.minimum(X[:, 2] - self.R_mm, 0.0) / self.opts.contact_tol_mm)
        out.append((f - self.f0) / (self.fsig * self.f0))
        if len(self.pp_box_views):  # soft walls keep unknown principal points inside the crop range
            c = pp[self.pp_box_slots]
            below = np.minimum(c - self.pp_box_lo, 0)
            above = np.maximum(c - self.pp_box_hi, 0)
            out.append(((below + above) / (0.02 * self.f0[self.group[self.pp_box_views]][:, None])).ravel())
        if len(self.pose_views):
            pv = self.pose_views
            dr = (Rotation.from_rotvec(cams[pv, :3]) * Rotation.from_rotvec(self.pose_r0).inv()).as_rotvec()
            out.append((dr / np.radians(self.pose_sig_deg)[:, None]).ravel())
            out.append(((cams[pv, 3:] - self.pose_t0) / self.pose_sig_mm[:, None]).ravel())
        else:
            out.append(X.mean(0) * 1e-3)  # weak gauge prior when no pose fixes the frame
        return np.concatenate(out)

    def sparsity(self) -> lil_matrix:
        m = len(self.residuals(self._dummy()))
        S = lil_matrix((m, self.n), dtype=np.int8)
        r = 0

        def xcols(i):
            return range(3 * i, 3 * i + 3)

        def camcols(v):
            return list(range(self.off_cam + 6 * v, self.off_cam + 6 * v + 6)) + \
                [self.off_f + self.group[v]]

        def ppcols(v):
            return range(self.off_pp + 2 * self.pp_slot[v], self.off_pp + 2 * self.pp_slot[v] + 2) \
                if v in self.pp_slot else []

        for k in range(len(self.ov)):
            cols = list(xcols(self.op[k])) + camcols(self.ov[k]) + list(ppcols(self.ov[k]))
            for rr in (r, r + 1):
                S[rr, cols] = 1
            r += 2
        for k in range(len(self.ov)):  # depth positivity
            S[r, list(xcols(self.op[k])) + camcols(self.ov[k])] = 1
            r += 1
        for k in np.where(self.has_r)[0]:
            S[r, list(xcols(self.op[k])) + camcols(self.ov[k])] = 1
            r += 1
        for ij in (self.rod_ij, self.con_ij, self.np_ij):
            for i, j in ij:
                S[r, list(xcols(i)) + list(xcols(j))] = 1
                r += 1
        if self.support_plane:
            for i in range(self.N):
                S[r, 3 * i + 2] = 1
                r += 1
        for g in range(self.G):
            S[r, self.off_f + g] = 1
            r += 1
        for k in self.pp_box_slots:
            for a in range(2):
                S[r, self.off_pp + 2 * int(k) + a] = 1
                r += 1
        if len(self.pose_views):
            for v in self.pose_views:
                for a in range(3):
                    S[r, list(range(self.off_cam + 6 * v, self.off_cam + 6 * v + 3))] = 1
                    r += 1
            for v in self.pose_views:
                for a in range(3):
                    S[r, self.off_cam + 6 * v + 3 + a] = 1
                    r += 1
        else:
            for a in range(3):
                S[r, list(range(a, self.n_x, 3))] = 1
                r += 1
        assert r == m, (r, m)
        return S

    def _dummy(self) -> np.ndarray:
        q = np.zeros(self.n)
        X, cams, f, pp = self.unpack(q)
        cams[:, 5] = 1000.0
        f[:] = self.f0
        return self.pack(X, cams, f, pp)


# --------------------------------------------------------------------------- initialisation
def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Similarity transform minimising ||s R src + t - dst||; O(k) for k points."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    d = np.ones(3)
    if np.linalg.det(U @ Vt) < 0:
        d[2] = -1
    R = U @ np.diag(d) @ Vt
    s = (S * d).sum() / (a ** 2).sum() * len(src) if with_scale else 1.0
    return s, R, md - s * R @ ms


def _camera_clouds(prob: _Problem) -> dict[int, dict[int, np.ndarray]]:
    """Per view: label index -> 3-D point in camera coordinates from apparent radius."""
    clouds: dict[int, dict[int, np.ndarray]] = {}
    for vi, v in enumerate(prob.p.views):
        if vi not in prob.active or not v.spheres:
            continue
        f = prob.f0[prob.group[vi]]
        c = prob._centres[vi]
        zs = {k: f * prob.R_mm / o.r for k, o in v.spheres.items() if o.r and o.r > 0}
        zmed = float(np.median(list(zs.values()))) if zs else 20.0 * prob.D * f / max(v.width, v.height)
        pts = {}
        for k, o in v.spheres.items():
            z = zs.get(k, zmed)
            pts[prob.lid[k]] = np.array([(o.x - c[0]) / f * z, (o.y - c[1]) / f * z, z])
        clouds[vi] = pts
    return clouds


def _initialise_posed(prob: _Problem, rng: np.random.Generator, jitter: float) -> np.ndarray:
    """All active views have pose priors: triangulate (>= 2 views) or place along the
    viewing ray at the radius-implied depth (1 view)."""
    cams = np.zeros((prob.V, 6))
    for k, v in enumerate(prob.pose_views):
        cams[v] = np.r_[prob.pose_r0[k], prob.pose_t0[k]]
    X = np.full((prob.N, 3), np.nan)
    for i in range(prob.N):
        m = prob.op == i
        rows = []
        single = None
        for vi, (u, w), r in zip(prob.ov[m], prob.ouv[m], prob.orad[m]):
            f = prob.f0[prob.group[vi]]
            c = prob._centres[vi]
            R = Rotation.from_rotvec(cams[vi, :3]).as_matrix()
            P = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]]) @ np.c_[R, cams[vi, 3:]]
            rows += [u * P[2] - P[0], w * P[2] - P[1]]
            if not np.isnan(r):
                z = f * prob.R_mm / r
                xc = np.array([(u - c[0]) / f * z, (w - c[1]) / f * z, z])
                single = R.T @ (xc - cams[vi, 3:])
        if len(rows) >= 4:
            _, _, Vt = np.linalg.svd(np.array(rows))
            X[i] = Vt[-1, :3] / Vt[-1, 3]
        elif single is not None:
            X[i] = single
    X = _place_unobserved(prob, X, rng)
    if jitter > 0:
        X = X + rng.normal(0, jitter, X.shape)
    pp = np.array([prob.pp_init(v) for v in prob.pp_view]
                  ).reshape(-1, 2)
    return prob.pack(X, cams, prob.f0.copy(), pp)


def _place_unobserved(prob: _Problem, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Place spheres without a position next to placed rod neighbours (iteratively)."""
    nb: dict[int, list[int]] = {i: [] for i in range(prob.N)}
    for i, j in prob.rod_ij:
        nb[i].append(j)
        nb[j].append(i)
    centroid = np.nanmean(X, 0)
    for _ in range(prob.N):
        missing = [i for i in range(prob.N) if np.isnan(X[i]).any()]
        if not missing:
            break
        progressed = False
        for i in missing:
            placed = [j for j in nb[i] if not np.isnan(X[j]).any()]
            if placed:
                m = np.mean(X[placed], 0)
                away = m - centroid
                away = away / (np.linalg.norm(away) + 1e-9)
                X[i] = m + away * 0.5 * prob.rod_len.mean() + rng.normal(0, 2.0, 3)
                progressed = True
        if not progressed:
            for i in missing:
                X[i] = centroid + rng.normal(0, prob.D, 3)
    return X


def _initialise(prob: _Problem, rng: np.random.Generator, jitter: float = 0.0) -> np.ndarray:
    if len(prob.pose_views) and set(prob.pose_views.tolist()) >= prob.active:
        return _initialise_posed(prob, rng, jitter)
    clouds = _camera_clouds(prob)
    if not clouds:
        raise ValueError("no observations")
    order = sorted(clouds, key=lambda v: -len(clouds[v]))
    world: dict[int, list[np.ndarray]] = {k: [p] for k, p in clouds[order[0]].items()}
    registered = {order[0]}
    pending = set(order[1:])
    while pending:
        best = max(pending, key=lambda v: len(set(clouds[v]) & set(world)))
        shared = sorted(set(clouds[best]) & set(world))
        if len(shared) < 3:
            break
        src = np.array([clouds[best][k] for k in shared])
        dst = np.array([np.mean(world[k], 0) for k in shared])
        s, R, t = _umeyama(src, dst)
        s = float(np.clip(s, 0.6, 1.6))
        for k, p in clouds[best].items():
            world.setdefault(k, []).append(s * R @ p + t)
        registered.add(best)
        pending.discard(best)

    X = np.full((prob.N, 3), np.nan)
    for k, ps in world.items():
        X[k] = np.mean(ps, 0)
    # scale so that rods have their nominal length
    if len(prob.rod_ij):
        ok = ~np.isnan(X[prob.rod_ij]).any(axis=(1, 2))
        if ok.any():
            d = np.linalg.norm(X[prob.rod_ij[ok, 0]] - X[prob.rod_ij[ok, 1]], axis=1)
            X *= np.median(prob.rod_len[ok] / np.maximum(d, 1e-9))
    X = _place_unobserved(prob, X, rng)
    if jitter > 0:
        X = X + rng.normal(0, jitter, X.shape)

    # cameras by PnP against the initial structure
    cams = np.zeros((prob.V, 6))
    pp = np.array([prob.pp_init(v) for v in prob.pp_view]
                  ).reshape(-1, 2)
    for vi, v in enumerate(prob.p.views):
        ks = [prob.lid[k] for k in v.spheres]
        if vi not in prob.active or len(ks) < 4:
            cams[vi] = [0, 0, 0, 0, 0, 30 * prob.D]
            continue
        f = prob.f0[prob.group[vi]]
        c = prob._centres[vi]
        K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]], float)
        img = np.array([[o.x, o.y] for o in v.spheres.values()], float)
        ok, rv, tv = cv2.solvePnP(X[ks].astype(float), img, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            rv, tv = cv2.solvePnPRefineLM(X[ks].astype(float), img, K, None, rv, tv)
            cams[vi] = np.r_[rv.ravel(), tv.ravel()]
        else:
            cams[vi] = [0, 0, 0, 0, 0, 30 * prob.D]
    return prob.pack(X, cams, prob.f0.copy(), pp)


# --------------------------------------------------------------------------- driver
def _rod_targets(project: Project, X: np.ndarray | None, lid: dict[str, int]) -> tuple[np.ndarray, list[Rod]]:
    parts = project.parts
    rods, targets = [], []
    for r in project.rods:
        t = r.type
        if t == "auto":
            if X is None:
                t = parts.default_rod
            else:
                t = parts.nearest_rod_type(float(np.linalg.norm(X[lid[r.a]] - X[lid[r.b]])))
        t = t or parts.default_rod
        rods.append(Rod(r.a, r.b, r.color, t))
        targets.append(parts.pitch_mm(t))
    return np.array(targets, float), rods


DENSE_PARAM_LIMIT = 600  # below this, dense finite differences + exact TR steps are fastest


def _run(prob: _Problem, q0: np.ndarray) -> object:
    """Trust-region least squares. Dense (exact SVD-based steps) for small problems,
    which converges in tens of evaluations; sparse finite differences with LSMR steps
    for large frameworks, where dense Jacobians (O(m n) memory, n evaluations each)
    become the bottleneck."""
    kw = dict(loss=prob.opts.loss, f_scale=prob.opts.loss_scale, x_scale="jac",
              max_nfev=prob.opts.max_nfev, method="trf")
    if prob.n > DENSE_PARAM_LIMIT:
        kw.update(jac_sparsity=prob.sparsity(), tr_solver="lsmr")
    return least_squares(prob.residuals, q0, **kw)


def _best_of_restarts(prob: _Problem, opts: SolveOptions, q_warm: np.ndarray | None = None):
    rng = np.random.default_rng(opts.seed)
    best = None
    scale = prob.rod_len.mean() if len(prob.rod_len) else prob.p.parts.pitch_mm()
    starts = [(None, 0.0)] + [(None, 0.12 * scale)] * max(0, opts.restarts - 1)
    if q_warm is not None:
        starts = [(q_warm, 0.0)] + starts[:1]
    for q, jit in starts:
        q0 = q if q is not None else _initialise(prob, rng, jit)
        res = _run(prob, q0)
        if best is None or res.cost < best.cost:
            best = res
    return best


def solve(project: Project, priors: dict[str, CameraPrior], opts: SolveOptions | None = None,
          log=print) -> Solution:
    """Reconstruct sphere centres and cameras from an annotated project."""
    opts = opts or SolveOptions()
    project = copy.deepcopy(project)  # outlier rejection edits observations
    for w in project.validate():
        log(f"warning: {w}")
    labels = project.labels
    lid = {k: i for i, k in enumerate(labels)}
    D = project.parts.sphere_diameter_mm

    # pass 1: user contacts only, rod types default/explicit; "auto" rods get a loose
    # length (30 % of pitch) so that their type can be read off the solved geometry
    targets, rods = _rod_targets(project, None, lid)
    contacts = list(project.contacts)
    auto = np.array([r.type == "auto" for r in project.rods], bool)
    sig1 = opts.rod_tol_rel * targets
    sig1[auto] = 0.3 * targets[auto]
    prob = _Problem(project, priors, opts, targets, contacts, rod_sigma=sig1)
    res = _best_of_restarts(prob, opts)
    X = prob.unpack(res.x)[0]

    # pass 2: rod-type classification, automatic contacts, outlier rejection
    targets, rods = _rod_targets(project, X, lid)
    # contacts are added iteratively: tightening one contact can bring a neighbour into range
    for _ in range(3 if project.auto_contacts else 1):
        added = _detect_contacts(project, labels, X, contacts, D, opts) if project.auto_contacts else []
        contacts += added
        prob = _Problem(project, priors, opts, targets, contacts)
        res = _best_of_restarts(prob, opts, q_warm=res.x)
        X = prob.unpack(res.x)[0]
        if not added:
            break
    dropped, removed = _reject_outliers(prob, res.x, opts)
    if dropped:
        log(f"dropped {len(dropped)} outlier observation(s): " +
            ", ".join(f"{v}:{k}" for v, k in dropped))
    if removed:
        log(f"removed {len(removed)} sphere(s) whose observations are all inconsistent "
            f"(false matches): {', '.join(removed)}")
        labels = project.labels
        lid = {k: i for i, k in enumerate(labels)}
        old_X = dict(zip(prob.labels, prob.unpack(res.x)[0]))
        targets, rods = _rod_targets(project, np.array([old_X[k] for k in labels]), lid)
        contacts = [c for c in contacts if c[0] in lid and c[1] in lid]
        prob = _Problem(project, priors, opts, targets, contacts)
        res = _best_of_restarts(prob, opts)
    elif dropped:
        prob = _Problem(project, priors, opts, targets, contacts)
        res = _run(prob, res.x)

    sol = _to_solution(prob, res.x, rods, contacts)
    sol.stats = _stats(prob, res, rods, contacts)
    sol.stats["frame"] = "marker" if len(prob.pose_views) else "camera"
    sol.stats["dropped_observations"] = [list(d) for d in dropped]
    sol.stats["removed_spheres"] = removed
    if opts.leave_one_out and len(project.views) >= 3:
        sol.stats["leave_one_view_out"] = _leave_one_out(project, priors, opts, targets, contacts, res.x, sol)
    return sol


def _detect_contacts(project: Project, labels, X, contacts, D, opts) -> list[tuple[str, str]]:
    """Unbonded sphere pairs closer than D (1 + contact_detect_rel): magnetic contacts."""
    bonded = {r.key for r in project.rods} | {frozenset(c) for c in contacts}
    out = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if frozenset((labels[i], labels[j])) in bonded:
                continue
            if np.linalg.norm(X[i] - X[j]) < D * (1 + opts.contact_detect_rel):
                out.append((labels[i], labels[j]))
    return out


def _reject_outliers(prob: _Problem, q: np.ndarray, opts: SolveOptions
                     ) -> tuple[list[tuple[str, str]], list[str]]:
    """Remove observations whose reprojection error exceeds ``outlier_sigma``; a sphere
    is kept in at least two photos, unless *all* its observations are outliers, in which
    case it is a false match and is removed with its rods and contacts (mutates project).
    Returns (dropped observations, removed spheres)."""
    r = prob.residuals(q)[: 2 * len(prob.ov)].reshape(-1, 2)
    e = np.linalg.norm(r, axis=1) / np.maximum(prob.ow, 1e-9)
    bad: dict[str, list[str]] = {}
    for k in np.where(e > opts.outlier_sigma)[0]:
        bad.setdefault(prob.labels[prob.op[k]], []).append(prob.p.views[prob.ov[k]].name)
    drop, removed = [], []
    for lab, views in bad.items():
        n_obs = sum(lab in w.spheres for w in prob.p.views)
        if len(views) == n_obs and n_obs <= 2:
            removed.append(lab)
        else:
            keep = max(0, n_obs - 2)
            drop += [(v, lab) for v in views[:keep]]
    for vname, lab in drop:
        for v in prob.p.views:
            if v.name == vname:
                v.spheres.pop(lab, None)
    if removed:
        from .edit import delete_sphere
        for lab in removed:
            delete_sphere(prob.p, lab)
    return drop, removed


def _to_solution(prob: _Problem, q: np.ndarray, rods: list[Rod], contacts) -> Solution:
    X, cams, f, pp = prob.unpack(q)
    cameras = [ViewCamera(v.name, cams[vi, :3].copy(), cams[vi, 3:].copy(), float(f[prob.group[vi]]),
                          prob.principal(vi, pp).copy(), v.width, v.height)
               for vi, v in enumerate(prob.p.views)]
    parts = prob.p.parts
    return Solution(list(prob.labels), X.copy(), cameras, rods, list(contacts), parts.sphere_diameter_mm,
                    {k: parts.pitch_mm(k) for k in parts.rod_types})


def _stats(prob: _Problem, res, rods, contacts) -> dict:
    X, cams, f, pp = prob.unpack(res.x)
    per_view = {}
    for vi, v in enumerate(prob.p.views):
        m = prob.ov == vi
        if not m.any():
            continue
        cam = ViewCamera(v.name, cams[vi, :3], cams[vi, 3:], f[prob.group[vi]], prob.principal(vi, pp),
                         v.width, v.height)
        uv, z = cam.project(X[prob.op[m]])
        err = np.linalg.norm(uv - prob.ouv[m], axis=1)
        rad = f[prob.group[vi]] * prob.R_mm / z
        worst = int(np.argmax(err))
        per_view[v.name] = {
            "n_obs": int(m.sum()),
            "rms_px": float(np.sqrt((err ** 2).mean())),
            "rms_rel_radius": float(np.sqrt(((err / rad) ** 2).mean())),
            "worst": [prob.labels[prob.op[m][worst]], float(err[worst])],
        }
    rod_err = (np.linalg.norm(X[prob.rod_ij[:, 0]] - X[prob.rod_ij[:, 1]], axis=1) - prob.rod_len) \
        if len(prob.rod_ij) else np.zeros(0)
    con_d = [float(np.linalg.norm(X[prob.lid[a]] - X[prob.lid[b]])) for a, b in contacts]
    nobs = prob.p.observation_count()
    return {
        "cost": float(res.cost), "nfev": int(res.nfev), "success": bool(res.success),
        "focal_px": f.tolist(),
        "views": per_view,
        "rod_length_error_mm": {"rms": float(np.sqrt((rod_err ** 2).mean())) if len(rod_err) else 0.0,
                                "max_abs": float(np.abs(rod_err).max()) if len(rod_err) else 0.0},
        "contacts": {f"{a}-{b}": d for (a, b), d in zip(contacts, con_d)},
        "observations_per_sphere": nobs,
    }


def _procrustes_shift(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    s, R, t = _umeyama(B, A, with_scale=False)
    return np.linalg.norm((B @ R.T + t) - A, axis=1)


def _leave_one_out(project, priors, opts, targets, contacts, q, sol) -> dict:
    out = {}
    for vi, v in enumerate(project.views):
        act = [k for k in range(len(project.views)) if k != vi]
        prob = _Problem(project, priors, opts, targets, contacts, active_views=act)
        res = _run(prob, q)
        Xv = prob.unpack(res.x)[0]
        e = _procrustes_shift(sol.X, Xv)
        out[v.name] = {"rms_mm": float(np.sqrt((e ** 2).mean())), "max_mm": float(e.max()),
                       "max_sphere": sol.labels[int(e.argmax())]}
    return out
