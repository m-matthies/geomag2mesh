"""Single-photo 3-D lifting of detected spheres using rod-length constraints.

Each sphere lies on its viewing ray, X_i = z_i b_i (b_i the bearing through its
image centre). The apparent radius gives z_i to ~10 % only, but every detected
rod fixes |X_i - X_j| = pitch to < 1 %. Solving all depths jointly with robust
rod residuals (spurious rods are down-weighted and finally discarded) turns
the noisy radius depths into a metric point cloud accurate to a few mm.

For cropped photos the principal point is unknown; rods in different
directions make it observable (each rod is a small calibration object), so it
is estimated within the range allowed by the original sensor size.

Unknowns: log z_i (n) and optionally the principal point (2). Residuals: radius
depth prior, rod lengths (soft-L1), sphere non-penetration, principal-point prior.
Cost per solve O(iterations x (n + rods + n^2)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .detect import ViewDetections


@dataclass
class LiftOptions:
    radius_sigma: float = 0.10     # relative depth error of the apparent radius
    rod_sigma: float = 0.01        # relative rod length tolerance
    rod_keep_tol: float = 0.08     # rods outside this relative error are discarded
    pp_sigma_rel: float = 0.25     # prior std of an unknown principal point / focal length
    nonpen: float = 0.97           # sphere centres at least this many diameters apart
    loss_scale: float = 2.0


@dataclass
class LiftedView:
    X: np.ndarray            # (n, 3) sphere centres, camera frame, mm
    principal: np.ndarray    # (2,) principal point used
    rods: list               # rods kept (i, j, color)
    rod_err: np.ndarray      # relative length error of kept rods
    cost: float


def lift_view(det: ViewDetections, focal_px: float, principal_guess, principal_known: bool,
              pitch: float, sphere_diameter: float, opts: LiftOptions | None = None,
              pp_bounds: tuple[np.ndarray, np.ndarray] | None = None) -> LiftedView:
    opts = opts or LiftOptions()
    n = len(det.spheres)
    xy = np.array([[s.x, s.y] for s in det.spheres], float).reshape(-1, 2)
    r = np.array([s.r for s in det.spheres], float)
    f = float(focal_px)
    z_r = f * (sphere_diameter / 2) / np.maximum(r, 1e-6)
    c0 = np.asarray(principal_guess, float)
    rods = [(d.i, d.j, d.color) for d in det.rods]
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], int).reshape(-1, 2)

    def unpack(x):
        lz = x[:n]
        c = c0 + (x[n:n + 2] if not principal_known else 0.0)
        return lz, c

    def points(lz, c):
        z = np.exp(lz)
        return np.c_[(xy - c) / f * z[:, None], z]

    def residuals(x, rods_):
        lz, c = unpack(x)
        X = points(lz, c)
        out = [(lz - np.log(z_r)) / opts.radius_sigma]
        if rods_:
            ij = np.array([(a, b) for a, b, _ in rods_])
            d = np.linalg.norm(X[ij[:, 0]] - X[ij[:, 1]], axis=1)
            out.append((d - pitch) / (opts.rod_sigma * pitch))
        if len(pairs):
            bonded = {frozenset((a, b)) for a, b, _ in rods_}
            m = np.array([frozenset(p) not in bonded for p in map(tuple, pairs)])
            d = np.linalg.norm(X[pairs[m, 0]] - X[pairs[m, 1]], axis=1) if m.any() else np.zeros(0)
            out.append(np.minimum(d - opts.nonpen * sphere_diameter, 0) / (0.02 * sphere_diameter))
        if not principal_known:
            out.append(x[n:n + 2] / (opts.pp_sigma_rel * f))
        return np.concatenate(out)

    x0 = np.r_[np.log(z_r), np.zeros(0 if principal_known else 2)]
    lb = np.full(len(x0), -np.inf)
    ub = np.full(len(x0), np.inf)
    if not principal_known and pp_bounds is not None:
        lb[n:] = pp_bounds[0] - c0
        ub[n:] = pp_bounds[1] - c0
        x0[n:] = np.clip(x0[n:], lb[n:] + 1e-6, ub[n:] - 1e-6)
    kept = list(rods)
    sol = None
    for _ in range(3):  # robust solve, discard inconsistent rods, re-solve
        sol = least_squares(residuals, x0, args=(kept,), loss="soft_l1", f_scale=opts.loss_scale,
                            bounds=(lb, ub), max_nfev=300)
        lz, c = unpack(sol.x)
        X = points(lz, c)
        errs = np.array([abs(np.linalg.norm(X[a] - X[b]) - pitch) / pitch for a, b, _ in kept])
        new = [rd for rd, e in zip(kept, errs) if e <= opts.rod_keep_tol]
        x0 = sol.x
        if len(new) == len(kept):
            break
        kept = new
    lz, c = unpack(sol.x)
    X = points(lz, c)
    errs = np.array([(np.linalg.norm(X[a] - X[b]) - pitch) / pitch for a, b, _ in kept])
    return LiftedView(X, c, kept, errs, float(sol.cost))
