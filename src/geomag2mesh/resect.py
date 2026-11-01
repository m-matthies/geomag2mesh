"""Camera resection against a partial 3-D model with unknown correspondences.

Given model spheres (3-D centres + rods with colours) and one photo's detections
(centres, radii, rods with colours), find the camera pose. Hypotheses come from
P3P on triples: three detections joined by detected rods (a path i-j-k) matched
to model rod paths with the same colours; when the photo has too few rod paths,
triples of the strongest detections are tried against all model triples.
Each P3P solution is pruned by the apparent radii (depth f R / r within
``depth_tol``) and scored by projecting the whole model:

  score = matched detections (centre within ``reproj_tol`` radii, radius within
          ``radius_tol``; one-to-one)
        + rod_weight x detected rods explained by a model rod of the same colour
        - rod_penalty x detected rods between matched spheres contradicting the model.

The best hypotheses are refined with PnP (LM) on their matches. This uses the
precise quantities (image centres, rods) for scoring; radii only prune.

Complexity: H hypotheses x O(N + n) projection and assignment, with H the number
of colour-compatible (detected path, model path) pairs, typically < 10^4.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import ViewDetections


@dataclass
class ResectOptions:
    reproj_tol: float = 0.35
    radius_tol: float = 0.25
    depth_tol: float = 0.30
    rod_weight: float = 0.7
    rod_penalty: float = 1.0
    max_triples: int = 60          # strongest detections used for rod-less triples
    max_hyp: int = 200000
    tri_tol: float = 0.35          # relative distance tolerance for rod-less triples


@dataclass
class Resection:
    rvec: np.ndarray
    tvec: np.ndarray
    matches: list[tuple[int, int]]   # (detection index, model index)
    score: float
    n_rods_ok: int
    n_rods_bad: int


class Model:
    def __init__(self, X: np.ndarray, rods: list[tuple[int, int, str]]):
        self.X = np.asarray(X, float)
        self.rods = {frozenset((a, b)): c for a, b, c in rods}
        nb: dict[int, list[tuple[int, str]]] = {}
        for a, b, c in rods:
            nb.setdefault(a, []).append((b, c))
            nb.setdefault(b, []).append((a, c))
        self.paths = [(i, j, k, ci, ck) for j, lst in nb.items()
                      for (i, ci), (k, ck) in itertools.permutations(lst, 2)]


def _score(model: Model, det: ViewDetections, K: np.ndarray, rv, tv, R_mm: float, o: ResectOptions):
    pts, _ = cv2.projectPoints(model.X, rv, tv, K, None)
    pts = pts.reshape(-1, 2)
    Rm = cv2.Rodrigues(rv)[0]
    z = (model.X @ Rm.T + tv.ravel())[:, 2]
    if (z <= 0).any():
        return -1e9, [], 0, 0
    rp = K[0, 0] * R_mm / z
    xy = np.array([[s.x, s.y] for s in det.spheres])
    r = np.array([s.r for s in det.spheres])
    d = np.linalg.norm(xy[:, None] - pts[None], axis=2) / r[:, None]
    rr = np.abs(r[:, None] / rp[None] - 1)
    ok = (d < o.reproj_tol) & (rr < o.radius_tol)
    if not ok.any():
        return 0.0, [], 0, 0
    C = np.where(ok, d + rr, 1e6)
    ri, ci = linear_sum_assignment(C)
    m = [(int(i), int(j)) for i, j in zip(ri, ci) if C[i, j] < 1e5]
    mm = dict(m)
    good = bad = 0
    for rd in det.rods:
        if rd.i in mm and rd.j in mm:
            col = model.rods.get(frozenset((mm[rd.i], mm[rd.j])))
            if col is not None and col == rd.color:
                good += 1
            else:
                bad += 1
    return len(m) + o.rod_weight * good - o.rod_penalty * bad, m, good, bad


def resect(model: Model, det: ViewDetections, focal_px: float, principal, R_mm: float,
           opts: ResectOptions | None = None, top_k: int = 5) -> list[Resection]:
    o = opts or ResectOptions()
    if len(det.spheres) < 3 or len(model.X) < 3:
        return []
    K = np.array([[focal_px, 0, principal[0]], [0, focal_px, principal[1]], [0, 0, 1.0]])
    xy = np.array([[s.x, s.y] for s in det.spheres], float)
    zr = focal_px * R_mm / np.array([s.r for s in det.spheres])
    # detection rod paths
    nb: dict[int, list[tuple[int, str]]] = {}
    for rd in det.rods:
        nb.setdefault(rd.i, []).append((rd.j, rd.color))
        nb.setdefault(rd.j, []).append((rd.i, rd.color))
    dpaths = [(i, j, k, ci, ck) for j, lst in nb.items() for (i, ci), (k, ck) in itertools.permutations(lst, 2)]
    by_col: dict[tuple[str, str], list] = {}
    for p in model.paths:
        by_col.setdefault((p[3], p[4]), []).append(p)
    pairs = []
    for (i, j, k, ci, ck) in dpaths:
        for (a, b, c, _, _) in by_col.get((ci, ck), []):
            pairs.append(((i, j, k), (a, b, c)))
    # rod-less fallback: strongest detections x all model triples (ordered), pruned by
    # consistent ordering of image distances is not possible; rely on radius pruning
    if len(pairs) < 50:
        # rod-less triples, pruned by pairwise distances: radius-based 3-D distances of
        # the detections must match the model's within ``tri_tol`` (depth scatter ~10 %)
        strong = sorted(range(len(det.spheres)), key=lambda i: -det.spheres[i].score)[:8]
        P = np.c_[(xy - np.asarray(principal)) / focal_px * zr[:, None], zr]
        Dm = np.linalg.norm(model.X[:, None] - model.X[None], axis=2)
        np.fill_diagonal(Dm, -1e9)
        for tri in itertools.combinations(strong, 3):
            i, j, k = tri
            dij, dik, djk = (np.linalg.norm(P[i] - P[j]), np.linalg.norm(P[i] - P[k]),
                             np.linalg.norm(P[j] - P[k]))
            tol = lambda d: o.tri_tol * max(d, 20.0)
            M1 = np.abs(Dm - dij) < tol(dij)
            M2 = np.abs(Dm - dik) < tol(dik)
            M3 = np.abs(Dm - djk) < tol(djk)
            cand = np.argwhere(M1[:, :, None] & M2[:, None, :] & M3[None, :, :])
            for a, b, c in cand:
                pairs.append((tri, (int(a), int(b), int(c))))
            if len(pairs) > o.max_hyp:
                break
    results: dict[frozenset, Resection] = {}
    for (di, mi) in pairs:
        obj = model.X[list(mi)].astype(np.float64)
        img = xy[list(di)].astype(np.float64)
        try:
            n, rvs, tvs = cv2.solveP3P(obj, img, K, None, flags=cv2.SOLVEPNP_P3P)
        except cv2.error:
            continue
        for rv, tv in zip(rvs, tvs):
            Rm = cv2.Rodrigues(rv)[0]
            zz = (obj @ Rm.T + tv.ravel())[:, 2]
            if (zz <= 0).any() or np.max(np.abs(np.log(zz / zr[list(di)]))) > o.depth_tol:
                continue
            s, m, g, b = _score(model, det, K, rv, tv, R_mm, o)
            if len(m) < 4:
                continue
            key = frozenset(m)
            if key not in results or s > results[key].score:
                results[key] = Resection(rv.ravel(), tv.ravel(), m, s, g, b)
    # refine the best by PnP on their matches, rescore
    out = []
    for res in sorted(results.values(), key=lambda r: -r.score)[: 3 * top_k]:
        di = [a for a, _ in res.matches]
        mi = [b for _, b in res.matches]
        rv, tv = cv2.solvePnPRefineLM(model.X[mi].astype(np.float64), xy[di].astype(np.float64), K, None,
                                      res.rvec.reshape(3, 1).copy(), res.tvec.reshape(3, 1).copy())
        s, m, g, b = _score(model, det, K, rv, tv, R_mm, o)
        out.append(Resection(rv.ravel(), tv.ravel(), m, s, g, b) if s >= res.score else res)
    dedup: dict[frozenset, Resection] = {}
    for r in out:
        k = frozenset(r.matches)
        if k not in dedup or r.score > dedup[k].score:
            dedup[k] = r
    return sorted(dedup.values(), key=lambda r: -r.score)[:top_k]
