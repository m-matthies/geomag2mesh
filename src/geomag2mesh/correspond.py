"""Cross-view correspondence: turn per-photo detections into a labelled project.

Principle. A steel sphere of known diameter D at depth z has apparent radius
r ~ f D / (2 z), so every photo yields a *metric* 3-D point cloud in its own
camera frame. Two photos of the same rigid framework are then related by a
rigid motion (up to a small scale error from the focal-length prior). The
motion is found by RANSAC over rod-path triples (i - j - k joined by two rods):
a triple in photo A is paired with a triple in photo B when the rod colours
agree and the i-k distances are compatible; each hypothesis is scored by the
number of mutually nearest points within a tolerance plus the number of
colour-consistent rods it explains. Pairwise matches are merged into global
sphere tracks with union-find, refusing merges that would put two spheres of
the same photo into one track.

Complexity. With T rod-path triples per photo the hypothesis count per photo
pair is O(T^2) (typically a few thousand); each costs O(n log n) for n
spheres (KD-tree nearest neighbours).

Failure modes. Strongly symmetric frameworks (a regular cube photographed so
that only rod colours distinguish faces) admit several equally good
alignments; few detections per photo (< 5) give no reliable hypotheses;
radius-based depth degrades for spheres with large occlusion.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .annotations import Obs, Project, Rod, View
from .detect import ViewDetections
from .images import CameraPrior
from .parts import PartCatalog
from .solve import _umeyama


@dataclass
class MatchOptions:
    triple_tol_rel: float = 0.30     # x pitch: i-k distance agreement of two triples
    min_inliers: int = 4
    scale_bounds: tuple[float, float] = (0.8, 1.25)
    rod_bonus: float = 0.5
    color_conflict_penalty: float = 1.0
    max_hypotheses: int = 20000
    radius_sigma_rel: float = 0.06   # relative error of apparent radius -> depth error
    lateral_sigma_rel: float = 0.15  # x sphere radius: centre localisation error in mm
    max_mahal: float = 3.0
    search_rel: float = 1.0          # x pitch: Euclidean pre-selection radius


@dataclass
class PairMatch:
    a: int
    b: int
    pairs: list[tuple[int, int]]     # (sphere index in view a, sphere index in view b)
    score: float
    transform: tuple[float, np.ndarray, np.ndarray]


def camera_cloud(det: ViewDetections, prior: CameraPrior, D: float) -> np.ndarray:
    """Metric 3-D sphere centres in the camera frame (principal point assumed central)."""
    f = prior.focal_px
    c = np.array([det.width / 2.0, det.height / 2.0])
    xy = np.array([[s.x, s.y] for s in det.spheres]).reshape(-1, 2)
    r = np.array([s.r for s in det.spheres])
    z = f * (D / 2) / r
    return np.c_[(xy - c) / f * z[:, None], z]


def _triples(det: ViewDetections) -> list[tuple[int, int, int, str, str]]:
    """Rod paths i - j - k with colours (c_ij, c_jk); both orientations included."""
    nb: dict[int, list[tuple[int, str]]] = {}
    for r in det.rods:
        nb.setdefault(r.i, []).append((r.j, r.color))
        nb.setdefault(r.j, []).append((r.i, r.color))
    out = []
    for j, lst in nb.items():
        for (i, ci), (k, ck) in itertools.permutations(lst, 2):
            out.append((i, j, k, ci, ck))
    return out


def _rod_set(det: ViewDetections) -> dict[frozenset, str]:
    return {frozenset((r.i, r.j)): r.color for r in det.rods}


def ray_covariances(P: np.ndarray, radius_sigma_rel: float, lateral_sigma_mm: float) -> np.ndarray:
    """Covariance of radius-based 3-D points: elongated along each viewing ray
    (depth std = z x relative radius error), isotropic lateral std otherwise."""
    u = P / np.linalg.norm(P, axis=1, keepdims=True)
    sz = P[:, 2] * radius_sigma_rel
    I = np.eye(3)[None]
    return lateral_sigma_mm ** 2 * I + (sz ** 2 - lateral_sigma_mm ** 2)[:, None, None] * u[:, :, None] * u[:, None, :]


def _evaluate(Pa, Pb, Ca, Cb, tree_b, s, R, t, search, rods_a, rods_b, opts):
    """Score an alignment: mutually best Mahalanobis matches within ``opts.max_mahal``."""
    Q = s * Pa @ R.T + t
    CaR = (s ** 2) * np.einsum("ij,njk,lk->nil", R, Ca, R)
    cand = tree_b.query_ball_point(Q, r=search)
    best_a: dict[int, tuple[float, int]] = {}
    best_b: dict[int, tuple[float, int]] = {}
    for i, js in enumerate(cand):
        for j in js:
            d = Q[i] - Pb[j]
            m2 = float(d @ np.linalg.solve(CaR[i] + Cb[j], d))
            if m2 <= opts.max_mahal ** 2:
                if i not in best_a or m2 < best_a[i][0]:
                    best_a[i] = (m2, j)
                if j not in best_b or m2 < best_b[j][0]:
                    best_b[j] = (m2, i)
    pairs = [(i, j) for i, (_, j) in best_a.items() if best_b[j][1] == i]
    m = dict(pairs)
    good = bad = 0
    for key, col in rods_a.items():
        i, k = tuple(key)
        if i in m and k in m:
            cb = rods_b.get(frozenset((m[i], m[k])))
            if cb is not None:
                good += cb == col
                bad += cb != col
    score = len(pairs) + opts.rod_bonus * good - opts.color_conflict_penalty * bad
    return score, pairs


def match_pair(da: ViewDetections, db: ViewDetections, Pa: np.ndarray, Pb: np.ndarray,
               pitch: float, opts: MatchOptions, a: int, b: int,
               sphere_radius_mm: float = 6.35) -> PairMatch | None:
    """Best rigid(+scale) alignment of view a onto view b. See module docstring."""
    if len(Pa) < 3 or len(Pb) < 3:
        return None
    ta, tb = _triples(da), _triples(db)
    tree_b = cKDTree(Pb)
    rods_a, rods_b = _rod_set(da), _rod_set(db)
    lat = opts.lateral_sigma_rel * sphere_radius_mm
    Ca = ray_covariances(Pa, opts.radius_sigma_rel, lat)
    Cb = ray_covariances(Pb, opts.radius_sigma_rel, lat)
    search = opts.search_rel * pitch
    best: PairMatch | None = None
    # index triples of view b by colour pair for O(1) candidate lookup
    by_col: dict[tuple[str, str], list] = {}
    for tr in tb:
        by_col.setdefault((tr[3], tr[4]), []).append(tr)
    n_h = 0
    for (i, j, k, ci, ck) in ta:
        dik = np.linalg.norm(Pa[i] - Pa[k])
        for (i2, j2, k2, _, _) in by_col.get((ci, ck), []):
            if abs(np.linalg.norm(Pb[i2] - Pb[k2]) - dik) > opts.triple_tol_rel * pitch:
                continue
            n_h += 1
            if n_h > opts.max_hypotheses:
                break
            s, R, t = _umeyama(Pa[[i, j, k]], Pb[[i2, j2, k2]])
            if not opts.scale_bounds[0] <= s <= opts.scale_bounds[1]:
                continue
            score, pairs = _evaluate(Pa, Pb, Ca, Cb, tree_b, s, R, t, search, rods_a, rods_b, opts)
            if len(pairs) >= 3 and (best is None or score > best.score):
                best = PairMatch(a, b, pairs, score, (s, R, t))
    if best is None:
        return None
    # refine on inliers (iterated closest point with fixed tolerance)
    for _ in range(3):
        ia = [p[0] for p in best.pairs]
        ib = [p[1] for p in best.pairs]
        s, R, t = _umeyama(Pa[ia], Pb[ib])
        s = float(np.clip(s, *opts.scale_bounds))
        score, pairs = _evaluate(Pa, Pb, Ca, Cb, tree_b, s, R, t, search, rods_a, rods_b, opts)
        if score < best.score:
            break
        best = PairMatch(a, b, pairs, score, (s, R, t))
    return best if len(best.pairs) >= opts.min_inliers else None


class _UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def members(self):
        groups: dict = {}
        for x in list(self.parent):
            groups.setdefault(self.find(x), []).append(x)
        return groups


def build_project(dets: list[ViewDetections], priors: dict[str, CameraPrior], parts: PartCatalog,
                  files: dict[str, str] | None = None, opts: MatchOptions | None = None,
                  log=print) -> Project:
    """Match detections across photos and emit an annotation project with track labels."""
    opts = opts or MatchOptions()
    D = parts.sphere_diameter_mm
    pitch = parts.pitch_mm()
    clouds = [camera_cloud(d, priors[d.name], D) for d in dets]

    matches: list[PairMatch] = []
    for a, b in itertools.combinations(range(len(dets)), 2):
        m = match_pair(dets[a], dets[b], clouds[a], clouds[b], pitch, opts, a, b, D / 2)
        if m:
            matches.append(m)
            log(f"match {dets[a].name} <-> {dets[b].name}: {len(m.pairs)} spheres, score {m.score:.1f}")
        else:
            log(f"match {dets[a].name} <-> {dets[b].name}: no consistent alignment")

    # merge strongest matches first; refuse merges that join two spheres of one view
    uf = _UnionFind()
    for v, d in enumerate(dets):
        for i in range(len(d.spheres)):
            uf.find((v, i))
    views_of: dict = {}

    def views_in(root):
        return views_of.setdefault(root, {root[0]})

    for m in sorted(matches, key=lambda m: -m.score):
        for i, j in m.pairs:
            ra, rb = uf.find((m.a, i)), uf.find((m.b, j))
            if ra == rb:
                continue
            va, vb = views_in(ra), views_in(rb)
            if va & vb:
                continue  # conflict: would merge two spheres from the same photo
            uf.parent[rb] = ra
            views_of[ra] = va | vb
            views_of.pop(rb, None)

    groups = uf.members()
    # rods per track pair, colour by vote
    track_of = {m: k for k, ms in groups.items() for m in ms}
    rod_votes: dict[frozenset, dict[str, float]] = {}
    for v, d in enumerate(dets):
        for r in d.rods:
            ta, tb = track_of[(v, r.i)], track_of[(v, r.j)]
            if ta != tb:
                rod_votes.setdefault(frozenset((ta, tb)), {}).setdefault(r.color, 0.0)
                rod_votes[frozenset((ta, tb))][r.color] += r.score
    # keep tracks seen in >= 2 views, or in one view with a rod to a multi-view track
    multi = {k for k, ms in groups.items() if len(ms) >= 2}
    keep = set(multi)
    for key in rod_votes:
        a, b = tuple(key)
        if a in multi or b in multi:
            keep |= {a, b}
    ordered = sorted(keep, key=lambda k: (-len(groups[k]), k))
    label = {k: f"S{n + 1:02d}" for n, k in enumerate(ordered)}

    views = []
    for v, d in enumerate(dets):
        sph = {}
        for i, s in enumerate(d.spheres):
            k = track_of[(v, i)]
            if k in label:
                sph[label[k]] = Obs(s.x, s.y, s.r, float(np.clip(s.score / 0.6, 0.3, 1.0)))
        views.append(View(d.name, (files or {}).get(d.name), d.width, d.height, sph))
    rods = []
    for key, votes in rod_votes.items():
        a, b = tuple(key)
        if a in label and b in label:
            rods.append(Rod(label[a], label[b], max(votes, key=votes.get), "auto"))
    log(f"tracks: {len(label)} spheres ({len(multi & keep)} seen in >= 2 photos), {len(rods)} rods")
    return Project(views=views, rods=rods, auto_contacts=True, parts=parts)


def verify_rods(project: Project, solution, tol_rel: float = 0.15) -> list[Rod]:
    """Rods whose solved length differs from every rod pitch by more than tol_rel."""
    pitches = project.parts.active_pitches()
    bad = []
    for r in project.rods:
        if r.a not in solution.labels or r.b not in solution.labels:
            continue
        d = float(np.linalg.norm(solution.X[solution.index(r.a)] - solution.X[solution.index(r.b)]))
        if min(abs(d - p) / p for p in pitches) > tol_rel:
            bad.append(r)
    return bad


# =========================================================================== known camera poses
@dataclass
class PosedCamera:
    """Pinhole camera with known pose (world -> camera: x_c = R x + t)."""
    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xc = np.atleast_2d(X) @ self.R.T + self.t
        uv = Xc @ self.K.T
        return uv[:, :2] / uv[:, 2:3], Xc[:, 2]

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.c_[self.R, self.t]


@dataclass
class PosedMatchOptions:
    reproj_tol_rel: float = 0.35   # x apparent radius
    radius_tol_rel: float = 0.25   # relative disagreement of apparent and predicted radius
    min_views: int | None = None  # None: 3 when >= 6 photos (two rays always 'intersect'), else 2
    min_height_mm: float = -3.0    # sphere centres may not lie below the mat (z up)
    rod_len_tol_rel: float = 0.035  # 3-D rod length check (1.4 mm at 39.7 mm pitch)
    complete_min_coverage: float = 0.5


def _triangulate(cams: list[PosedCamera], uvs: list[np.ndarray]) -> np.ndarray:
    """Linear (DLT) triangulation from >= 2 views. O(v)."""
    A = []
    for c, (u, v) in zip(cams, uvs):
        P = c.P
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]


def match_with_poses(dets: list[ViewDetections], cams: list[PosedCamera], parts: PartCatalog,
                     opts: PosedMatchOptions | None = None, log=print):
    """Multi-view sphere correspondence when camera poses are known (marker mat).

    Algorithm: for every detection pair from two views, triangulate; keep the
    hypothesis when it reprojects within ``reproj_tol_rel`` x radius in both views,
    the predicted apparent radius f R / z agrees with both measured radii, and it lies
    above the mat. Each hypothesis then collects support from all other views
    (nearest compatible detection). Hypotheses are accepted greedily by support,
    each detection being used at most once. Complexity O(V^2 n^2 V) for V views
    with n detections each.

    Returns (tracks, X): tracks is a list of {view_index: detection_index}; X the
    triangulated centres (mm, world frame).
    """
    opts = opts or PosedMatchOptions()
    Rmm = parts.sphere_radius_mm
    V = len(dets)
    min_views = opts.min_views or (3 if V >= 6 else 2)
    uv = [np.array([[s.x, s.y] for s in d.spheres]).reshape(-1, 2) for d in dets]
    rr = [np.array([s.r for s in d.spheres]) for d in dets]

    def compatible(X, v, i) -> float | None:
        p, z = cams[v].project(X)
        if z[0] <= 0:
            return None
        rpred = cams[v].K[0, 0] * Rmm / z[0]
        e = np.linalg.norm(p[0] - uv[v][i])
        if e > opts.reproj_tol_rel * rr[v][i] or abs(rr[v][i] - rpred) > opts.radius_tol_rel * rpred:
            return None
        return e / rr[v][i]

    hyps = []
    for a, b in itertools.combinations(range(V), 2):
        for i in range(len(uv[a])):
            for j in range(len(uv[b])):
                X = _triangulate([cams[a], cams[b]], [uv[a][i], uv[b][j]])
                if X[2] < opts.min_height_mm:
                    continue
                ea, eb = compatible(X, a, i), compatible(X, b, j)
                if ea is None or eb is None:
                    continue
                members = {a: i, b: j}
                for v in range(V):
                    if v in members or not len(uv[v]):
                        continue
                    p, _ = cams[v].project(X)
                    k = int(np.argmin(np.linalg.norm(uv[v] - p[0], axis=1)))
                    if compatible(X, v, k) is not None:
                        members[v] = k
                if len(members) > 2:
                    X = _triangulate([cams[v] for v in members], [uv[v][k] for v, k in members.items()])
                errs = [compatible(X, v, k) for v, k in members.items()]
                if any(e is None for e in errs) or len(members) < min_views:
                    continue
                hyps.append((len(members), -float(np.mean(errs)), members, X))
    hyps.sort(key=lambda h: (h[0], h[1]), reverse=True)
    used: set[tuple[int, int]] = set()
    tracks, Xs = [], []
    for n, _, members, X in hyps:
        keys = {(v, k) for v, k in members.items()}
        if keys & used:
            continue
        used |= keys
        tracks.append(members)
        Xs.append(X)
    log(f"posed matching: {len(hyps)} hypotheses, {len(tracks)} spheres "
        f"({sum(len(t) >= 3 for t in tracks)} seen in >= 3 photos)")
    return tracks, np.array(Xs).reshape(-1, 3)


def project_from_tracks(dets: list[ViewDetections], cams: list[PosedCamera], tracks, X: np.ndarray,
                        parts: PartCatalog, images_lab: list[np.ndarray] | None = None,
                        files: dict[str, str] | None = None,
                        opts: PosedMatchOptions | None = None, log=print) -> Project:
    """Labelled project from posed tracks: rods from detections whose 3-D length fits a
    part, optionally completed by colour evidence between triangulated spheres."""
    from .detect import DetectOptions, strip_color
    opts = opts or PosedMatchOptions()
    labels = [f"S{n + 1:02d}" for n in range(len(tracks))]
    where = {(v, k): t for t, members in enumerate(tracks) for v, k in members.items()}
    pitches = parts.active_pitches()

    def fits(t1, t2):
        d = float(np.linalg.norm(X[t1] - X[t2]))
        return min(abs(d - p) / p for p in pitches) <= opts.rod_len_tol_rel

    detected: dict[frozenset, dict[str, float]] = {}
    for v, d in enumerate(dets):
        for r in d.rods:
            t1, t2 = where.get((v, r.i)), where.get((v, r.j))
            if t1 is None or t2 is None or t1 == t2 or not fits(t1, t2):
                continue
            detected.setdefault(frozenset((t1, t2)), {}).setdefault(r.color, 0.0)
            detected[frozenset((t1, t2))][r.color] += r.score

    votes: dict[frozenset, dict[str, float]] = {}
    if images_lab is None:
        votes = detected
    else:
        # Multi-view colour vote for every sphere pair at a rod pitch: a view supports
        # the rod when the projected strip shows one palette colour, and contradicts it
        # when the strip is clearly visible but uncoloured or differently coloured.
        dopt = DetectOptions()
        names = list(parts.palette)
        proj = [cam.project(X) for cam in cams]
        for t1, t2 in itertools.combinations(range(len(tracks)), 2):
            key = frozenset((t1, t2))
            if not fits(t1, t2):
                continue
            support: dict[str, float] = dict(detected.get(key, {}))
            n_sup = 1 if key in detected else 0
            n_con = 0
            for v, cam in enumerate(cams):
                p, z = proj[v]
                r = cam.K[0, 0] * parts.sphere_radius_mm / z
                zmax = max(z[t1], z[t2])
                others = [(p[k, 0], p[k, 1], r[k]) for k in range(len(X))
                          if k not in (t1, t2) and z[k] < zmax]
                res = strip_color(images_lab[v], p[t1], p[t2], r[t1], r[t2], others, names, parts, dopt)
                if res is None:
                    continue
                col, cov, vis = res
                if cov >= opts.complete_min_coverage:
                    support[col] = support.get(col, 0.0) + cov
                    n_sup += 1
                elif vis >= 0.7 and cov < 0.2:
                    n_con += 1
            need = 2  # independent colour evidence from two views (a 2-D detection counts as one)
            if support and n_sup >= need and n_con <= n_sup // 2:
                votes[key] = support
    log(f"rods: {len(votes)} accepted ({sum(k in detected for k in votes)} also detected in 2-D)")
    views = []
    for v, d in enumerate(dets):
        sph = {}
        for (vv, k), t in where.items():
            if vv == v:
                s = d.spheres[k]
                sph[labels[t]] = Obs(s.x, s.y, s.r, float(np.clip(s.score / 0.6, 0.3, 1.0)))
        views.append(View(d.name, (files or {}).get(d.name), d.width, d.height, sph))
    rods = [Rod(labels[a], labels[b], max(c, key=c.get), "auto")
            for key, c in votes.items() for a, b in [tuple(key)]]
    return Project(views=views, rods=rods, auto_contacts=True, parts=parts)
