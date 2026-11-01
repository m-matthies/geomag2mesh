"""Mat-free automatic reconstruction by incremental resection (no marker mat).

Pipeline
--------
1. Seed: every photo is lifted to a metric 3-D sphere cloud on its own
   (``lift.lift_view``: radius depths refined by detected rods). For each ordered
   photo pair (A, B), photo B is resected against A's cloud
   (``resect.resect``: P3P hypotheses from colour-matched rod paths, scored by
   projecting the whole cloud). The best few pairs become seed hypotheses.
2. Grow: the seed pair is bundle-adjusted (``solve.solve``); every remaining photo
   is resected against the current model, the most confident one is added, and the
   model is re-solved, until all photos are registered or none resects reliably.
3. Finish: with all cameras known, sphere correspondences and rods are rebuilt from
   scratch by posed triangulation and multi-view rod colour voting
   (``correspond.match_with_poses`` / ``project_from_tracks``), which recovers
   spheres and rods the per-photo detector missed.
4. Every seed hypothesis is carried through 2-3; the reconstruction that explains
   the photos best (matched observations, consistent rods, residuals) wins.

Why resection instead of pairwise cloud alignment: sphere *depths* from apparent
radius scatter by ~10 %, about one rod pitch, which makes cloud-to-cloud alignment
of a lattice-like framework ambiguous. Resection scores hypotheses with the precise
image centres (0.1 sphere radii) against a 3-D model, and the model sharpens as
photos are added.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .annotations import Obs, Project, Rod, View
from .detect import ViewDetections, lab_image
from .images import CameraPrior, ImageRecord
from .lift import lift_view
from .parts import PartCatalog
from .resect import Model, ResectOptions, resect
from .solve import SolveOptions, solve


@dataclass
class IncrementalOptions:
    seed_tol: float = 1.0           # resection tolerance (x radius) against a single-photo cloud
    grow_tol: float = 0.45          # resection tolerance against the bundle-adjusted model
    min_matches: int = 4
    n_seeds: int = 3
    min_seed_score: float = 6.0


def _principal(rec: ImageRecord) -> tuple[float, float]:
    pr = rec.prior
    if pr.principal is not None:
        return pr.principal
    return (rec.width / 2.0, rec.height / 2.0)


class _Builder:
    """Labelled observations accumulated from resections."""

    def __init__(self, recs, dets, parts: PartCatalog):
        self.recs, self.dets, self.parts = recs, dets, parts
        self.label_of: dict[tuple[int, int], str] = {}   # (view, detection) -> label
        self.views: list[int] = []
        self.pp: dict[int, tuple[float, float]] = {}      # principal point chosen per view
        self.n = 0

    def new_label(self) -> str:
        self.n += 1
        return f"S{self.n:02d}"

    def add_view(self, v: int, matches: dict[int, str], pp: tuple[float, float]):
        """matches: detection index -> existing label; other detections get new labels."""
        self.views.append(v)
        self.pp[v] = tuple(pp)
        used = set()
        for i in range(len(self.dets[v].spheres)):
            lab = matches.get(i)
            if lab is None or lab in used:
                lab = self.new_label()
            used.add(lab)
            self.label_of[(v, i)] = lab

    def project(self, fixed_intrinsics: bool = True) -> Project:
        """Labelled project. With ``fixed_intrinsics`` (used while growing) the focal
        length is held at the prior and the principal point at its prior/centre: with few
        multi-view spheres, a free principal point trades off against camera rotation."""
        views = []
        for v in self.views:
            d, rec = self.dets[v], self.recs[v]
            sph = {self.label_of[(v, i)]: Obs(s.x, s.y, s.r, float(np.clip(s.score / 0.6, 0.3, 1.0)))
                   for i, s in enumerate(d.spheres)}
            pr = rec.prior
            if fixed_intrinsics:
                cam = {"focal_px": pr.focal_px, "focal_sigma_rel": 0.01, "principal_known": True,
                       "camera_id": f"{pr.camera_id}|{rec.name}", "principal": list(self.pp[v])}
            else:
                cam = {"focal_px": pr.focal_px, "focal_sigma_rel": pr.focal_sigma_rel,
                       "principal_known": pr.principal_known, "camera_id": pr.camera_id}
                if pr.principal is not None:
                    cam["principal"] = list(pr.principal)
            views.append(View(rec.name, str(rec.path), rec.width, rec.height, sph, cam))
        rods: dict[frozenset, dict[str, float]] = {}
        for v in self.views:
            for rd in self.dets[v].rods:
                a, b = self.label_of[(v, rd.i)], self.label_of[(v, rd.j)]
                if a != b:
                    rods.setdefault(frozenset((a, b)), {}).setdefault(rd.color, 0.0)
                    rods[frozenset((a, b))][rd.color] += 1.0
        rod_list = [Rod(*sorted(k), max(c, key=c.get)) for k, c in rods.items()]
        return Project(views=views, rods=rod_list, auto_contacts=True, parts=self.parts)


def _solve(proj: Project, log):
    from .images import priors_for_project
    from .pipeline import solve_project
    sol, proj = solve_project(proj, SolveOptions(restarts=2), verify=True, log=lambda *_: None)
    return sol, proj


def _model_from_solution(sol) -> Model:
    return Model(sol.X, [(sol.index(r.a), sol.index(r.b), r.color) for r in sol.rods])


def _resect_into(sol, det, rec, tol) :
    model = _model_from_solution(sol)
    cam = next(c for c in sol.cameras if c.name == rec.name) if rec.name in [c.name for c in sol.cameras] else None
    pp = _principal(rec)
    res = resect(model, det, rec.prior.focal_px, pp, rec_radius(sol), ResectOptions(reproj_tol=tol, radius_tol=0.3))
    return res


def resect_any_pp(model: Model, det, rec, R_mm: float, ropts: ResectOptions, top_k: int = 2):
    """Resect over the photo's principal-point candidates (one if known; a grid over the
    feasible crop offsets otherwise). Returns [(Resection, principal)] best first."""
    out = []
    for pp in rec.prior.principal_candidates(rec.width, rec.height, nx=3, ny=4):
        for r in resect(model, det, rec.prior.focal_px, pp, R_mm, ropts, top_k=top_k):
            out.append((r, pp))
    out.sort(key=lambda t: -t[0].score)
    # keep distinct correspondence sets only
    seen, uniq = set(), []
    for r, pp in out:
        k = frozenset(r.matches)
        if k not in seen:
            seen.add(k)
            uniq.append((r, pp))
    return uniq[:top_k]


def rec_radius(sol) -> float:
    return sol.sphere_diameter_mm / 2


def seeds(recs, dets, parts: PartCatalog, opts: IncrementalOptions, log=print):
    """Ranked (score, a, b, resection) seed hypotheses."""
    lifts = {}
    for v, (rec, d) in enumerate(zip(recs, dets)):
        if len(d.spheres) >= 3:
            lifts[v] = lift_view(d, rec.prior.focal_px, _principal(rec), True, parts.pitch_mm(),
                                 parts.sphere_diameter_mm)
    out = []
    for a, b in itertools.permutations(lifts, 2):
        lv = lifts[a]
        model = Model(lv.X, lv.rods)
        res = resect_any_pp(model, dets[b], recs[b], parts.sphere_radius_mm,
                            ResectOptions(reproj_tol=opts.seed_tol, radius_tol=0.3), top_k=2)
        for r, pp in res[:2]:
            if len(r.matches) >= opts.min_matches:
                out.append((r.score, a, b, r, pp))
    out.sort(key=lambda t: -t[0])
    log("seed pairs: " + ", ".join(f"{recs[a].name}->{recs[b].name} {s:.1f}" for s, a, b, *_ in out[:6]))
    return out


def reconstruct_from_seed(recs, dets, parts, seed, opts: IncrementalOptions, log=print):
    score, a, b, r, pp_b = seed
    B = _Builder(recs, dets, parts)
    B.add_view(a, {}, _principal(recs[a]))
    B.add_view(b, {i: B.label_of[(a, j)] for i, j in r.matches}, pp_b)
    sol, proj = _solve(B.project(), log)
    remaining = [v for v in range(len(recs)) if v not in (a, b)]
    while remaining:
        best = None
        for v in remaining:
            res = resect_any_pp(_model_from_solution(sol), dets[v], recs[v], parts.sphere_radius_mm,
                                ResectOptions(reproj_tol=opts.grow_tol, radius_tol=0.3), top_k=2)
            if res and len(res[0][0].matches) >= opts.min_matches:
                margin = res[0][0].score - (res[1][0].score if len(res) > 1 else 0.0)
                key = res[0][0].score + 0.5 * margin
                if best is None or key > best[0]:
                    best = (key, v, res[0][0], res[0][1])
        if best is None:
            break
        _, v, rr, pp = best
        B.add_view(v, {i: sol.labels[j] for i, j in rr.matches}, pp)
        remaining.remove(v)
        sol, proj = _solve(B.project(), log)
    return sol, proj, [recs[v].name for v in B.views]


def explain_score(sol, proj) -> float:
    """How well a reconstruction explains the photos: matched observations in >= 2
    photos and rods, penalised by reprojection error (in sphere radii)."""
    nobs = sol.stats.get("observations_per_sphere", {})
    multi = sum(n for n in nobs.values() if n >= 2)
    rms = np.mean([v["rms_rel_radius"] for v in sol.stats.get("views", {}).values()] or [9])
    return multi + 0.5 * len(sol.rods) - 10.0 * max(0.0, rms - 0.25) - 2 * len(sol.stats.get("dropped_observations", []))


def reconstruct(recs, dets, parts: PartCatalog, opts: IncrementalOptions | None = None, log=print):
    """Best mat-free reconstruction over the top seed hypotheses: (solution, project, cameras)."""
    opts = opts or IncrementalOptions()
    cands = [s for s in seeds(recs, dets, parts, opts, log) if s[0] >= opts.min_seed_score][: opts.n_seeds]
    if not cands:
        return None
    best = None
    for seed in cands:
        try:
            sol, proj, order = reconstruct_from_seed(recs, dets, parts, seed, opts, log)
        except Exception as e:  # a degenerate seed must not abort the others
            log(f"seed {recs[seed[1]].name}->{recs[seed[2]].name} failed: {e}")
            continue
        sc = explain_score(sol, proj)
        log(f"seed {recs[seed[1]].name}->{recs[seed[2]].name}: registered {len(order)} photos, "
            f"{len(sol.labels)} spheres, {len(sol.rods)} rods, explain score {sc:.1f}")
        if best is None or sc > best[0]:
            best = (sc, sol, proj, order)
    return best


def finish(recs, dets, parts: PartCatalog, sol, log=print, pose_sigma_deg: float = 1.5,
           pose_sigma_mm: float = 8.0):
    """Rebuild correspondences and rods with the registered cameras (posed triangulation
    + multi-view rod colour voting), then solve with pose priors and free intrinsics.
    Returns (solution, project) or None if fewer than two photos are registered."""
    from .correspond import PosedCamera, PosedMatchOptions, match_with_poses, project_from_tracks
    from .pipeline import solve_project
    cams_by = {c.name: c for c in sol.cameras}
    idx = [i for i, r in enumerate(recs) if r.name in cams_by]
    if len(idx) < 2:
        return None
    cams, rr, dd = [], [], []
    for i in idx:
        c = cams_by[recs[i].name]
        K = np.array([[c.focal_px, 0, c.principal[0]], [0, c.focal_px, c.principal[1]], [0, 0, 1.0]])
        cams.append(PosedCamera(c.name, K, c.R, c.tvec))
        rr.append(recs[i])
        dd.append(dets[i])
    # no mat: the world frame is arbitrary, so the below-the-mat test does not apply
    po = PosedMatchOptions(min_height_mm=-1e9, reproj_tol_rel=0.45, radius_tol_rel=0.3)
    tracks, X = match_with_poses(dd, cams, parts, po, log=log)
    labs = [lab_image(r.load_rgb()[0]) for r in rr]
    files = {r.name: str(r.path) for r in rr}
    proj = project_from_tracks(dd, cams, tracks, X, parts, labs, files, po, log=log)
    for v, r, c in zip(proj.views, rr, cams):
        pr = r.prior
        v.camera = {"focal_px": float(c.K[0, 0]), "focal_sigma_rel": pr.focal_sigma_rel,
                    "principal_known": pr.principal_known, "camera_id": pr.camera_id,
                    "rvec": Rotation.from_matrix(c.R).as_rotvec().tolist(), "tvec": c.t.tolist(),
                    "pose_sigma_deg": pose_sigma_deg, "pose_sigma_mm": pose_sigma_mm}
        if pr.principal is not None:
            v.camera["principal"] = list(pr.principal)
        else:
            v.camera["principal_init"] = [float(c.K[0, 2]), float(c.K[1, 2])]
    return solve_project(proj, SolveOptions(restarts=2), verify=True, log=log)


# =========================================================================== anchor-based (joint)
def _build_project(recs, dets, parts, assign: dict[tuple[int, int], str], pp_init: dict[int, tuple],
                   intr: dict[int, tuple] | None = None) -> Project:
    """Project from a detection -> label assignment. Intrinsics: focal and principal point
    free (photos cropped by unknown amounts), starting from ``pp_init`` / ``intr``."""
    views = []
    rods: dict[frozenset, dict[str, float]] = {}
    for v, (rec, d) in enumerate(zip(recs, dets)):
        sph = {}
        for i, s in enumerate(d.spheres):
            lab = assign.get((v, i))
            if lab is not None and lab not in sph:
                sph[lab] = Obs(s.x, s.y, s.r, float(np.clip(s.score / 0.6, 0.3, 1.0)))
        pr = rec.prior
        f0 = intr[v][0] if intr and v in intr else pr.focal_px
        cam = {"focal_px": f0, "focal_sigma_rel": pr.focal_sigma_rel, "camera_id": pr.camera_id}
        if pr.principal is not None:
            cam.update(principal=list(pr.principal), principal_known=True)
        else:
            init = intr[v][1] if intr and v in intr else pp_init.get(v, _principal(rec))
            cam.update(principal_known=False, principal_init=[float(init[0]), float(init[1])])
        views.append(View(rec.name, str(rec.path), rec.width, rec.height, sph, cam))
        for rd in d.rods:
            a, b = assign.get((v, rd.i)), assign.get((v, rd.j))
            if a and b and a != b:
                rods.setdefault(frozenset((a, b)), {}).setdefault(rd.color, 0.0)
                rods[frozenset((a, b))][rd.color] += 1.0
    rod_list = [Rod(*sorted(k), max(c, key=c.get)) for k, c in rods.items()]
    return Project(views=views, rods=rod_list, auto_contacts=True, parts=parts)


def reconstruct_anchor(recs, dets, parts: PartCatalog, anchor: int, lifts, opts: IncrementalOptions,
                       iterations: int = 3, log=print):
    """Tracks from resecting every photo against the anchor photo's lifted cloud, then
    alternate joint bundle adjustment (free intrinsics) and re-resection of every photo
    against the improved model."""
    from .pipeline import solve_project
    lv = lifts[anchor]
    amodel = Model(lv.X, lv.rods)
    assign: dict[tuple[int, int], str] = {(anchor, i): f"A{i:02d}" for i in range(len(dets[anchor].spheres))}
    pp_init: dict[int, tuple] = {anchor: _principal(recs[anchor])}
    n_new = 0
    for v in range(len(recs)):
        if v == anchor:
            continue
        res = resect_any_pp(amodel, dets[v], recs[v], parts.sphere_radius_mm,
                            ResectOptions(reproj_tol=opts.seed_tol, radius_tol=0.3), top_k=1)
        matched = {}
        if res and len(res[0][0].matches) >= opts.min_matches:
            r, pp = res[0]
            matched = {i: f"A{j:02d}" for i, j in r.matches}
            pp_init[v] = pp
        for i in range(len(dets[v].spheres)):
            if i in matched:
                assign[(v, i)] = matched[i]
            else:
                n_new += 1
                assign[(v, i)] = f"N{n_new:03d}"
    sol, proj = solve_project(_build_project(recs, dets, parts, assign, pp_init),
                              SolveOptions(restarts=2), verify=True, log=lambda *_: None)
    for _ in range(iterations):
        cams = {c.name: c for c in sol.cameras}
        model = _model_from_solution(sol)
        new_assign: dict[tuple[int, int], str] = {}
        intr = {}
        n_new = 0
        for v, (rec, d) in enumerate(zip(recs, dets)):
            c = cams.get(rec.name)
            if c is None:
                continue
            intr[v] = (c.focal_px, tuple(c.principal))
            res = resect(model, d, c.focal_px, tuple(c.principal), parts.sphere_radius_mm,
                         ResectOptions(reproj_tol=opts.grow_tol, radius_tol=0.3), top_k=1)
            matched = {i: sol.labels[j] for i, j in res[0].matches} if res else {}
            for i in range(len(d.spheres)):
                if i in matched:
                    new_assign[(v, i)] = matched[i]
                else:
                    n_new += 1
                    new_assign[(v, i)] = f"M{n_new:03d}"
        if new_assign == assign:
            break
        assign = new_assign
        sol, proj = solve_project(_build_project(recs, dets, parts, assign, pp_init, intr),
                                  SolveOptions(restarts=1), verify=True, log=lambda *_: None)
    return sol, proj


def reconstruct_joint(recs, dets, parts: PartCatalog, opts: IncrementalOptions | None = None, log=print):
    """Try every photo as anchor; keep the reconstruction that explains the photos best."""
    opts = opts or IncrementalOptions()
    lifts = {v: lift_view(d, r.prior.focal_px, _principal(r), True, parts.pitch_mm(), parts.sphere_diameter_mm)
             for v, (r, d) in enumerate(zip(recs, dets)) if len(d.spheres) >= 3}
    best = None
    for a in lifts:
        try:
            sol, proj = reconstruct_anchor(recs, dets, parts, a, lifts, opts, log=log)
        except Exception as e:
            log(f"anchor {recs[a].name} failed: {e}")
            continue
        sc = explain_score(sol, proj)
        log(f"anchor {recs[a].name}: {len(sol.labels)} spheres, {len(sol.rods)} rods, explain score {sc:.1f}")
        if best is None or sc > best[0]:
            best = (sc, sol, proj, recs[a].name)
    return best
