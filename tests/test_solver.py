"""Solver on synthetic observations with known ground truth."""
import numpy as np
from scipy.spatial.transform import Rotation

from geomag2mesh import synthetic as syn
from geomag2mesh.solve import SolveOptions, _umeyama, solve


def _errors(truth, sol):
    return syn.aligned_errors(truth, sol.labels, sol.X)


def test_twin_cube_uncropped(twin_cube):
    pts, edges, contacts = twin_cube
    proj, priors, truth = syn.make_project(pts, edges, n_views=4, seed=3)
    sol = solve(proj, priors, SolveOptions(), log=lambda *_: None)
    e = _errors(truth, sol)
    assert np.sqrt((e ** 2).mean()) < 0.6 and e.max() < 1.2
    found = {frozenset(c) for c in sol.contacts}
    assert found == {frozenset(c) for c in contacts}


def test_twin_cube_with_cropped_views(twin_cube):
    pts, edges, _ = twin_cube
    proj, priors, truth = syn.make_project(pts, edges, n_views=4, crop_views=(1, 2), seed=4)
    sol = solve(proj, priors, SolveOptions(), log=lambda *_: None)
    assert np.sqrt((_errors(truth, sol) ** 2).mean()) < 1.0


def test_without_radii_still_converges(twin_cube):
    """Radii are optional: initialisation falls back to median depth."""
    pts, edges, _ = twin_cube
    proj, priors, truth = syn.make_project(pts, edges, n_views=5, seed=7)
    for v in proj.views:
        for o in v.spheres.values():
            o.r = None
    sol = solve(proj, priors, SolveOptions(restarts=6), log=lambda *_: None)
    assert np.sqrt((_errors(truth, sol) ** 2).mean()) < 1.0


def test_pose_priors_fix_frame(twin_cube):
    """With pose priors the solution is in the prior frame: no alignment needed."""
    pts, edges, _ = twin_cube
    proj, priors, truth = syn.make_project(pts, edges, n_views=4, seed=5)
    # recover the true poses by solving once, then perturb slightly and use as priors
    ref = solve(proj, priors, SolveOptions(), log=lambda *_: None)
    A = np.array([truth[k] for k in ref.labels])
    _, Rw, tw = _umeyama(ref.X, A, with_scale=False)  # ref frame -> truth frame
    rng = np.random.default_rng(0)
    for cam, (name, pr) in zip(ref.cameras, priors.items()):
        Rc = cam.R @ Rw.T                      # truth frame -> camera
        tc = cam.tvec - Rc @ tw
        pr.rvec = tuple(Rotation.from_matrix(Rc).as_rotvec() + rng.normal(0, 1e-3, 3))
        pr.tvec = tuple(tc + rng.normal(0, 0.3, 3))
    sol = solve(proj, priors, SolveOptions(), log=lambda *_: None)
    assert sol.stats["frame"] == "marker"
    err = np.linalg.norm(sol.X - np.array([truth[k] for k in sol.labels]), axis=1)
    assert np.median(err) < 1.0


def test_outlier_observation_is_rejected(twin_cube):
    pts, edges, _ = twin_cube
    proj, priors, truth = syn.make_project(pts, edges, n_views=5, seed=8)
    v = proj.views[0]
    k = next(iter(v.spheres))
    v.spheres[k].x += 400.0  # gross labelling error
    sol = solve(proj, priors, SolveOptions(), log=lambda *_: None)
    assert [v.name, k] in sol.stats["dropped_observations"]
    assert np.sqrt((_errors(truth, sol) ** 2).mean()) < 1.0


def test_long_rods_auto_type(catalog):
    pts, edges = syn.cube(catalog.pitch_mm("long"), shear=0.15, rng=np.random.default_rng(1))
    proj, priors, truth = syn.make_project(pts, edges, n_views=4, seed=9, distance=450)
    proj.parts.mixed_rods = True  # type unknown a priori: classify from geometry
    for r in proj.rods:
        r.type = "auto"
    sol = solve(proj, priors, SolveOptions(), log=lambda *_: None)
    assert {r.type for r in sol.rods} == {"long"}
    assert np.sqrt((_errors(truth, sol) ** 2).mean()) < 1.0


def test_crop_with_known_offset_is_invariant(twin_cube):
    """Cropping a photo and declaring the crop offset as principal point must not change
    the solution (regression: the annotation sigma once scaled with the image size)."""
    import copy
    pts, edges, _ = twin_cube
    proj, priors, truth = syn.make_project(pts, edges, n_views=4, seed=3)
    ref = solve(proj, priors, SolveOptions(restarts=1), log=lambda *_: None)
    cropped, cpri = copy.deepcopy(proj), copy.deepcopy(priors)
    v = cropped.views[0]
    ox, oy = 200.0, 900.0
    for o in v.spheres.values():
        o.x, o.y = o.x - ox, o.y - oy
    v.width, v.height = v.width - 400, v.height - 1300
    pr = cpri[v.name]
    pr.principal = (proj.views[0].width / 2 - ox, proj.views[0].height / 2 - oy)
    sol = solve(cropped, cpri, SolveOptions(restarts=1), log=lambda *_: None)
    A = np.array([ref.X[ref.index(k)] for k in sol.labels])
    _, R, t = _umeyama(sol.X, A, with_scale=False)
    assert np.linalg.norm(sol.X @ R.T + t - A, axis=1).max() < 1e-3
