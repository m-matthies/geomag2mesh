"""End-to-end checks: rendered photos on the marker mat, and the real example photos."""
import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import EXAMPLE


def _score(truth_path, sol):
    truth = json.loads(truth_path.read_text())
    T = np.array(list(truth["centres"].values()))
    names = list(truth["centres"])
    d = np.linalg.norm(sol.X[:, None] - T[None], axis=2)
    lab = {k: names[d[i].argmin()] for i, k in enumerate(sol.labels) if d[i].min() < 6.35}
    gt = {frozenset((a, b)) for a, b, _ in truth["rods"]}
    rods_ok = [r.a in lab and r.b in lab and frozenset((lab[r.a], lab[r.b])) in gt for r in sol.rods]
    return lab, d.min(1), rods_ok


@pytest.mark.slow
def test_mat_photos_end_to_end(mat_photos, tmp_path):
    """Automatic run on 12 rendered photos: what it reports must be right, and most of
    the framework must be found."""
    d, board = mat_photos
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "geomag2mesh", "run", *map(str, sorted(d.glob("*.jpg"))), "-o", str(out),
           "--mat", "--squares", f"{board.squares_x}x{board.squares_y}",
           "--square-mm", str(board.square_mm), "--marker-mm", str(board.marker_mm)]
    subprocess.run(cmd, check=True, capture_output=True)
    from geomag2mesh.solve import Solution
    sol = Solution.load(out / "solution.json")
    lab, err, rods_ok = _score(d / "truth.json", sol)
    # measured (12 rendered photos): 12/16 spheres, max error 0.5 mm, 12/24 rods, no false parts
    assert len(set(lab.values())) >= 12 and len(lab) == len(sol.labels)  # no false spheres
    assert np.median(err) < 1.0 and err.max() < 1.5        # mm, in the mat frame
    assert sum(rods_ok) >= 12 and np.mean(rods_ok) >= 0.9   # rods found; precision


@pytest.mark.skipif(not (EXAMPLE / "photos").exists(), reason="example photos not present")
def test_example_annotations_reproduce_reference(tmp_path):
    from geomag2mesh.annotations import Project
    from geomag2mesh.images import priors_for_project
    from geomag2mesh.solve import Solution, SolveOptions, _umeyama, solve
    p = Project.load(EXAMPLE / "annotations.json")
    sol = solve(p, priors_for_project(p), SolveOptions(), log=lambda *_: None)
    ref = Solution.load(EXAMPLE / "reference_solution.json")
    A = np.array([ref.X[ref.index(k)] for k in sol.labels])
    _, R, t = _umeyama(sol.X, A, with_scale=False)
    e = np.linalg.norm(sol.X @ R.T + t - A, axis=1)
    assert e.max() < 1.0  # mm; reference stored with an earlier solver revision
    assert len(sol.contacts) == 4
