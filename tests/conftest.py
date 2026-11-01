import json
from pathlib import Path

import numpy as np
import pytest

from geomag2mesh import synthetic as syn
from geomag2mesh.parts import PartCatalog

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "twin_cube"


@pytest.fixture
def catalog():
    return PartCatalog()


@pytest.fixture
def twin_cube(catalog):
    return syn.twin_cube(catalog.pitch_mm(), catalog.sphere_diameter_mm)


@pytest.fixture(scope="session")
def mat_photos(tmp_path_factory):
    """Rendered photos of a twin cube on a marker mat (8 views) + ground truth."""
    import cv2
    from scipy.spatial.transform import Rotation

    from geomag2mesh.markers import BoardSpec
    cat = PartCatalog()
    pts, edges, contacts = syn.twin_cube(cat.pitch_mm(), cat.sphere_diameter_mm)
    X = np.array(list(pts.values()))
    X -= X.mean(0)
    X[:, 2] -= X[:, 2].min() - cat.sphere_radius_mm
    X = Rotation.from_euler("z", 23, degrees=True).apply(X)
    pts = {k: X[i] for i, k in enumerate(pts)}
    cols = ["dark_blue", "light_blue", "magenta", "orange"]
    rods = [(a, b, cols[i % 4]) for i, (a, b) in enumerate(edges)]
    board = BoardSpec(11, 8, 24.0, 18.0)
    imgs, _ = syn.render_scene(pts, rods, board=board, n_views=12, seed=2)
    d = tmp_path_factory.mktemp("mat_photos")
    for i, im in enumerate(imgs):
        cv2.imwrite(str(d / f"syn_{i:02d}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    (d / "truth.json").write_text(json.dumps({"centres": {k: v.tolist() for k, v in pts.items()},
                                              "rods": rods, "contacts": contacts}))
    return d, board
