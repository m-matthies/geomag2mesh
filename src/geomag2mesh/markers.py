"""Printable ChArUco marker mat and per-photo camera pose from it.

Placing the framework on a printed ChArUco board makes the reconstruction
well posed: each photo's pose and the metric scale follow from the board
corners (OpenCV ``CharucoDetector`` + ``solvePnP``), and the board plane is
the table plane (z = 0, z up). The framework may cover the centre of the mat;
ChArUco interpolates corners from the visible markers only.

Print at 100 % scale ("actual size") and check one square with a ruler; enter
the measured square size if it differs (``--square-mm``).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DICTS = {"5x5_250": cv2.aruco.DICT_5X5_250, "4x4_250": cv2.aruco.DICT_4X4_250,
         "6x6_250": cv2.aruco.DICT_6X6_250}


@dataclass
class BoardSpec:
    squares_x: int = 9
    squares_y: int = 6
    square_mm: float = 30.0
    marker_mm: float = 22.0
    dictionary: str = "5x5_250"

    def board(self) -> cv2.aruco.CharucoBoard:
        d = cv2.aruco.getPredefinedDictionary(DICTS[self.dictionary])
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_mm,
                                      self.marker_mm, d)

    def corners_3d(self) -> np.ndarray:
        """Chessboard corner positions (mm) in the board frame, centred, z = 0 (z up)."""
        pts = np.asarray(self.board().getChessboardCorners(), float).reshape(-1, 3)
        pts[:, :2] -= pts[:, :2].mean(0)
        # OpenCV board frame has y pointing "down the image"; flip y and z for a
        # right-handed frame with z up out of the mat.
        pts[:, 1] *= -1
        return pts

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict | None) -> "BoardSpec | None":
        return cls(**d) if d else None


def render_board(spec: BoardSpec, dpi: int = 300, margin_mm: float = 10.0) -> np.ndarray:
    """Board image (uint8 gray) at exact physical scale for the given dpi."""
    px_per_mm = dpi / 25.4
    w = spec.squares_x * spec.square_mm + 2 * margin_mm
    h = spec.squares_y * spec.square_mm + 2 * margin_mm
    size = (round(w * px_per_mm), round(h * px_per_mm))
    return spec.board().generateImage(size, marginSize=round(margin_mm * px_per_mm), borderBits=1)


def save_board_pdf(spec: BoardSpec, path: str, dpi: int = 300) -> tuple[float, float]:
    """Write a PDF/PNG whose embedded resolution prints at actual size. Returns (w, h) mm."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(render_board(spec, dpi))
    draw = ImageDraw.Draw(img)
    note = (f"geomag2mesh ChArUco mat  {spec.squares_x}x{spec.squares_y} squares  "
            f"square {spec.square_mm:g} mm  marker {spec.marker_mm:g} mm  dict {spec.dictionary}  "
            "- print at 100 %")
    draw.text((10, img.height - 30), note, fill=0)
    img.save(path, resolution=dpi)
    return img.width / dpi * 25.4, img.height / dpi * 25.4


def detect_board(rgb: np.ndarray, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray]:
    """ChArUco corner ids and pixel positions (sub-pixel). Empty arrays if none found."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    det = cv2.aruco.CharucoDetector(spec.board())
    corners, ids, _, _ = det.detectBoard(gray)
    if ids is None or len(ids) == 0:
        return np.zeros(0, int), np.zeros((0, 2))
    return ids.ravel().astype(int), corners.reshape(-1, 2).astype(float)


def board_pose(ids: np.ndarray, corners: np.ndarray, spec: BoardSpec, focal_px: float,
               principal: tuple[float, float], min_corners: int = 6):
    """Camera pose (rvec, tvec; world = board frame, mm) and RMS reprojection error, or None."""
    if len(ids) < min_corners:
        return None
    obj = spec.corners_3d()[ids]
    K = np.array([[focal_px, 0, principal[0]], [0, focal_px, principal[1]], [0, 0, 1.0]])
    ok, rv, tv = cv2.solvePnP(obj, corners, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(obj, corners, K, None, rv, tv)
    proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - corners) ** 2).sum(1).mean()))
    return rv.ravel(), tv.ravel(), rms
