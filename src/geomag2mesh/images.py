"""Image loading and camera-intrinsics priors from EXIF.

Handles HEIC/HEIF (via pillow-heif), JPEG and PNG; applies the EXIF
orientation so pixel coordinates refer to the upright image.

Intrinsics model: pinhole, square pixels, no distortion (phone pipelines
already remove most lens distortion in the delivered image).

Assumptions encoded here:

* ``FocalLengthIn35mmFilm`` refers to the 43.27 mm full-frame diagonal.
* An image whose size equals the EXIF original size (either orientation)
  is uncropped: principal point at the centre.
* An image with the original aspect ratio but smaller size is a uniformly
  downscaled full frame: focal length scales, principal point at the centre.
* Any other size is treated as an unscaled crop: focal length in original
  pixels, principal point unknown (estimated by the solver).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # optional HEIC support
    import pillow_heif

    pillow_heif.register_heif_opener()
    HAVE_HEIF = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_HEIF = False

FULL_FRAME_DIAGONAL_MM = 43.267
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp"}


@dataclass
class CameraPrior:
    """Prior on intrinsics for one image, in that image's pixel frame."""

    focal_px: float
    focal_sigma_rel: float   # relative 1-sigma of the prior
    principal_known: bool    # True: principal point fixed (image centre unless ``principal`` is set)
    camera_id: str           # images with equal id share one focal length
    rvec: tuple | None = None        # optional pose prior (world -> camera), e.g. from markers
    tvec: tuple | None = None
    pose_sigma_deg: float = 0.2
    pose_sigma_mm: float = 1.0
    principal: tuple | None = None   # known principal point (px), e.g. a crop with known offset
    orig_size: tuple | None = None   # (w, h) of the uncropped sensor frame (either orientation)
    principal_init: tuple | None = None  # starting value for an unknown principal point

    def principal_candidates(self, width: int, height: int, nx: int = 5, ny: int = 6,
                             margin: float = 0.05) -> list[tuple[float, float]]:
        """Principal points to try: the known one, or a grid over all positions a crop of
        this size can have inside the original frame (plus a small margin)."""
        if self.principal is not None:
            return [tuple(self.principal)]
        if self.principal_known or self.orig_size is None:
            return [(width / 2.0, height / 2.0)]
        out: list[tuple[float, float]] = []
        a, b = self.orig_size
        for W0, H0 in {(a, b), (b, a)}:       # every orientation the crop fits in
            if width > W0 * (1 + margin) or height > H0 * (1 + margin):
                continue
            mx, my = margin * W0, margin * H0
            xs = np.linspace(W0 / 2 - (W0 - width) - mx, W0 / 2 + mx, nx)
            ys = np.linspace(H0 / 2 - (H0 - height) - my, H0 / 2 + my, ny)
            out += [(float(x), float(y)) for x in xs for y in ys]
        return out or [(width / 2.0, height / 2.0)]

    @property
    def has_pose(self) -> bool:
        return self.rvec is not None and self.tvec is not None


@dataclass
class ImageRecord:
    name: str
    path: Path
    width: int
    height: int
    prior: CameraPrior

    def load_rgb(self, max_side: int | None = None) -> tuple[np.ndarray, float]:
        """Return (RGB uint8 array, scale) where scale = returned / native pixels."""
        im = _open_upright(self.path).convert("RGB")
        s = 1.0
        if max_side and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        return np.asarray(im), s


def _open_upright(path: Path) -> Image.Image:
    im = Image.open(path)
    return ImageOps.exif_transpose(im)


def _exif_info(im: Image.Image) -> dict:
    exif = im.getexif()
    sub = exif.get_ifd(0x8769) if exif else {}
    return {
        "make": exif.get(271), "model": exif.get(272),
        "focal_mm": sub.get(37386), "focal_35mm": sub.get(41989),
        "orig_w": sub.get(40962), "orig_h": sub.get(40963),
        "lens": sub.get(42036),
    }


def camera_prior(width: int, height: int, info: dict) -> CameraPrior:
    """Derive an intrinsics prior from image size and EXIF fields."""
    cam_id = "|".join(str(info.get(k)) for k in ("make", "model", "lens", "focal_mm"))
    f35, ow, oh = info.get("focal_35mm"), info.get("orig_w"), info.get("orig_h")
    if not f35 or not ow or not oh:
        # No usable EXIF: a typical phone/compact field of view (~65 deg diagonal).
        diag = float(np.hypot(width, height))
        return CameraPrior(0.78 * diag, 0.35, True, cam_id + f"|{width}x{height}")
    ow, oh = int(ow), int(oh)
    f_orig = float(f35) / FULL_FRAME_DIAGONAL_MM * float(np.hypot(ow, oh))
    if {width, height} == {ow, oh}:
        return CameraPrior(f_orig, 0.08, True, cam_id)
    aspect_img = max(width, height) / min(width, height)
    aspect_orig = max(ow, oh) / min(ow, oh)
    if abs(aspect_img / aspect_orig - 1) < 0.005:
        s = max(width, height) / max(ow, oh)
        return CameraPrior(f_orig * s, 0.08, True, cam_id + f"|scaled{width}x{height}")
    # Cropped: assume unscaled crop (focal in original pixels); widen the prior
    # because a crop that was also rescaled cannot be identified from metadata.
    return CameraPrior(f_orig, 0.15, False, cam_id, orig_size=(ow, oh))


def load_record(path: str | Path) -> ImageRecord:
    path = Path(path)
    im = _open_upright(path)
    info = _exif_info(Image.open(path))
    w, h = im.size
    return ImageRecord(path.stem, path, w, h, camera_prior(w, h, info))


def find_images(paths: list[str | Path]) -> list[Path]:
    """Expand files and directories into a sorted list of image files."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES)
        elif p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
        else:
            raise FileNotFoundError(f"not an image or directory: {p}")
    if any(p.suffix.lower() in {".heic", ".heif"} for p in out) and not HAVE_HEIF:
        raise RuntimeError("HEIC input requires pillow-heif (pip install pillow-heif)")
    return out


def priors_for_project(project) -> dict[str, CameraPrior]:
    """Camera priors for every view: explicit ``camera`` entry, else EXIF of the image
    file (rescaled to the annotation frame), else a generic field-of-view guess."""
    out: dict[str, CameraPrior] = {}
    for v in project.views:
        if v.camera:
            c = v.camera
            out[v.name] = CameraPrior(float(c["focal_px"]), float(c.get("focal_sigma_rel", 0.1)),
                                      bool(c.get("principal_known", True)),
                                      str(c.get("camera_id", v.name)),
                                      tuple(c["rvec"]) if c.get("rvec") is not None else None,
                                      tuple(c["tvec"]) if c.get("tvec") is not None else None,
                                      float(c.get("pose_sigma_deg", 0.2)),
                                      float(c.get("pose_sigma_mm", 1.0)),
                                      tuple(map(float, c["principal"])) if c.get("principal") else None,
                                      principal_init=tuple(map(float, c["principal_init"]))
                                      if c.get("principal_init") else None)
            if out[v.name].principal is not None:
                out[v.name].principal_known = True
            continue
        path = project.image_path(v)
        if path is not None and path.exists():
            rec = load_record(path)
            s = max(v.width, v.height) / max(rec.width, rec.height)
            pr = rec.prior
            out[v.name] = CameraPrior(pr.focal_px * s, pr.focal_sigma_rel, pr.principal_known,
                                      pr.camera_id)
        else:
            out[v.name] = camera_prior(v.width, v.height, {})
    return out
