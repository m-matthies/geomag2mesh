"""Annotation model: which sphere is where in which photo, and which rods exist.

The annotation file is the single interface between the automatic front end
(detection + matching), manual editing, and the solver. Format (JSON)::

    {
      "format": "geomag2mesh/annotations@1",
      "parts": {...optional PartCatalog overrides...},
      "views": [
        {"name": "IMG_1078", "file": "IMG_1078.HEIC", "width": 3024, "height": 4032,
         "spheres": {"A": [x, y, r_or_null, weight], ...}}
      ],
      "rods": [{"a": "A", "b": "B", "color": "dark_blue", "type": "short"}],
      "contacts": [["C", "D"]],
      "auto_contacts": true
    }

Pixel coordinates refer to the upright (EXIF-rotated) image at the stated
width/height. ``r`` is the apparent sphere radius in pixels (optional, but it
gives the solver a depth cue); ``weight`` in (0, 1] down-weights partially
occluded spheres. Sphere labels are arbitrary strings.

A view may carry a ``camera`` block that overrides the EXIF-derived prior::

    "camera": {"focal_px": 3028.6, "focal_sigma_rel": 0.08, "camera_id": "phone",
               "principal": [1512.0, 676.0]}

``principal`` fixes the principal point of a photo cropped by a known offset
(original centre minus crop origin); without it, cropped photos get a free
principal point. ``rvec``/``tvec`` (+ ``pose_sigma_deg``/``pose_sigma_mm``) give
a pose prior, as written by the marker-mat workflow.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .parts import PartCatalog

FORMAT = "geomag2mesh/annotations@1"


@dataclass
class Obs:
    x: float
    y: float
    r: float | None = None
    w: float = 1.0


@dataclass
class View:
    name: str
    file: str | None
    width: int
    height: int
    spheres: dict[str, Obs] = field(default_factory=dict)
    camera: dict | None = None  # optional prior: focal_px, focal_sigma_rel, principal_known, camera_id


@dataclass
class Rod:
    a: str
    b: str
    color: str = "unknown"
    type: str | None = None  # None: catalogue default, "auto": classify after solving

    @property
    def key(self) -> frozenset:
        return frozenset((self.a, self.b))


@dataclass
class Project:
    views: list[View]
    rods: list[Rod]
    contacts: list[tuple[str, str]] = field(default_factory=list)
    auto_contacts: bool = True
    parts: PartCatalog = field(default_factory=PartCatalog)
    extra_spheres: list[str] = field(default_factory=list)  # never observed
    support_plane: bool = False  # world frame is the mat (z up, z = 0); spheres stay at z >= R
    base_dir: Path = field(default_factory=Path.cwd)

    # ------------------------------------------------------------------ queries
    @property
    def labels(self) -> list[str]:
        """All sphere labels, in a stable order (first appearance)."""
        seen: dict[str, None] = {}
        for v in self.views:
            for k in v.spheres:
                seen.setdefault(k)
        for r in self.rods:
            seen.setdefault(r.a)
            seen.setdefault(r.b)
        for a, b in self.contacts:
            seen.setdefault(a)
            seen.setdefault(b)
        for k in self.extra_spheres:
            seen.setdefault(k)
        return list(seen)

    def observation_count(self) -> dict[str, int]:
        n = {k: 0 for k in self.labels}
        for v in self.views:
            for k in v.spheres:
                n[k] += 1
        return n

    def degree(self) -> dict[str, int]:
        d = {k: 0 for k in self.labels}
        for r in self.rods:
            d[r.a] += 1
            d[r.b] += 1
        return d

    def image_path(self, view: View) -> Path | None:
        if not view.file:
            return None
        p = Path(view.file)
        return p if p.is_absolute() else self.base_dir / p

    # --------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Return human-readable problems; raise ValueError for fatal ones."""
        fatal, warn = [], []
        keys = set()
        for r in self.rods:
            if r.a == r.b:
                fatal.append(f"rod {r.a}-{r.b} joins a sphere to itself")
            if r.key in keys:
                fatal.append(f"duplicate rod {r.a}-{r.b}")
            keys.add(r.key)
            if r.type not in (None, "auto") and r.type not in self.parts.rod_types:
                fatal.append(f"rod {r.a}-{r.b}: unknown rod type {r.type!r}")
        for a, b in self.contacts:
            if frozenset((a, b)) in keys:
                fatal.append(f"{a}-{b} is listed both as rod and as contact")
        if len(self.views) < 2:
            warn.append("fewer than two views: depth relies on sphere size only")
        nobs, deg = self.observation_count(), self.degree()
        for k in self.labels:
            if nobs[k] == 0 and deg[k] < 3:
                warn.append(f"sphere {k}: unobserved and only {deg[k]} rod(s): position undetermined")
            elif nobs[k] == 1 and deg[k] == 0:
                warn.append(f"sphere {k}: seen in one view without rods: depth from sphere size only")
        if fatal:
            raise ValueError("; ".join(fatal))
        return warn

    # ----------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "parts": self.parts.to_dict(),
            "views": [{
                "name": v.name, "file": v.file, "width": v.width, "height": v.height,
                "spheres": {k: [round(o.x, 2), round(o.y, 2),
                                None if o.r is None else round(o.r, 2), round(o.w, 3)]
                            for k, o in v.spheres.items()},
                **({"camera": v.camera} if v.camera else {}),
            } for v in self.views],
            "rods": [{"a": r.a, "b": r.b, "color": r.color,
                      **({"type": r.type} if r.type else {})} for r in self.rods],
            "contacts": [list(c) for c in self.contacts],
            "auto_contacts": self.auto_contacts,
            "extra_spheres": self.extra_spheres,
            "support_plane": self.support_plane,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        for v in d["views"]:  # store image paths relative to the annotation file
            if v["file"]:
                try:
                    v["file"] = str(Path(self.base_dir / v["file"]).resolve()
                                    .relative_to(path.parent.resolve()))
                except ValueError:
                    v["file"] = str(Path(self.base_dir / v["file"]).resolve())
        path.write_text(json.dumps(d, indent=1))

    @classmethod
    def from_dict(cls, d: dict, base_dir: Path | None = None) -> "Project":
        if d.get("format", FORMAT) != FORMAT:
            raise ValueError(f"unsupported annotation format {d.get('format')!r}")
        views = []
        for v in d["views"]:
            sph = {}
            for k, val in v.get("spheres", {}).items():
                x, y = float(val[0]), float(val[1])
                r = None if len(val) < 3 or val[2] is None else float(val[2])
                w = 1.0 if len(val) < 4 else float(val[3])
                sph[str(k)] = Obs(x, y, r, w)
            views.append(View(v["name"], v.get("file"), int(v["width"]), int(v["height"]), sph,
                              v.get("camera")))
        rods = []
        for r in d.get("rods", []):
            if isinstance(r, dict):
                rods.append(Rod(str(r["a"]), str(r["b"]), r.get("color", "unknown"), r.get("type")))
            else:  # compact form [a, b, color?, type?]
                rods.append(Rod(str(r[0]), str(r[1]), *(list(r[2:4]))))
        return cls(views=views, rods=rods,
                   contacts=[(str(a), str(b)) for a, b in d.get("contacts", [])],
                   auto_contacts=bool(d.get("auto_contacts", True)),
                   parts=PartCatalog.from_dict(d.get("parts")),
                   extra_spheres=[str(s) for s in d.get("extra_spheres", [])],
                   support_plane=bool(d.get("support_plane", False)),
                   base_dir=base_dir or Path.cwd())

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()), base_dir=path.parent)
