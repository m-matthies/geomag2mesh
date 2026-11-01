"""Physical part catalogue for Geomag-style magnetic construction sets.

All lengths are in millimetres. The centre-to-centre distance of two spheres
joined by one rod (the *pitch*) is ``rod_length + sphere_diameter``: the flat
magnetic rod tips touch the sphere surfaces on the line between the centres.

Default dimensions (hjreggel.net/gmtrcply/magsticks, accessed 2026-10-29):
sphere 12.7 mm; short ("original"/PRO) rod 27.0 mm x 7.4 mm;
long (current "Geomag"/PRO-L) rod 58.2 mm x 7.6 mm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Reference rod colours in CIE L*a*b* (D65). The first four were measured on
# iPhone photographs of Geomag Classic rods under indoor light; the remaining
# entries are approximate placeholders and should be re-measured for a new set
# (``geomag2mesh palette`` prints measured values for annotated rods).
DEFAULT_PALETTE: dict[str, tuple[float, float, float]] = {
    "dark_blue": (31.0, 30.0, -60.0),
    "light_blue": (48.0, -1.0, -32.0),
    "magenta": (23.0, 27.0, -2.0),
    "orange": (56.0, 36.0, 40.0),
    "red": (40.0, 55.0, 35.0),        # approximate
    "green": (50.0, -45.0, 30.0),     # approximate
    "yellow": (80.0, 0.0, 70.0),      # approximate
}

# sRGB used when rendering a rod of a given colour class in exported meshes.
RENDER_RGB: dict[str, tuple[int, int, int]] = {
    "dark_blue": (32, 52, 196),
    "light_blue": (32, 150, 214),
    "magenta": (196, 38, 128),
    "orange": (236, 96, 40),
    "red": (210, 30, 40),
    "green": (40, 170, 70),
    "yellow": (240, 200, 30),
    "unknown": (130, 130, 140),
}
STEEL_RGB = (196, 198, 204)


@dataclass(frozen=True)
class RodType:
    """A rod length class."""

    name: str
    length_mm: float
    diameter_mm: float


@dataclass
class PartCatalog:
    """Sphere and rod dimensions plus the colour palette used for detection."""

    sphere_diameter_mm: float = 12.7
    rod_types: dict[str, RodType] = field(default_factory=lambda: {
        "short": RodType("short", 27.0, 7.4),
        "long": RodType("long", 58.2, 7.6),
    })
    default_rod: str = "short"
    mixed_rods: bool = False  # True: a build may combine rod types (detection considers all)
    palette: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: dict(DEFAULT_PALETTE))

    @property
    def sphere_radius_mm(self) -> float:
        return self.sphere_diameter_mm / 2.0

    def pitch_mm(self, rod_type: str | None = None) -> float:
        """Centre-to-centre distance of two spheres joined by a rod of this type."""
        rt = self.rod_types[rod_type or self.default_rod]
        return rt.length_mm + self.sphere_diameter_mm

    @property
    def active_rod_types(self) -> list[str]:
        """Rod types a build may contain: all when ``mixed_rods``, else the default only.
        Restricting matters: e.g. a cube's body diagonal (39.7 mm x sqrt 3 = 68.8 mm)
        is within 3 % of the long-rod pitch (70.9 mm)."""
        return list(self.rod_types) if self.mixed_rods else [self.default_rod]

    def active_pitches(self) -> list[float]:
        return [self.pitch_mm(k) for k in self.active_rod_types]

    def nearest_rod_type(self, centre_distance_mm: float) -> str:
        """Active rod type whose pitch is closest to an observed centre distance."""
        return min(self.active_rod_types,
                   key=lambda k: abs(self.pitch_mm(k) - centre_distance_mm))

    def to_dict(self) -> dict:
        return {
            "sphere_diameter_mm": self.sphere_diameter_mm,
            "default_rod": self.default_rod,
            "mixed_rods": self.mixed_rods,
            "rod_types": {k: {"length_mm": v.length_mm, "diameter_mm": v.diameter_mm}
                          for k, v in self.rod_types.items()},
            "palette": {k: list(v) for k, v in self.palette.items()},
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "PartCatalog":
        if not d:
            return cls()
        cat = cls()
        cat.sphere_diameter_mm = float(d.get("sphere_diameter_mm", cat.sphere_diameter_mm))
        if "rod_types" in d:
            cat.rod_types = {k: RodType(k, float(v["length_mm"]), float(v.get("diameter_mm", 7.4)))
                             for k, v in d["rod_types"].items()}
        cat.default_rod = d.get("default_rod", cat.default_rod)
        cat.mixed_rods = bool(d.get("mixed_rods", False))
        if cat.default_rod not in cat.rod_types:
            raise ValueError(f"default_rod {cat.default_rod!r} not among rod_types")
        if "palette" in d:
            cat.palette = {k: tuple(map(float, v)) for k, v in d["palette"].items()}
        return cat
