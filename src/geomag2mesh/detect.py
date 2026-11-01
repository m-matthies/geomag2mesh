"""Automatic detection of steel spheres and coloured rods in one photograph.

Spheres: Hough-circle candidates scored by (i) radial edge support on the
circle, (ii) low colour saturation inside (polished steel), (iii) strong
intensity variation inside (environment reflections); then a per-image size
consensus and non-maximum suppression that still allows overlapping spheres
at different depths.

Rods: for every sphere pair whose image distance is compatible with one rod
(using each sphere's apparent radius as a scale), a strip along the
centre-to-centre segment is sampled; the rod is accepted when enough of the
unoccluded strip is classified as one palette colour.

Detection runs on a downscaled copy (``work_side`` px on the long side); all
results are returned in native pixel coordinates of the upright image.

Limitations (by construction): spheres hidden behind other parts are not
found; strongly specular rods can be split into several colour classes;
cluttered, saturated backgrounds produce false rod evidence. Results should be
checked on the overlay images that the CLI writes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .parts import PartCatalog


@dataclass
class DetectOptions:
    work_side: int = 1280
    r_min_rel: float = 0.012      # sphere radius search range / long side (working image)
    r_max_rel: float = 0.075
    hough_param1: float = 100.0
    hough_param2: float = 22.0
    min_score: float = 0.30
    size_consensus: tuple[float, float] = (0.55, 1.5)
    chroma_min: float = 18.0      # Lab chroma for "coloured" pixels
    color_max_dist: float = 28.0  # max Lab distance (a*, b*, 0.35 L*) to a palette entry
    rod_min_coverage: float = 0.55
    rod_min_visible: float = 0.30
    rod_len_tol: float = 0.30
    sat_weight: float = 0.3       # weight of the low-saturation cue (steel mirrors rods)
    dup_rel: float = 0.6          # NMS: similar-size circles closer than this x radius are one sphere
    mono_weight: float = 0.8      # penalty for discs filled by one rod colour
    local_search: bool = False
    use_rod_segments: bool = True
    refine_native: bool = True     # final local Hough refinement at full resolution
    rod_to_radius: float = 0.9    # sphere radius / visible rod width (12.7 / 7.4 / 2, rounded up)
    support_bonus: float = 0.12   # score added per rod segment pointing at a circle (max 3)
    min_circle_score: float = 0.12
    min_visible_arc: float = 0.25   # overlapping detection: uncovered fraction of its circle
    min_visible_edge: float = 0.55  # ... and the coherent-edge fraction on that part
    radius_px: tuple[float, float] | None = None  # expected radius range (working px), e.g. from mat pose     # relative tolerance on the 3-D rod length from radius depth


@dataclass
class SphereDet:
    x: float
    y: float
    r: float
    score: float


@dataclass
class RodDet:
    i: int
    j: int
    color: str
    score: float


@dataclass
class ViewDetections:
    name: str
    width: int
    height: int
    spheres: list[SphereDet] = field(default_factory=list)
    rods: list[RodDet] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"name": self.name, "width": self.width, "height": self.height,
                "spheres": [[round(s.x, 2), round(s.y, 2), round(s.r, 2), round(s.score, 3)]
                            for s in self.spheres],
                "rods": [[r.i, r.j, r.color, round(r.score, 3)] for r in self.rods]}

    @classmethod
    def from_dict(cls, d: dict) -> "ViewDetections":
        return cls(d["name"], d["width"], d["height"],
                   [SphereDet(*s) for s in d["spheres"]], [RodDet(*r) for r in d["rods"]])


# --------------------------------------------------------------------------- colour
def lab_image(rgb: np.ndarray) -> np.ndarray:
    """CIE L*a*b* (float32, L in 0..100, a/b centred at 0) from uint8 RGB."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def classify_colors(lab_px: np.ndarray, palette: dict[str, tuple], opts: DetectOptions) -> np.ndarray:
    """Palette index per pixel (-1: uncoloured or no palette entry close enough)."""
    names = list(palette)
    P = np.array([palette[k] for k in names], np.float32)
    w = np.array([0.35, 1.0, 1.0], np.float32)  # lightness varies most with shading
    d = np.linalg.norm((lab_px[:, None, :] - P[None]) * w, axis=2)
    idx = d.argmin(1)
    chroma = np.hypot(lab_px[:, 1], lab_px[:, 2])
    idx[(d.min(1) > opts.color_max_dist) | (chroma < opts.chroma_min)] = -1
    return idx


# --------------------------------------------------------------------------- spheres
def _stub_score(lab: np.ndarray, x: float, y: float, r: float, palette, opts) -> float:
    """Longest same-colour arc (degrees / 30, clipped to 1) on rings just outside the
    circle: every sphere in a framework has rod stubs at its rim, while specular
    highlights inside a sphere and table texture do not."""
    H, W = lab.shape[:2]
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    best = 0.0
    for k in (1.18, 1.35):
        px = (x + k * r * np.cos(th)).round().astype(int)
        py = (y + k * r * np.sin(th)).round().astype(int)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        cls = np.full(len(th), -1)
        if ok.any():
            cls[ok] = classify_colors(lab[py[ok], px[ok]], palette, opts)
        # longest circular run of one class
        run, prev = 0, -2
        cc = np.r_[cls, cls]
        for c in cc:
            run = run + 1 if (c == prev and c >= 0) else (1 if c >= 0 else 0)
            prev = c
            best = max(best, min(run, len(th)) * 3.0)
    return float(np.clip(best / 30.0, 0, 1))


@dataclass
class _Ctx:
    """Per-image arrays shared by candidate scoring."""
    rgb: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray
    blur: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    gmag: np.ndarray
    gref: float
    palette: dict
    opts: DetectOptions

    @classmethod
    def build(cls, rgb: np.ndarray, palette: dict, opts: DetectOptions) -> "_Ctx":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.hypot(gx, gy)
        return cls(rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), lab_image(rgb), blur, gx, gy, gmag,
                   float(np.percentile(gmag, 90) + 1e-6), palette, opts)


def score_circle(ctx: _Ctx, x: float, y: float, r: float) -> tuple[float, dict]:
    """Sphere likelihood in [0, 1] for a circle hypothesis, with its components."""
    H, W = ctx.blur.shape
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    px = (x + r * np.cos(th)).round().astype(int)
    py = (y + r * np.sin(th)).round().astype(int)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    if ok.mean() < 0.6:
        return 0.0, {}
    g = ctx.gmag[py[ok], px[ok]]
    radial = np.abs(ctx.gx[py[ok], px[ok]] * np.cos(th[ok]) + ctx.gy[py[ok], px[ok]] * np.sin(th[ok]))
    s_edge = float(((radial > 0.35 * ctx.gref) & (radial > 0.7 * g)).mean())
    x0, x1 = max(0, int(x - 0.75 * r)), min(W, int(x + 0.75 * r) + 1)
    y0, y1 = max(0, int(y - 0.75 * r)), min(H, int(y + 0.75 * r) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= (0.75 * r) ** 2
    if inside.sum() < 10:
        return 0.0, {}
    sat = ctx.hsv[y0:y1, x0:x1, 1][inside]
    val = ctx.hsv[y0:y1, x0:x1, 2][inside].astype(np.float32)
    s_sat = float(np.clip((110 - np.median(sat)) / 70, 0, 1))
    s_hi = float(np.clip((np.percentile(val, 99) - 140) / 80, 0, 1))
    s_tex = float(np.clip(val.std() / 40, 0, 1))
    # printed mat: disc filled by pure black and pure white in comparable amounts
    dark, bright = float((val < 45).mean()), float((val > 215).mean())
    s_mat = float(np.clip((min(dark, bright) - 0.08) / 0.12, 0, 1))
    # a rod segment is filled by one palette colour; a sphere mirrors a mixture
    labd = ctx.lab[y0:y1, x0:x1][inside]
    cls = classify_colors(labd[:: max(1, len(labd) // 400)], ctx.palette, ctx.opts)
    dom = np.bincount(cls[cls >= 0]).max() / len(cls) if (cls >= 0).any() else 0.0
    s_mono = float(np.clip((dom - 0.35) / 0.3, 0, 1))
    # steel mirrors nearby rods, so saturation is only a weak cue
    o = ctx.opts
    base = (s_edge * (1 - o.sat_weight + o.sat_weight * s_sat) * (0.3 + 0.7 * s_hi)
            * (0.6 + 0.4 * s_tex) * (1.0 - o.mono_weight * s_mono) * (1.0 - s_mat))
    s_stub = _stub_score(ctx.lab, x, y, r, ctx.palette, ctx.opts) if base >= 0.1 else 0.0
    parts = dict(edge=s_edge, sat=s_sat, hi=s_hi, tex=s_tex, mono=s_mono, mat=s_mat, stub=s_stub)
    return base * (0.25 + 0.75 * s_stub), parts


def hough_candidates(ctx: _Ctx) -> list[tuple[float, float, float]]:
    H, W = ctx.blur.shape
    L = max(H, W)
    o = ctx.opts
    rmin, rmax = (o.radius_px if o.radius_px else (o.r_min_rel * L, o.r_max_rel * L))
    cands = []
    bands = np.geomspace(rmin, rmax, 4)
    for lo, hi in zip(bands[:-1], bands[1:]):  # radius bands keep the accumulator conditioned
        c = cv2.HoughCircles(ctx.blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(4, 0.4 * lo),
                             param1=o.hough_param1, param2=o.hough_param2,
                             minRadius=int(lo), maxRadius=int(hi))
        if c is not None:
            cands += [tuple(map(float, x)) for x in c[0]]
    return cands


def suppress(cands: list[SphereDet], dup_rel: float = 0.6) -> list[SphereDet]:
    """Non-maximum suppression for spheres. O(n^2).

    1. Highlight rings: a circle lying inside a clearly larger circle with at least
       60 % of its score is a reflection inside that sphere and is dropped.
    2. Greedy by score: a circle whose centre lies within ``dup_rel`` x radius of a
       kept circle of similar size, or within 0.5 x radius regardless of size, is a
       duplicate. Spheres overlapping at different depths (centre offset >~ 0.6 r)
       survive; ``occlusion_check`` judges them by their visible outline."""
    keep = []
    for s in cands:
        inner = False
        for k in cands:
            if k is s or k.r < s.r / 0.75:
                continue
            d = np.hypot(s.x - k.x, s.y - k.y)
            if d + 0.6 * s.r < k.r and k.score >= 0.6 * s.score:
                inner = True
                break
        if not inner:
            keep.append(s)
    kept: list[SphereDet] = []
    for s in sorted(keep, key=lambda s: -s.score):
        dup = False
        for k in kept:
            d = np.hypot(s.x - k.x, s.y - k.y)
            similar = max(s.r, k.r) / min(s.r, k.r) < 1.35
            if (d < dup_rel * max(s.r, k.r) and similar) or d < 0.5 * max(s.r, k.r):
                dup = True
                break
        if not dup:
            kept.append(s)
    return kept


def detect_spheres(rgb: np.ndarray, opts: DetectOptions, palette: dict | None = None,
                   segs: list | None = None) -> list[SphereDet]:
    """Sphere detections in working-image coordinates.

    Evidence: circle appearance (``score_circle``) plus *rod-end support*: the number
    of distinct rod segments whose outward axis points at the circle (every sphere of
    a framework terminates rods). Circles centred on a rod body are rejected. Rod
    ends also propose candidates where the circle evidence alone is too weak (dark
    spheres reflecting a dark table). O(C s) for C candidates, s pixels per disc."""
    from .rods import end_support, on_rod_body
    ctx = _Ctx.build(rgb, palette or PartCatalog().palette, opts)
    segs = segs or []
    cands = list(hough_candidates(ctx))
    if segs:
        w_ref = float(np.median([g.width for g in segs]))
        for g in segs:
            for p, u in g.ends():
                for k in (0.6, 0.85, 1.1):
                    r0 = opts.rod_to_radius * w_ref
                    for rr in (0.8 * r0, r0, 1.25 * r0):
                        c = p + u * k * rr
                        cands.append((float(c[0]), float(c[1]), float(rr)))
    out = []
    for x, y, r in cands:
        if segs and on_rod_body(segs, x, y, r):
            continue
        sc, _ = score_circle(ctx, x, y, r)
        if sc < opts.min_circle_score:  # rod support boosts circles, it cannot create them
            continue
        sup = len({si for si, _ in end_support(segs, x, y, r)}) if segs else 0
        total = sc + opts.support_bonus * min(sup, 3)
        if total >= opts.min_score * 0.6:
            out.append(SphereDet(x, y, r, total))
    if not out:
        return []
    if opts.radius_px:  # size known from the mat pose: no consensus step needed
        lo_r, hi_r = opts.radius_px
        out = [s for s in out if lo_r <= s.r <= hi_r and s.score >= opts.min_score]
    else:
        top = sorted(out, key=lambda s: -s.score)[:12]
        rmed = float(np.median([s.r for s in top]))
        lo, hi = opts.size_consensus
        out = [s for s in out if lo * rmed <= s.r <= hi * rmed and s.score >= opts.min_score]
    out = suppress(out, opts.dup_rel)
    out = occlusion_check(ctx, out, opts)
    if opts.local_search:
        out = suppress([local_search(ctx, d) for d in out], opts.dup_rel)
    return out


def occlusion_check(ctx: _Ctx, dets: list[SphereDet], opts: DetectOptions) -> list[SphereDet]:
    """Accept detections in score order; a candidate overlapping already accepted discs
    must show its own outline on the uncovered part of its circle: a sphere partly
    behind another has a sharp visible crescent, while a near-duplicate's extra arc
    falls on rods or table. O(n^2 m) for n detections and m arc samples."""
    H, W = ctx.blur.shape
    th = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    kept: list[SphereDet] = []

    def outline(d):  # full-circle coherent edge fraction: highest for the true sphere
        _, comp = score_circle(ctx, d.x, d.y, d.r)
        return comp.get("edge", 0.0)

    for d in sorted(dets, key=lambda s: -(outline(s) + 0.25 * s.score)):
        px = d.x + d.r * np.cos(th)
        py = d.y + d.r * np.sin(th)
        free = np.ones(len(th), bool)
        for k in kept:
            free &= np.hypot(px - k.x, py - k.y) > 1.05 * k.r
        if free.all():
            kept.append(d)
            continue
        if free.mean() < opts.min_visible_arc:
            continue
        xi, yi = px.round().astype(int), py.round().astype(int)
        ok = free & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        if ok.sum() < 6:
            continue
        g = ctx.gmag[yi[ok], xi[ok]]
        radial = np.abs(ctx.gx[yi[ok], xi[ok]] * np.cos(th[ok]) + ctx.gy[yi[ok], xi[ok]] * np.sin(th[ok]))
        coherent = float(((radial > 0.35 * ctx.gref) & (radial > 0.7 * g)).mean())
        if coherent >= opts.min_visible_edge:
            kept.append(d)
    return kept


def rods_from_segments(segs: list, spheres: list[SphereDet]) -> list[RodDet]:
    """A segment whose two ends point at two different detected spheres is a rod
    between them; collinear pieces of one rod merge (same sphere pair)."""
    from .rods import end_support
    att: dict[tuple[int, int], tuple[int, float]] = {}
    for k, sp in enumerate(spheres):
        for si, ei in end_support(segs, sp.x, sp.y, sp.r):
            p, u = segs[si].ends()[ei]
            err = abs(float((np.array([sp.x, sp.y]) - p) @ u) - 0.8 * sp.r)
            if (si, ei) not in att or err < att[(si, ei)][1]:
                att[(si, ei)] = (k, err)
    out: dict[frozenset, RodDet] = {}
    for si, g in enumerate(segs):
        a, b = att.get((si, 0)), att.get((si, 1))
        if a is None or b is None or a[0] == b[0]:
            continue
        key = frozenset((a[0], b[0]))
        if key not in out or g.n_px > out[key].score:
            out[key] = RodDet(min(a[0], b[0]), max(a[0], b[0]), g.color, float(g.n_px))
    rods = list(out.values())
    for r in rods:
        r.score = 1.0
    return rods


def local_search(ctx: _Ctx, d: SphereDet, steps: int = 2) -> SphereDet:
    """Coordinate ascent of the circle score over centre and radius (Hough centres
    are biased where rods meet the sphere outline). O(steps * 75) score evaluations."""
    best = d
    for _ in range(steps):
        improved = False
        for dx in (-0.12, 0.0, 0.12):
            for dy in (-0.12, 0.0, 0.12):
                for dr in (0.88, 1.0, 1.12):
                    if dx == dy == 0 and dr == 1.0:
                        continue
                    x, y, r = best.x + dx * best.r, best.y + dy * best.r, best.r * dr
                    sc, _ = score_circle(ctx, x, y, r)
                    if sc > best.score + 1e-3:
                        best, improved = SphereDet(x, y, r, sc), True
        if not improved:
            break
    return best


def refine_native(gray: np.ndarray, d: SphereDet, search: float = 0.3) -> SphereDet:
    """Local Hough refinement of one sphere on the full-resolution image (native px):
    the silhouette of a polished sphere is sharp at full resolution, while reflections
    inside the disc dominate edge fits on downscaled images. Picks the circle with
    radius in [0.8, 1.25] r closest to the estimate. O(patch area)."""
    H, W = gray.shape
    r = d.r
    x0, y0 = int(max(0, d.x - 1.6 * r)), int(max(0, d.y - 1.6 * r))
    x1, y1 = int(min(W, d.x + 1.6 * r)), int(min(H, d.y + 1.6 * r))
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0 or min(patch.shape) < 10:
        return d
    cs = cv2.HoughCircles(patch, cv2.HOUGH_GRADIENT, dp=1, minDist=max(2, r / 3), param1=60, param2=20,
                          minRadius=int(0.8 * r), maxRadius=int(1.25 * r) + 1)
    if cs is None:
        return d
    best = None
    for cx, cy, cr in cs[0][:12]:
        cx, cy = cx + x0, cy + y0
        dist = np.hypot(cx - d.x, cy - d.y)
        if dist < search * r and (best is None or dist + 0.5 * abs(cr - r) < best[0]):
            best = (dist + 0.5 * abs(cr - r), cx, cy, cr)
    if best is None:
        return d
    return SphereDet(float(best[1]), float(best[2]), float(best[3]), d.score)


def refine_sphere(gray: np.ndarray, s: SphereDet) -> SphereDet:
    """Sub-pixel refinement by fitting a circle to strong radial edges near the
    detected circle (algebraic Kasa fit, O(k) for k edge points)."""
    H, W = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    pts = []
    for t in th:
        best, bp = 0.0, None
        for rr in np.linspace(0.8 * s.r, 1.2 * s.r, 13):
            x, y = s.x + rr * np.cos(t), s.y + rr * np.sin(t)
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                g = abs(gx[yi, xi] * np.cos(t) + gy[yi, xi] * np.sin(t))
                if g > best:
                    best, bp = g, (x, y)
        if bp is not None and best > 0:
            pts.append((*bp, best))
    if len(pts) < 20:
        return s
    P = np.array(pts)
    keep = P[:, 2] > np.percentile(P[:, 2], 40)  # occluded arcs have weak edges
    x, y = P[keep, 0], P[keep, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    cx, cy, c = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)[0]
    r = float(np.sqrt(max(c + cx ** 2 + cy ** 2, 0)))
    if np.hypot(cx - s.x, cy - s.y) > 0.3 * s.r or not 0.8 * s.r < r < 1.2 * s.r:
        return s
    return SphereDet(float(cx), float(cy), r, s.score)


# --------------------------------------------------------------------------- rods
def detect_rods(rgb: np.ndarray, spheres: list[SphereDet], parts: PartCatalog,
                opts: DetectOptions, focal_px: float) -> list[RodDet]:
    """Rod hypotheses between sphere pairs. O(n^2 s) for n spheres and s strip samples."""
    H, W = rgb.shape[:2]
    lab = lab_image(rgb)
    names = list(parts.palette)
    D = parts.sphere_diameter_mm
    pitches = sorted(parts.active_pitches())
    rod_r_rel = min(rt.diameter_mm for rt in parts.rod_types.values()) / D  # rod radius / sphere radius
    n = len(spheres)
    if n < 2:
        return []
    xy = np.array([[s.x, s.y] for s in spheres])
    rr = np.array([s.r for s in spheres])
    z = focal_px * (D / 2) / rr                     # depth from apparent radius (mm)
    P3 = np.c_[(xy - [W / 2, H / 2]) / focal_px * z[:, None], z]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            d3 = float(np.linalg.norm(P3[i] - P3[j]))
            if not any(abs(d3 - p) < opts.rod_len_tol * p for p in pitches):
                continue
            others = [(xy[k, 0], xy[k, 1], rr[k]) for k in range(n) if k not in (i, j)]
            res = strip_color(lab, xy[i], xy[j], rr[i], rr[j], others, names, parts, opts, rod_r_rel)
            if res is not None and res[1] >= opts.rod_min_coverage:
                out.append(RodDet(i, j, res[0], float(res[1])))
    return _prune_rods(out, xy, rr)


def strip_color(lab: np.ndarray, a: np.ndarray, b: np.ndarray, ra: float, rb: float,
                others: list[tuple[float, float, float]], names: list[str], parts: PartCatalog,
                opts: DetectOptions, rod_r_rel: float | None = None):
    """Dominant palette colour along the image segment between two sphere discs.

    Returns (colour, coverage, visible_fraction) or None when the segment is too
    short or too occluded to judge. Samples inside ``others`` discs (x, y, r) are
    treated as occluded. O(s * |others|) for s samples."""
    H, W = lab.shape[:2]
    if rod_r_rel is None:
        rod_r_rel = min(rt.diameter_mm for rt in parts.rod_types.values()) / parts.sphere_diameter_mm
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-6:
        return None
    u = (b - a) / L
    nrm = np.array([-u[1], u[0]])
    t0, t1 = ra * 1.05, L - rb * 1.05
    rmean = 0.5 * (ra + rb)
    if t1 - t0 < 0.3 * rmean:
        return None  # foreshortened rod hidden behind its spheres
    ts = np.linspace(t0, t1, 40)
    half = 0.35 * rod_r_rel * rmean
    offs = np.linspace(-half, half, 3)
    samp = ((a + ts[:, None] * u)[:, None, :] + offs[None, :, None] * nrm).reshape(-1, 2)
    xi, yi = samp[:, 0].round().astype(int), samp[:, 1].round().astype(int)
    inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    occl = np.zeros(len(samp), bool)
    for ox, oy, orr in others:
        occl |= np.hypot(samp[:, 0] - ox, samp[:, 1] - oy) < orr
    vis = inb & ~occl
    if vis.mean() < opts.rod_min_visible:
        return None
    cls = classify_colors(lab[yi[vis], xi[vis]], parts.palette, opts)
    if (cls >= 0).sum() == 0:
        return None
    counts = np.bincount(cls[cls >= 0], minlength=len(names))
    best = int(counts.argmax())
    return names[best], float(counts[best] / vis.sum()), float(vis.mean())


def _prune_rods(rods: list[RodDet], xy: np.ndarray, rr: np.ndarray) -> list[RodDet]:
    """Two rods leaving one sphere in nearly the same image direction share their strip;
    keep the shorter one (a rod ends at the first sphere along its line)."""
    rods = sorted(rods, key=lambda r: -r.score)
    keep: list[RodDet] = []
    for r in rods:
        ok = True
        for k in keep:
            common = {r.i, r.j} & {k.i, k.j}
            if not common:
                continue
            c = common.pop()
            o1 = r.j if r.i == c else r.i
            o2 = k.j if k.i == c else k.i
            v1, v2 = xy[o1] - xy[c], xy[o2] - xy[c]
            cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            if cos > 0.97:
                if np.linalg.norm(v1) > np.linalg.norm(v2):
                    ok = False
                else:
                    keep.remove(k)
                break
        if ok:
            keep.append(r)
    return keep


# --------------------------------------------------------------------------- driver
def detect_view(name: str, rgb_native_loader, native_size: tuple[int, int], focal_px_native: float,
                parts: PartCatalog, opts: DetectOptions | None = None,
                radius_px_native: tuple[float, float] | None = None) -> ViewDetections:
    """Detect spheres and rods; ``rgb_native_loader(max_side)`` returns (rgb, scale)."""
    opts = opts or DetectOptions()
    rgb, s = rgb_native_loader(opts.work_side)
    if radius_px_native is not None:
        from dataclasses import replace
        opts = replace(opts, radius_px=(radius_px_native[0] * s, radius_px_native[1] * s))
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (0, 0), 1.0)
    segs = []
    if opts.use_rod_segments:
        from .rods import detect_rod_segments
        segs = detect_rod_segments(rgb, parts.palette)
    sph = [refine_sphere(gray, d) for d in detect_spheres(rgb, opts, parts.palette, segs)]
    rods = detect_rods(rgb, sph, parts, opts, focal_px_native * s)
    if segs:  # rods seen as segments are more reliable than the pairwise strip test
        seg_rods = rods_from_segments(segs, sph)
        have = {frozenset((r.i, r.j)) for r in seg_rods}
        rods = seg_rods + [r for r in rods if frozenset((r.i, r.j)) not in have]
    W, H = native_size
    native = [SphereDet(d.x / s, d.y / s, d.r / s, d.score) for d in sph]
    if opts.refine_native and s < 0.999:
        full, _ = rgb_native_loader(None)
        g = cv2.GaussianBlur(cv2.cvtColor(full, cv2.COLOR_RGB2GRAY), (0, 0), 2.0)
        native = [refine_native(g, d) for d in native]
    return ViewDetections(name, W, H, native, rods)
