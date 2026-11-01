"""Rod segment detection from colour, and rod-end support for sphere detection.

Rods are the most salient parts of a Geomag photo: saturated, uniformly coloured,
straight bars. This module

1. classifies saturated pixels into rod colours with a per-photo k-means
   initialised from the palette (absorbs white balance and shading shifts);
2. erodes each colour mask (rim pixels of shaded rods drift in hue) and splits
   connected components into straight segments by sequential RANSAC line fitting
   with a band of the measured rod half-width, cutting runs at gaps;
3. exposes, for every segment end, the outward axis: the projected sphere centre
   lies on the projected rod axis (a 3-D line through the sphere centre projects
   to a 2-D line through its image), at a distance between ~0.2 r (strong
   foreshortening) and ~1.3 r from the visible end.

Complexity: k-means O(I K P) on P saturated pixels (subsampled); RANSAC
O(T S) per component for T trials on S sampled pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .parts import PartCatalog


@dataclass
class RodSegment:
    p0: np.ndarray        # end point (working px)
    p1: np.ndarray        # other end point
    width: float          # full visible width (px)
    color: str
    n_px: int

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def axis(self) -> np.ndarray:
        d = self.p1 - self.p0
        return d / (np.linalg.norm(d) + 1e-9)

    def ends(self):
        """(end point, outward unit direction) for both ends."""
        a = self.axis
        return [(self.p0, -a), (self.p1, a)]


@dataclass
class RodOptions:
    chroma_min: float = 16.0
    max_dist: float = 30.0
    min_class_frac: float = 0.01
    erode_px: int = 2
    min_aspect: float = 1.8      # segment length / width
    band: float = 1.15           # RANSAC band (x half-width)
    gap: float = 1.2             # break a run at gaps longer than this x width
    trials: int = 150
    min_px_frac: float = 0.25    # of the pixels a square of width^2 would have
    min_fill: float = 0.55       # pixels / (length x width) of a segment
    min_width_rel: float = 0.45  # x reference width (median of the longest segments)
    seed: int = 0


def adaptive_classes(lab: np.ndarray, palette: dict, opts: RodOptions):
    """Per-pixel colour class (index into palette names, -1 none) and the adapted centres."""
    names = list(palette)
    C = np.array([palette[k] for k in names], np.float32)
    w = np.array([0.35, 1.0, 1.0], np.float32)
    X = lab.reshape(-1, 3)
    chroma = np.hypot(X[:, 1], X[:, 2])
    idx = np.where(chroma > opts.chroma_min)[0]
    cls = np.full(len(X), -1, int)
    if not len(idx):
        return cls.reshape(lab.shape[:2]), C
    rng = np.random.default_rng(opts.seed)
    sub = idx if len(idx) < 60000 else rng.choice(idx, 60000, replace=False)
    Y = X[sub]
    active = np.ones(len(C), bool)
    for _ in range(10):
        d = np.linalg.norm((Y[:, None] - C[None]) * w, axis=2)
        d[:, ~active] = 1e9
        a = d.argmin(1)
        for k in range(len(C)):
            sel = a == k
            if sel.sum() < opts.min_class_frac * len(Y):
                active[k] = False
                continue
            C[k] = Y[sel].mean(0)
    Y = X[idx]
    d = np.linalg.norm((Y[:, None] - C[None]) * w, axis=2)
    d[:, ~active] = 1e9
    a = d.argmin(1)
    ok = d.min(1) < opts.max_dist
    cls[idx[ok]] = a[ok]
    return cls.reshape(lab.shape[:2]), C


def _segments_from_component(pts: np.ndarray, half_w: float, color: str, opts: RodOptions,
                             rng) -> list[RodSegment]:
    out = []
    remaining = pts
    w = 2 * half_w
    min_px = opts.min_px_frac * w * w * opts.min_aspect
    for _ in range(8):
        if len(remaining) < min_px:
            break
        S = remaining if len(remaining) <= 2500 else remaining[rng.choice(len(remaining), 2500, replace=False)]
        best, best_n = None, 0
        for _t in range(opts.trials):
            i, j = rng.choice(len(S), 2, replace=False)
            d = S[j] - S[i]
            L = np.linalg.norm(d)
            if L < w:
                continue
            u = d / L
            n = np.array([-u[1], u[0]])
            cnt = int((np.abs((S - S[i]) @ n) < opts.band * half_w).sum())
            if cnt > best_n:
                best_n, best = cnt, (S[i], u)
        if best is None:
            break
        p, u = best
        n = np.array([-u[1], u[0]])
        inl = np.abs((remaining - p) @ n) < opts.band * half_w
        P = remaining[inl]
        if len(P) < min_px:
            break
        # refine axis by PCA
        m = P.mean(0)
        _, _, Vt = np.linalg.svd(P - m, full_matrices=False)
        u = Vt[0]
        n = np.array([-u[1], u[0]])
        inl = np.abs((remaining - m) @ n) < opts.band * half_w
        P = remaining[inl]
        t = (P - m) @ u
        order = np.argsort(t)
        ts = t[order]
        # split into runs at gaps
        cuts = np.where(np.diff(ts) > opts.gap * w)[0]
        starts = np.r_[0, cuts + 1]
        ends = np.r_[cuts, len(ts) - 1]
        used = np.zeros(len(P), bool)
        for s0, e0 in zip(starts, ends):
            run = order[s0:e0 + 1]
            if len(run) < min_px:
                continue
            tr = t[run]
            lo, hi = np.percentile(tr, 1), np.percentile(tr, 99)
            wid = 2 * np.percentile(np.abs((P[run] - m) @ n), 90)
            if hi - lo < opts.min_aspect * max(wid, 1):
                continue
            fill = len(run) / max((hi - lo) * max(wid, 1.0), 1.0)
            if fill < opts.min_fill:
                continue
            used[run] = True
            out.append(RodSegment(m + lo * u, m + hi * u, float(wid), color, int(len(run))))
        keep = np.ones(len(remaining), bool)
        keep[np.where(inl)[0]] = False
        remaining = remaining[keep]
        if not used.any():
            break
    return out


def detect_rod_segments(rgb: np.ndarray, palette: dict, opts: RodOptions | None = None,
                        lab: np.ndarray | None = None) -> list[RodSegment]:
    """Rod segments in working-image pixel coordinates."""
    from .detect import lab_image
    opts = opts or RodOptions()
    if lab is None:
        lab = lab_image(cv2.GaussianBlur(rgb, (0, 0), 1.0))
    cls, _ = adaptive_classes(lab, palette, opts)
    names = list(palette)
    rng = np.random.default_rng(opts.seed)
    segs: list[RodSegment] = []
    H, W = cls.shape
    min_area = (0.004 * max(H, W)) ** 2 * 6
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for k, name in enumerate(names):
        m = (cls == k).astype(np.uint8)
        if m.sum() < min_area:
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        if opts.erode_px:
            m = cv2.erode(m, kernel, iterations=opts.erode_px)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        for c in range(1, n):
            if stats[c, cv2.CC_STAT_AREA] < min_area:
                continue
            ys, xs = np.nonzero(lbl == c)
            dvals = dist[ys, xs]
            half_w = float(np.percentile(dvals, 95)) + opts.erode_px
            if half_w < 2:
                continue
            pts = np.c_[xs, ys].astype(float)
            segs += _segments_from_component(pts, half_w, name, opts, rng)
    if segs:  # thin fringes (hue drift at shaded rims) are far narrower than real rods
        longest = sorted(segs, key=lambda g: -g.length)[:max(3, len(segs) // 4)]
        w_ref = float(np.median([g.width for g in longest]))
        segs = [g for g in segs if g.width >= opts.min_width_rel * w_ref
                and g.length >= opts.min_aspect * g.width]
    return segs


def end_support(segs: list[RodSegment], x: float, y: float, r: float,
                near: float = 0.25, far: float = 1.45, lateral: float = 0.45) -> list[tuple[int, int]]:
    """Segment ends whose outward axis points at a circle (x, y, r): the centre lies
    ahead of the end, between ``near`` r and ``far`` r along the axis and within
    ``lateral`` r of it. Returns (segment index, end index)."""
    out = []
    c = np.array([x, y])
    for si, s in enumerate(segs):
        # sphere radius ~ 0.86 x rod width at equal depth; allow for depth differences
        if not 0.5 <= r / (0.86 * max(s.width, 1e-6)) <= 1.7:
            continue
        for ei, (p, u) in enumerate(s.ends()):
            v = c - p
            along = float(v @ u)
            perp = abs(float(v @ np.array([-u[1], u[0]])))
            if near * r <= along <= far * r and perp <= lateral * r:
                out.append((si, ei))
    return out


def on_rod_body(segs: list[RodSegment], x: float, y: float, r: float) -> bool:
    """True if a circle centre lies inside a rod segment's body (typical false positive)."""
    c = np.array([x, y])
    for s in segs:
        u = s.axis
        v = c - s.p0
        t = float(v @ u)
        perp = abs(float(v @ np.array([-u[1], u[0]])))
        if 0.15 * s.length < t < 0.85 * s.length and perp < 0.6 * s.width:
            return True
    return False
