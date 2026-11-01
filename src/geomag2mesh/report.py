"""Diagnostics: reprojection overlays and a quality verdict for a reconstruction."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .annotations import Project
from .parts import RENDER_RGB
from .solve import Solution

# Thresholds of the quality gate (dimensionless unless stated).
GOOD_RMS_REL_RADIUS = 0.25   # reprojection RMS / apparent sphere radius, per view
CHECK_RMS_REL_RADIUS = 0.50
GOOD_ROD_RMS_REL = 0.01      # rod length RMS / pitch
MIN_MULTIVIEW_FRAC = 0.7     # spheres observed in >= 2 photos
UNRELIABLE_MULTIVIEW_FRAC = 0.6
MAX_WEAK_FRAC = 0.2          # spheres with <= 1 rod (open ends are rare in frameworks)
MAX_LOW_FRAC = 0.5           # spheres with <= 2 rods (closed polyhedral frameworks have >= 3)


def quality(sol: Solution, project: Project) -> dict:
    """Verdict 'good' | 'check' | 'unreliable' with the reasons."""
    reasons, level = [], 0
    for name, v in sol.stats.get("views", {}).items():
        e = v["rms_rel_radius"]
        if e > CHECK_RMS_REL_RADIUS:
            level = max(level, 2)
            reasons.append(f"{name}: reprojection RMS {e:.2f} sphere radii")
        elif e > GOOD_RMS_REL_RADIUS:
            level = max(level, 1)
            reasons.append(f"{name}: reprojection RMS {e:.2f} sphere radii")
    pitch = project.parts.pitch_mm()
    rod = sol.stats.get("rod_length_error_mm", {}).get("rms", 0.0) / pitch
    if rod > 3 * GOOD_ROD_RMS_REL:
        level = max(level, 2)
        reasons.append(f"rod lengths deviate {100 * rod:.1f} % RMS from part length")
    elif rod > GOOD_ROD_RMS_REL:
        level = max(level, 1)
        reasons.append(f"rod lengths deviate {100 * rod:.1f} % RMS")
    nobs = sol.stats.get("observations_per_sphere", {})
    if nobs:
        frac = np.mean([n >= 2 for n in nobs.values()])
        if frac < MIN_MULTIVIEW_FRAC:
            level = max(level, 2 if frac < UNRELIABLE_MULTIVIEW_FRAC else 1)
            reasons.append(f"only {100 * frac:.0f} % of spheres seen in >= 2 photos "
                           "(single-photo spheres are unverified)")
    nd = len(sol.stats.get("dropped_observations", []))
    if nd > 0.15 * max(1, sum(nobs.values())):
        level = max(level, 2)
        reasons.append(f"{nd} observations rejected as outliers (inconsistent labelling?)")
    deg = {k: 0 for k in sol.labels}
    for r in sol.rods:
        deg[r.a] += 1
        deg[r.b] += 1
    loose = [k for k, d in deg.items() if d == 0]
    if loose:
        level = max(level, 1)
        reasons.append(f"{len(loose)} sphere(s) without any rod: {', '.join(loose[:8])}")
    # Completeness: in a closed framework nearly every sphere carries >= 2 rods.
    weak = [k for k, d in deg.items() if d <= 1]
    low = [k for k, d in deg.items() if d <= 2]
    if len(sol.labels) and (len(weak) / len(sol.labels) > MAX_WEAK_FRAC
                            or len(low) / len(sol.labels) > MAX_LOW_FRAC):
        level = max(level, 1)
        reasons.append(f"possibly incomplete: {len(low)} of {len(sol.labels)} spheres carry <= 2 rods "
                       f"(mean {2 * len(sol.rods) / len(sol.labels):.1f} rods per sphere; closed frameworks "
                       "have >= 3); add photos or annotate missing parts")
    return {"verdict": ["good", "check", "unreliable"][level], "reasons": reasons}


def overlays(sol: Solution, project: Project, out_dir: str | Path, max_side: int = 1400) -> list[Path]:
    """Draw the solved model over each photo: rods in their colour, sphere outlines
    (white), labels, and lines from annotated to projected centres (red)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for v, cam in zip(project.views, sol.cameras):
        p = project.image_path(v)
        if p is None or not p.exists():
            continue
        from .images import load_record
        rec = load_record(p)
        rgb, s = rec.load_rgb(max_side)
        s *= rec.width / v.width  # annotation frame -> displayed image
        im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ov = im.copy()
        uv, z = cam.project(sol.X)
        rad = cam.focal_px * sol.sphere_diameter_mm / 2 / z
        for r in sol.rods:
            a, b = sol.index(r.a), sol.index(r.b)
            col = RENDER_RGB.get(r.color, RENDER_RGB["unknown"])[::-1]
            w = max(2, int(0.25 * s * (rad[a] + rad[b]) / 2))
            cv2.line(ov, tuple((uv[a] * s).astype(int)), tuple((uv[b] * s).astype(int)), col, w)
        for i in np.argsort(-z):
            c = tuple((uv[i] * s).astype(int))
            cv2.circle(ov, c, max(2, int(rad[i] * s)), (255, 255, 255), 2)
            cv2.putText(ov, sol.labels[i], (c[0] - 8, c[1] + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.4, rad[i] * s / 40), (0, 255, 255), 2)
        for k, o in v.spheres.items():
            if k in sol.labels:
                i = sol.index(k)
                cv2.line(ov, (int(o.x * s), int(o.y * s)), tuple((uv[i] * s).astype(int)), (0, 0, 255), 2)
        out = cv2.addWeighted(ov, 0.75, im, 0.25, 0)
        path = out_dir / f"check_{v.name}.jpg"
        cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)
    return paths


def detection_overlay(rgb: np.ndarray, det, path: str | Path, max_side: int = 1400) -> None:
    """Draw raw detections (spheres with index, rods with colour initial)."""
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    im = cv2.cvtColor(cv2.resize(rgb, (int(w * s), int(h * s))), cv2.COLOR_RGB2BGR)
    for r in det.rods:
        a, b = det.spheres[r.i], det.spheres[r.j]
        col = RENDER_RGB.get(r.color, RENDER_RGB["unknown"])[::-1]
        cv2.line(im, (int(a.x * s), int(a.y * s)), (int(b.x * s), int(b.y * s)), col, 3)
    for i, d in enumerate(det.spheres):
        cv2.circle(im, (int(d.x * s), int(d.y * s)), int(d.r * s), (0, 255, 0), 2)
        cv2.putText(im, str(i), (int(d.x * s) - 6, int(d.y * s) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)
    cv2.imwrite(str(path), im, [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_summary(sol: Solution, project: Project, q: dict, out: str | Path) -> None:
    out = Path(out)
    lines = ["# Reconstruction report", "",
             f"Verdict: **{q['verdict']}**", ""]
    lines += [f"- {r}" for r in q["reasons"]] or ["- no issues found"]
    st = sol.stats
    lines += ["", "| photo | observations | RMS px | RMS / sphere radius | worst |",
              "|---|---:|---:|---:|---|"]
    for name, v in st.get("views", {}).items():
        lines.append(f"| {name} | {v['n_obs']} | {v['rms_px']:.1f} | {v['rms_rel_radius']:.2f} | "
                     f"{v['worst'][0]} ({v['worst'][1]:.0f} px) |")
    lines += ["", f"Spheres: {len(sol.labels)}; rods: {len(sol.rods)}; contacts: {len(sol.contacts)}",
              f"Rod length error: RMS {st['rod_length_error_mm']['rms']:.2f} mm, "
              f"max {st['rod_length_error_mm']['max_abs']:.2f} mm",
              f"Focal length(s): {', '.join(f'{f:.0f}' for f in st['focal_px'])} px; frame: {st.get('frame')}"]
    if "leave_one_view_out" in st:
        lines += ["", "Leave-one-photo-out stability (rigidly aligned centre shift):"]
        lines += [f"- without {k}: RMS {v['rms_mm']:.2f} mm, max {v['max_mm']:.2f} mm ({v['max_sphere']})"
                  for k, v in st["leave_one_view_out"].items()]
    out.write_text("\n".join(lines) + "\n")
    (out.with_suffix(".json")).write_text(json.dumps({"quality": q, "stats": st}, indent=1))
