"""End-to-end orchestration used by the CLI."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .annotations import Project
from .correspond import (MatchOptions, PosedCamera, build_project, match_with_poses,
                         project_from_tracks, verify_rods)
from .detect import DetectOptions, ViewDetections, detect_view, lab_image
from .images import ImageRecord, load_record, priors_for_project
from .markers import BoardSpec, board_pose, detect_board
from .parts import PartCatalog
from .solve import Solution, SolveOptions, solve


def expected_radius(rec: ImageRecord, pose, parts: PartCatalog, spread=(0.6, 1.7)) -> tuple[float, float]:
    """Apparent sphere radius range (native px) from the camera distance to the mat origin."""
    R = Rotation.from_rotvec(pose[0]).as_matrix()
    cam_centre = -R.T @ np.asarray(pose[1])
    r0 = rec.prior.focal_px * parts.sphere_radius_mm / float(np.linalg.norm(cam_centre))
    return spread[0] * r0, spread[1] * r0


def detect_all(records: list[ImageRecord], parts: PartCatalog, opts: DetectOptions | None = None,
               poses: dict | None = None, log=print) -> list[ViewDetections]:
    out = []
    for rec in records:
        rr = expected_radius(rec, poses[rec.name], parts) if poses and rec.name in poses else None
        d = detect_view(rec.name, rec.load_rgb, (rec.width, rec.height), rec.prior.focal_px, parts, opts,
                        radius_px_native=rr)
        log(f"detect {rec.name}: {len(d.spheres)} spheres, {len(d.rods)} rods")
        out.append(d)
    return out


def save_detections(dets: list[ViewDetections], path: Path) -> None:
    path.write_text(json.dumps([d.to_dict() for d in dets], indent=1))


def load_detections(path: Path) -> list[ViewDetections]:
    return [ViewDetections.from_dict(d) for d in json.loads(path.read_text())]


def board_poses(records: list[ImageRecord], spec: BoardSpec, max_side: int = 2400,
                log=print) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """Camera pose per photo from the marker mat (photos without a usable mat are skipped).
    Cropped photos use the image centre as principal point (a small pose bias the
    bundle adjustment absorbs through a looser prior)."""
    found = {}
    for rec in records:
        rgb, s = rec.load_rgb(max_side)
        ids, corners = detect_board(rgb, spec)
        found[rec.name] = (ids, corners / s)
    _calibrate_focal(records, found, spec, log)
    out = {}
    for rec in records:
        ids, corners = found[rec.name]
        pose = board_pose(ids, corners, spec, rec.prior.focal_px, (rec.width / 2, rec.height / 2))
        if pose is None:
            log(f"mat {rec.name}: not found ({len(ids)} corners)")
            continue
        out[rec.name] = pose
        log(f"mat {rec.name}: {len(ids)} corners, reprojection {pose[2]:.2f} px")
    return out


def _calibrate_focal(records, found, spec, log, min_corners: int = 8) -> None:
    """Without EXIF focal length (wide prior), estimate it from the mat: planar views with
    a known principal point determine f (OpenCV calibrateCamera, principal point, aspect
    ratio and distortion fixed). Updates the priors of uncropped photos in place."""
    import cv2
    groups: dict[tuple, list] = {}
    for r in records:
        ids, c = found[r.name]
        if r.prior.focal_sigma_rel > 0.2 and r.prior.principal_known and len(ids) >= min_corners:
            groups.setdefault((r.width, r.height), []).append((r, ids, c))
    obj3d = spec.corners_3d()
    for (w, h), items in groups.items():
        if len(items) < 2:
            continue
        objp = [obj3d[ids].astype(np.float32) for _, ids, _ in items]
        imgp = [c.astype(np.float32).reshape(-1, 1, 2) for _, _, c in items]
        f0 = items[0][0].prior.focal_px
        K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1.0]])
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
                 cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 |
                 cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
        rms, K, *_ = cv2.calibrateCamera(objp, imgp, (w, h), K0, np.zeros(5), flags=flags)
        f = float(K[0, 0])
        log(f"focal length from mat ({len(items)} photos {w}x{h}): {f:.0f} px (reprojection {rms:.2f} px)")
        for r, _, _ in items:
            r.prior.focal_px, r.prior.focal_sigma_rel = f, 0.02


def auto_project(records: list[ImageRecord], dets: list[ViewDetections], parts: PartCatalog,
                 base_dir: Path, match_opts: MatchOptions | None = None,
                 poses: dict | None = None, log=print) -> Project:
    """Draft annotations from detections: posed triangulation when >= 2 photos have mat
    poses, otherwise rigid alignment of radius-based point clouds."""
    priors = {r.name: r.prior for r in records}
    if poses and len(poses) >= 2:
        return _posed_project(records, dets, parts, base_dir, poses, log)
    files = {}
    for r in records:
        try:
            files[r.name] = str(r.path.resolve().relative_to(base_dir.resolve()))
        except ValueError:
            files[r.name] = str(r.path.resolve())
    proj = build_project(dets, priors, parts, files, match_opts, log)
    proj.base_dir = base_dir
    for v in proj.views:  # carry intrinsics priors so the file is self-contained
        pr = priors[v.name]
        v.camera = {"focal_px": pr.focal_px, "focal_sigma_rel": pr.focal_sigma_rel,
                    "principal_known": pr.principal_known, "camera_id": pr.camera_id}
    return proj


def _posed_project(records, dets, parts, base_dir, poses, log) -> Project:
    keep = [i for i, r in enumerate(records) if r.name in poses]
    if len(keep) < len(records):
        log(f"ignoring {len(records) - len(keep)} photo(s) without a detected mat")
    recs = [records[i] for i in keep]
    dets = [dets[i] for i in keep]
    cams = []
    for r in recs:
        rv, tv, _ = poses[r.name]
        f = r.prior.focal_px
        K = np.array([[f, 0, r.width / 2], [0, f, r.height / 2], [0, 0, 1.0]])
        cams.append(PosedCamera(r.name, K, Rotation.from_rotvec(rv).as_matrix(), np.asarray(tv)))
    tracks, X = match_with_poses(dets, cams, parts, log=log)
    labs = [lab_image(r.load_rgb()[0]) for r in recs]
    files = {}
    for r in recs:
        try:
            files[r.name] = str(r.path.resolve().relative_to(base_dir.resolve()))
        except ValueError:
            files[r.name] = str(r.path.resolve())
    proj = project_from_tracks(dets, cams, tracks, X, parts, labs, files, log=log)
    proj.base_dir = base_dir
    proj.support_plane = True  # mat poses define the frame: the mat is z = 0
    for v, r in zip(proj.views, recs):
        rv, tv, rms = poses[r.name]
        pr = r.prior
        # pose uncertainty grows with board reprojection error and with an unknown principal point
        sd_deg = 0.05 + 0.05 * rms + (0.5 if not pr.principal_known else 0.0)
        sd_mm = 0.3 + 0.3 * rms + (3.0 if not pr.principal_known else 0.0)
        v.camera = {"focal_px": pr.focal_px, "focal_sigma_rel": pr.focal_sigma_rel,
                    "principal_known": pr.principal_known, "camera_id": pr.camera_id,
                    "rvec": [float(x) for x in rv], "tvec": [float(x) for x in tv],
                    "pose_sigma_deg": sd_deg, "pose_sigma_mm": sd_mm}
    return proj


def solve_project(project: Project, opts: SolveOptions | None = None, verify: bool = True,
                  log=print) -> tuple[Solution, Project]:
    """Solve; when ``verify``, iteratively remove rods whose solved length matches no
    rod type (spurious detections), solving with loose rod tolerance first."""
    opts = opts or SolveOptions()
    priors = priors_for_project(project)
    if verify and project.rods:
        loose = replace(opts, rod_tol_rel=0.08, restarts=1, leave_one_out=False)
        for _ in range(4):
            sol = solve(project, priors, loose, log=lambda *_: None)
            _sync_removed(project, sol, log)
            bad = verify_rods(project, sol)
            if not bad:
                break
            log(f"removing {len(bad)} rod(s) inconsistent with part lengths: "
                + ", ".join(f"{r.a}-{r.b}" for r in bad))
            keys = {r.key for r in bad}
            project.rods = [r for r in project.rods if r.key not in keys]
        # every sphere of a framework carries rods; rod-less spheres are false matches
        deg = project.degree()
        lonely = [k for k, d in deg.items() if d == 0]
        if lonely and len(lonely) < len(deg):
            from .edit import delete_sphere
            for k in lonely:
                delete_sphere(project, k)
            log(f"removed {len(lonely)} sphere(s) without any rod: {', '.join(lonely)}")
    sol = solve(project, priors, opts, log=log)
    _sync_removed(project, sol, lambda *_: None)
    return sol, project


def _sync_removed(project: Project, sol: Solution, log) -> None:
    """Apply sphere removals made by the solver (false matches) to the project."""
    from .edit import delete_sphere
    gone = [k for k in project.labels if k not in sol.labels]
    for k in gone:
        delete_sphere(project, k)
    if gone:
        log(f"removed {len(gone)} falsely matched sphere(s): {', '.join(gone)}")


def load_records(paths) -> list[ImageRecord]:
    return [load_record(p) for p in paths]
