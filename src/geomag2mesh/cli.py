"""Command-line interface: ``geomag2mesh <command> ...``.

Typical workflows
-----------------
Fully automatic (best with the printed marker mat under the model)::

    geomag2mesh mat -o mat.pdf                      # print at 100 %, once
    geomag2mesh run photos/ -o out/ --mat           # detect, match, solve, mesh, report

Automatic draft, then manual correction::

    geomag2mesh auto photos/ -o out/                # writes out/annotations.json + overlays
    geomag2mesh annotate out/annotations.json --solution out/solution.json
    geomag2mesh solve out/annotations.json -o out/

Manual from scratch::

    geomag2mesh init photos/ -o ann.json && geomag2mesh annotate ann.json
    geomag2mesh solve ann.json -o out/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__


def _parts(args, base=None):
    from .parts import PartCatalog
    cat = base or PartCatalog()
    if getattr(args, "parts", None):
        cat = PartCatalog.from_dict(json.loads(Path(args.parts).read_text()))
    _apply_rod_arg(cat, getattr(args, "rod", None))
    return cat


def _apply_rod_arg(cat, rod: str | None) -> None:
    if rod == "mixed":
        cat.mixed_rods = True
    elif rod:
        cat.default_rod, cat.mixed_rods = rod, False


def _board(args):
    from .markers import BoardSpec
    sx, sy = (int(v) for v in args.squares.lower().split("x"))
    return BoardSpec(sx, sy, args.square_mm, args.marker_mm, args.dictionary)


def _add_board_args(ap):
    ap.add_argument("--squares", default="9x6", help="mat squares, e.g. 9x6")
    ap.add_argument("--square-mm", type=float, default=30.0)
    ap.add_argument("--marker-mm", type=float, default=22.0)
    ap.add_argument("--dictionary", default="5x5_250", choices=["4x4_250", "5x5_250", "6x6_250"])


def cmd_mat(args):
    from .markers import save_board_pdf
    w, h = save_board_pdf(_board(args), args.output, args.dpi)
    print(f"wrote {args.output} ({w:.0f} x {h:.0f} mm); print at 100 % and check one square = "
          f"{args.square_mm:g} mm")


def _auto(args, out: Path):
    from .images import find_images
    from .pipeline import auto_project, board_poses, detect_all, load_records, save_detections
    from .report import detection_overlay
    recs = load_records(find_images(args.photos))
    if len(recs) < 2:
        sys.exit("need at least two photos")
    parts = _parts(args)
    out.mkdir(parents=True, exist_ok=True)
    poses = board_poses(recs, _board(args)) if args.mat else None
    if not args.mat:
        print("note: matching without the marker mat is experimental; check the overlays "
              "(see README, 'Capture protocol')")
    elif len(poses) < 2:
        print("note: the mat was found in fewer than two photos; falling back to matching "
              "without camera poses (experimental)")
    dets = detect_all(recs, parts, poses=poses)
    save_detections(dets, out / "detections.json")
    for r, d in zip(recs, dets):
        detection_overlay(r.load_rgb(1400)[0], _scaled(d, r, 1400), out / f"detect_{r.name}.jpg")
    proj = auto_project(recs, dets, parts, out, poses=poses)
    proj.save(out / "annotations.json")
    print(f"wrote {out / 'annotations.json'} ({len(proj.labels)} spheres, {len(proj.rods)} rods)")
    return proj


def _scaled(det, rec, max_side):
    from copy import deepcopy
    s = min(1.0, max_side / max(rec.width, rec.height))
    d = deepcopy(det)
    for sp in d.spheres:
        sp.x, sp.y, sp.r = sp.x * s, sp.y * s, sp.r * s
    return d


def _solve_and_export(proj, out: Path, args):
    from .mesh import export
    from .pipeline import solve_project
    from .report import overlays, quality, write_summary
    from .solve import SolveOptions
    opts = SolveOptions(leave_one_out=args.loo, restarts=args.restarts)
    sol, proj = solve_project(proj, opts, verify=not args.no_verify)
    sol.save(out / "solution.json")
    q = quality(sol, proj)
    paths = export(sol, proj.parts, out, stem=args.stem)
    overlays(sol, proj, out)
    write_summary(sol, proj, q, out / "report.md")
    print(f"verdict: {q['verdict']}")
    for r in q["reasons"]:
        print(f"  - {r}")
    print("wrote " + ", ".join(str(p) for p in paths.values()) + f", {out / 'report.md'}")
    if q["verdict"] == "unreliable":
        print("The automatic labelling is probably wrong somewhere. Inspect check_*.jpg, fix with "
              "'geomag2mesh annotate', then re-run 'geomag2mesh solve'.")
    return sol


def cmd_auto(args):
    _auto(args, Path(args.output))


def cmd_run(args):
    out = Path(args.output)
    proj = _auto(args, out)
    _solve_and_export(proj, out, args)


def cmd_solve(args):
    from .annotations import Project
    proj = Project.load(args.annotations)
    _apply_rod_arg(proj.parts, args.rod)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    _solve_and_export(proj, out, args)


def cmd_init(args):
    from .annotations import Project, View
    from .images import find_images, load_record
    out = Path(args.output)
    views = []
    for p in find_images(args.photos):
        rec = load_record(p)
        try:
            f = str(p.resolve().relative_to(out.parent.resolve()))
        except ValueError:
            f = str(p.resolve())
        views.append(View(rec.name, f, rec.width, rec.height))
    proj = Project(views=views, rods=[], parts=_parts(args))
    proj.base_dir = out.parent
    proj.save(out)
    print(f"wrote {out} with {len(views)} photos; next: geomag2mesh annotate {out}")


def cmd_annotate(args):
    from .annotate import run
    run(Path(args.annotations), Path(args.solution) if args.solution else None)


def cmd_edit(args):
    from . import edit
    from .annotations import Project
    p = Project.load(args.annotations)
    for old, new in args.rename or []:
        print(edit.rename(p, old, new))
    for lab in args.delete or []:
        print(edit.delete_sphere(p, lab))
    for a, b, *col in args.rod or []:
        print(edit.toggle_rod(p, a, b, col[0] if col else "unknown"))
    p.save(args.annotations)


def cmd_palette(args):
    """Median L*a*b* of annotated rods per colour name (to calibrate a new set)."""
    from .annotations import Project
    from .detect import lab_image
    from .images import load_record
    p = Project.load(args.annotations)
    samples: dict[str, list] = {}
    for v in p.views:
        path = p.image_path(v)
        if path is None or not path.exists():
            continue
        rec = load_record(path)
        rgb, s = rec.load_rgb(1600)
        s *= rec.width / v.width
        lab = lab_image(rgb)
        for r in p.rods:
            if r.a in v.spheres and r.b in v.spheres and r.color != "unknown":
                a, b = v.spheres[r.a], v.spheres[r.b]
                for t in np.linspace(0.4, 0.6, 7):
                    x, y = int((a.x + (b.x - a.x) * t) * s), int((a.y + (b.y - a.y) * t) * s)
                    if 0 <= y < lab.shape[0] and 0 <= x < lab.shape[1]:
                        samples.setdefault(r.color, []).append(lab[y, x])
    pal = {k: [round(float(x), 1) for x in np.median(np.array(v), 0)] for k, v in samples.items()}
    print(json.dumps({"palette": pal}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="geomag2mesh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_parts(p):
        p.add_argument("--rod", choices=["short", "long", "mixed"],
                       help="rod length used in the build (default short); 'mixed' allows both")
        p.add_argument("--parts", help="JSON file with a PartCatalog (sizes, palette)")

    def common_solve(p):
        p.add_argument("--loo", action="store_true", help="leave-one-photo-out uncertainty")
        p.add_argument("--restarts", type=int, default=3)
        p.add_argument("--no-verify", action="store_true", help="keep rods with implausible lengths")
        p.add_argument("--stem", default="model", help="output file stem")

    p = sub.add_parser("mat", help="write the printable marker mat")
    p.add_argument("-o", "--output", default="geomag_mat.pdf")
    p.add_argument("--dpi", type=int, default=300)
    _add_board_args(p)
    p.set_defaults(func=cmd_mat)

    for name, fn, hlp in (("auto", cmd_auto, "detect + match: draft annotations"),
                          ("run", cmd_run, "auto + solve + mesh + report")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("photos", nargs="+", help="image files or directories")
        p.add_argument("-o", "--output", required=True, help="output directory")
        p.add_argument("--mat", action="store_true", help="photos show the marker mat")
        _add_board_args(p)
        common_parts(p)
        if name == "run":
            common_solve(p)
        p.set_defaults(func=fn)

    p = sub.add_parser("solve", help="solve annotations, export mesh and report")
    p.add_argument("annotations")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--rod", choices=["short", "long", "mixed"])
    common_solve(p)
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("init", help="empty annotation file for manual annotation")
    p.add_argument("photos", nargs="+")
    p.add_argument("-o", "--output", required=True)
    common_parts(p)
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("annotate", help="interactive editor (desktop window)")
    p.add_argument("annotations")
    p.add_argument("--solution", help="solution.json to draw predicted sphere positions")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("edit", help="scriptable label/rod edits")
    p.add_argument("annotations")
    p.add_argument("--rename", nargs=2, action="append", metavar=("OLD", "NEW"),
                   help="rename; merges when NEW exists")
    p.add_argument("--delete", action="append", metavar="LABEL")
    p.add_argument("--rod", nargs="+", action="append", metavar="A B [COLOUR]", help="toggle a rod")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("palette", help="measure rod colours from annotated rods")
    p.add_argument("annotations")
    p.set_defaults(func=cmd_palette)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
