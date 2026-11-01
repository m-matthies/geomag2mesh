# geomag2mesh

[![tests](https://github.com/m-matthies/geomag2mesh/actions/workflows/tests.yml/badge.svg)](https://github.com/m-matthies/geomag2mesh/actions/workflows/tests.yml)

Reconstruct a 3-D mesh of a Geomag-style magnetic construction (steel spheres joined by
equal-length magnetic rods) from ordinary photographs, and export it as GLB, STL and OBJ
in millimetres.

The tool exploits what makes these objects special instead of treating them as generic
photogrammetry targets. Thin, specular parts photographed from few views defeat dense
multi-view stereo.

* Every sphere has the same known diameter (12.7 mm), so its apparent radius gives its
  depth: `z = f R / r`.
* Every rod has a known length, so sphere centres joined by a rod are exactly one *pitch*
  apart (27.0 + 12.7 = 39.7 mm for short rods).
* Spheres that touch directly are exactly one diameter apart; the solver detects and
  enforces such contacts.

A model is therefore just sphere centres plus a graph. The solver fits that model to the
photographs by bundle adjustment with rigid-part constraints, and the mesh is generated
from it.

## Install

```bash
git clone https://github.com/m-matthies/geomag2mesh.git && cd geomag2mesh
pip install -e ".[all,dev]"        # numpy, scipy, opencv-python-headless, pillow, trimesh,
                                # + pillow-heif (iPhone HEIC), manifold3d (watertight STL),
                                # + matplotlib (interactive annotator)
pytest -q                       # 15 tests, about 75 s (includes rendered end-to-end runs)
```

Python >= 3.10. CI tests Python 3.10-3.14 on Linux, Python 3.13 on macOS and Windows, and
the minimum declared dependency versions on Python 3.10.
OpenCV must be a single distribution: having both `opencv-python` and
`opencv-contrib-python` installed can shadow modules.

## Capture protocol (decides whether automation works)

1. Print the marker mat at 100 % scale: `geomag2mesh mat -o mat.pdf` (A4, 9 x 6 squares
   of 30 mm). For larger builds use A3: `--squares 14x10 --square-mm 28 --marker-mm 21`.
   Measure one square with a ruler and pass the measured value as `--square-mm`.
2. Put the build on the mat, away from other coloured objects.
3. Take **12-16 photos** walking around it at two heights (about 30 deg and 55 deg
   elevation). Keep the whole build and a good part of the mat in frame, and keep the
   build at a similar size in each photo.
4. Do **not** crop or resize. Original phone files (HEIC or JPEG) carry the focal length
   in EXIF. Without EXIF the focal length is calibrated from the mat. If you must crop
   (e.g. for privacy), record the offset as the view's `camera.principal` in the
   annotation file (see `examples/twin_cube/annotations.json`, photo IMG_1078); a crop
   with a known offset leaves the reconstruction unchanged.
5. Diffuse light helps. Avoid strong coloured reflections close to the build.

## Workflows

```bash
# Fully automatic (with mat)
geomag2mesh run photos/ -o out/ --mat [--squares 9x6 --square-mm 30 --marker-mm 22]

# Automatic draft, then correct by hand
geomag2mesh auto photos/ -o out/ [--mat]
geomag2mesh annotate out/annotations.json --solution out/solution.json   # desktop window
geomag2mesh solve out/annotations.json -o out/

# Fully manual (any photos, also without mat)
geomag2mesh init photos/ -o ann.json
geomag2mesh annotate ann.json
geomag2mesh solve ann.json -o out/ --loo

# Scriptable corrections
geomag2mesh edit ann.json --rename S07 S03   # merge two tracks
geomag2mesh edit ann.json --delete S12 --rod S01 S04 magenta

# Calibrate the colour palette for a different set (prints JSON for --parts)
geomag2mesh palette ann.json
```

Options: `--rod short|long|mixed` (rod length used in the build; default `short`;
`mixed` lets the solver classify each rod) and `--parts parts.json` (sphere and rod sizes,
colour palette).

Outputs in `out/`:

| file | content |
|---|---|
| `model.glb` | coloured mesh, PBR materials (metallic spheres) |
| `model.stl` | boolean union per rigid body (bodies joined only by magnetic contact stay separate), watertight |
| `model.obj` | per-vertex colours |
| `model_centres.json` | sphere centres (mm, z up, support plane at z = 0), rods, contacts |
| `solution.json` | centres, cameras, statistics (input for `annotate --solution`) |
| `annotations.json` | labelled observations (editable) |
| `report.md` / `report.json` | verdict, per-photo residuals, rod-length errors, optional leave-one-photo-out stability |
| `check_*.jpg` | solved model drawn over each photo |
| `detect_*.jpg` | raw detections |

### Quality gate

Each run ends with a verdict (`good`, `check` or `unreliable`) and its reasons:
- reprojection RMS per photo relative to the sphere radius (good <= 0.25, unreliable > 0.5);
- rod-length consistency;
- the fraction of spheres seen in at least two photos (unreliable < 60 %);
- rejected observations;
- rod-less spheres;
- completeness (closed frameworks have >= 3 rods per sphere).

The thresholds are heuristics that I set on the test data below. They are not calibrated
error rates.

## How it works

| stage | method | complexity |
|---|---|---|
| mat pose | OpenCV ChArUco corners -> IPPE PnP + LM; focal length by `calibrateCamera` (principal point fixed) when EXIF is missing | O(markers) |
| sphere detection | Hough circles in 3 radius bands, scored by coherent radial edges x low saturation x specular highlight x texture x (1 - single-colour fill) x (1 - mat pattern) x rod-stub test on rings outside the rim; size gate from the mat pose (or a per-image consensus); NMS that allows depth overlap | O(C p) for C candidates, p disc pixels |
| rod detection | for sphere pairs whose radius-based 3-D distance fits a rod pitch: palette classification (CIE Lab) of a strip between the discs, excluding samples occluded by other spheres | O(n^2 s) |
| matching, with mat | DLT triangulation of detection pairs; accept when reprojection < 0.35 r and the predicted radius f R / z agrees within 25 % in every supporting photo; greedy selection by support; >= 3 photos per sphere when >= 6 photos; rods from 2-D detections and multi-view colour voting, restricted to pairs whose 3-D distance is within 3.5 % of the pitch | O(V^3 n^2) |
| matching, without mat (experimental) | per-photo metric point clouds from sphere radii; RANSAC over rod-path triples (colour- and distance-compatible); Mahalanobis inliers with depth-elongated covariances; union-find track merging that refuses same-photo conflicts | O(T^2 n log n) per photo pair |
| solve | robust (soft-L1) trust-region bundle adjustment: reprojection, apparent radius, rod lengths (1 % sigma), contacts, one-sided non-penetration, focal prior, mat pose priors; no shape template: initialised from radius depth + Umeyama registration, or by triangulation when poses are known; automatic contact detection, outlier rejection, removal of false-match spheres, rod-type classification | dense TR for <= 600 parameters (converges in tens of evaluations), sparse LSMR above |
| mesh | icospheres + capped cylinders (rod ends embedded 5 mm from centres so unions are manifold); table frame from the mat, or from the convex-hull face opposite the cameras' up direction | O(parts) |

## Validation

All numbers are from this repository's test data. They are my own measurements, not
published benchmarks.

**Hand-annotated real photos** (`examples/twin_cube`, 4 iPhone XR photos, all cropped;
IMG_1078 by a known offset, the other three with unknown offsets):
- Two sheared cubes, 16 spheres, 24 rods, 4 sphere contacts.
- The template-free solver reproduces the first, separately coded reconstruction to
  0.55 mm RMS (0.99 mm max). Rod lengths deviate from the part length by 0.14 mm RMS.
- All 4 contacts are found automatically. The contact distance was left free in a test
  solve, and the fitted sphere-diameter-to-pitch ratio was 0.317 against 0.320 from the
  part sizes.
- Leaving out any one photo moves the centres by 0.27-0.87 mm RMS (2.1 mm at most, for the
  mostly hidden sphere L).

**Rendered photos on the mat** (twin cube, simple shaded rendering with fixed random seed;
see `synthetic.render_scene` and `tests/test_pipeline.py`):

| photos | spheres found (of 16) | max centre error | rods found (of 24) | false spheres / rods |
|---:|---:|---:|---:|---:|
| 6 | 8 | 0.6 mm | 5 | 0 / 0 |
| 12 | 12 | 0.5 mm | 12 | 0 / 0 |
| 16 | 15 | 0.5 mm | 16 | 0 / 0 |

Rod colours were correct for every reported rod. Errors are absolute, in the mat frame,
with no alignment. Missing parts are the occluded inner spheres and rods of the cube pair.

**Automatic run on the 4 real photos, without mat:** fails. Tracks mix different spheres
and the result is about 40 mm RMS off. The quality gate reports `unreliable`. Sphere
detection on those photos finds 33 of 44 clearly visible spheres with 25 false positives.
Detected rods have the correct colour every time, but more than half of the rod
detections are spurious (28 of 50).

## Limitations and failure modes

* Automatic reconstruction without the mat is experimental and fails on wide-baseline,
  heavily self-occluding photo sets like the example. The mat turns sphere matching into
  triangulation, which is what makes automation reliable.
* Spheres hidden in every photo cannot be detected. Annotate them as `extra_spheres` with
  rods (3 rods determine a position up to reflection), or take more photos.
* The automatic path reports only parts it can verify. Expect missing inner rods and
  spheres (the quality gate says "possibly incomplete"), then add them in the annotator.
* The default colour palette was measured on one Geomag Classic set under indoor light.
  Red, green and yellow entries are placeholders. Calibrate with `geomag2mesh palette`.
* Rod-length coincidences: a cube's body diagonal (68.8 mm) is within 3 % of the long-rod
  pitch (70.9 mm). Mixed-rod builds (`--rod mixed`) are therefore more ambiguous.
* Panels, plates and non-Geomag parts are not modelled. Rods are straight cylinders,
  without the moulded grip texture.
* The rendered validation images are much simpler than real photographs (no
  inter-reflections, ideal colours). The numbers above are an upper bound for real
  photos taken on the mat. I have not yet validated the mat workflow on real photographs.
* The interactive annotator (`annotate`) needs a desktop session. Its editing operations
  are unit-tested; the window itself is not tested automatically.

## File format

See `src/geomag2mesh/annotations.py` for the annotation schema. All pixel coordinates refer
to the upright (EXIF-rotated) image. Sphere labels are arbitrary strings.

## References

* Geomag part dimensions (sphere 12.7 mm; rods 27.0 x 7.4 mm and 58.2 x 7.6 mm):
  H. J. Reggel, *GMTRCPLY: Magnetic sticks*, https://hjreggel.net/gmtrcply/magsticks/index.html
* S. Umeyama, "Least-squares estimation of transformation parameters between two point
  patterns", *IEEE TPAMI* 13(4):376-380, 1991. doi:10.1109/34.88573
* B. Triggs, P. McLauchlan, R. Hartley, A. Fitzgibbon, "Bundle adjustment: a modern
  synthesis", *Vision Algorithms: Theory and Practice*, LNCS 1883, 2000.
  doi:10.1007/3-540-44480-7_21
* R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.,
  Cambridge University Press, 2004. doi:10.1017/CBO9780511811685
* S. Garrido-Jurado et al., "Automatic generation and detection of highly reliable fiducial
  markers under occlusion", *Pattern Recognition* 47(6):2280-2292, 2014.
  doi:10.1016/j.patcog.2014.01.005
* T. Collins, A. Bartoli, "Infinitesimal plane-based pose estimation" (IPPE), *IJCV*
  109:252-286, 2014. doi:10.1007/s11263-014-0725-5

## Licence

MIT, see `LICENSE`.
