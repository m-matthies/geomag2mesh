"""Pure editing operations on a Project (used by the CLI ``edit`` command and the
interactive annotator). Each function mutates the project in place and returns a
short description of what changed, or raises ValueError."""
from __future__ import annotations

import numpy as np

from .annotations import Obs, Project, Rod


def _view(p: Project, name: str):
    for v in p.views:
        if v.name == name:
            return v
    raise ValueError(f"no photo named {name!r}")


def set_sphere(p: Project, view: str, label: str, x: float, y: float,
               r: float | None = None, w: float = 1.0) -> str:
    v = _view(p, view)
    old = v.spheres.get(label)
    v.spheres[label] = Obs(float(x), float(y), r if r is not None else (old.r if old else None), w)
    return f"{'moved' if old else 'placed'} {label} in {view}"


def delete_sphere(p: Project, label: str, view: str | None = None) -> str:
    """Remove a sphere from one photo, or everywhere (with its rods and contacts)."""
    if view is not None:
        v = _view(p, view)
        if v.spheres.pop(label, None) is None:
            raise ValueError(f"{label} not annotated in {view}")
        return f"removed {label} from {view}"
    if label not in p.labels:
        raise ValueError(f"no sphere {label!r}")
    for v in p.views:
        v.spheres.pop(label, None)
    p.rods = [r for r in p.rods if label not in (r.a, r.b)]
    p.contacts = [c for c in p.contacts if label not in c]
    p.extra_spheres = [s for s in p.extra_spheres if s != label]
    return f"deleted sphere {label}"


def rename(p: Project, old: str, new: str) -> str:
    """Rename a label everywhere; if ``new`` exists the two spheres are merged
    (observations of ``old`` win only in photos where ``new`` is absent)."""
    if old == new:
        return "nothing to do"
    if old not in p.labels:
        raise ValueError(f"no sphere {old!r}")
    merged = new in p.labels
    for v in p.views:
        if old in v.spheres:
            o = v.spheres.pop(old)
            v.spheres.setdefault(new, o)
    seen, rods = set(), []
    for r in p.rods:
        a, b = (new if r.a == old else r.a), (new if r.b == old else r.b)
        if a == b or frozenset((a, b)) in seen:
            continue
        seen.add(frozenset((a, b)))
        rods.append(Rod(a, b, r.color, r.type))
    p.rods = rods
    p.contacts = [(new if a == old else a, new if b == old else b) for a, b in p.contacts
                  if {new if a == old else a, new if b == old else b} != {new}]
    p.extra_spheres = [new if s == old else s for s in p.extra_spheres]
    return f"{'merged' if merged else 'renamed'} {old} -> {new}"


def toggle_rod(p: Project, a: str, b: str, color: str = "unknown", rod_type: str | None = None) -> str:
    if a == b:
        raise ValueError("a rod needs two different spheres")
    key = frozenset((a, b))
    for r in p.rods:
        if r.key == key:
            p.rods.remove(r)
            return f"removed rod {a}-{b}"
    p.rods.append(Rod(a, b, color, rod_type))
    for s in (a, b):
        if s not in p.labels:
            p.extra_spheres.append(s)
    return f"added rod {a}-{b} ({color})"


def set_rod_color(p: Project, a: str, b: str, color: str) -> str:
    for r in p.rods:
        if r.key == frozenset((a, b)):
            r.color = color
            return f"rod {a}-{b} colour {color}"
    raise ValueError(f"no rod {a}-{b}")


def nearest_sphere(p: Project, view: str, x: float, y: float, max_rel: float = 1.0) -> str | None:
    """Label of the annotated sphere whose disc contains (x, y), if any."""
    v = _view(p, view)
    best, bd = None, np.inf
    for k, o in v.spheres.items():
        d = np.hypot(o.x - x, o.y - y)
        rad = o.r if o.r else 0.02 * max(v.width, v.height)
        if d <= max_rel * rad and d < bd:
            best, bd = k, d
    return best


def next_label(p: Project, prefix: str = "S") -> str:
    used = set(p.labels)
    n = 1
    while f"{prefix}{n:02d}" in used:
        n += 1
    return f"{prefix}{n:02d}"
