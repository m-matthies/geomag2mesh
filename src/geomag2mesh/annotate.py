"""Interactive annotation editor (matplotlib window; needs a desktop session).

Controls (shown in the window title):
  n / p        next / previous photo
  left click   on a sphere: select it; elsewhere: place the *active* label there
  right click  remove the clicked sphere from this photo
  l            type a new active label in the terminal (empty = next free label)
  m            make the selected sphere the active label (to place it in other photos)
  r            rod mode: click two spheres to add/remove the rod between them
  c            set the colour of the last rod (typed in the terminal)
  h            toggle half weight (partly hidden) for the selected sphere
  g            rename/merge the selected label (typed in the terminal)
  s            save        q   save and quit

If a solution file is given, the solved model is drawn faintly so hidden or
missing spheres can be placed where the model predicts them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import edit
from .annotations import Project
from .images import load_record
from .parts import RENDER_RGB


class Annotator:
    def __init__(self, project: Project, path: Path, solution=None, max_side: int = 1600):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.p, self.path, self.sol = project, path, solution
        self.vi = 0
        self.active = edit.next_label(project)
        self.selected: str | None = None
        self.rod_mode, self.rod_first, self.last_rod = False, None, None
        self.max_side = max_side
        self._img_cache: dict[str, tuple[np.ndarray, float]] = {}
        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw()

    # ------------------------------------------------------------------ drawing
    def image(self, v):
        if v.name not in self._img_cache:
            path = self.p.image_path(v)
            rec = load_record(path)
            rgb, s = rec.load_rgb(self.max_side)
            self._img_cache[v.name] = (rgb, s * rec.width / v.width)
        return self._img_cache[v.name]

    def draw(self):
        ax = self.ax
        ax.clear()
        v = self.p.views[self.vi]
        rgb, s = self.image(v)
        ax.imshow(rgb)
        if self.sol is not None and v.name in [c.name for c in self.sol.cameras]:
            cam = [c for c in self.sol.cameras if c.name == v.name][0]
            uv, z = cam.project(self.sol.X)
            rad = cam.focal_px * self.sol.sphere_diameter_mm / 2 / z
            for i, k in enumerate(self.sol.labels):
                ax.add_patch(self.plt.Circle(uv[i] * s, rad[i] * s, fill=False, ls=":", ec="w", alpha=0.6))
                ax.text(*(uv[i] * s), k, color="w", alpha=0.6, fontsize=7, ha="center")
        for r in self.p.rods:
            if r.a in v.spheres and r.b in v.spheres:
                a, b = v.spheres[r.a], v.spheres[r.b]
                col = np.array(RENDER_RGB.get(r.color, RENDER_RGB["unknown"])) / 255
                ax.plot([a.x * s, b.x * s], [a.y * s, b.y * s], color=col, lw=3, alpha=0.9)
        for k, o in v.spheres.items():
            rad = (o.r or 0.02 * max(v.width, v.height)) * s
            ec = "yellow" if k == self.selected else ("lime" if o.w >= 1 else "orange")
            ax.add_patch(self.plt.Circle((o.x * s, o.y * s), rad, fill=False, ec=ec, lw=2))
            ax.text(o.x * s, o.y * s, k, color="cyan", fontsize=9, ha="center", va="center",
                    fontweight="bold")
        mode = "ROD MODE" if self.rod_mode else f"active label {self.active}"
        ax.set_title(f"{v.name} ({self.vi + 1}/{len(self.p.views)})  |  {mode}  |  "
                     "n/p photo, click place/select, right-click remove, l label, m use selected, "
                     "r rods, c colour, h half weight, g rename, s save, q quit", fontsize=8)
        ax.set_axis_off()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------- events
    def _to_native(self, ev):
        _, s = self.image(self.p.views[self.vi])
        return ev.xdata / s, ev.ydata / s

    def on_click(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None or self.fig.canvas.toolbar.mode:
            return
        v = self.p.views[self.vi]
        x, y = self._to_native(ev)
        hit = edit.nearest_sphere(self.p, v.name, x, y)
        if ev.button == 3 and hit:
            print(edit.delete_sphere(self.p, hit, v.name))
        elif ev.button == 1 and self.rod_mode:
            if hit is None:
                return
            if self.rod_first is None:
                self.rod_first = hit
                print(f"rod from {hit} ...")
            else:
                print(edit.toggle_rod(self.p, self.rod_first, hit))
                self.last_rod, self.rod_first = (self.rod_first, hit), None
        elif ev.button == 1:
            if hit:
                self.selected = hit
            else:
                r = self._guess_radius(v)
                print(edit.set_sphere(self.p, v.name, self.active, x, y, r))
                self.selected = self.active
        self.draw()

    def _guess_radius(self, v):
        rs = [o.r for o in v.spheres.values() if o.r]
        return float(np.median(rs)) if rs else None

    def on_key(self, ev):
        k = ev.key
        if k in ("n", "right"):
            self.vi = (self.vi + 1) % len(self.p.views)
        elif k in ("p", "left"):
            self.vi = (self.vi - 1) % len(self.p.views)
        elif k == "l":
            lab = input("active label (empty = next free): ").strip()
            self.active = lab or edit.next_label(self.p)
        elif k == "m" and self.selected:
            self.active = self.selected
        elif k == "r":
            self.rod_mode, self.rod_first = not self.rod_mode, None
        elif k == "c" and self.last_rod:
            col = input(f"colour for rod {self.last_rod[0]}-{self.last_rod[1]} "
                        f"({', '.join(self.p.parts.palette)}): ").strip()
            if col:
                print(edit.set_rod_color(self.p, *self.last_rod, col))
        elif k == "h" and self.selected:
            o = self.p.views[self.vi].spheres.get(self.selected)
            if o:
                o.w = 0.5 if o.w >= 1 else 1.0
        elif k == "g" and self.selected:
            new = input(f"rename/merge {self.selected} into: ").strip()
            if new:
                print(edit.rename(self.p, self.selected, new))
                self.selected = new
        elif k == "s":
            self.p.save(self.path)
            print(f"saved {self.path}")
        elif k == "q":
            self.p.save(self.path)
            print(f"saved {self.path}")
            self.plt.close(self.fig)
            return
        self.draw()


def run(annotations: Path, solution: Path | None = None) -> None:
    from .solve import Solution
    p = Project.load(annotations)
    sol = Solution.load(solution) if solution and Path(solution).exists() else None
    Annotator(p, Path(annotations), sol)
    import matplotlib.pyplot as plt
    plt.show()
