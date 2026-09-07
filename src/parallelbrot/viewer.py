"""Interactive pan-and-zoom viewer built on matplotlib.

The C++ viewer in this repository is a separate GLFW binary that the wheels
do not ship, since it needs OpenGL and a display at import time. The compute
core is fast enough to drive a window from Python instead: a 1280x720 frame
at the default iteration count costs a few milliseconds.
"""

from __future__ import annotations

from . import COLOR_SCHEMES, render

_ZOOM_STEP = 1.3
_PAN_KEYS = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}


class Viewer:
    """Mutable view state plus the matplotlib event handlers that drive it."""

    def __init__(self, width, height, center, zoom, max_iterations, color_scheme, backend):
        self.width = width
        self.height = height
        self.center = list(center)
        self.zoom = zoom
        self.max_iterations = max_iterations
        self.scheme = COLOR_SCHEMES.index(color_scheme)
        self.backend = backend
        self._drag = None

    # ── coordinate mapping ──────────────────────────────────
    # Mirrors the kernels, except that row 0 is the top here: render()
    # returns the image already flipped for display.

    def _scale(self):
        return 4.0 / self.zoom

    def _complex_at(self, px, py):
        scale = self._scale()
        return (
            self.center[0] + scale * (self.width / self.height) * (px / self.width - 0.5),
            self.center[1] + scale * (0.5 - py / self.height),
        )

    def _frame(self):
        return render(
            self.width, self.height,
            center=tuple(self.center),
            zoom=self.zoom,
            max_iterations=self.max_iterations,
            color_scheme=COLOR_SCHEMES[self.scheme],
            backend=self.backend,
        )

    def _title(self):
        return (f"parallelbrot — zoom {self.zoom:.4g} — {self.max_iterations} iter "
                f"— {COLOR_SCHEMES[self.scheme]}")

    # ── event handlers ──────────────────────────────────────

    def _refresh(self):
        self.image.set_data(self._frame())
        self.axes.set_title(self._title(), fontsize=9)
        self.image.figure.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.xdata is None:
            return
        # Keep the point under the cursor fixed while the scale changes.
        before = self._complex_at(event.xdata, event.ydata)
        self.zoom *= _ZOOM_STEP if event.button == "up" else 1 / _ZOOM_STEP
        after = self._complex_at(event.xdata, event.ydata)
        self.center[0] += before[0] - after[0]
        self.center[1] += before[1] - after[1]
        self._refresh()

    def _on_press(self, event):
        if event.xdata is not None:
            self._drag = (event.xdata, event.ydata)

    def _on_release(self, _event):
        self._drag = None

    def _on_motion(self, event):
        if self._drag is None or event.xdata is None:
            return
        origin = self._complex_at(*self._drag)
        current = self._complex_at(event.xdata, event.ydata)
        self.center[0] += origin[0] - current[0]
        self.center[1] += origin[1] - current[1]
        self._drag = (event.xdata, event.ydata)
        self._refresh()

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            import matplotlib.pyplot as plt
            plt.close(self.image.figure)
            return
        elif key in ("+", "="):
            self.max_iterations = min(self.max_iterations * 2, 1 << 16)
        elif key == "-":
            self.max_iterations = max(self.max_iterations // 2, 2)
        elif key == "c":
            self.scheme = (self.scheme + 1) % len(COLOR_SCHEMES)
        elif key == "r":
            self.center, self.zoom, self.max_iterations = [-0.5, 0.0], 1.0, 128
        elif key in _PAN_KEYS:
            dx, dy = _PAN_KEYS[key]
            step = self._scale() * 0.1
            self.center[0] += dx * step * (self.width / self.height)
            self.center[1] += dy * step
        else:
            return
        self._refresh()

    def show(self):
        import matplotlib.pyplot as plt

        figure, self.axes = plt.subplots(figsize=(self.width / 100, self.height / 100))
        self.image = self.axes.imshow(self._frame(), interpolation="nearest")
        self.axes.set_axis_off()
        self.axes.set_title(self._title(), fontsize=9)
        figure.tight_layout()

        for event, handler in (
            ("scroll_event", self._on_scroll),
            ("button_press_event", self._on_press),
            ("button_release_event", self._on_release),
            ("motion_notify_event", self._on_motion),
            ("key_press_event", self._on_key),
        ):
            figure.canvas.mpl_connect(event, handler)

        plt.show()


def run(**kwargs) -> int:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print(
            "parallelbrot: the interactive viewer needs matplotlib.\n"
            "  pip install 'parallelbrot[viewer]'   (or: pip install matplotlib)",
        )
        return 1

    print(
        "Controls: drag to pan · scroll to zoom · arrows to pan · "
        "+/- iterations · C colours · R reset · Q quit"
    )
    Viewer(**kwargs).show()
    return 0
