"""Tests for the interactive viewer, driven without opening a window."""

from types import SimpleNamespace

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from parallelbrot.viewer import Viewer  # noqa: E402


def event(x=None, y=None, button=None, key=None):
    return SimpleNamespace(xdata=x, ydata=y, button=button, key=key)


@pytest.fixture
def viewer():
    v = Viewer(160, 120, (-0.5, 0.0), 1.0, 64, "fire", "auto")
    figure, v.axes = plt.subplots()
    v.image = v.axes.imshow(v._frame())
    yield v
    plt.close(figure)


def test_scroll_keeps_the_point_under_the_cursor(viewer):
    before = viewer._complex_at(40.0, 30.0)
    viewer._on_scroll(event(40.0, 30.0, "up"))
    after = viewer._complex_at(40.0, 30.0)
    assert viewer.zoom > 1.0
    assert before == pytest.approx(after, abs=1e-12)


def test_drag_translates_by_the_dragged_distance(viewer):
    start = viewer._complex_at(20.0, 20.0)
    viewer._on_press(event(20.0, 20.0))
    viewer._on_motion(event(60.0, 50.0))
    viewer._on_release(event())
    assert start == pytest.approx(viewer._complex_at(60.0, 50.0), abs=1e-12)


def test_keys_change_state(viewer):
    viewer._on_key(event(key="+"))
    assert viewer.max_iterations == 128
    viewer._on_key(event(key="-"))
    assert viewer.max_iterations == 64

    before = viewer.scheme
    viewer._on_key(event(key="c"))
    assert viewer.scheme != before

    viewer._on_key(event(key="right"))
    assert viewer.center[0] > -0.5

    viewer._on_key(event(key="r"))
    assert (viewer.center, viewer.zoom, viewer.max_iterations) == ([-0.5, 0.0], 1.0, 128)


def test_unknown_key_is_ignored(viewer):
    state = (list(viewer.center), viewer.zoom, viewer.max_iterations, viewer.scheme)
    viewer._on_key(event(key="z"))
    viewer._on_key(event(key=None))
    assert (list(viewer.center), viewer.zoom, viewer.max_iterations, viewer.scheme) == state


def test_events_outside_the_axes_do_not_crash(viewer):
    viewer._on_scroll(event(None, None, "up"))
    viewer._on_motion(event(None, None))
    viewer._on_press(event(None, None))
    assert viewer.zoom == 1.0
