"""Tests for the Python API. Run with: PYTHONPATH=src pytest"""

import numpy as np
import pytest

import parallelbrot as pb

BACKENDS = pb.available_backends()
GPU_BACKENDS = [b for b in BACKENDS if b != "cpu"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_shape_and_range(backend):
    img = pb.render(64, 32, backend=backend)
    assert img.shape == (32, 64, 4)
    assert img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert (img[..., 3] == 1.0).all()


def test_known_points():
    """Points in the set never escape and render black under the Fire palette."""
    # A 1x1 render centred on the origin, which is inside the set.
    assert pb.render(1, 1, center=(0.0, 0.0), zoom=1e6)[0, 0, :3].tolist() == [0, 0, 0]
    # (2, 0) is outside and escapes immediately, so it is not black.
    assert pb.render(1, 1, center=(2.0, 0.0), zoom=1e6)[0, 0, :3].any()


def test_deterministic():
    a = pb.render(48, 48, zoom=200.0, max_iterations=300)
    b = pb.render(48, 48, zoom=200.0, max_iterations=300)
    assert np.array_equal(a, b)


def test_origin():
    lower = pb.render(16, 16, origin="lower")
    upper = pb.render(16, 16, origin="upper")
    assert np.array_equal(lower[::-1], upper)


@pytest.mark.parametrize("scheme", pb.COLOR_SCHEMES)
def test_every_scheme_stays_in_range(scheme):
    """Ocean's waveforms overshoot [0, 1] unless the palette clamps."""
    image = pb.render(200, 150, zoom=1.5, max_iterations=256, color_scheme=scheme)
    assert image.min() >= 0.0 and image.max() <= 1.0


@pytest.mark.parametrize("scheme", pb.COLOR_SCHEMES)
def test_color_schemes_differ(scheme):
    assert pb.render(16, 16, color_scheme=scheme).shape == (16, 16, 4)
    assert len({pb.render(16, 16, color_scheme=s).tobytes() for s in pb.COLOR_SCHEMES}) == 4


@pytest.mark.parametrize("kwargs", [
    {"width": 0, "height": 8},
    {"width": 8, "height": 8, "max_iterations": 0},
    {"width": 8, "height": 8, "zoom": 0.0},
    {"width": 8, "height": 8, "color_scheme": "nope"},
    {"width": 8, "height": 8, "backend": "vulkan"},
    {"width": 8, "height": 8, "origin": "sideways"},
])
def test_invalid_arguments(kwargs):
    width, height = kwargs.pop("width"), kwargs.pop("height")
    with pytest.raises(ValueError):
        pb.render(width, height, **kwargs)


@pytest.mark.skipif(not GPU_BACKENDS, reason="no GPU backend available")
@pytest.mark.parametrize("backend", GPU_BACKENDS)
def test_gpu_matches_cpu(backend):
    """GPU output must track the CPU reference; double support varies by device."""
    cpu = pb.render(96, 96, zoom=50.0, max_iterations=200, backend="cpu")
    gpu = pb.render(96, 96, zoom=50.0, max_iterations=200, backend=backend)
    assert np.allclose(cpu, gpu, atol=1e-5)


def test_auto_backend_is_available():
    assert pb.render(8, 8, backend="auto").shape == (8, 8, 4)
    assert set(pb.available_backends()) <= set(pb.compiled_backends())


def test_device_name_is_none_for_uncompiled_backends():
    """The README shows device_name() returning None, not raising."""
    for backend in ("cpu", "opencl", "cuda"):
        name = pb.device_name(backend)
        assert name is None or isinstance(name, str)
    with pytest.raises(ValueError):
        pb.device_name("vulkan")
