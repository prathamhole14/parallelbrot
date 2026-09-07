"""GPU-accelerated Mandelbrot rendering with CPU, OpenCL and CUDA backends."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import struct
import zlib

import numpy as np

from . import _core
from ._core import available_backends, compiled_backends, has_openmp

__all__ = [
    "render",
    "save_png",
    "available_backends",
    "compiled_backends",
    "has_openmp",
    "device_name",
    "COLOR_SCHEMES",
    "__version__",
]

try:
    __version__ = version("parallelbrot")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0.dev0"

COLOR_SCHEMES = ("ultra_fractal", "fire", "ocean", "psychedelic")

_RENDERERS = {
    "cpu": _core.render_cpu,
    "opencl": getattr(_core, "render_opencl", None),
    "cuda": getattr(_core, "render_cuda", None),
}


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        return available_backends()[-1]
    if backend not in _RENDERERS:
        raise ValueError(f"unknown backend {backend!r}; expected auto or one of {tuple(_RENDERERS)}")
    if _RENDERERS[backend] is None:
        raise RuntimeError(f"backend {backend!r} was not compiled into this build")
    return backend


def _resolve_scheme(color_scheme: str | int) -> int:
    if isinstance(color_scheme, int):
        if not 0 <= color_scheme < len(COLOR_SCHEMES):
            raise ValueError(f"color_scheme must be in 0..{len(COLOR_SCHEMES) - 1}")
        return color_scheme
    try:
        return COLOR_SCHEMES.index(color_scheme)
    except ValueError:
        raise ValueError(
            f"unknown color_scheme {color_scheme!r}; expected one of {COLOR_SCHEMES}"
        ) from None


def render(
    width: int,
    height: int,
    *,
    center: tuple[float, float] = (-0.5, 0.0),
    zoom: float = 1.0,
    max_iterations: int = 128,
    color_scheme: str | int = "fire",
    backend: str = "auto",
    origin: str = "upper",
) -> np.ndarray:
    """Render the Mandelbrot set to an ``(height, width, 4)`` float32 RGBA array.

    Values are in [0, 1]. ``backend`` may be ``"auto"`` (fastest available),
    ``"cpu"``, ``"opencl"`` or ``"cuda"``.

    The backends compute with the imaginary axis increasing upwards, so row 0
    is the bottom of the image. ``origin="upper"`` (the default) flips that to
    the usual image convention, returning a reversed view rather than a copy.
    """
    name = _resolve_backend(backend)
    buffer = _RENDERERS[name](
        width, height,
        center_x=center[0], center_y=center[1], zoom=zoom,
        max_iterations=max_iterations, color_scheme=_resolve_scheme(color_scheme),
    )
    image = np.frombuffer(buffer, dtype=np.float32).reshape(height, width, 4)

    if origin == "upper":
        return image[::-1]
    if origin == "lower":
        return image
    raise ValueError(f"origin must be 'upper' or 'lower', not {origin!r}")


def device_name(backend: str) -> str | None:
    """Name of the device a GPU backend would use, or None if unavailable.

    A backend that was not compiled into this build has no device, so it
    reports None rather than raising; only an unrecognised name is an error.
    """
    if backend not in _RENDERERS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {tuple(_RENDERERS)}")
    getter = getattr(_core, f"{backend}_device_name", None)
    return getter() if getter is not None else None


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def save_png(path: str, image: np.ndarray) -> None:
    """Write an ``(H, W, 4)`` float array in [0, 1] as an 8-bit RGBA PNG.

    Hand-rolled rather than via Pillow: it is a dozen lines against the
    standard library, and the alternative is a dependency heavier than this
    package purely so the CLI can save its output.
    """
    height, width, _ = image.shape
    rgba = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    # Each PNG scanline is prefixed with a filter-type byte; 0 means "none".
    scanlines = np.hstack(
        [np.zeros((height, 1), np.uint8), np.ascontiguousarray(rgba).reshape(height, width * 4)]
    )
    with open(path, "wb") as f:
        f.write(
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(scanlines.tobytes(), 6))
            + _chunk(b"IEND", b"")
        )
