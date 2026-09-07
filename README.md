# parallelbrot

[![PyPI](https://img.shields.io/pypi/v/parallelbrot.svg)](https://pypi.org/project/parallelbrot/)
[![Python](https://img.shields.io/pypi/pyversions/parallelbrot.svg)](https://pypi.org/project/parallelbrot/)
[![Wheels](https://github.com/prathamhole14/parallelbrot/actions/workflows/wheels.yml/badge.svg)](https://github.com/prathamhole14/parallelbrot/actions/workflows/wheels.yml)

Mandelbrot set renderer with CPU, OpenCL and CUDA backends — a Python package
for generating fractal images, and an interactive real-time viewer.

![Mandelbrot Renderer Demo](docs/mandelbrot.gif)

---

## Install

```bash
pip install parallelbrot
# or
uv add parallelbrot
```

Wheels are `abi3`, so one wheel per platform covers CPython 3.9 and up —
the wheel is built on 3.9 and installs unchanged on 3.14.

| Platform | Wheel |
|----------|-------|
| Linux x86_64, aarch64 | manylinux_2_28 |
| macOS arm64 | 15.0+ |
| macOS x86_64 | 14.0+ |
| Windows x64 | ✅ |

The macOS floors come from Homebrew's `libomp`, which the wheels bundle for
multi-threaded rendering. On older macOS, install from source or conda-forge.

---

## Command line

```bash
parallelbrot                                  # 1920x1080 -> mandelbrot.png
parallelbrot -o seahorse.png -s 3840x2160 \
    -c -0.743644,0.131826 -z 4000 -i 1000 --colors ocean
parallelbrot --info                           # what backends are usable here
parallelbrot -I                               # interactive pan-and-zoom window
```

`-I` opens a live viewer: drag to pan, scroll to zoom on the cursor, arrows to
pan, `+`/`-` for iterations, `C` to cycle palettes, `R` to reset, `Q` to quit.
It needs matplotlib — `pip install 'parallelbrot[viewer]'`.

| Option | Default | Meaning |
|--------|---------|---------|
| `-o`, `--output` | `mandelbrot.png` | Output path |
| `-s`, `--size` | `1920x1080` | `WIDTHxHEIGHT` |
| `-c`, `--center` | `-0.5,0.0` | `REAL,IMAG` |
| `-z`, `--zoom` | `1.0` | Magnification |
| `-i`, `--iterations` | `128` | Escape limit |
| `--colors` | `fire` | Palette |
| `-b`, `--backend` | `auto` | `auto`, `cpu`, `opencl`, `cuda` |
| `-I`, `--interactive` | | Open a window instead of writing a file |
| `-q`, `--quiet` | | Suppress the summary line |

PNG writing is built in, so the CLI needs nothing beyond NumPy.

---

## Python API

```python
import parallelbrot as pb

image = pb.render(1920, 1080)                 # (1080, 1920, 4) float32 RGBA
```

Zoom in on the seahorse valley, and save it:

```python
import parallelbrot as pb

image = pb.render(
    1920, 1080,
    center=(-0.743644, 0.131826),
    zoom=4000,
    max_iterations=1000,
    color_scheme="fire",
)
pb.save_png("seahorse.png", image)
```

`save_png` needs nothing beyond NumPy. The array is ordinary `float32`, so
Pillow, matplotlib or imageio all work on it too if you already have them:

```python
import matplotlib.pyplot as plt

plt.imshow(pb.render(800, 600, zoom=200, center=(-0.75, 0.1)))
plt.axis("off")
plt.show()
```

### `render()`

| Argument | Default | Meaning |
|----------|---------|---------|
| `width`, `height` | — | Output size in pixels |
| `center` | `(-0.5, 0.0)` | Point on the complex plane at the image centre |
| `zoom` | `1.0` | Magnification; the view spans `4 / zoom` |
| `max_iterations` | `128` | Escape limit — raise it as you zoom in |
| `color_scheme` | `"fire"` | `ultra_fractal`, `fire`, `ocean` or `psychedelic` |
| `backend` | `"auto"` | `auto`, `cpu`, `opencl` or `cuda` |
| `origin` | `"upper"` | `upper` puts row 0 at the top, as images expect |

Returns an `(height, width, 4)` float32 array with values in `[0, 1]`.

### Choosing a backend

```python
import parallelbrot as pb

pb.available_backends()   # ('cpu',) — compiled in and a device is present
pb.compiled_backends()    # ('cpu', 'opencl') — compiled in, device or not
pb.device_name("opencl")  # 'NVIDIA GeForce RTX 4070' or None
pb.has_openmp()           # is the CPU backend multi-threaded?
```

`backend="auto"` picks the fastest backend that has a working device, falling
back to the CPU.

Rendering releases the GIL, so calls from separate threads run in parallel:

```python
import parallelbrot as pb
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as pool:                       # renders a zoom
    frames = list(pool.map(                              # sequence in parallel
        lambda z: pb.render(640, 480, zoom=z), [2**i for i in range(12)]
    ))
```

---

## Which backends do you get?

| | CPU (OpenMP) | OpenCL | CUDA |
|---|---|---|---|
| `pip install` / `uv add` | ✅ | — | — |
| conda-forge | ✅ | ✅ | ✅ |
| built from source | ✅ | if headers found | if `nvcc` found |

PyPI wheels are CPU-only on purpose. Linking OpenCL makes the extension
require an ICD loader at import time, which would break the package on
machines without a GPU driver — including for people who only wanted the CPU
backend. conda can declare that loader as a dependency, so the GPU builds live
there.

The CPU backend is not a toy: OpenMP parallelises across rows with dynamic
scheduling, since interior points cost far more than exterior ones. A
1920×1080 frame at 1000 iterations takes **37 ms** on 16 cores, against 506 ms
single-threaded.

---

## Interactive viewer

A real-time pan-and-zoom window, built separately from the Python package.

```bash
make install-deps-ubuntu     # or -fedora, -arch, -macos
make opencl && bin/mandelbrot_opencl   # any GPU — recommended
make cuda   && bin/mandelbrot_cuda     # NVIDIA only, fastest
make cpu    && bin/mandelbrot_cpu      # always works
```

| Input | Action |
|-------|--------|
| Mouse drag | Pan |
| Mouse wheel | Zoom, centred on the cursor |
| Arrow keys | Pan |
| `+` / `-` | Increase / decrease iterations |
| `C` | Cycle colour schemes |
| `R` | Reset view |
| `Esc` | Quit |

`make help` lists every target; `make check-opencl` and `make check-cuda`
report what your machine can do.

### Viewer dependencies

| Component | Linux | macOS | Windows (MSYS2) |
|-----------|-------|-------|-----------------|
| Compiler | `g++` (GCC ≥ 9) | Apple Clang | MinGW-w64 g++ |
| Build | `make`, `pkg-config` | `make`, `pkg-config` | GNU make, `pkg-config` |
| OpenGL | `libgl1-mesa-dev`, `libglew-dev` | built-in | `mingw-w64-x86_64-glew` |
| GLFW 3 | `libglfw3-dev` | `brew install glfw` | `mingw-w64-x86_64-glfw` |
| OpenCL | `ocl-icd-opencl-dev` + driver | built-in framework | `mingw-w64-x86_64-opencl-icd` |
| CUDA *(optional)* | `nvidia-cuda-toolkit` + driver 470+ | not supported | CUDA Toolkit |

GPU drivers: NVIDIA needs the proprietary driver or `nvidia-opencl-dev`; AMD
needs `rocm-opencl-runtime` or `mesa-opencl-icd`; Intel needs
`intel-opencl-icd`. macOS provides OpenCL itself.

---

## Building from source

```bash
git clone https://github.com/prathamhole14/parallelbrot
cd parallelbrot

uv sync                      # venv + dev dependencies, package built editable
uv run pytest
uv run python -c "import parallelbrot as p; print(p.compiled_backends())"
```

Without uv:

```bash
pip install -U meson meson-python ninja
pip install -e . --no-build-isolation    # rebuilds the C++ on import
pytest
```

`--no-build-isolation` matters. A plain `pip install -e .` records the path to
the `ninja` inside pip's temporary build environment, which pip then deletes,
so every later import fails with `FileNotFoundError: .../pip-build-env-*/ninja`.
Reinstall with the flag above to repair it.

Build options are Meson features, so absent tooling degrades the build rather
than failing it:

```bash
uv sync -C setup-args=-Dopencl=enabled    # fail if OpenCL is missing
uv sync -C setup-args=-Dopenmp=disabled   # single-threaded CPU backend
python -m build --wheel                   # a cp3XX-abi3 wheel
```

macOS needs `brew install libomp` for a multi-threaded CPU backend — Apple
Clang ships without OpenMP. `pb.has_openmp()` tells you which you got.

### Layout

```
include/parallelbrot/core.hpp   Shared View struct + backend entry points
src/core/                       Compute cores — no windowing, no Python
src/cpu|opencl|cuda/            Interactive frontends + device kernels
src/python/                     CPython extension (limited API)
src/parallelbrot/               Python package
meson.build, pyproject.toml     Wheel build
Makefile                        Interactive viewer binaries
```

The compute cores carry no GLFW, OpenGL or Python headers, so the same code
serves the viewer, the Python package and the tests.

---

## License

MIT — see [LICENSE](LICENSE).
