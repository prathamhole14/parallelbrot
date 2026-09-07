# parallelbrot

GPU-accelerated Mandelbrot renderer (CPU / OpenCL / CUDA) being packaged for
PyPI and conda-forge.

## Code conventions

- **Minimal, correct, concise.** Prefer the smallest change that is fully
  correct. No speculative abstraction, no defensive scaffolding for cases that
  cannot occur, no helpers with one caller.
- Comment *why*, not *what*. Skip comments that restate the code.
- Match the surrounding style rather than importing a new one.
- No `Co-Authored-By` trailers in commit messages.

## Layout

```
include/parallelbrot/core.hpp   Shared View struct + backend entry points
src/core/                       Compute cores — no GLFW, no OpenGL, no Python
src/cpu|opencl|cuda/            Interactive GLFW frontends + device kernels
src/python/                     CPython extension module
src/parallelbrot/               Python package
scripts/embed_kernel.sh         Bakes the .cl source into a header at build time
meson.build + pyproject.toml    Wheel build (meson-python) -> build/<tag>/
Makefile                        Interactive GUI binaries -> bin/
```

The split matters: anything in `src/core/` must stay free of windowing and
Python headers so the same code serves the GUI, the CLI and the extension.

## Invariants

- The OpenCL and CUDA kernels take the same arguments as `View` and must stay
  in sync with it.
- Kernels compute in `double`; devices without `cl_khr_fp64` are rejected at
  init rather than silently producing wrong pixels.
- Buffers are row-major RGBA float32 in [0, 1], row 0 at the **bottom**
  (OpenGL convention). The Python layer flips to top-row-first by default.
- The extension targets `Py_LIMITED_API` 3.9. Do not call CPython APIs
  outside the limited set, and do not add NumPy's C API — the Python layer
  wraps the returned bytearray instead. Three things must agree: `limited_api`
  in meson.build, `limited-api` in pyproject.toml, and the `#ifndef` fallback
  in the extension source. meson-python errors out if any extension ends up
  version-tagged.
- abi3 wheels take their interpreter tag from the *building* Python, so CI
  must build on the oldest supported version (3.9) for one wheel to cover
  everything above it.
- Renders release the GIL, so the cached GPU renderers are mutex-guarded.

## Build & test

```bash
make cpu | opencl | cuda        # interactive frontends
pip install -e .                # editable install; rebuilds C++ on import
python -m build --wheel         # cp3XX-abi3 wheel
pytest                          # tests
```

OpenCL and OpenMP are `auto` features: absent tooling degrades the build
rather than failing it. Force with `-Dopencl=enabled`.
