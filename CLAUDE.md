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
- The extension targets `Py_LIMITED_API` 3.9 (`cp39-abi3`). Do not call
  CPython APIs outside the limited set, and do not add NumPy's C API — the
  Python layer wraps the returned bytearray instead.
- Renders release the GIL, so the cached GPU renderers are mutex-guarded.

## Build & test

```bash
make cpu | opencl | cuda        # interactive frontends
scripts/build_ext_dev.sh        # extension in place (add --opencl)
PYTHONPATH=src pytest           # tests
```
