# parallelbrot

GPU-accelerated Mandelbrot set renderer with OpenCL (cross-platform), CUDA (NVIDIA), and CPU backends.

![Mandelbrot Renderer Demo](docs/mandelbrot.gif)

---

## Quick Start

### 1 — Install dependencies

**Linux — Ubuntu / Debian**
```bash
make install-deps-ubuntu      # OpenCL + OpenGL + GLFW + GLEW
make install-cuda-ubuntu      # CUDA toolkit (NVIDIA only)
```
For other distros, replace `ubuntu` with `fedora` or `arch`.

### 2 — Build & Run

```bash
make opencl && bin/mandelbrot_opencl   # Any GPU — recommended
make cuda   && bin/mandelbrot_cuda     # NVIDIA only (fastest)
make cpu    && bin/mandelbrot_cpu      # CPU fallback
make all                                 # Build opencl + cpu
make clean                              # Remove build artifacts
```

> **Windows:** executables are built as `bin/mandelbrot_*.exe` automatically.

---

## Controls

| Key / Input      | Action                         |
|------------------|--------------------------------|
| Mouse drag       | Pan                            |
| Mouse wheel      | Zoom (centred on cursor)       |
| Arrow keys       | Pan                            |
| `+` / `-`        | Increase / decrease iterations |
| `C`              | Cycle colour schemes           |
| `R`              | Reset view                     |
| `ESC`            | Quit                           |

---

## Requirements

| Component | Linux | macOS | Windows (MSYS2) |
|-----------|-------|-------|-----------------|
| Compiler  | `g++` (GCC ≥ 9) | Apple Clang / GCC | MinGW-w64 g++ |
| Build tool | `make`, `pkg-config` | `make`, `pkg-config` | GNU make, `pkg-config` |
| OpenGL | `libgl1-mesa-dev`, `libglew-dev` | Built-in framework | `mingw-w64-x86_64-glew` |
| GLFW 3 | `libglfw3-dev` | `brew install glfw` | `mingw-w64-x86_64-glfw` |
| OpenCL | `ocl-icd-opencl-dev` + GPU driver | Built-in framework | `mingw-w64-x86_64-opencl-icd` + GPU driver |
| CUDA *(optional)* | `nvidia-cuda-toolkit` + driver 470+ | *(not supported)* | NVIDIA CUDA Toolkit |

### GPU driver notes

- **NVIDIA (Linux):** install the proprietary driver or `nvidia-opencl-dev`.  
  OpenCL will automatically use NVIDIA CUDA via the ICD loader.
- **AMD (Linux):** install the ROCm stack (`rocm-opencl-runtime`) or Mesa OpenCL (`mesa-opencl-icd`).
- **Intel (Linux):** install `intel-opencl-icd`.
- **macOS:** OpenCL is provided by the OS; no extra driver needed.

### Check your GPU

```bash
make check-opencl    # list OpenCL devices
make check-cuda      # check CUDA / nvcc version
```

---

## Makefile Reference

```bash
make help            # full target list
```

---

## License

MIT — see [LICENSE](LICENSE).
