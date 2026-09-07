/*
    parallelbrot — backend-agnostic rendering core.

    This header is deliberately free of any windowing, OpenGL or GUI
    dependency. Everything here computes Mandelbrot pixels into a caller
    supplied buffer, so the same code can drive an interactive window, a
    command line renderer, or a Python extension module.

    Buffer layout for every backend: row-major, `height` rows of `width`
    pixels, 4 floats per pixel (R, G, B, A) in the range [0, 1].
    Row 0 is the bottom row, matching the OpenGL texture convention the
    interactive frontends use.
*/

#ifndef PARALLELBROT_CORE_HPP
#define PARALLELBROT_CORE_HPP

#include <cstddef>
#include <string>

namespace parallelbrot {

// Colour palettes. Values match the kernel-side `color_scheme` argument
// used by the OpenCL and CUDA backends, so the enum can be passed straight
// through to the device.
enum ColorScheme {
    COLOR_ULTRA_FRACTAL = 0,
    COLOR_FIRE          = 1,
    COLOR_OCEAN         = 2,
    COLOR_PSYCHEDELIC   = 3,
    COLOR_SCHEME_COUNT  = 4
};

// Everything needed to describe one frame. Defaults match the values the
// interactive renderers start with.
struct View {
    int    width          = 1280;
    int    height         = 720;
    double center_x       = -0.5;
    double center_y       = 0.0;
    double zoom           = 1.0;
    int    max_iterations = 128;
    int    color_scheme   = COLOR_FIRE;

    std::size_t pixel_count() const {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    }

    // Number of floats in an output buffer for this view (4 per pixel).
    std::size_t float_count() const { return pixel_count() * 4; }

    std::size_t byte_size() const { return float_count() * sizeof(float); }

    bool valid() const { return width > 0 && height > 0 && max_iterations > 0; }
};

// ────────────────────────────────────────────────────────────
// CPU backend — always available.
// ────────────────────────────────────────────────────────────

// Renders `view` into `out`, which must hold at least view.float_count()
// floats. Parallelised with OpenMP when the compiler supports it.
void render_cpu(const View& view, float* out);

// Maps a single iteration count to an RGB triple using the given scheme.
// Exposed so non-CPU backends can be checked against it in tests.
void map_color(int iterations, int max_iterations, int color_scheme,
               float& r, float& g, float& b);

// ────────────────────────────────────────────────────────────
// OpenCL backend — compiled in when PARALLELBROT_WITH_OPENCL is defined.
// ────────────────────────────────────────────────────────────
#ifdef PARALLELBROT_WITH_OPENCL

// Owns the OpenCL platform/device/context/queue/program and a device side
// output buffer that grows on demand. Construction is cheap; initialize()
// does the real work and reports failure through `error` rather than
// throwing, so callers can fall back to the CPU backend.
class OpenCLRenderer {
public:
    OpenCLRenderer();
    ~OpenCLRenderer();

    OpenCLRenderer(const OpenCLRenderer&)            = delete;
    OpenCLRenderer& operator=(const OpenCLRenderer&) = delete;

    // Selects a device (preferring NVIDIA, then any GPU, then anything),
    // builds the embedded kernel, and readies a command queue.
    // Returns false and fills `error` on failure.
    bool initialize(std::string* error = nullptr);

    bool is_initialized() const { return initialized_; }

    // Human readable name of the device in use; empty until initialized.
    const std::string& device_name() const { return device_name_; }

    // Computes `view` on the device and reads the result back into `out`.
    // Reallocates the device buffer if the view grew since the last call.
    bool render(const View& view, float* out, std::string* error = nullptr);

    // True when the selected device advertises cl_khr_fp64. The kernel uses
    // double precision, so this is a hard requirement.
    bool supports_double() const { return supports_double_; }

private:
    struct Impl;
    Impl*       impl_;
    bool        initialized_    = false;
    bool        supports_double_ = false;
    std::string device_name_;
};

// Convenience probe: true when at least one usable OpenCL device exists.
bool opencl_available();

#endif  // PARALLELBROT_WITH_OPENCL

// ────────────────────────────────────────────────────────────
// CUDA backend — compiled in when PARALLELBROT_WITH_CUDA is defined.
// ────────────────────────────────────────────────────────────
#ifdef PARALLELBROT_WITH_CUDA

// Owns the CUDA device selection and a device side float4 buffer.
// The interactive frontend needs the raw device pointer so it can blit
// straight into an OpenGL texture, so device_buffer() is public alongside
// the host-copy render() used by headless callers.
class CudaRenderer {
public:
    CudaRenderer();
    ~CudaRenderer();

    CudaRenderer(const CudaRenderer&)            = delete;
    CudaRenderer& operator=(const CudaRenderer&) = delete;

    bool initialize(std::string* error = nullptr);
    bool is_initialized() const { return initialized_; }

    const std::string& device_name() const { return device_name_; }

    // Computes `view` into the internal device buffer. No host transfer.
    // Use this together with device_buffer() for OpenGL interop.
    bool render_device(const View& view, std::string* error = nullptr);

    // Computes `view` and copies the result back into `out`.
    bool render(const View& view, float* out, std::string* error = nullptr);

    // Device pointer holding the most recent render_device() result.
    // Typed as void* so this header stays usable without the CUDA headers.
    void* device_buffer() const;

private:
    struct Impl;
    Impl*       impl_;
    bool        initialized_ = false;
    std::string device_name_;
};

bool cuda_available();

#endif  // PARALLELBROT_WITH_CUDA

}  // namespace parallelbrot

#endif  // PARALLELBROT_CORE_HPP
