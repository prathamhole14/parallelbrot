/*
    parallelbrot — CUDA rendering core.

    Device selection, device memory management and kernel dispatch, with no
    OpenGL and no windowing. The kernel launch itself already lived behind a
    plain C entry point in src/cuda/mandelbrot_kernel.cu; this file adds the
    allocation and transfer around it.

    Two render paths are offered on purpose. The interactive frontend maps an
    OpenGL texture and copies device-to-device, so it wants the raw device
    pointer and never pays for a host round trip. Headless callers want the
    pixels in host memory. Both share one device buffer.
*/

#include "parallelbrot/core.hpp"

#include <cuda_runtime.h>

#include <string>

// Defined in src/cuda/mandelbrot_kernel.cu
extern "C" {
    void launch_mandelbrot_kernel(float4* d_output,
                                  int width,
                                  int height,
                                  double center_x,
                                  double center_y,
                                  double zoom,
                                  int max_iterations,
                                  int color_scheme);
}

namespace parallelbrot {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
    return false;
}

bool fail_cuda(std::string* error, const char* what, cudaError_t err) {
    return fail(error, std::string(what) + ": " + cudaGetErrorString(err));
}

}  // namespace

struct CudaRenderer::Impl {
    float4*     buffer       = nullptr;
    std::size_t buffer_bytes = 0;   // Current device buffer capacity.

    // Grows the device buffer on demand; capacity is a high-water mark so a
    // shrinking window does not trigger a reallocation every frame.
    bool ensure_buffer(std::size_t bytes, std::string* error) {
        if (buffer != nullptr && bytes <= buffer_bytes) {
            return true;
        }
        if (buffer != nullptr) {
            cudaFree(buffer);
            buffer = nullptr;
            buffer_bytes = 0;
        }
        cudaError_t err = cudaMalloc(&buffer, bytes);
        if (err != cudaSuccess) {
            buffer = nullptr;
            return fail_cuda(error, "Failed to allocate CUDA memory", err);
        }
        buffer_bytes = bytes;
        return true;
    }

    void release() {
        if (buffer != nullptr) {
            cudaFree(buffer);
            buffer = nullptr;
            buffer_bytes = 0;
        }
    }
};

CudaRenderer::CudaRenderer() : impl_(new Impl()) {}

CudaRenderer::~CudaRenderer() {
    impl_->release();
    delete impl_;
}

bool CudaRenderer::initialize(std::string* error) {
    if (initialized_) {
        return true;
    }

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return fail_cuda(error, "CUDA initialization failed", err);
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return fail_cuda(error, "Failed to get device properties", err);
    }

    device_name_ = prop.name;
    initialized_ = true;
    return true;
}

bool CudaRenderer::render_device(const View& view, std::string* error) {
    if (!initialized_) {
        return fail(error, "CUDA renderer is not initialized");
    }
    if (!view.valid()) {
        return fail(error, "Invalid view");
    }
    if (!impl_->ensure_buffer(view.byte_size(), error)) {
        return false;
    }

    launch_mandelbrot_kernel(impl_->buffer,
                             view.width,
                             view.height,
                             view.center_x,
                             view.center_y,
                             view.zoom,
                             view.max_iterations,
                             view.color_scheme);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return fail_cuda(error, "Mandelbrot kernel launch failed", err);
    }
    return true;
}

bool CudaRenderer::render(const View& view, float* out, std::string* error) {
    if (out == nullptr) {
        return fail(error, "Invalid output buffer");
    }
    if (!render_device(view, error)) {
        return false;
    }

    cudaError_t err = cudaMemcpy(out, impl_->buffer, view.byte_size(),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return fail_cuda(error, "Failed to copy result to host", err);
    }
    return true;
}

void* CudaRenderer::device_buffer() const {
    return static_cast<void*>(impl_->buffer);
}

bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace parallelbrot
