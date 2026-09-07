/*
    parallelbrot — OpenCL rendering core.

    Device discovery, context/queue creation, kernel compilation and the
    compute-plus-readback path, with no OpenGL and no windowing. The kernel
    source is compiled into the binary (see scripts/embed_kernel.sh) rather
    than loaded from disk, so this works regardless of the working directory
    or install location.
*/

#include "parallelbrot/core.hpp"
#include "parallelbrot/opencl_kernel_source.hpp"

#include <CL/cl.h>

#include <cstring>
#include <string>
#include <vector>

namespace parallelbrot {
namespace {

// Reports an error through the optional out-parameter. Callers that pass
// nullptr simply get a false return.
bool fail(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
    return false;
}

std::string cl_error_string(cl_int err) {
    return std::to_string(static_cast<int>(err));
}

// Queries a string property of a platform or device into a std::string.
template <typename Id, typename InfoFn, typename Param>
std::string query_string(Id id, InfoFn fn, Param param) {
    std::size_t size = 0;
    if (fn(id, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return std::string();
    }
    std::vector<char> buffer(size);
    if (fn(id, param, size, buffer.data(), nullptr) != CL_SUCCESS) {
        return std::string();
    }
    return std::string(buffer.data());
}

// Picks a device, preferring an NVIDIA platform, then any GPU, then any
// device at all. Mirrors the selection the interactive frontend used.
bool select_device(cl_platform_id* out_platform, cl_device_id* out_device,
                   std::string* error) {
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        return fail(error, "No OpenCL platforms found");
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    if (clGetPlatformIDs(num_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return fail(error, "Failed to get OpenCL platforms");
    }

    cl_platform_id platform = platforms[0];
    for (const auto& candidate : platforms) {
        std::string name = query_string(candidate, clGetPlatformInfo, CL_PLATFORM_NAME);
        if (name.find("NVIDIA") != std::string::npos) {
            platform = candidate;
            break;
        }
    }

    cl_uint num_devices = 0;
    cl_device_type type = CL_DEVICE_TYPE_GPU;
    if (clGetDeviceIDs(platform, type, 0, nullptr, &num_devices) != CL_SUCCESS || num_devices == 0) {
        type = CL_DEVICE_TYPE_ALL;
        if (clGetDeviceIDs(platform, type, 0, nullptr, &num_devices) != CL_SUCCESS || num_devices == 0) {
            return fail(error, "No OpenCL devices found");
        }
    }

    std::vector<cl_device_id> devices(num_devices);
    if (clGetDeviceIDs(platform, type, num_devices, devices.data(), nullptr) != CL_SUCCESS) {
        return fail(error, "Failed to enumerate OpenCL devices");
    }

    *out_platform = platform;
    *out_device   = devices[0];
    return true;
}

}  // namespace

struct OpenCLRenderer::Impl {
    cl_platform_id  platform = nullptr;
    cl_device_id    device   = nullptr;
    cl_context      context  = nullptr;
    cl_command_queue queue   = nullptr;
    cl_program      program  = nullptr;
    cl_kernel       kernel   = nullptr;
    cl_mem          buffer   = nullptr;
    std::size_t     buffer_bytes = 0;   // Current device buffer capacity.

    // Grows the device buffer when a larger view arrives. Shrinking is not
    // worth a reallocation, so capacity is a high-water mark.
    bool ensure_buffer(std::size_t bytes, std::string* error) {
        if (buffer != nullptr && bytes <= buffer_bytes) {
            return true;
        }
        if (buffer != nullptr) {
            clReleaseMemObject(buffer);
            buffer = nullptr;
            buffer_bytes = 0;
        }
        cl_int err = CL_SUCCESS;
        buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
        if (err != CL_SUCCESS) {
            buffer = nullptr;
            return fail(error, "Failed to create OpenCL buffer: " + cl_error_string(err));
        }
        buffer_bytes = bytes;
        return true;
    }

    void release() {
        if (kernel)  { clReleaseKernel(kernel);         kernel  = nullptr; }
        if (program) { clReleaseProgram(program);        program = nullptr; }
        if (buffer)  { clReleaseMemObject(buffer);       buffer  = nullptr; }
        if (queue)   { clReleaseCommandQueue(queue);     queue   = nullptr; }
        if (context) { clReleaseContext(context);        context = nullptr; }
        buffer_bytes = 0;
    }
};

OpenCLRenderer::OpenCLRenderer() : impl_(new Impl()) {}

OpenCLRenderer::~OpenCLRenderer() {
    impl_->release();
    delete impl_;
}

bool OpenCLRenderer::initialize(std::string* error) {
    if (initialized_) {
        return true;
    }

    if (!select_device(&impl_->platform, &impl_->device, error)) {
        return false;
    }

    device_name_ = query_string(impl_->device, clGetDeviceInfo, CL_DEVICE_NAME);

    // The kernel computes in double precision. Without cl_khr_fp64 the build
    // would fail with a confusing compiler log, so check up front.
    std::string extensions = query_string(impl_->device, clGetDeviceInfo, CL_DEVICE_EXTENSIONS);
    supports_double_ = extensions.find("cl_khr_fp64") != std::string::npos;
    if (!supports_double_) {
        return fail(error, "OpenCL device '" + device_name_ +
                           "' lacks cl_khr_fp64 (double precision), which the kernel requires");
    }

    cl_int err = CL_SUCCESS;
    impl_->context = clCreateContext(nullptr, 1, &impl_->device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        return fail(error, "Failed to create OpenCL context: " + cl_error_string(err));
    }

    impl_->queue = clCreateCommandQueue(impl_->context, impl_->device, 0, &err);
    if (err != CL_SUCCESS) {
        return fail(error, "Failed to create OpenCL command queue: " + cl_error_string(err));
    }

    const char* source_ptr  = opencl_kernel_source();
    const std::size_t source_size = std::strlen(source_ptr);

    impl_->program = clCreateProgramWithSource(impl_->context, 1, &source_ptr, &source_size, &err);
    if (err != CL_SUCCESS) {
        return fail(error, "Failed to create OpenCL program: " + cl_error_string(err));
    }

    err = clBuildProgram(impl_->program, 1, &impl_->device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::size_t log_size = 0;
        clGetProgramBuildInfo(impl_->program, impl_->device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, '\0');
        clGetProgramBuildInfo(impl_->program, impl_->device, CL_PROGRAM_BUILD_LOG,
                              log_size, log.data(), nullptr);
        return fail(error, std::string("OpenCL compilation error: ") + log.data());
    }

    impl_->kernel = clCreateKernel(impl_->program, "mandelbrot_buffer", &err);
    if (err != CL_SUCCESS) {
        return fail(error, "Failed to create OpenCL kernel: " + cl_error_string(err));
    }

    initialized_ = true;
    return true;
}

bool OpenCLRenderer::render(const View& view, float* out, std::string* error) {
    if (!initialized_) {
        return fail(error, "OpenCL renderer is not initialized");
    }
    if (!view.valid() || out == nullptr) {
        return fail(error, "Invalid view or output buffer");
    }
    if (!impl_->ensure_buffer(view.byte_size(), error)) {
        return false;
    }

    // Copied into locals because clSetKernelArg takes the address of each
    // argument and the View is const.
    int    width          = view.width;
    int    height         = view.height;
    double center_x       = view.center_x;
    double center_y       = view.center_y;
    double zoom           = view.zoom;
    int    max_iterations = view.max_iterations;
    int    color_scheme   = view.color_scheme;

    cl_int err  = clSetKernelArg(impl_->kernel, 0, sizeof(cl_mem), &impl_->buffer);
    err |= clSetKernelArg(impl_->kernel, 1, sizeof(int),    &width);
    err |= clSetKernelArg(impl_->kernel, 2, sizeof(int),    &height);
    err |= clSetKernelArg(impl_->kernel, 3, sizeof(double), &center_x);
    err |= clSetKernelArg(impl_->kernel, 4, sizeof(double), &center_y);
    err |= clSetKernelArg(impl_->kernel, 5, sizeof(double), &zoom);
    err |= clSetKernelArg(impl_->kernel, 6, sizeof(int),    &max_iterations);
    err |= clSetKernelArg(impl_->kernel, 7, sizeof(int),    &color_scheme);
    if (err != CL_SUCCESS) {
        return fail(error, "Failed to set kernel arguments: " + cl_error_string(err));
    }

    std::size_t global_work_size[2] = {(std::size_t)width, (std::size_t)height};
    err = clEnqueueNDRangeKernel(impl_->queue, impl_->kernel, 2, nullptr,
                                 global_work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return fail(error, "Failed to execute kernel: " + cl_error_string(err));
    }

    err = clEnqueueReadBuffer(impl_->queue, impl_->buffer, CL_TRUE, 0,
                              view.byte_size(), out, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return fail(error, "Failed to read buffer: " + cl_error_string(err));
    }

    return true;
}

bool opencl_available() {
    cl_platform_id platform = nullptr;
    cl_device_id   device   = nullptr;
    return select_device(&platform, &device, nullptr);
}

}  // namespace parallelbrot
