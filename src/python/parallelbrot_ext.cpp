/*
    parallelbrot — CPython extension module.

    Built against the limited API so one cp39-abi3 wheel loads on CPython 3.9
    onwards. That rules out pybind11 and nanobind, whose stable-ABI support
    starts at 3.12, but the surface here is small enough not to need them.

    Pixels come back as a bytearray, not a NumPy array: NumPy's C API is not
    part of the limited API and would pin the build to a NumPy ABI
    generation. np.frombuffer() wraps the result zero-copy on the Python side.
*/

#define PY_SSIZE_T_CLEAN
#ifndef Py_LIMITED_API
#   define Py_LIMITED_API 0x03090000  /* CPython 3.9 */
#endif

#include <Python.h>

#include "parallelbrot/core.hpp"

#include <mutex>
#include <string>

namespace {

// Bounds width * height * 4 floats well below the Py_ssize_t limit.
const int kMaxDimension = 100000;

const char* const kKeywords[] = {
    "width", "height", "center_x", "center_y",
    "zoom", "max_iterations", "color_scheme", nullptr
};

bool parse_view(PyObject* args, PyObject* kwargs, parallelbrot::View* v) {
    int w = 0, h = 0, iters = 128, scheme = parallelbrot::COLOR_FIRE;
    double cx = -0.5, cy = 0.0, zoom = 1.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|dddii",
                                     const_cast<char**>(kKeywords),
                                     &w, &h, &cx, &cy, &zoom, &iters, &scheme)) {
        return false;
    }
    if (w <= 0 || h <= 0 || w > kMaxDimension || h > kMaxDimension) {
        PyErr_Format(PyExc_ValueError, "width and height must be in 1..%d", kMaxDimension);
        return false;
    }
    if (iters <= 0) {
        PyErr_SetString(PyExc_ValueError, "max_iterations must be positive");
        return false;
    }
    if (zoom <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "zoom must be positive");
        return false;
    }
    if (scheme < 0 || scheme >= parallelbrot::COLOR_SCHEME_COUNT) {
        PyErr_Format(PyExc_ValueError, "color_scheme must be in 0..%d",
                     parallelbrot::COLOR_SCHEME_COUNT - 1);
        return false;
    }

    *v = parallelbrot::View{w, h, cx, cy, zoom, iters, scheme};
    return true;
}

// Uninitialised bytearray sized for `v`; CPython mallocs the storage, so it
// is aligned for float.
PyObject* new_buffer(const parallelbrot::View& v, float** out) {
    PyObject* buf = PyByteArray_FromStringAndSize(
        nullptr, static_cast<Py_ssize_t>(v.byte_size()));
    if (buf != nullptr) {
        *out = reinterpret_cast<float*>(PyByteArray_AsString(buf));
    }
    return buf;
}

PyObject* to_tuple(const char* const* names, int count) {
    PyObject* tuple = PyTuple_New(count);
    if (tuple == nullptr) return nullptr;
    for (int i = 0; i < count; ++i) {
        PyObject* item = PyUnicode_FromString(names[i]);
        if (item == nullptr || PyTuple_SetItem(tuple, i, item) != 0) {
            Py_DECREF(tuple);
            return nullptr;
        }
    }
    return tuple;
}

// ── GPU backend plumbing ────────────────────────────────────
#if defined(PARALLELBROT_WITH_OPENCL) || defined(PARALLELBROT_WITH_CUDA)

// Device setup costs tens of milliseconds, so each renderer is built once and
// reused. The mutex is required because renders run with the GIL released.
template <class Renderer>
class Lazy {
public:
    // Runs fn under the lock, initialising the renderer on first use.
    template <class Fn>
    bool with(Fn&& fn, std::string* error) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!tried_) {
            tried_ = true;
            Renderer* r = new (std::nothrow) Renderer();
            if (r == nullptr) {
                init_error_ = "out of memory allocating renderer";
            } else if (r->initialize(&init_error_)) {
                renderer_ = r;
            } else {
                delete r;
            }
        }
        if (renderer_ == nullptr) {
            *error = init_error_;
            return false;
        }
        return fn(*renderer_);
    }

private:
    std::mutex  mutex_;
    Renderer*   renderer_ = nullptr;
    bool        tried_    = false;
    std::string init_error_;
};

template <class Lazy>
PyObject* render_gpu(Lazy& lazy, PyObject* args, PyObject* kwargs) {
    parallelbrot::View view;
    if (!parse_view(args, kwargs, &view)) return nullptr;

    float* out = nullptr;
    PyObject* buf = new_buffer(view, &out);
    if (buf == nullptr) return nullptr;

    bool ok = false;
    std::string error;
    Py_BEGIN_ALLOW_THREADS
    ok = lazy.with([&](auto& r) { return r.render(view, out, &error); }, &error);
    Py_END_ALLOW_THREADS

    if (!ok) {
        Py_DECREF(buf);
        PyErr_SetString(PyExc_RuntimeError, error.c_str());
        return nullptr;
    }
    return buf;
}

template <class Lazy>
PyObject* device_name(Lazy& lazy) {
    std::string name, error;
    Py_BEGIN_ALLOW_THREADS
    lazy.with([&](auto& r) { name = r.device_name(); return true; }, &error);
    Py_END_ALLOW_THREADS

    if (name.empty()) Py_RETURN_NONE;
    return PyUnicode_FromString(name.c_str());
}

#endif  // any GPU backend

// ── Backend entry points ────────────────────────────────────

PyObject* render_cpu(PyObject*, PyObject* args, PyObject* kwargs) {
    parallelbrot::View view;
    if (!parse_view(args, kwargs, &view)) return nullptr;

    float* out = nullptr;
    PyObject* buf = new_buffer(view, &out);
    if (buf == nullptr) return nullptr;

    // The core touches no Python state, so other threads may run.
    Py_BEGIN_ALLOW_THREADS
    parallelbrot::render_cpu(view, out);
    Py_END_ALLOW_THREADS

    return buf;
}

#ifdef PARALLELBROT_WITH_OPENCL
Lazy<parallelbrot::OpenCLRenderer> g_opencl;

PyObject* render_opencl(PyObject*, PyObject* args, PyObject* kwargs) {
    return render_gpu(g_opencl, args, kwargs);
}
PyObject* opencl_device_name(PyObject*, PyObject*) { return device_name(g_opencl); }
#endif

#ifdef PARALLELBROT_WITH_CUDA
Lazy<parallelbrot::CudaRenderer> g_cuda;

PyObject* render_cuda(PyObject*, PyObject* args, PyObject* kwargs) {
    return render_gpu(g_cuda, args, kwargs);
}
PyObject* cuda_device_name(PyObject*, PyObject*) { return device_name(g_cuda); }
#endif

// ── Introspection ───────────────────────────────────────────

// Backends compiled into this extension, regardless of whether a device
// is present.
PyObject* compiled_backends(PyObject*, PyObject*) {
    const char* names[3] = {"cpu"};
    int n = 1;
#ifdef PARALLELBROT_WITH_CUDA
    names[n++] = "cuda";
#endif
#ifdef PARALLELBROT_WITH_OPENCL
    names[n++] = "opencl";
#endif
    return to_tuple(names, n);
}

// Backends that are compiled in and have a usable device right now,
// fastest first.
PyObject* available_backends(PyObject*, PyObject*) {
    bool cuda = false, opencl = false;
    Py_BEGIN_ALLOW_THREADS
#ifdef PARALLELBROT_WITH_CUDA
    cuda = parallelbrot::cuda_available();
#endif
#ifdef PARALLELBROT_WITH_OPENCL
    opencl = parallelbrot::opencl_available();
#endif
    Py_END_ALLOW_THREADS

    const char* names[3] = {"cpu"};
    int n = 1;
    if (cuda)   names[n++] = "cuda";
    if (opencl) names[n++] = "opencl";
    return to_tuple(names, n);
}

PyObject* has_openmp(PyObject*, PyObject*) {
#ifdef _OPENMP
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}

#define KWFUNC(f) reinterpret_cast<PyCFunction>(reinterpret_cast<void(*)()>(f))

PyMethodDef kMethods[] = {
    {"render_cpu", KWFUNC(render_cpu), METH_VARARGS | METH_KEYWORDS,
     "render_cpu(width, height, center_x=-0.5, center_y=0.0, zoom=1.0, "
     "max_iterations=128, color_scheme=1) -> bytearray"},
#ifdef PARALLELBROT_WITH_OPENCL
    {"render_opencl", KWFUNC(render_opencl), METH_VARARGS | METH_KEYWORDS,
     "render_opencl(width, height, ...) -> bytearray"},
    {"opencl_device_name", opencl_device_name, METH_NOARGS,
     "opencl_device_name() -> str | None"},
#endif
#ifdef PARALLELBROT_WITH_CUDA
    {"render_cuda", KWFUNC(render_cuda), METH_VARARGS | METH_KEYWORDS,
     "render_cuda(width, height, ...) -> bytearray"},
    {"cuda_device_name", cuda_device_name, METH_NOARGS,
     "cuda_device_name() -> str | None"},
#endif
    {"compiled_backends", compiled_backends, METH_NOARGS,
     "compiled_backends() -> tuple[str, ...]"},
    {"available_backends", available_backends, METH_NOARGS,
     "available_backends() -> tuple[str, ...]"},
    {"has_openmp", has_openmp, METH_NOARGS, "has_openmp() -> bool"},
    {nullptr, nullptr, 0, nullptr}
};

int module_exec(PyObject* m) {
    return (PyModule_AddIntConstant(m, "COLOR_ULTRA_FRACTAL", parallelbrot::COLOR_ULTRA_FRACTAL) < 0
         || PyModule_AddIntConstant(m, "COLOR_FIRE",          parallelbrot::COLOR_FIRE) < 0
         || PyModule_AddIntConstant(m, "COLOR_OCEAN",         parallelbrot::COLOR_OCEAN) < 0
         || PyModule_AddIntConstant(m, "COLOR_PSYCHEDELIC",   parallelbrot::COLOR_PSYCHEDELIC) < 0)
        ? -1 : 0;
}

PyModuleDef_Slot kSlots[] = {
    {Py_mod_exec, reinterpret_cast<void*>(module_exec)},
    {0, nullptr}
};

PyModuleDef kModule = {
    PyModuleDef_HEAD_INIT,
    "parallelbrot._core",
    "Compiled Mandelbrot rendering backends (CPU, OpenCL, CUDA).",
    0, kMethods, kSlots, nullptr, nullptr, nullptr
};

}  // namespace

extern "C" PyMODINIT_FUNC PyInit__core(void) {
    return PyModuleDef_Init(&kModule);
}
