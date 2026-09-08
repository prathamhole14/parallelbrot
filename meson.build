project('parallelbrot', 'cpp',
  version: '0.2.1',
  license: 'MIT',
  meson_version: '>=1.3.0',
  default_options: ['cpp_std=c++17', 'buildtype=release', 'warning_level=2'],
)

py = import('python').find_installation(pure: false)

sources = ['src/python/parallelbrot_ext.cpp', 'src/core/mandelbrot_cpu_core.cpp']
deps = []
args = []

# OpenMP parallelises the CPU core. Apple Clang ships without it, so a build
# lacking OpenMP is degraded but valid; has_openmp() reports which you got.
openmp = dependency('openmp', required: get_option('openmp'))
if openmp.found()
  deps += openmp
endif

# The kernel is baked into the binary as a raw string literal, so nothing has
# to locate a .cl file at runtime. include_directories('.') covers the build
# directory this lands in.
opencl = dependency('OpenCL', required: get_option('opencl'))
if opencl.found()
  sources += custom_target('opencl_kernel_source.hpp',
    input: 'src/opencl/mandelbrot_kernel.cl',
    output: 'opencl_kernel_source.hpp',
    command: [find_program('scripts/embed_kernel.sh'), '@INPUT@', '@OUTPUT@'],
  )
  sources += 'src/core/mandelbrot_opencl_core.cpp'
  args += ['-DPARALLELBROT_WITH_OPENCL', '-DCL_TARGET_OPENCL_VERSION=120']
  deps += opencl
endif

# limited_api makes meson define Py_LIMITED_API, name the file .abi3.so and
# link python3.dll on Windows, which is what tags the wheel cp39-abi3.
py.extension_module('_core',
  sources,
  include_directories: include_directories('include', '.'),
  dependencies: deps,
  cpp_args: args,
  limited_api: '3.9',
  subdir: 'parallelbrot',
  install: true,
)

py.install_sources(
  ['src/parallelbrot/__init__.py', 'src/parallelbrot/cli.py', 'src/parallelbrot/viewer.py', 'src/parallelbrot/py.typed'],
  subdir: 'parallelbrot',
)
