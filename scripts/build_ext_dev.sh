#!/bin/sh
# Builds the extension in place for quick iteration. Superseded by
# meson-python once pyproject.toml lands; kept for editing without a
# full reinstall.
#
#   usage: scripts/build_ext_dev.sh [--opencl] [--cuda]

set -e
cd "$(dirname "$0")/.."

DEFS=""
LIBS=""
SRC="src/core/mandelbrot_cpu_core.cpp"

for arg in "$@"; do
    case "$arg" in
        --opencl)
            sh scripts/embed_kernel.sh src/opencl/mandelbrot_kernel.cl \
                include/parallelbrot/opencl_kernel_source.hpp
            DEFS="$DEFS -DPARALLELBROT_WITH_OPENCL -DCL_TARGET_OPENCL_VERSION=120"
            LIBS="$LIBS -lOpenCL"
            SRC="$SRC src/core/mandelbrot_opencl_core.cpp"
            ;;
        --cuda) echo "$0: build CUDA via meson; nvcc is not wired in here" >&2; exit 1 ;;
        *)      echo "$0: unknown option $arg" >&2; exit 1 ;;
    esac
done

INCLUDE=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
OUT=src/parallelbrot/_core.abi3.so

g++ -std=c++17 -O3 -Wall -Wextra -fPIC -shared -fvisibility=hidden \
    -Iinclude -I"$INCLUDE" $DEFS -fopenmp \
    src/python/parallelbrot_ext.cpp $SRC -o "$OUT" $LIBS -fopenmp

echo "built $OUT"
