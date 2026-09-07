# Makefile for Mandelbrot Renderer
# Supports: Linux, macOS (Homebrew), Windows (MSYS2/MinGW)
# Requires: OpenGL, GLFW3, GLEW, OpenCL (+ CUDA for nvidia targets)

CC        = g++
NVCC      = nvcc
CFLAGS    = -std=c++17 -O3 -Wall -Wextra -DCL_TARGET_OPENCL_VERSION=120 -Iinclude
NVCCFLAGS = -std=c++17 -O3 -Iinclude

# ────────────────────────────────────────────────────────────
# OS Detection
# ────────────────────────────────────────────────────────────
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)

# ──────────────── Linux ────────────────
ifeq ($(UNAME_S),Linux)
    CFLAGS     += -D_GNU_SOURCE
    EXE        :=
    # Use pkg-config for accurate per-distro flags; fallback to sensible defaults
    HAVE_PKG   := $(shell command -v pkg-config 2>/dev/null)
    ifneq ($(HAVE_PKG),)
        CORE_LIBS  := $(shell pkg-config --libs glfw3 glew gl 2>/dev/null || echo "-lGL -lGLEW -lglfw")
        X11_LIBS   := $(shell pkg-config --libs x11 xrandr 2>/dev/null || echo "-lX11 -lXrandr")
    else
        CORE_LIBS  := -lGL -lGLEW -lglfw
        X11_LIBS   := -lX11 -lXrandr
    endif
    EXTRA_LIBS  := -lpthread -ldl $(X11_LIBS)
    LIBS        := $(CORE_LIBS) -lOpenCL $(EXTRA_LIBS)
    CPU_LIBS    := $(CORE_LIBS) $(EXTRA_LIBS)
    CUDA_LIBS   := $(CORE_LIBS) -lcuda -lcudart $(EXTRA_LIBS)
    NVCCFLAGS   += -arch=sm_50
endif

# ──────────────── macOS (Homebrew) ────────────────
ifeq ($(UNAME_S),Darwin)
    EXE         :=
    BREW_PREFIX := $(shell brew --prefix 2>/dev/null || echo /opt/homebrew)
    CFLAGS      += -I$(BREW_PREFIX)/include
    FWORK       := -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
    CORE_LIBS   := -L$(BREW_PREFIX)/lib -lglfw -lGLEW
    EXTRA_LIBS  := $(FWORK) -lpthread
    # macOS ships OpenCL as a framework
    LIBS        := $(CORE_LIBS) -framework OpenCL $(EXTRA_LIBS)
    CPU_LIBS    := $(CORE_LIBS) $(EXTRA_LIBS)
    CUDA_LIBS   := $(CORE_LIBS) -lcuda -lcudart $(EXTRA_LIBS)
    NVCCFLAGS   += -arch=sm_50
endif

# ──────────────── Windows (MSYS2 / MinGW) ────────────────
ifeq ($(OS),Windows_NT)
    EXE        := .exe
    CFLAGS     += -D_WIN32
    HAVE_PKG   := $(shell command -v pkg-config 2>/dev/null)
    ifneq ($(HAVE_PKG),)
        CORE_LIBS  := $(shell pkg-config --libs glfw3 glew 2>/dev/null || echo "-lglfw3 -lGLEW")
    else
        CORE_LIBS  := -lglfw3 -lGLEW
    endif
    EXTRA_LIBS  := -lopengl32 -lgdi32 -lpthread
    LIBS        := $(CORE_LIBS) -lOpenCL $(EXTRA_LIBS)
    CPU_LIBS    := $(CORE_LIBS) $(EXTRA_LIBS)
    CUDA_LIBS   := $(CORE_LIBS) -lcuda -lcudart $(EXTRA_LIBS)
    NVCCFLAGS   += -arch=sm_50
endif

# ────────────────────────────────────────────────────────────
# OpenMP — parallelises the CPU core across rows
# ────────────────────────────────────────────────────────────
# Apple Clang has no built-in OpenMP, so it stays off by default on macOS.
# To enable it there: brew install libomp, then
#   make cpu OPENMP_FLAGS="-Xpreprocessor -fopenmp -lomp"
ifeq ($(UNAME_S),Darwin)
    OPENMP_FLAGS ?=
else
    OPENMP_FLAGS ?= -fopenmp
endif

# ────────────────────────────────────────────────────────────
# Paths & Targets
# ────────────────────────────────────────────────────────────
SRC_DIR    = src
INC_DIR    = include
OPENCL_DIR = $(SRC_DIR)/opencl
CUDA_DIR   = $(SRC_DIR)/cuda
CPU_DIR    = $(SRC_DIR)/cpu
CORE_DIR   = $(SRC_DIR)/core
BUILD_DIR  = bin

TARGET_OC   = $(BUILD_DIR)/mandelbrot_opencl$(EXE)
TARGET_CPU  = $(BUILD_DIR)/mandelbrot_cpu$(EXE)
TARGET_CUDA = $(BUILD_DIR)/mandelbrot_cuda$(EXE)

# Backend-agnostic cores (no GLFW/OpenGL)
CORE_CPU     = $(CORE_DIR)/mandelbrot_cpu_core.cpp
CORE_OC      = $(CORE_DIR)/mandelbrot_opencl_core.cpp
CORE_CUDA    = $(CORE_DIR)/mandelbrot_cuda_core.cu

# Interactive frontends (windowing and presentation only)
SOURCES_OC   = $(OPENCL_DIR)/mandelbrot_opencl.cpp
SOURCES_CPU  = $(CPU_DIR)/mandelbrot_cpu.cpp
SOURCES_CUDA = $(CUDA_DIR)/mandelbrot_cuda.cpp
CUDA_KERNEL  = $(CUDA_DIR)/mandelbrot_kernel.cu
KERNEL_FILE  = $(OPENCL_DIR)/mandelbrot_kernel.cl

# The .cl source is baked into the binary as a raw string literal so that
# nothing has to locate it on disk at runtime.
KERNEL_HEADER = $(BUILD_DIR)/gen/opencl_kernel_source.hpp
EMBED_SCRIPT  = scripts/embed_kernel.sh

# ────────────────────────────────────────────────────────────
# Build Rules
# ────────────────────────────────────────────────────────────
.PHONY: all opencl cpu cuda clean run help kernel-header \
        install-deps-ubuntu install-deps-fedora install-deps-arch install-deps-macos \
        install-cuda-ubuntu install-cuda-fedora install-cuda-arch \
        check-opencl check-cuda release

all: $(TARGET_OC) $(TARGET_CPU)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Regenerate the embedded kernel header whenever the .cl source changes.
$(KERNEL_HEADER): $(KERNEL_FILE) $(EMBED_SCRIPT) | $(BUILD_DIR)
	sh $(EMBED_SCRIPT) $(KERNEL_FILE) $@

kernel-header: $(KERNEL_HEADER)

$(TARGET_OC): $(SOURCES_OC) $(CORE_OC) $(CORE_CPU) $(KERNEL_HEADER) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(dir $(KERNEL_HEADER)) -DPARALLELBROT_WITH_OPENCL $(OPENMP_FLAGS) \
	    $(SOURCES_OC) $(CORE_OC) $(CORE_CPU) -o $@ $(LIBS)

$(TARGET_CPU): $(SOURCES_CPU) $(CORE_CPU) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) $(SOURCES_CPU) $(CORE_CPU) -o $@ $(CPU_LIBS)

$(TARGET_CUDA): $(SOURCES_CUDA) $(CUDA_KERNEL) $(CORE_CUDA) $(CORE_CPU) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DPARALLELBROT_WITH_CUDA \
	    $(CUDA_KERNEL) $(CORE_CUDA) $(SOURCES_CUDA) $(CORE_CPU) -o $@ $(CUDA_LIBS)

opencl:    $(TARGET_OC)
cpu:       $(TARGET_CPU)
cuda:      $(TARGET_CUDA)

debug: CFLAGS += -g -DDEBUG
debug: $(TARGET_OC)

release: CFLAGS += -DNDEBUG -march=native -flto
release: $(TARGET_OC)

run: $(TARGET_OC)
	./$(TARGET_OC)

clean:
	rm -rf $(BUILD_DIR)

# ────────────────────────────────────────────────────────────
# Dependency Installation Helpers
# ────────────────────────────────────────────────────────────

install-deps-ubuntu:
	sudo apt update
	sudo apt install -y build-essential pkg-config \
	    libgl1-mesa-dev libglu1-mesa-dev \
	    libglfw3-dev libglew-dev \
	    opencl-headers ocl-icd-opencl-dev \
	    mesa-opencl-icd
	@echo "  (NVIDIA GPU users: also run 'make install-cuda-ubuntu')"

install-cuda-ubuntu:
	sudo apt update
	sudo apt install -y nvidia-cuda-toolkit
	@echo "For the latest CUDA toolkit visit: https://developer.nvidia.com/cuda-downloads"

install-deps-fedora:
	sudo dnf install -y gcc-c++ pkgconfig \
	    mesa-libGL-devel mesa-libGLU-devel \
	    glfw-devel glew-devel \
	    opencl-headers ocl-icd-devel \
	    mesa-libOpenCL

install-cuda-fedora:
	@echo "Install NVIDIA CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
	@echo "Then: sudo dnf install -y cuda-toolkit"

install-deps-arch:
	sudo pacman -S --needed base-devel pkgconf \
	    mesa glu glfw-x11 glew \
	    opencl-headers ocl-icd mesa-opencl-icd

install-cuda-arch:
	sudo pacman -S --needed cuda

install-deps-macos:
	@which brew >/dev/null 2>&1 || (echo "Install Homebrew first: https://brew.sh" && exit 1)
	brew install pkg-config glfw glew

# ────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────

check-opencl:
	@echo "Checking OpenCL devices:"
	@which clinfo >/dev/null 2>&1 && clinfo || echo "Install clinfo to see OpenCL device information"

check-cuda:
	@echo "Checking CUDA devices:"
	@which nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found. Install NVIDIA drivers."
	@which nvcc >/dev/null 2>&1 \
	    && echo "NVCC: $$(nvcc --version | grep -o 'V[0-9]*\.[0-9]*\.[0-9]*')" \
	    || echo "nvcc not found. Install CUDA Toolkit (run: make install-cuda-ubuntu/fedora/arch)."

# ────────────────────────────────────────────────────────────
# Help
# ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Mandelbrot Renderer — Build Targets"
	@echo "─────────────────────────────────────────────────"
	@echo "  make all          Build OpenCL + CPU backends (default)"
	@echo "  make opencl       OpenCL GPU backend (any GPU)"
	@echo "  make cpu          CPU-only fallback backend"
	@echo "  make cuda         CUDA GPU backend (NVIDIA only)"

	@echo "  make clean        Remove build artifacts"
	@echo "  make run          Build and run the OpenCL version"
	@echo ""
	@echo "Dependency Installation"
	@echo "─────────────────────────────────────────────────"
	@echo "  make install-deps-ubuntu   Ubuntu / Debian"
	@echo "  make install-deps-fedora   Fedora / RHEL"
	@echo "  make install-deps-arch     Arch Linux"
	@echo "  make install-deps-macos    macOS (Homebrew)"
	@echo "  make install-cuda-ubuntu   CUDA on Ubuntu"
	@echo "  make install-cuda-fedora   CUDA on Fedora"
	@echo "  make install-cuda-arch     CUDA on Arch"
	@echo ""
	@echo "Diagnostics"
	@echo "─────────────────────────────────────────────────"
	@echo "  make check-opencl   List OpenCL devices"
	@echo "  make check-cuda     Check CUDA / nvcc"
	@echo ""
