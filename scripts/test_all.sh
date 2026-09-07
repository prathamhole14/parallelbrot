#!/bin/bash

# Test script for Mandelbrot Renderer
# Tests all available implementations

echo "=== Mandelbrot Renderer Test Script ==="
echo

echo "1. Checking system compatibility..."
echo "OpenCL devices:"
make check-opencl | head -10
echo
echo "CUDA devices:"
make check-cuda
echo

echo "2. Building all versions..."
echo "Building OpenCL version..."
make opencl > /dev/null 2>&1 && echo "✓ OpenCL version built successfully" || echo "✗ OpenCL version failed to build"

echo "Building CPU version..."
make cpu > /dev/null 2>&1 && echo "✓ CPU version built successfully" || echo "✗ CPU version failed to build" 

echo "Building CUDA version..."
if make cuda > /dev/null 2>&1; then
    echo "✓ CUDA version built successfully"
    CUDA_AVAILABLE=true
else
    echo "✗ CUDA version failed to build (CUDA toolkit not installed)"
    CUDA_AVAILABLE=false
fi
echo

echo "3. Available executables:"
ls -la bin/mandelbrot_* 2>/dev/null | grep -E "^-rwx" | awk '{print "  " $9 " (" $5 " bytes)"}' || echo "  No executables found"
echo

echo "4. Recommended version for your system:"
if [ "$CUDA_AVAILABLE" = true ] && nvidia-smi > /dev/null 2>&1; then
    echo "  🚀 Use bin/mandelbrot_cuda (CUDA - Maximum NVIDIA performance)"
elif make check-opencl 2>/dev/null | grep -q "NVIDIA\|AMD\|Intel"; then
    echo "  ⚡ Use bin/mandelbrot_opencl (OpenCL - Good GPU performance)" 
else
    echo "  💻 Use bin/mandelbrot_cpu (CPU - Fallback option)"
fi

echo
echo "5. Quick start:"
echo "  bin/mandelbrot_opencl   # Run OpenCL version (recommended for most)"
echo "  bin/mandelbrot_cuda     # Run CUDA version (NVIDIA only, max performance)"
echo "  bin/mandelbrot_cpu      # Run CPU version (always works)"
echo
echo "Press Ctrl+C to exit any running program."
