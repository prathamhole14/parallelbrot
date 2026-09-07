/*
    CUDA Mandelbrot Kernel
    High-performance GPU computation of the Mandelbrot set using CUDA
*/

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

// Enhanced color mapping functions with multiple beautiful palettes

// Ultra Fractal inspired color scheme - deep blues to bright oranges
__device__ float4 mapColor_UltraFractal(int iterations, int max_iterations) {
    if (iterations == max_iterations) {
        return make_float4(0.0f, 0.0f, 0.0f, 1.0f); // Black for points in the set
    }
    
    float t = (float)iterations / (float)max_iterations;
    float smooth_t = t + 1.0f - logf(logf(sqrtf(4.0f)))/logf(2.0f); // Smooth coloring
    smooth_t = fmodf(smooth_t * 0.05f, 1.0f); // Slower color cycling
    
    float r, g, b;
    
    if (smooth_t < 0.16f) {
        // Deep blue to cyan
        float local_t = smooth_t / 0.16f;
        r = 0.0f;
        g = local_t * 0.7f;
        b = 0.5f + local_t * 0.5f;
    } else if (smooth_t < 0.42f) {
        // Cyan to yellow
        float local_t = (smooth_t - 0.16f) / 0.26f;
        r = local_t;
        g = 0.7f + local_t * 0.3f;
        b = 1.0f - local_t;
    } else if (smooth_t < 0.6425f) {
        // Yellow to red
        float local_t = (smooth_t - 0.42f) / 0.2225f;
        r = 1.0f;
        g = 1.0f - local_t * 0.5f;
        b = 0.0f;
    } else if (smooth_t < 0.8575f) {
        // Red to magenta
        float local_t = (smooth_t - 0.6425f) / 0.215f;
        r = 1.0f;
        g = local_t * 0.5f;
        b = local_t;
    } else {
        // Magenta back to blue
        float local_t = (smooth_t - 0.8575f) / 0.1425f;
        r = 1.0f - local_t;
        g = 0.5f - local_t * 0.5f;
        b = 1.0f;
    }
    
    return make_float4(r, g, b, 1.0f);
}

// Fire/Heat color scheme - black through red, orange, yellow to white
__device__ float4 mapColor_Fire(int iterations, int max_iterations) {
    if (iterations == max_iterations) {
        return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    
    float t = (float)iterations / (float)max_iterations;
    t = powf(t, 0.8f); // Enhance contrast
    
    float r, g, b;
    
    if (t < 0.25f) {
        // Black to dark red
        float local_t = t / 0.25f;
        r = local_t * 0.8f;
        g = 0.0f;
        b = 0.0f;
    } else if (t < 0.5f) {
        // Dark red to bright red
        float local_t = (t - 0.25f) / 0.25f;
        r = 0.8f + local_t * 0.2f;
        g = local_t * 0.3f;
        b = 0.0f;
    } else if (t < 0.75f) {
        // Red to orange/yellow
        float local_t = (t - 0.5f) / 0.25f;
        r = 1.0f;
        g = 0.3f + local_t * 0.7f;
        b = local_t * 0.5f;
    } else {
        // Orange to white
        float local_t = (t - 0.75f) / 0.25f;
        r = 1.0f;
        g = 1.0f;
        b = 0.5f + local_t * 0.5f;
    }
    
    return make_float4(r, g, b, 1.0f);
}

// Ocean color scheme - deep blue through turquoise to white foam
__device__ float4 mapColor_Ocean(int iterations, int max_iterations) {
    if (iterations == max_iterations) {
        return make_float4(0.0f, 0.0f, 0.1f, 1.0f); // Deep blue for set points
    }
    
    float t = (float)iterations / (float)max_iterations;
    t = sinf(t * 3.14159f * 0.5f); // Smoother distribution
    
    // The waveforms overshoot [0, 1]; OpenGL clamped on upload, but callers
    // reading the buffer directly need the values already in range.
    float r = __saturatef(0.1f + t * (0.3f + 0.7f * sinf(t * 6.28f)));
    float g = __saturatef(0.2f + t * (0.6f + 0.4f * cosf(t * 4.0f)));
    float b = __saturatef(0.4f + t * (0.6f + 0.4f * sinf(t * 8.0f)));
    
    return make_float4(r, g, b, 1.0f);
}

// Psychedelic rainbow with enhanced saturation
__device__ float4 mapColor_Psychedelic(int iterations, int max_iterations) {
    if (iterations == max_iterations) {
        return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    
    float t = (float)iterations / (float)max_iterations;
    t = fmodf(t * 3.0f, 1.0f); // Triple the frequency for more color bands
    
    float r = 0.5f + 0.5f * sinf(2.0f * 3.14159f * t + 0.0f);
    float g = 0.5f + 0.5f * sinf(2.0f * 3.14159f * t + 2.09f);
    float b = 0.5f + 0.5f * sinf(2.0f * 3.14159f * t + 4.19f);
    
    // Enhance saturation
    float max_val = fmaxf(fmaxf(r, g), b);
    if (max_val > 0.5f) {
        r = r * 1.2f;
        g = g * 1.2f; 
        b = b * 1.2f;
    }
    
    return make_float4(fminf(r, 1.0f), fminf(g, 1.0f), fminf(b, 1.0f), 1.0f);
}

// Main color mapping function - selects scheme based on parameter
__device__ float4 mapColor(int iterations, int max_iterations, int color_scheme) {
    switch (color_scheme) {
        case 0: return mapColor_UltraFractal(iterations, max_iterations);
        case 1: return mapColor_Fire(iterations, max_iterations);
        case 2: return mapColor_Ocean(iterations, max_iterations);
        case 3: return mapColor_Psychedelic(iterations, max_iterations);
        default: return mapColor_Fire(iterations, max_iterations); // Fire as default
    }
}

// Optimized Mandelbrot kernel with unrolled iterations
__device__ int mandelbrot_iterations(double real, double imag, int max_iterations) {
    double z_real = 0.0;
    double z_imag = 0.0;
    int iterations = 0;
    
    // Unroll the first few iterations for better performance
    #pragma unroll 4
    for (int i = 0; i < max_iterations && iterations < max_iterations; i += 4) {
        // Iteration 1
        if (iterations < max_iterations) {
            double z_real_sq = z_real * z_real;
            double z_imag_sq = z_imag * z_imag;
            if (z_real_sq + z_imag_sq > 4.0) break;
            double temp = z_real_sq - z_imag_sq + real;
            z_imag = 2.0 * z_real * z_imag + imag;
            z_real = temp;
            iterations++;
        }
        
        // Iteration 2
        if (iterations < max_iterations) {
            double z_real_sq = z_real * z_real;
            double z_imag_sq = z_imag * z_imag;
            if (z_real_sq + z_imag_sq > 4.0) break;
            double temp = z_real_sq - z_imag_sq + real;
            z_imag = 2.0 * z_real * z_imag + imag;
            z_real = temp;
            iterations++;
        }
        
        // Iteration 3
        if (iterations < max_iterations) {
            double z_real_sq = z_real * z_real;
            double z_imag_sq = z_imag * z_imag;
            if (z_real_sq + z_imag_sq > 4.0) break;
            double temp = z_real_sq - z_imag_sq + real;
            z_imag = 2.0 * z_real * z_imag + imag;
            z_real = temp;
            iterations++;
        }
        
        // Iteration 4
        if (iterations < max_iterations) {
            double z_real_sq = z_real * z_real;
            double z_imag_sq = z_imag * z_imag;
            if (z_real_sq + z_imag_sq > 4.0) break;
            double temp = z_real_sq - z_imag_sq + real;
            z_imag = 2.0 * z_real * z_imag + imag;
            z_real = temp;
            iterations++;
        }
    }
    
    return iterations;
}

// CUDA kernel for Mandelbrot computation - buffer version
__global__ void mandelbrot_cuda_buffer(float4* output,
                                      int width,
                                      int height,
                                      double center_x,
                                      double center_y,
                                      double zoom,
                                      int max_iterations,
                                      int color_scheme) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate complex coordinates
    double scale = 4.0 / zoom;
    double aspect_ratio = (double)width / (double)height;
    
    double real = center_x + scale * aspect_ratio * ((double)x / (double)width - 0.5);
    double imag = center_y + scale * ((double)y / (double)height - 0.5);
    
    // Compute Mandelbrot iterations
    int iterations = mandelbrot_iterations(real, imag, max_iterations);
    
    // Map iteration count to color
    float4 color = mapColor(iterations, max_iterations, color_scheme);
    
    // Write to output buffer
    int index = y * width + x;
    output[index] = color;
}

// CUDA kernel for Mandelbrot computation - OpenGL interop version
__global__ void mandelbrot_cuda_texture(cudaSurfaceObject_t surface,
                                       int width,
                                       int height,
                                       double center_x,
                                       double center_y,
                                       double zoom,
                                       int max_iterations,
                                       int color_scheme) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate complex coordinates
    double scale = 4.0 / zoom;
    double aspect_ratio = (double)width / (double)height;
    
    double real = center_x + scale * aspect_ratio * ((double)x / (double)width - 0.5);
    double imag = center_y + scale * ((double)y / (double)height - 0.5);
    
    // Compute Mandelbrot iterations
    int iterations = mandelbrot_iterations(real, imag, max_iterations);
    
    // Map iteration count to color
    float4 color = mapColor(iterations, max_iterations, color_scheme);
    
    // Write to surface (OpenGL texture)
    surf2Dwrite(color, surface, x * sizeof(float4), y);
}

// Host function to launch the kernel
extern "C" {
    void launch_mandelbrot_kernel(float4* d_output,
                                 int width,
                                 int height,
                                 double center_x,
                                 double center_y,
                                 double zoom,
                                 int max_iterations,
                                 int color_scheme) {
        
        // Calculate grid and block dimensions
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        // Launch kernel
        mandelbrot_cuda_buffer<<<gridSize, blockSize>>>(
            d_output, width, height, center_x, center_y, zoom, max_iterations, color_scheme
        );
        
        // Synchronize to ensure completion
        cudaDeviceSynchronize();
    }
    
    void launch_mandelbrot_texture_kernel(cudaSurfaceObject_t surface,
                                         int width,
                                         int height,
                                         double center_x,
                                         double center_y,
                                         double zoom,
                                         int max_iterations,
                                         int color_scheme) {
        
        // Calculate grid and block dimensions
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        // Launch kernel
        mandelbrot_cuda_texture<<<gridSize, blockSize>>>(
            surface, width, height, center_x, center_y, zoom, max_iterations, color_scheme
        );
        
        // Synchronize to ensure completion
        cudaDeviceSynchronize();
    }
}
