/*
    OpenCL Mandelbrot Kernel
    High-performance GPU computation of the Mandelbrot set
*/

// Enhanced color mapping functions with multiple beautiful palettes

// Ultra Fractal inspired color scheme - deep blues to bright oranges
float4 mapColor_UltraFractal(int iterations, int max_iterations) {
    if (iterations == max_iterations) {
        return (float4)(0.0f, 0.0f, 0.0f, 1.0f); // Black for points in the set
    }
    
    float t = (float)iterations / (float)max_iterations;
    float smooth_t = t + 1.0f - log(log(sqrt(4.0f)))/log(2.0f); // Smooth coloring
    smooth_t = fmod(smooth_t * 0.05f, 1.0f); // Slower color cycling
    
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
    
    return (float4)(r, g, b, 1.0f);
}

// Fire/Heat color scheme - black through red, orange, yellow to white
float4 mapColor_Fire(int iterations, int max_iterations) {
    if (iterations == max_iterations) {
        return (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    }
    
    float t = (float)iterations / (float)max_iterations;
    t = pow(t, 0.8f); // Enhance contrast
    
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
    
    return (float4)(r, g, b, 1.0f);
}

// Ocean color scheme - deep blue through turquoise to white foam
float4 mapColor_Ocean(int iterations, int max_iterations) {
    if (iterations == max_iterations) {
        return (float4)(0.0f, 0.0f, 0.1f, 1.0f); // Deep blue for set points
    }
    
    float t = (float)iterations / (float)max_iterations;
    t = sin(t * 3.14159f * 0.5f); // Smoother distribution
    
    // The waveforms overshoot [0, 1]; OpenGL clamped on upload, but callers
    // reading the buffer directly need the values already in range.
    float r = clamp(0.1f + t * (0.3f + 0.7f * sin(t * 6.28f)), 0.0f, 1.0f);
    float g = clamp(0.2f + t * (0.6f + 0.4f * cos(t * 4.0f)), 0.0f, 1.0f);
    float b = clamp(0.4f + t * (0.6f + 0.4f * sin(t * 8.0f)), 0.0f, 1.0f);
    
    return (float4)(r, g, b, 1.0f);
}

// Psychedelic rainbow with enhanced saturation
float4 mapColor_Psychedelic(int iterations, int max_iterations) {
    if (iterations == max_iterations) {
        return (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    }
    
    float t = (float)iterations / (float)max_iterations;
    t = fmod(t * 3.0f, 1.0f); // Triple the frequency for more color bands
    
    float r = 0.5f + 0.5f * sin(2.0f * 3.14159f * t + 0.0f);
    float g = 0.5f + 0.5f * sin(2.0f * 3.14159f * t + 2.09f);
    float b = 0.5f + 0.5f * sin(2.0f * 3.14159f * t + 4.19f);
    
    // Enhance saturation
    float max_val = fmax(fmax(r, g), b);
    if (max_val > 0.5f) {
        r = r * 1.2f;
        g = g * 1.2f; 
        b = b * 1.2f;
    }
    
    return (float4)(clamp(r, 0.0f, 1.0f), clamp(g, 0.0f, 1.0f), clamp(b, 0.0f, 1.0f), 1.0f);
}

// Main color mapping function - selects color scheme based on parameter
float4 mapColor(int iterations, int max_iterations, int color_scheme) {
    switch (color_scheme) {
        case 1:
            return mapColor_Fire(iterations, max_iterations);
        case 2:
            return mapColor_Ocean(iterations, max_iterations);
        case 3:
            return mapColor_Psychedelic(iterations, max_iterations);
        default:
            return mapColor_UltraFractal(iterations, max_iterations);
    }
}

// Buffer version - outputs to a global memory buffer
__kernel void mandelbrot_buffer(__global float4* output,
                               const int width,
                               const int height,
                               const double center_x,
                               const double center_y, 
                               const double zoom,
                               const int max_iterations,
                               const int color_scheme) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) {
        return;
    }
    
    // Calculate complex coordinates
    double scale = 4.0 / zoom;
    double aspect_ratio = (double)width / (double)height;
    
    double real = center_x + scale * aspect_ratio * ((double)x / (double)width - 0.5);
    double imag = center_y + scale * ((double)y / (double)height - 0.5);
    
    // Mandelbrot iteration
    double z_real = 0.0;
    double z_imag = 0.0;
    int iterations = 0;
    
    while (iterations < max_iterations) {
        double z_real_sq = z_real * z_real;
        double z_imag_sq = z_imag * z_imag;
        
        if (z_real_sq + z_imag_sq > 4.0) {
            break;
        }
        
        double temp = z_real_sq - z_imag_sq + real;
        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = temp;
        
        iterations++;
    }
    
    // Map iteration count to color
    float4 color = mapColor(iterations, max_iterations, color_scheme);
    
    // Write to output buffer
    int index = y * width + x;
    output[index] = color;
}

