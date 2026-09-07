/*
    parallelbrot — CPU rendering core.

    Pure computation: no GLFW, no OpenGL, no windowing. The iteration loop
    and the colour palettes were lifted verbatim out of the interactive CPU
    frontend so that both the window and any headless caller produce
    identical pixels.
*/

#include "parallelbrot/core.hpp"

#include <algorithm>
#include <cmath>

namespace parallelbrot {
namespace {

// ── Palette 0: Ultra Fractal ────────────────────────────────
void map_color_ultra_fractal(int iterations, int max_iterations,
                             float& r, float& g, float& b) {
    if (iterations == max_iterations) {
        r = g = b = 0.0f;  // Black for points in the set
        return;
    }

    float t = (float)iterations / (float)max_iterations;
    float smooth_t = t + 1.0f - log(log(sqrt(4.0f)))/log(2.0f); // Smooth coloring
    smooth_t = fmod(smooth_t * 0.05f, 1.0f); // Slower color cycling

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
}

// ── Palette 1: Fire ─────────────────────────────────────────
void map_color_fire(int iterations, int max_iterations,
                    float& r, float& g, float& b) {
    if (iterations == max_iterations) {
        r = g = b = 0.0f;
        return;
    }

    float t = (float)iterations / (float)max_iterations;
    t = pow(t, 0.8f); // Enhance contrast

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
}

// ── Palette 2: Ocean ────────────────────────────────────────
void map_color_ocean(int iterations, int max_iterations,
                     float& r, float& g, float& b) {
    if (iterations == max_iterations) {
        r = 0.0f; g = 0.0f; b = 0.1f; // Deep blue for set points
        return;
    }

    float t = (float)iterations / (float)max_iterations;
    t = sin(t * 3.14159f * 0.5f); // Smoother distribution

    r = 0.1f + t * (0.3f + 0.7f * sin(t * 6.28f));
    g = 0.2f + t * (0.6f + 0.4f * cos(t * 4.0f));
    b = 0.4f + t * (0.6f + 0.4f * sin(t * 8.0f));
}

// ── Palette 3: Psychedelic ──────────────────────────────────
void map_color_psychedelic(int iterations, int max_iterations,
                           float& r, float& g, float& b) {
    if (iterations == max_iterations) {
        r = g = b = 0.0f;
        return;
    }

    float t = (float)iterations / (float)max_iterations;
    t = fmod(t * 3.0f, 1.0f); // Triple the frequency for more color bands

    r = 0.5f + 0.5f * sin(2.0f * 3.14159f * t + 0.0f);
    g = 0.5f + 0.5f * sin(2.0f * 3.14159f * t + 2.09f);
    b = 0.5f + 0.5f * sin(2.0f * 3.14159f * t + 4.19f);

    // Enhance saturation
    float max_val = std::max({r, g, b});
    if (max_val > 0.5f) {
        r = std::min(r * 1.2f, 1.0f);
        g = std::min(g * 1.2f, 1.0f);
        b = std::min(b * 1.2f, 1.0f);
    }
}

}  // namespace

void map_color(int iterations, int max_iterations, int color_scheme,
               float& r, float& g, float& b) {
    switch (color_scheme) {
        case COLOR_ULTRA_FRACTAL: map_color_ultra_fractal(iterations, max_iterations, r, g, b); break;
        case COLOR_FIRE:          map_color_fire(iterations, max_iterations, r, g, b);          break;
        case COLOR_OCEAN:         map_color_ocean(iterations, max_iterations, r, g, b);         break;
        case COLOR_PSYCHEDELIC:   map_color_psychedelic(iterations, max_iterations, r, g, b);   break;
        default:                  map_color_ultra_fractal(iterations, max_iterations, r, g, b); break;
    }
}

void render_cpu(const View& view, float* out) {
    if (!view.valid() || out == nullptr) {
        return;
    }

    const int    width          = view.width;
    const int    height         = view.height;
    const double center_x       = view.center_x;
    const double center_y       = view.center_y;
    const int    max_iterations = view.max_iterations;
    const int    color_scheme   = view.color_scheme;

    const double scale        = 4.0 / view.zoom;
    const double aspect_ratio = (double)width / (double)height;

    // Rows are independent, so this parallelises cleanly. Dynamic scheduling
    // matters here: interior points run the full iteration count while
    // exterior points bail out early, so a static split leaves threads idle.
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
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
            float r, g, b;
            map_color(iterations, max_iterations, color_scheme, r, g, b);

            // Write to buffer
            std::size_t index = ((std::size_t)y * (std::size_t)width + (std::size_t)x) * 4;
            out[index + 0] = r;     // R
            out[index + 1] = g;     // G
            out[index + 2] = b;     // B
            out[index + 3] = 1.0f;  // A
        }
    }
}

}  // namespace parallelbrot
