#ifndef LLM_TENSOR_H
#define LLM_TENSOR_H

// Tensor operations for LLM inference.
// Supports CPU (with optional OpenMP) and GPU (CUDA) backends.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gguf.h"

#ifdef LLM_USE_OPENMP
#include <omp.h>
#endif

#ifdef LLM_USE_CUDA
// Forward declarations for CUDA kernels (defined in cuda_kernels.cu)
void cuda_matmul(float* out, const float* a, const float* b, int M, int N, int K);
void cuda_rmsnorm(float* out, const float* x, const float* w, int n, float eps);
void cuda_softmax(float* x, int n);
void cuda_silu_elementwise_mul(float* out, const float* gate, const float* up, int n);
void cuda_rope(float* q, float* k, int dim, int head_dim, int pos, float theta);
void cuda_add(float* out, const float* a, const float* b, int n);
#endif

// ---- Half-float conversion ----

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            exponent = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        result = sign | 0x7F800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

// ---- Dequantization ----

// Dequantize Q4_0 block: 32 elements stored as 1 f16 scale + 16 bytes of nibbles
inline void dequantize_q4_0(const void* src, float* dst, int64_t n) {
    const uint8_t* data = static_cast<const uint8_t*>(src);
    int64_t num_blocks = n / 32;
    for (int64_t b = 0; b < num_blocks; b++) {
        uint16_t scale_h;
        memcpy(&scale_h, data, 2);
        float scale = fp16_to_fp32(scale_h);
        data += 2;
        for (int j = 0; j < 16; j++) {
            uint8_t byte = data[j];
            dst[b * 32 + j]      = ((int)(byte & 0x0F) - 8) * scale;
            dst[b * 32 + j + 16] = ((int)(byte >> 4) - 8) * scale;
        }
        data += 16;
    }
}

// Dequantize Q8_0 block: 32 elements stored as 1 f16 scale + 32 int8 values
inline void dequantize_q8_0(const void* src, float* dst, int64_t n) {
    const uint8_t* data = static_cast<const uint8_t*>(src);
    int64_t num_blocks = n / 32;
    for (int64_t b = 0; b < num_blocks; b++) {
        uint16_t scale_h;
        memcpy(&scale_h, data, 2);
        float scale = fp16_to_fp32(scale_h);
        data += 2;
        for (int j = 0; j < 32; j++) {
            dst[b * 32 + j] = static_cast<int8_t>(data[j]) * scale;
        }
        data += 32;
    }
}

// Convert F16 array to F32
inline void dequantize_f16(const void* src, float* dst, int64_t n) {
    const uint16_t* data = static_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32(data[i]);
    }
}

// Dequantize any supported type to F32
inline void dequantize(const void* src, float* dst, int64_t n, GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(dst, src, n * sizeof(float));
            break;
        case GGML_TYPE_F16:
            dequantize_f16(src, dst, n);
            break;
        case GGML_TYPE_Q4_0:
            dequantize_q4_0(src, dst, n);
            break;
        case GGML_TYPE_Q8_0:
            dequantize_q8_0(src, dst, n);
            break;
        default:
            fprintf(stderr, "Error: unsupported quantization type %s for dequantization\n",
                    ggml_type_name(type));
            break;
    }
}

// ---- Backend selection ----

enum class Backend {
    CPU,
    CUDA,
};

inline const char* backend_name(Backend b) {
    switch (b) {
        case Backend::CPU: return "CPU";
        case Backend::CUDA: return "CUDA";
    }
    return "unknown";
}

// ---- CPU math operations ----

// Matrix multiply: out[M x N] = a[M x K] * b[K x N]
inline void cpu_matmul(float* out, const float* a, const float* b, int M, int N, int K) {
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}

// Optimized vector-matrix multiply: out[N] = x[K] * w[K x N]
// (single token inference path, w stored as [N x K] transposed)
inline void cpu_matmul_transposed(float* out, const float* x, const float* w, int N, int K) {
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        const float* row = w + i * K;
        for (int k = 0; k < K; k++) {
            sum += x[k] * row[k];
        }
        out[i] = sum;
    }
}

// RMS normalization: out = x * w / rms(x)
inline void cpu_rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

// Softmax in-place
inline void cpu_softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// SiLU(x) = x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// SiLU element-wise multiply: out = silu(gate) * up
inline void cpu_silu_elementwise_mul(float* out, const float* gate, const float* up, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = silu(gate[i]) * up[i];
    }
}

// Rotary positional embeddings (RoPE)
inline void cpu_rope(float* q, float* k, int dim, int head_dim, int pos, float theta) {
    for (int i = 0; i < dim; i += 2) {
        int head_offset = i % head_dim;
        float freq = 1.0f / powf(theta, static_cast<float>(head_offset) / head_dim);
        float angle = pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        // Apply to query
        float q0 = q[i], q1 = q[i + 1];
        q[i]     = q0 * cos_val - q1 * sin_val;
        q[i + 1] = q0 * sin_val + q1 * cos_val;

        // Apply to key
        float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * cos_val - k1 * sin_val;
        k[i + 1] = k0 * sin_val + k1 * cos_val;
    }
}

// Element-wise add: out = a + b
inline void cpu_add(float* out, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

// ---- Unified compute dispatcher ----

struct Compute {
    Backend backend;

    explicit Compute(Backend b) : backend(b) {}

    void matmul(float* out, const float* a, const float* b, int M, int N, int K) {
#ifdef LLM_USE_CUDA
        if (backend == Backend::CUDA) {
            cuda_matmul(out, a, b, M, N, K);
            return;
        }
#endif
        cpu_matmul(out, a, b, M, N, K);
    }

    void matmul_transposed(float* out, const float* x, const float* w, int N, int K) {
#ifdef LLM_USE_CUDA
        if (backend == Backend::CUDA) {
            // For CUDA, use standard matmul with M=1
            cuda_matmul(out, x, w, 1, N, K);
            return;
        }
#endif
        cpu_matmul_transposed(out, x, w, N, K);
    }

    void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
#ifdef LLM_USE_CUDA
        if (backend == Backend::CUDA) {
            cuda_rmsnorm(out, x, w, n, eps);
            return;
        }
#endif
        cpu_rmsnorm(out, x, w, n, eps);
    }

    void softmax(float* x, int n) {
#ifdef LLM_USE_CUDA
        if (backend == Backend::CUDA) {
            cuda_softmax(x, n);
            return;
        }
#endif
        cpu_softmax(x, n);
    }

    void silu_mul(float* out, const float* gate, const float* up, int n) {
#ifdef LLM_USE_CUDA
        if (backend == Backend::CUDA) {
            cuda_silu_elementwise_mul(out, gate, up, n);
            return;
        }
#endif
        cpu_silu_elementwise_mul(out, gate, up, n);
    }

    void rope(float* q, float* k, int dim, int head_dim, int pos, float theta) {
#ifdef LLM_USE_CUDA
        if (backend == Backend::CUDA) {
            cuda_rope(q, k, dim, head_dim, pos, theta);
            return;
        }
#endif
        cpu_rope(q, k, dim, head_dim, pos, theta);
    }

    void add(float* out, const float* a, const float* b, int n) {
#ifdef LLM_USE_CUDA
        if (backend == Backend::CUDA) {
            cuda_add(out, a, b, n);
            return;
        }
#endif
        cpu_add(out, a, b, n);
    }
};

#endif // LLM_TENSOR_H
