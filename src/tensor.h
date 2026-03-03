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
bool cuda_check_gpu();
void cuda_matmul(float* out, const float* a, const float* b, int M, int N, int K);
void cuda_matmul_transposed_weight(float* out, const float* x, const float* w, int N, int K);
void cuda_rmsnorm(float* out, const float* x, const float* w, int n, float eps);
void cuda_softmax(float* x, int n);
void cuda_silu_elementwise_mul(float* out, const float* gate, const float* up, int n);
void cuda_rope(float* q, float* k, int q_dim, int k_dim, int head_dim, int pos, float theta);
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

// ---- FP8 conversion ----

// Convert FP8 E4M3FN (1 sign + 4 exponent + 3 mantissa, bias=7, no Inf) to FP32.
// NaN is represented only by 0x7F (+NaN) and 0xFF (-NaN).
inline float fp8_e4m3_to_fp32(uint8_t h) {
    uint32_t s = static_cast<uint32_t>(h >> 7) << 31;
    uint32_t e = static_cast<uint32_t>((h >> 3) & 0x0F);
    uint32_t m = static_cast<uint32_t>(h & 0x07);

    if (e == 0x0F && m == 0x07) {
        // NaN (0x7F or 0xFF)
        uint32_t nan_val = s | 0x7FC00000u;
        float f; memcpy(&f, &nan_val, sizeof(f)); return f;
    }

    uint32_t result;
    if (e == 0) {
        if (m == 0) {
            result = s; // ±0
        } else {
            // Subnormal: value = (-1)^s * 2^(-6) * (0.m)_2 = m * 2^(-9)
            // Normalize: find position of leading 1 bit in m (0..2)
            int pos = 2;
            while (pos > 0 && !(m & (1u << pos))) pos--;
            int exp_val = pos - 9;
            uint32_t frac = m & ((1u << pos) - 1u);
            result = s | (static_cast<uint32_t>(exp_val + 127) << 23) | (frac << (23 - pos));
        }
    } else {
        // Normal: value = (-1)^s * 2^(e-7) * (1 + m/8)
        // FP32: biased_exp = e + 120, mantissa in bits [22:20]
        result = s | ((e + 120u) << 23) | (m << 20);
    }
    float f; memcpy(&f, &result, sizeof(f)); return f;
}

// Convert FP8 E5M2 (1 sign + 5 exponent + 2 mantissa, bias=15) to FP32.
// Supports Inf and NaN following the E5M2 specification.
inline float fp8_e5m2_to_fp32(uint8_t h) {
    uint32_t s = static_cast<uint32_t>(h >> 7) << 31;
    uint32_t e = static_cast<uint32_t>((h >> 2) & 0x1F);
    uint32_t m = static_cast<uint32_t>(h & 0x03);

    uint32_t result;
    if (e == 0x1F) {
        result = (m == 0) ? (s | 0x7F800000u)  // ±Inf
                          : (s | 0x7FC00000u);  // NaN
    } else if (e == 0) {
        if (m == 0) {
            result = s; // ±0
        } else {
            // Subnormal: value = (-1)^s * 2^(-14) * (0.m)_2 = m * 2^(-16)
            int pos = 1;
            while (pos > 0 && !(m & (1u << pos))) pos--;
            int exp_val = pos - 16;
            uint32_t frac = m & ((1u << pos) - 1u);
            result = s | (static_cast<uint32_t>(exp_val + 127) << 23) | (frac << (23 - pos));
        }
    } else {
        // Normal: value = (-1)^s * 2^(e-15) * (1 + m/4)
        // FP32: biased_exp = e + 112, mantissa in bits [22:21]
        result = s | ((e + 112u) << 23) | (m << 21);
    }
    float f; memcpy(&f, &result, sizeof(f)); return f;
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

// ---- K-quant helper ----

// Extract a 6-bit scale and 6-bit min from the packed 12-byte scales field.
// Used by Q4_K and Q5_K: 8 (d,m) pairs packed as:
//   bytes 0-3:  lower 6 bits of d[0..3]  (bits 6-7 = upper bits of d[4..7])
//   bytes 4-7:  lower 6 bits of m[0..3]  (bits 6-7 = upper bits of m[4..7])
//   bytes 8-11: lower 4 bits of d[4..7] in bits 0-3, lower 4 bits of m[4..7] in bits 4-7
static inline void get_scale_min_k4(int j, const uint8_t* scales,
                                    uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = scales[j]   & 63;
        m  = scales[j+4] & 63;
    } else {
        sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4);
        m  = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4);
    }
}

// ---- K-quant dequantization (block size = 256 elements) ----

// Q2_K: 2-bit quants, block = scales(16)+qs(64)+d_f16(2)+dmin_f16(2) = 84 bytes
inline void dequantize_q2_k(const void* src, float* dst, int64_t n) {
    const size_t block_bytes = 84;
    const int    QK = 256;
    int64_t nb = n / QK;
    const uint8_t* data = static_cast<const uint8_t*>(src);

    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* blk   = data + b * block_bytes;
        const uint8_t* sc    = blk;       // scales: 16 bytes at offset 0
        const uint8_t* qs    = blk + 16;  // quants: 64 bytes at offset 16
        uint16_t d_h, dmin_h;
        memcpy(&d_h,    blk + 80, 2);
        memcpy(&dmin_h, blk + 82, 2);
        float d    = fp16_to_fp32(d_h);
        float dmin = fp16_to_fp32(dmin_h);

        float* y = dst + static_cast<size_t>(b) * QK;
        const uint8_t* q = qs;
        int is = 0;
        for (int n_ = 0; n_ < QK; n_ += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc0 = sc[is++];
                float dl = d * (sc0 & 0xF), ml = dmin * (sc0 >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((q[l]    >> shift) & 3) - ml;
                uint8_t sc1 = sc[is++];
                dl = d * (sc1 & 0xF); ml = dmin * (sc1 >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((q[l+16] >> shift) & 3) - ml;
                shift += 2;
                if (shift == 8) { shift = 0; q += 32; }
            }
        }
    }
}

// Q3_K: 3-bit quants, block = hmask(32)+qs(64)+scales(12)+d_f16(2) = 110 bytes
// Scales are 16 × 6-bit values packed into 12 bytes with a specific bit layout.
inline void dequantize_q3_k(const void* src, float* dst, int64_t n) {
    const size_t block_bytes = 110;
    const int    QK = 256;
    int64_t nb = n / QK;
    const uint8_t* data = static_cast<const uint8_t*>(src);

    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* blk  = data + b * block_bytes;
        const uint8_t* hm   = blk;        // hmask:  32 bytes at offset 0
        const uint8_t* qs   = blk + 32;   // quants: 64 bytes at offset 32
        const uint8_t* sc   = blk + 96;   // scales: 12 bytes at offset 96
        uint16_t d_h;
        memcpy(&d_h, blk + 108, 2);
        float d = fp16_to_fp32(d_h);

        // Unpack 16 × 6-bit signed scales from 12 bytes.
        // Packing (from ggml-quants reference):
        //   sc[0..3]  → bits 0-1 of scales[0,4,8,12]  and  bits 2-3 of scales[0,4,8,12]
        // Use the standard ggml extraction pattern:
        const uint32_t kmask1 = 0x03030303u;
        uint32_t tmp[3];
        memcpy(tmp, sc, 12);
        int8_t scales_i8[16];
        {
            uint32_t a0 = ((tmp[0]     ) & kmask1) | (((tmp[2]     ) & kmask1) << 4);
            uint32_t a1 = (((tmp[0]>>2)) & kmask1) | (((tmp[2]>>2) ) & kmask1) << 4;
            uint32_t a2 = ((tmp[1]     ) & kmask1) | (((tmp[2]>>4) ) & kmask1) << 4;
            uint32_t a3 = (((tmp[1]>>2)) & kmask1) | (((tmp[2]>>6) ) & kmask1) << 4;
            // subtract 32 from each byte
            uint32_t sub = 0x20202020u;
            a0 -= sub; a1 -= sub; a2 -= sub; a3 -= sub;
            memcpy(scales_i8 +  0, &a0, 4);
            memcpy(scales_i8 +  4, &a1, 4);
            memcpy(scales_i8 +  8, &a2, 4);
            memcpy(scales_i8 + 12, &a3, 4);
        }

        float* y = dst + static_cast<size_t>(b) * QK;
        const uint8_t* q = qs;
        uint8_t m = 1;
        for (int n_ = 0; n_ < QK; n_ += 128) {
            int sc_off = (n_ == 0) ? 0 : 8;
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl0 = d * scales_i8[sc_off + j*2 + 0];
                float dl1 = d * scales_i8[sc_off + j*2 + 1];
                int base = n_ + j * 32;
                for (int l = 0; l < 16; l++)
                    y[base + l]      = dl0 * (static_cast<int>((q[l]    >> shift) & 3) - (hm[l]    & m ? 0 : 4));
                for (int l = 0; l < 16; l++)
                    y[base + 16 + l] = dl1 * (static_cast<int>((q[l+16] >> shift) & 3) - (hm[l+16] & m ? 0 : 4));
                shift += 2;
                if (shift == 8) { shift = 0; q += 32; }
                m <<= 1;
            }
        }
    }
}

// Q4_K: 4-bit quants, block = d_f16(2)+dmin_f16(2)+scales(12)+qs(128) = 144 bytes
inline void dequantize_q4_k(const void* src, float* dst, int64_t n) {
    const size_t block_bytes = 144;
    const int    QK = 256;
    int64_t nb = n / QK;
    const uint8_t* data = static_cast<const uint8_t*>(src);

    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* blk = data + b * block_bytes;
        uint16_t d_h, dmin_h;
        memcpy(&d_h,    blk,     2);
        memcpy(&dmin_h, blk + 2, 2);
        float d    = fp16_to_fp32(d_h);
        float dmin = fp16_to_fp32(dmin_h);
        const uint8_t* scales = blk + 4;   // 12 bytes
        const uint8_t* qs     = blk + 16;  // 128 bytes

        float* y = dst + static_cast<size_t>(b) * QK;
        int is = 0;
        for (int j = 0; j < QK; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is++, scales, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is++, scales, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            const uint8_t* q = qs + (j >> 1);  // 32 bytes per 64-element group
            for (int l = 0; l < 32; l++) y[j + l]      = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) y[j + l + 32] = d2 * (q[l] >>  4) - m2;
        }
    }
}

// Q5_K: 5-bit quants, block = d_f16(2)+dmin_f16(2)+scales(12)+qh(32)+qs(128) = 176 bytes
inline void dequantize_q5_k(const void* src, float* dst, int64_t n) {
    const size_t block_bytes = 176;
    const int    QK = 256;
    int64_t nb = n / QK;
    const uint8_t* data = static_cast<const uint8_t*>(src);

    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* blk = data + b * block_bytes;
        uint16_t d_h, dmin_h;
        memcpy(&d_h,    blk,     2);
        memcpy(&dmin_h, blk + 2, 2);
        float d    = fp16_to_fp32(d_h);
        float dmin = fp16_to_fp32(dmin_h);
        const uint8_t* scales = blk + 4;   // 12 bytes
        const uint8_t* qh     = blk + 16;  // 32 bytes (1 high bit per element)
        const uint8_t* ql     = blk + 48;  // 128 bytes (4 low bits per element)

        float* y = dst + static_cast<size_t>(b) * QK;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is++, scales, sc, m);
            float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is++, scales, sc, m);
            float d2 = d * sc, m2 = dmin * m;
            const uint8_t* q = ql + (j >> 1);
            for (int l = 0; l < 32; l++) {
                y[j + l]      = d1 * ((q[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                y[j + l + 32] = d2 * ((q[l] >>  4) + (qh[l] & u2 ? 16 : 0)) - m2;
            }
            u1 <<= 2; u2 <<= 2;
        }
    }
}

// Q6_K: 6-bit quants, block = ql(128)+qh(64)+scales_i8(16)+d_f16(2) = 210 bytes
inline void dequantize_q6_k(const void* src, float* dst, int64_t n) {
    const size_t block_bytes = 210;
    const int    QK = 256;
    int64_t nb = n / QK;
    const uint8_t* data = static_cast<const uint8_t*>(src);

    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* blk = data + b * block_bytes;
        const uint8_t* ql  = blk;        // 128 bytes: lower 4 bits
        const uint8_t* qh  = blk + 128;  //  64 bytes: upper 2 bits
        const int8_t*  sc  = reinterpret_cast<const int8_t*>(blk + 192); // 16 bytes
        uint16_t d_h;
        memcpy(&d_h, blk + 208, 2);
        float d = fp16_to_fp32(d_h);

        float* y = dst + static_cast<size_t>(b) * QK;
        for (int n_ = 0; n_ < QK; n_ += 128) {
            const uint8_t* ql_ = ql + (n_ >> 1);
            const uint8_t* qh_ = qh + (n_ >> 2);
            const int8_t*  sc_ = sc + (n_ >> 4);
            float*          y_ = y  + n_;
            for (int l = 0; l < 32; l++) {
                int is = l >> 4;
                int8_t q1 = static_cast<int8_t>((ql_[l]      & 0xF) | (((qh_[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = static_cast<int8_t>((ql_[l + 32] & 0xF) | (((qh_[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = static_cast<int8_t>((ql_[l]      >>  4) | (((qh_[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = static_cast<int8_t>((ql_[l + 32] >>  4) | (((qh_[l] >> 6) & 3) << 4)) - 32;
                y_[l +  0] = d * sc_[is + 0] * q1;
                y_[l + 32] = d * sc_[is + 2] * q2;
                y_[l + 64] = d * sc_[is + 4] * q3;
                y_[l + 96] = d * sc_[is + 6] * q4;
            }
        }
    }
}

// Convert FP8 E4M3FN array to F32
inline void dequantize_f8_e4m3(const void* src, float* dst, int64_t n) {
    const uint8_t* data = static_cast<const uint8_t*>(src);
    for (int64_t i = 0; i < n; i++) {
        dst[i] = fp8_e4m3_to_fp32(data[i]);
    }
}

// Convert FP8 E5M2 array to F32
inline void dequantize_f8_e5m2(const void* src, float* dst, int64_t n) {
    const uint8_t* data = static_cast<const uint8_t*>(src);
    for (int64_t i = 0; i < n; i++) {
        dst[i] = fp8_e5m2_to_fp32(data[i]);
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
        case GGML_TYPE_Q2_K:
            dequantize_q2_k(src, dst, n);
            break;
        case GGML_TYPE_Q3_K:
            dequantize_q3_k(src, dst, n);
            break;
        case GGML_TYPE_Q4_K:
            dequantize_q4_k(src, dst, n);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_q5_k(src, dst, n);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_q6_k(src, dst, n);
            break;
        case GGML_TYPE_F8_E4M3:
            dequantize_f8_e4m3(src, dst, n);
            break;
        case GGML_TYPE_F8_E5M2:
            dequantize_f8_e5m2(src, dst, n);
            break;
        default:
            fprintf(stderr, "Error: unsupported quantization type %s for dequantization\n",
                    ggml_type_name(type));
            break;
    }
}

// ---- Lightweight quantized weight handle ----

// Holds a raw (potentially quantized) weight tensor.  For Q4_0 and Q8_0 the
// data pointer points directly into the GGUF file's owned_data buffer, so no
// extra allocation or dequantization is needed at load time.  For F32 the
// pointer may also point into the file buffer.  F16 is converted to F32 at
// load time (stored in weight_storage) because no fused F16 matmul is
// provided.
struct QuantWeight {
    const void* data = nullptr;
    GGMLType type = GGML_TYPE_F32;
    bool valid() const { return data != nullptr; }
};

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

// Fused Q8_0 dequantize + transposed matmul: out[N] = x[K] · W[N×K]
// W is in Q8_0 format: for every 32-element block, 2 bytes f16 scale then 32
// int8 values (34 bytes total per block).  Weights are never expanded to F32
// in memory — each scale+value pair is consumed once, keeping weight traffic
// at ~1 byte/element instead of 4 bytes/element.
// The scale is hoisted out of the inner accumulation: the 32 products
// x[j]*vals[j] are summed as floats first, then multiplied by the block
// scale once, reducing multiply count from 2×32 to 32+1 per block.
inline void cpu_matmul_transposed_q8_0(float* out, const float* x,
                                       const void* w, int N, int K) {
    const int bytes_per_block = 34; // 2-byte f16 scale + 32 int8 values
    const int blocks_per_row = K / 32;
    const uint8_t* wptr = static_cast<const uint8_t*>(w);
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        const uint8_t* row = wptr +
            static_cast<size_t>(i) * blocks_per_row * bytes_per_block;
        for (int b = 0; b < blocks_per_row; b++) {
            uint16_t scale_h;
            memcpy(&scale_h, row, 2);
            float scale = fp16_to_fp32(scale_h);
            const int8_t* vals = reinterpret_cast<const int8_t*>(row + 2);
            const float* xb = x + b * 32;
            // Accumulate the unscaled block dot product first, then apply
            // the block scale once — reduces multiplications from 64 to 33.
            float block_sum = 0.0f;
            for (int j = 0; j < 32; j++) {
                block_sum += xb[j] * static_cast<float>(vals[j]);
            }
            sum += block_sum * scale;
            row += bytes_per_block;
        }
        out[i] = sum;
    }
}

// Fused Q4_0 dequantize + transposed matmul: out[N] = x[K] · W[N×K]
// W is in Q4_0 format: for every 32-element block, 2 bytes f16 scale then 16
// nibble bytes (18 bytes total per block).
// The scale is hoisted out of the inner accumulation: the 32 unscaled
// products are summed first, then multiplied by the block scale once,
// reducing multiply count from 64 to 33 per block.
inline void cpu_matmul_transposed_q4_0(float* out, const float* x,
                                       const void* w, int N, int K) {
    const int bytes_per_block = 18; // 2-byte f16 scale + 16 nibble bytes
    const int blocks_per_row = K / 32;
    const uint8_t* wptr = static_cast<const uint8_t*>(w);
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        const uint8_t* row = wptr +
            static_cast<size_t>(i) * blocks_per_row * bytes_per_block;
        for (int b = 0; b < blocks_per_row; b++) {
            uint16_t scale_h;
            memcpy(&scale_h, row, 2);
            float scale = fp16_to_fp32(scale_h);
            const uint8_t* nibbles = row + 2;
            const float* xb = x + b * 32;
            // Accumulate the unscaled block dot product first, then apply
            // the block scale once — reduces multiplications from 64 to 33.
            float block_sum = 0.0f;
            for (int j = 0; j < 16; j++) {
                uint8_t byte = nibbles[j];
                // Q4_0 stores 4-bit unsigned values in range [0,15], offset by
                // 8 so that the represented range is [-8, 7].
                block_sum += xb[j]      * static_cast<float>(static_cast<int>(byte & 0x0F) - 8);
                block_sum += xb[j + 16] * static_cast<float>(static_cast<int>(byte >> 4)  - 8);
            }
            sum += block_sum * scale;
            row += bytes_per_block;
        }
        out[i] = sum;
    }
}

// Fused F16 dequantize + transposed matmul: out[N] = x[K] · W[N×K]
// W is stored as 2 bytes per element in F16 format (no block structure).
// Direct computation: each F16 value is converted to F32 inline during the dot
// product, keeping weight traffic at 2 bytes/element instead of 4 bytes/element
// and avoiding materialization of a full F32 weight buffer.
inline void cpu_matmul_transposed_f16(float* out, const float* x,
                                      const void* w, int N, int K) {
    const uint16_t* wptr = static_cast<const uint16_t*>(w);
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        const uint16_t* row = wptr + static_cast<size_t>(i) * K;
        for (int k = 0; k < K; k++) {
            sum += x[k] * fp16_to_fp32(row[k]);
        }
        out[i] = sum;
    }
}

// Fused FP8 E4M3FN dequantize + transposed matmul: out[N] = x[K] · W[N×K]
// W is stored as 1 byte per element in FP8 E4M3FN format (no block structure).
// Direct computation: each FP8 value is converted to F32 inline during the dot
// product, keeping weight traffic at 1 byte/element instead of 4 bytes/element
// and avoiding materialization of a full F32 weight buffer.
inline void cpu_matmul_transposed_f8_e4m3(float* out, const float* x,
                                           const void* w, int N, int K) {
    const uint8_t* wptr = static_cast<const uint8_t*>(w);
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        const uint8_t* row = wptr + static_cast<size_t>(i) * K;
        for (int k = 0; k < K; k++) {
            sum += x[k] * fp8_e4m3_to_fp32(row[k]);
        }
        out[i] = sum;
    }
}

// Fused FP8 E5M2 dequantize + transposed matmul: out[N] = x[K] · W[N×K]
// W is stored as 1 byte per element in FP8 E5M2 format (no block structure).
// Direct computation: each FP8 value is converted to F32 inline during the dot
// product, keeping weight traffic at 1 byte/element instead of 4 bytes/element.
inline void cpu_matmul_transposed_f8_e5m2(float* out, const float* x,
                                           const void* w, int N, int K) {
    const uint8_t* wptr = static_cast<const uint8_t*>(w);
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        const uint8_t* row = wptr + static_cast<size_t>(i) * K;
        for (int k = 0; k < K; k++) {
            sum += x[k] * fp8_e5m2_to_fp32(row[k]);
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
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < n; i++) {
        out[i] = silu(gate[i]) * up[i];
    }
}

// Rotary positional embeddings (RoPE) - interleaved style (LLaMA)
// Pairs consecutive elements: (0,1), (2,3), ...
// q_dim and k_dim may differ when using GQA (grouped query attention).
// freqs[j] = 1 / theta^(2j / head_dim), length = head_dim/2
inline void cpu_rope_apply(float* q, float* k, int q_dim, int k_dim, int head_dim,
                           int pos, const float* freqs) {
    int q_heads = q_dim / head_dim;
    int k_heads = k_dim / head_dim;

    // Apply to query
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int h = 0; h < q_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float angle = pos * freqs[i / 2];
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            float q0 = qh[i], q1 = qh[i + 1];
            qh[i]     = q0 * cos_val - q1 * sin_val;
            qh[i + 1] = q0 * sin_val + q1 * cos_val;
        }
    }

    // Apply to key
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int h = 0; h < k_heads; h++) {
        float* kh = k + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float angle = pos * freqs[i / 2];
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            float k0 = kh[i], k1 = kh[i + 1];
            kh[i]     = k0 * cos_val - k1 * sin_val;
            kh[i + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
}

// Wrapper that computes the frequency table on the fly (for standalone use / tests).
inline void cpu_rope(float* q, float* k, int q_dim, int k_dim, int head_dim, int pos, float theta) {
    int half_dim = head_dim / 2;
    std::vector<float> freqs(half_dim);
    for (int i = 0; i < half_dim; i++) {
        freqs[i] = 1.0f / powf(theta, static_cast<float>(2 * i) / head_dim);
    }
    cpu_rope_apply(q, k, q_dim, k_dim, head_dim, pos, freqs.data());
}

// Rotary positional embeddings (RoPE) - neox/halved style (Qwen, GPT-NeoX)
// Pairs elements at distance head_dim/2: (0, head_dim/2), (1, head_dim/2+1), ...
// freqs[i] = 1 / theta^(2i / head_dim), length = head_dim/2
inline void cpu_rope_neox_apply(float* q, float* k, int q_dim, int k_dim, int head_dim,
                                int pos, const float* freqs) {
    int half_dim = head_dim / 2;

    // Apply to query
    int q_heads = q_dim / head_dim;
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int h = 0; h < q_heads; h++) {
        float* qh = q + h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            float angle = pos * freqs[i];
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            float q0 = qh[i], q1 = qh[i + half_dim];
            qh[i]            = q0 * cos_val - q1 * sin_val;
            qh[i + half_dim] = q1 * cos_val + q0 * sin_val;
        }
    }

    // Apply to key
    int k_heads = k_dim / head_dim;
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int h = 0; h < k_heads; h++) {
        float* kh = k + h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            float angle = pos * freqs[i];
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            float k0 = kh[i], k1 = kh[i + half_dim];
            kh[i]            = k0 * cos_val - k1 * sin_val;
            kh[i + half_dim] = k1 * cos_val + k0 * sin_val;
        }
    }
}

// Wrapper that computes the frequency table on the fly (for standalone use / tests).
inline void cpu_rope_neox(float* q, float* k, int q_dim, int k_dim, int head_dim, int pos, float theta) {
    int half_dim = head_dim / 2;
    std::vector<float> freqs(half_dim);
    for (int i = 0; i < half_dim; i++) {
        freqs[i] = 1.0f / powf(theta, static_cast<float>(2 * i) / head_dim);
    }
    cpu_rope_neox_apply(q, k, q_dim, k_dim, head_dim, pos, freqs.data());
}

// Element-wise add: out = a + b
inline void cpu_add(float* out, const float* a, const float* b, int n) {
    #ifdef LLM_USE_OPENMP
    #pragma omp parallel for
    #endif
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
            cuda_matmul_transposed_weight(out, x, w, N, K);
            return;
        }
#endif
        cpu_matmul_transposed(out, x, w, N, K);
    }

    // Transposed matmul dispatching on the weight's quantization type.
    // For CPU, uses fused dequantize+accumulate kernels that keep weights in
    // their compact format throughout (no temporary F32 expansion).
    // For CUDA, falls back to an on-the-fly dequantize into a temporary buffer
    // since the CUDA kernels currently expect float* inputs.
    void matmul_transposed_q(float* out, const float* x,
                             const QuantWeight& w, int N, int K) {
#ifdef LLM_USE_CUDA
        if (backend == Backend::CUDA) {
            if (w.type == GGML_TYPE_F32) {
                cuda_matmul_transposed_weight(out, x,
                    static_cast<const float*>(w.data), N, K);
            } else {
                std::vector<float> tmp(static_cast<size_t>(N) * K);
                dequantize(w.data, tmp.data(),
                           static_cast<int64_t>(N) * K, w.type);
                cuda_matmul_transposed_weight(out, x, tmp.data(), N, K);
            }
            return;
        }
#endif
        switch (w.type) {
            case GGML_TYPE_F32:
                cpu_matmul_transposed(out, x,
                    static_cast<const float*>(w.data), N, K);
                break;
            case GGML_TYPE_F16:
                cpu_matmul_transposed_f16(out, x, w.data, N, K);
                break;
            case GGML_TYPE_Q8_0:
                cpu_matmul_transposed_q8_0(out, x, w.data, N, K);
                break;
            case GGML_TYPE_Q4_0:
                cpu_matmul_transposed_q4_0(out, x, w.data, N, K);
                break;
            case GGML_TYPE_F8_E4M3:
                cpu_matmul_transposed_f8_e4m3(out, x, w.data, N, K);
                break;
            case GGML_TYPE_F8_E5M2:
                cpu_matmul_transposed_f8_e5m2(out, x, w.data, N, K);
                break;
            default: {
                // Generic fallback for other quantized types
                std::vector<float> tmp(static_cast<size_t>(N) * K);
                dequantize(w.data, tmp.data(),
                           static_cast<int64_t>(N) * K, w.type);
                cpu_matmul_transposed(out, x, tmp.data(), N, K);
                break;
            }
        }
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

    // When precomputed_freqs is non-null (CPU path), it is used directly to
    // avoid recomputing the frequency table on every token.  theta is still
    // required for the CUDA path which computes freqs on the device.
    void rope(float* q, float* k, int q_dim, int k_dim, int head_dim, int pos,
              float theta, const float* precomputed_freqs = nullptr, bool neox = false) {
#ifdef LLM_USE_CUDA
        if (backend == Backend::CUDA) {
            cuda_rope(q, k, q_dim, k_dim, head_dim, pos, theta);
            return;
        }
#endif
        if (precomputed_freqs) {
            if (neox) {
                cpu_rope_neox_apply(q, k, q_dim, k_dim, head_dim, pos, precomputed_freqs);
            } else {
                cpu_rope_apply(q, k, q_dim, k_dim, head_dim, pos, precomputed_freqs);
            }
        } else {
            if (neox) {
                cpu_rope_neox(q, k, q_dim, k_dim, head_dim, pos, theta);
            } else {
                cpu_rope(q, k, q_dim, k_dim, head_dim, pos, theta);
            }
        }
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
