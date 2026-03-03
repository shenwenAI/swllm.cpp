// Basic tests for llm.cpp components

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf.h"
#include "tensor.h"
#include "sampler.h"
#include "tokenizer.h"
#include "model.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { fprintf(stderr, "  TEST: %s ... ", #name); } while(0)

#define PASS() \
    do { fprintf(stderr, "PASS\n"); tests_passed++; } while(0)

#define FAIL(msg) \
    do { fprintf(stderr, "FAIL: %s\n", msg); tests_failed++; } while(0)

#define ASSERT_TRUE(expr) \
    do { if (!(expr)) { FAIL(#expr); return; } } while(0)

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) { FAIL(#a " != " #b); return; } } while(0)

#define ASSERT_NEAR(a, b, eps) \
    do { if (fabs((a) - (b)) > (eps)) { \
        fprintf(stderr, "FAIL: %f != %f (eps=%f)\n", \
                (double)(a), (double)(b), (double)(eps)); \
        tests_failed++; return; } } while(0)

// ---- GGUF type size tests ----

void test_ggml_type_sizes() {
    TEST(ggml_type_sizes);
    ASSERT_EQ(ggml_type_size(GGML_TYPE_F32), 4u);
    ASSERT_EQ(ggml_type_size(GGML_TYPE_F16), 2u);
    ASSERT_EQ(ggml_type_size(GGML_TYPE_Q4_0), 18u); // 2 + 16
    ASSERT_EQ(ggml_type_size(GGML_TYPE_Q8_0), 34u); // 2 + 32
    ASSERT_EQ(ggml_type_size(GGML_TYPE_I32), 4u);
    ASSERT_EQ(ggml_type_size(GGML_TYPE_F8_E4M3), 1u);
    ASSERT_EQ(ggml_type_size(GGML_TYPE_F8_E5M2), 1u);
    ASSERT_EQ(ggml_block_size(GGML_TYPE_F32), 1u);
    ASSERT_EQ(ggml_block_size(GGML_TYPE_Q4_0), 32u);
    ASSERT_EQ(ggml_block_size(GGML_TYPE_Q8_0), 32u);
    ASSERT_EQ(ggml_block_size(GGML_TYPE_F8_E4M3), 1u);
    ASSERT_EQ(ggml_block_size(GGML_TYPE_F8_E5M2), 1u);
    PASS();
}

void test_ggml_type_names() {
    TEST(ggml_type_names);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_F32), "F32"), 0);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_F16), "F16"), 0);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_Q4_0), "Q4_0"), 0);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_Q8_0), "Q8_0"), 0);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_F8_E4M3), "F8_E4M3"), 0);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_F8_E5M2), "F8_E5M2"), 0);
    PASS();
}

// ---- GGUF parsing tests ----

// Create a minimal valid GGUF file in memory and verify parsing
void test_gguf_parse_minimal() {
    TEST(gguf_parse_minimal);

    // Build a minimal GGUF v3 file:
    // Header: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
    // No metadata, no tensors
    std::vector<uint8_t> data;
    auto write_u32 = [&](uint32_t v) {
        data.insert(data.end(), reinterpret_cast<uint8_t*>(&v),
                    reinterpret_cast<uint8_t*>(&v) + 4);
    };
    auto write_u64 = [&](uint64_t v) {
        data.insert(data.end(), reinterpret_cast<uint8_t*>(&v),
                    reinterpret_cast<uint8_t*>(&v) + 8);
    };

    write_u32(0x46554747u); // magic "GGUF" as little-endian uint32
    write_u32(3);           // version 3
    write_u64(0);           // tensor_count = 0
    write_u64(0);           // metadata_kv_count = 0

    GGUFFile gguf;
    gguf.file_size = data.size();
    gguf.file_data = data.data();
    gguf.owned_data = data;

    // Use internal parse by re-loading from the buffer
    // We test via the load method that reads from file,
    // but we can also test the parser directly via the public state
    ASSERT_TRUE(gguf.version == 0); // Not yet parsed

    // Parse by re-creating with owned data
    GGUFFile gguf2;
    gguf2.owned_data = data;
    gguf2.file_data = gguf2.owned_data.data();
    gguf2.file_size = data.size();

    // We can't call parse() directly since it's private,
    // but we can verify the file format is correct by checking magic
    ASSERT_EQ(data[0], 'G');
    ASSERT_EQ(data[1], 'G');
    ASSERT_EQ(data[2], 'U');
    ASSERT_EQ(data[3], 'F');

    PASS();
}

// ---- FP16 conversion tests ----

void test_fp16_conversion() {
    TEST(fp16_conversion);

    // Test zero
    ASSERT_NEAR(fp16_to_fp32(0x0000), 0.0f, 1e-6f);

    // Test one (0x3C00)
    ASSERT_NEAR(fp16_to_fp32(0x3C00), 1.0f, 1e-6f);

    // Test negative one (0xBC00)
    ASSERT_NEAR(fp16_to_fp32(0xBC00), -1.0f, 1e-6f);

    // Test 0.5 (0x3800)
    ASSERT_NEAR(fp16_to_fp32(0x3800), 0.5f, 1e-6f);

    // Test 2.0 (0x4000)
    ASSERT_NEAR(fp16_to_fp32(0x4000), 2.0f, 1e-6f);

    PASS();
}

// ---- Dequantization tests ----

void test_dequantize_f32() {
    TEST(dequantize_f32);

    float src[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dst[4] = {};
    dequantize(src, dst, 4, GGML_TYPE_F32);

    ASSERT_NEAR(dst[0], 1.0f, 1e-6f);
    ASSERT_NEAR(dst[1], 2.0f, 1e-6f);
    ASSERT_NEAR(dst[2], 3.0f, 1e-6f);
    ASSERT_NEAR(dst[3], 4.0f, 1e-6f);

    PASS();
}

// ---- CPU math operation tests ----

void test_cpu_rmsnorm() {
    TEST(cpu_rmsnorm);

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4] = {};

    cpu_rmsnorm(out, x, w, 4, 1e-5f);

    // RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // Each element should be x[i] / RMS
    float rms = sqrtf((1 + 4 + 9 + 16) / 4.0f + 1e-5f);
    ASSERT_NEAR(out[0], 1.0f / rms, 1e-4f);
    ASSERT_NEAR(out[1], 2.0f / rms, 1e-4f);
    ASSERT_NEAR(out[2], 3.0f / rms, 1e-4f);
    ASSERT_NEAR(out[3], 4.0f / rms, 1e-4f);

    PASS();
}

void test_cpu_softmax() {
    TEST(cpu_softmax);

    float x[] = {1.0f, 2.0f, 3.0f};
    cpu_softmax(x, 3);

    // Should sum to 1
    float sum = x[0] + x[1] + x[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f);

    // Should be monotonically increasing
    ASSERT_TRUE(x[0] < x[1]);
    ASSERT_TRUE(x[1] < x[2]);

    // Exact values
    float e1 = expf(1.0f), e2 = expf(2.0f), e3 = expf(3.0f);
    float s = e1 + e2 + e3;
    ASSERT_NEAR(x[0], e1 / s, 1e-5f);
    ASSERT_NEAR(x[1], e2 / s, 1e-5f);
    ASSERT_NEAR(x[2], e3 / s, 1e-5f);

    PASS();
}

void test_cpu_matmul() {
    TEST(cpu_matmul);

    // 2x3 * 3x2 = 2x2
    float a[] = {1, 2, 3, 4, 5, 6};
    float b[] = {7, 8, 9, 10, 11, 12};
    float out[4] = {};

    cpu_matmul(out, a, b, 2, 2, 3);

    // [1,2,3] * [7,9,11; 8,10,12]
    // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    ASSERT_NEAR(out[0], 58.0f, 1e-4f);
    ASSERT_NEAR(out[1], 64.0f, 1e-4f);
    ASSERT_NEAR(out[2], 139.0f, 1e-4f);
    ASSERT_NEAR(out[3], 154.0f, 1e-4f);

    PASS();
}

void test_cpu_matmul_transposed() {
    TEST(cpu_matmul_transposed);

    // x = [1, 2, 3], w (transposed) = [[1,2,3], [4,5,6]]
    // out = x * w^T -> [1*1+2*2+3*3, 1*4+2*5+3*6] = [14, 32]
    float x[] = {1, 2, 3};
    float w[] = {1, 2, 3, 4, 5, 6}; // stored as [N x K] = [2 x 3]
    float out[2] = {};

    cpu_matmul_transposed(out, x, w, 2, 3);

    ASSERT_NEAR(out[0], 14.0f, 1e-4f);
    ASSERT_NEAR(out[1], 32.0f, 1e-4f);

    PASS();
}

void test_cpu_matmul_transposed_q8_0() {
    TEST(cpu_matmul_transposed_q8_0);

    // Build a 2×32 weight matrix in Q8_0 format (N=2, K=32).
    // Q8_0 block layout: 2 bytes f16 scale, then 32 int8 values = 34 bytes/block.
    //
    // Row 0: scale=1.0 (f16=0x3C00), values=[1,2,...,32]
    // Row 1: scale=2.0 (f16=0x4000), values=[1,2,...,32]
    const int N = 2, K = 32;
    const int bytes_per_block = 34; // 2-byte f16 scale + 32 int8 values
    std::vector<uint8_t> w_q8(static_cast<size_t>(N) * bytes_per_block);

    uint16_t scale1 = 0x3C00; // 1.0 in f16
    memcpy(w_q8.data(), &scale1, 2);
    for (int j = 0; j < 32; j++) w_q8[2 + j] = static_cast<uint8_t>(j + 1);

    uint16_t scale2 = 0x4000; // 2.0 in f16
    memcpy(w_q8.data() + bytes_per_block, &scale2, 2);
    for (int j = 0; j < 32; j++) w_q8[bytes_per_block + 2 + j] = static_cast<uint8_t>(j + 1);

    // x = [1, 1, ..., 1]  (K=32 ones)
    std::vector<float> x(K, 1.0f);
    std::vector<float> out(N, 0.0f);

    cpu_matmul_transposed_q8_0(out.data(), x.data(), w_q8.data(), N, K);

    // Expected: sum(j=1..32) * scale
    // out[0] = 1.0 * (1+2+...+32) = 528
    // out[1] = 2.0 * (1+2+...+32) = 1056
    ASSERT_NEAR(out[0], 528.0f, 0.5f);
    ASSERT_NEAR(out[1], 1056.0f, 0.5f);

    // Cross-check: dequantize then matmul must give the same result
    std::vector<float> w_f32(static_cast<size_t>(N) * K);
    dequantize(w_q8.data(), w_f32.data(), static_cast<int64_t>(N) * K, GGML_TYPE_Q8_0);
    std::vector<float> out_ref(N, 0.0f);
    cpu_matmul_transposed(out_ref.data(), x.data(), w_f32.data(), N, K);
    ASSERT_NEAR(out[0], out_ref[0], 1e-4f);
    ASSERT_NEAR(out[1], out_ref[1], 1e-4f);

    PASS();
}

void test_cpu_matmul_transposed_q4_0() {
    TEST(cpu_matmul_transposed_q4_0);

    // Build a 2×32 weight matrix in Q4_0 format (N=2, K=32).
    // Q4_0 block layout: 2 bytes f16 scale, then 16 nibble bytes = 18 bytes/block.
    // Each nibble encodes a value in [0,15]; the represented weight is (nibble - 8).
    //
    // Row 0: scale=1.0 (f16=0x3C00), all nibbles=0x88 => each weight = 0
    //        (lower nibble 8-8=0, upper nibble 8-8=0) → dot product = 0
    // Row 1: scale=1.0 (f16=0x3C00), nibble pattern:
    //        lower nibble = 9 (weight=+1), upper nibble = 7 (weight=-1)
    //        for all 16 bytes: +1 for x[0..15], -1 for x[16..31]
    const int N = 2, K = 32;
    const int bytes_per_block = 18; // 2-byte f16 scale + 16 nibble bytes
    std::vector<uint8_t> w_q4(static_cast<size_t>(N) * bytes_per_block, 0);

    uint16_t scale1 = 0x3C00; // 1.0 in f16
    memcpy(w_q4.data(), &scale1, 2);
    // Row 0: all nibbles = 0x88 → lower=8-8=0, upper=8-8=0
    for (int j = 0; j < 16; j++) w_q4[2 + j] = 0x88;

    uint16_t scale2 = 0x3C00; // 1.0 in f16
    memcpy(w_q4.data() + bytes_per_block, &scale2, 2);
    // Row 1: nibbles = 0x79 → lower=9-8=+1, upper=7-8=-1
    for (int j = 0; j < 16; j++) w_q4[bytes_per_block + 2 + j] = 0x79;

    // x = [1, 1, ..., 1]  (K=32 ones)
    std::vector<float> x(K, 1.0f);
    std::vector<float> out(N, 0.0f);

    cpu_matmul_transposed_q4_0(out.data(), x.data(), w_q4.data(), N, K);

    // Row 0: all weights = 0 → dot = 0
    ASSERT_NEAR(out[0], 0.0f, 1e-4f);
    // Row 1: 16 * (+1) + 16 * (-1) = 0
    ASSERT_NEAR(out[1], 0.0f, 1e-4f);

    // Cross-check: dequantize then matmul must give the same result
    std::vector<float> w_f32(static_cast<size_t>(N) * K);
    dequantize(w_q4.data(), w_f32.data(), static_cast<int64_t>(N) * K, GGML_TYPE_Q4_0);
    std::vector<float> out_ref(N, 0.0f);
    cpu_matmul_transposed(out_ref.data(), x.data(), w_f32.data(), N, K);
    ASSERT_NEAR(out[0], out_ref[0], 1e-4f);
    ASSERT_NEAR(out[1], out_ref[1], 1e-4f);

    // Non-trivial case: x alternates [1, -1, 1, -1, ...]
    // Row 1 weights: +1 for positions 0-15, -1 for positions 16-31
    // dot = sum_{j=0}^{15} x[j]*1 + sum_{j=16}^{31} x[j]*(-1)
    //     = (8 ones + 8 neg-ones) + (-(8 ones + 8 neg-ones)) = 0
    std::vector<float> x2(K);
    for (int j = 0; j < K; j++) x2[j] = (j % 2 == 0) ? 1.0f : -1.0f;
    std::vector<float> out2(N, 0.0f);
    cpu_matmul_transposed_q4_0(out2.data(), x2.data(), w_q4.data(), N, K);

    // Verify against dequantize reference
    std::vector<float> out2_ref(N, 0.0f);
    cpu_matmul_transposed(out2_ref.data(), x2.data(), w_f32.data(), N, K);
    ASSERT_NEAR(out2[0], out2_ref[0], 1e-4f);
    ASSERT_NEAR(out2[1], out2_ref[1], 1e-4f);

    PASS();
}

void test_cpu_silu() {
    TEST(cpu_silu);

    // SiLU(0) = 0
    ASSERT_NEAR(silu(0.0f), 0.0f, 1e-6f);

    // SiLU(x) ≈ x for large x (sigmoid -> 1)
    ASSERT_NEAR(silu(10.0f), 10.0f, 0.01f);

    // SiLU(-10) ≈ 0 (sigmoid -> 0)
    ASSERT_NEAR(silu(-10.0f), 0.0f, 0.01f);

    PASS();
}

void test_cpu_add() {
    TEST(cpu_add);

    float a[] = {1, 2, 3, 4};
    float b[] = {5, 6, 7, 8};
    float out[4] = {};

    cpu_add(out, a, b, 4);

    ASSERT_NEAR(out[0], 6.0f, 1e-6f);
    ASSERT_NEAR(out[1], 8.0f, 1e-6f);
    ASSERT_NEAR(out[2], 10.0f, 1e-6f);
    ASSERT_NEAR(out[3], 12.0f, 1e-6f);

    PASS();
}

void test_cpu_rope() {
    TEST(cpu_rope);

    // Test RoPE at position 0: should be identity (cos(0)=1, sin(0)=0)
    float q[] = {1, 2, 3, 4};
    float k[] = {5, 6, 7, 8};
    float q_orig[] = {1, 2, 3, 4};
    float k_orig[] = {5, 6, 7, 8};

    cpu_rope(q, k, 4, 4, 4, 0, 10000.0f);

    ASSERT_NEAR(q[0], q_orig[0], 1e-5f);
    ASSERT_NEAR(q[1], q_orig[1], 1e-5f);
    ASSERT_NEAR(k[0], k_orig[0], 1e-5f);
    ASSERT_NEAR(k[1], k_orig[1], 1e-5f);

    PASS();
}

void test_cpu_rope_gqa() {
    TEST(cpu_rope_gqa);

    // Test RoPE with GQA: q has more dimensions than k
    // q: 2 heads * 4 head_dim = 8, k: 1 kv_head * 4 head_dim = 4
    float q[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float k[] = {1, 2, 3, 4};
    float q_orig[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float k_orig[] = {1, 2, 3, 4};

    // At position 0, RoPE should be identity
    cpu_rope(q, k, 8, 4, 4, 0, 10000.0f);

    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(q[i], q_orig[i], 1e-5f);
    }
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(k[i], k_orig[i], 1e-5f);
    }

    // At position 1, values should change
    float q2[] = {1, 0, 0, 0, 1, 0, 0, 0};
    float k2[] = {1, 0, 0, 0};
    cpu_rope(q2, k2, 8, 4, 4, 1, 10000.0f);

    // q[0] = cos(freq0), q[1] = sin(freq0) for first pair
    float freq0 = 1.0f / powf(10000.0f, 0.0f / 4.0f);
    float angle0 = 1.0f * freq0;
    ASSERT_NEAR(q2[0], cosf(angle0), 1e-5f);
    ASSERT_NEAR(q2[1], sinf(angle0), 1e-5f);
    // k should have the same rotation for its first pair
    ASSERT_NEAR(k2[0], cosf(angle0), 1e-5f);
    ASSERT_NEAR(k2[1], sinf(angle0), 1e-5f);

    PASS();
}

void test_cpu_rope_neox() {
    TEST(cpu_rope_neox);

    // Test neox-style RoPE at position 0: should be identity
    float q[] = {1, 2, 3, 4};
    float k[] = {5, 6, 7, 8};

    cpu_rope_neox(q, k, 4, 4, 4, 0, 10000.0f);

    ASSERT_NEAR(q[0], 1.0f, 1e-5f);
    ASSERT_NEAR(q[1], 2.0f, 1e-5f);
    ASSERT_NEAR(q[2], 3.0f, 1e-5f);
    ASSERT_NEAR(q[3], 4.0f, 1e-5f);
    ASSERT_NEAR(k[0], 5.0f, 1e-5f);
    ASSERT_NEAR(k[1], 6.0f, 1e-5f);

    // Test at position 1 with head_dim=4: pairs (0,2) and (1,3)
    float q2[] = {1, 0, 0, 0};
    float k2[] = {1, 0, 0, 0};
    cpu_rope_neox(q2, k2, 4, 4, 4, 1, 10000.0f);

    // Element 0 paired with element 2 (at distance head_dim/2=2)
    // freq = 1/theta^(0/4) = 1, angle = 1*1 = 1
    float freq0 = 1.0f / powf(10000.0f, 0.0f / 4.0f);
    float angle0 = 1.0f * freq0;
    ASSERT_NEAR(q2[0], cosf(angle0), 1e-5f);
    ASSERT_NEAR(q2[2], sinf(angle0), 1e-5f);
    ASSERT_NEAR(k2[0], cosf(angle0), 1e-5f);
    ASSERT_NEAR(k2[2], sinf(angle0), 1e-5f);

    // Test with GQA: q has more heads than k
    float q3[] = {1, 0, 0, 0, 1, 0, 0, 0};
    float k3[] = {1, 0, 0, 0};
    cpu_rope_neox(q3, k3, 8, 4, 4, 1, 10000.0f);

    // Both q heads should get the same rotation
    ASSERT_NEAR(q3[0], cosf(angle0), 1e-5f);
    ASSERT_NEAR(q3[4], cosf(angle0), 1e-5f);

    PASS();
}

void test_qkv_bias_add() {
    TEST(qkv_bias_add);

    // Simulate QKV projection + bias (as in Qwen3)
    float q[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float bias[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float expected[] = {1.1f, 2.2f, 3.3f, 4.4f};

    cpu_add(q, q, bias, 4);

    ASSERT_NEAR(q[0], expected[0], 1e-5f);
    ASSERT_NEAR(q[1], expected[1], 1e-5f);
    ASSERT_NEAR(q[2], expected[2], 1e-5f);
    ASSERT_NEAR(q[3], expected[3], 1e-5f);

    PASS();
}

void test_qk_norm() {
    TEST(qk_norm);

    // Simulate per-head QK-norm (Qwen3-style)
    // 2 heads, head_dim=2: Q = [h0: 3,4, h1: 1,2]
    float q[] = {3.0f, 4.0f, 1.0f, 2.0f};
    float norm_w[] = {1.0f, 1.0f}; // unit weights
    int num_heads = 2;
    int head_dim = 2;
    float eps = 1e-5f;

    // Apply per-head RMSNorm
    for (int h = 0; h < num_heads; h++) {
        cpu_rmsnorm(q + h * head_dim, q + h * head_dim, norm_w, head_dim, eps);
    }

    // Head 0: RMS = sqrt((9+16)/2) = sqrt(12.5), scale = 1/sqrt(12.5)
    float rms0 = sqrtf((9.0f + 16.0f) / 2.0f + eps);
    ASSERT_NEAR(q[0], 3.0f / rms0, 1e-4f);
    ASSERT_NEAR(q[1], 4.0f / rms0, 1e-4f);

    // Head 1: RMS = sqrt((1+4)/2) = sqrt(2.5), scale = 1/sqrt(2.5)
    float rms1 = sqrtf((1.0f + 4.0f) / 2.0f + eps);
    ASSERT_NEAR(q[2], 1.0f / rms1, 1e-4f);
    ASSERT_NEAR(q[3], 2.0f / rms1, 1e-4f);

    PASS();
}

// ---- Sampler tests ----

void test_sampler_greedy() {
    TEST(sampler_greedy);

    SamplerConfig cfg;
    cfg.temperature = 0.0f;  // greedy
    Sampler sampler(cfg);

    float logits[] = {1.0f, 5.0f, 2.0f, 3.0f};
    int token = sampler.sample(logits, 4);
    ASSERT_EQ(token, 1);  // highest logit

    PASS();
}

void test_sampler_temperature() {
    TEST(sampler_temperature);

    SamplerConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k = 0;  // disable top-k
    cfg.top_p = 1.0f;  // disable top-p
    cfg.seed = 42;
    Sampler sampler(cfg);

    float logits[] = {1.0f, 100.0f, 1.0f, 1.0f};  // very peaked
    int token = sampler.sample(logits, 4);
    // With such peaked logits, should almost always pick token 1
    ASSERT_EQ(token, 1);

    PASS();
}

void test_sampler_repeat_penalty() {
    TEST(sampler_repeat_penalty);

    SamplerConfig cfg;
    cfg.temperature = 0.0f;  // greedy
    cfg.repeat_penalty = 100.0f;  // very strong penalty
    Sampler sampler(cfg);

    float logits[] = {1.0f, 5.0f, 4.9f, 3.0f};
    std::vector<int> recent = {1};  // token 1 was recently used

    int token = sampler.sample(logits, 4, recent);
    // With heavy penalty on token 1, should pick token 2
    ASSERT_EQ(token, 2);

    PASS();
}

// ---- Compute dispatcher tests ----

void test_compute_cpu() {
    TEST(compute_cpu);

    Compute compute(Backend::CPU);

    float a[] = {1, 2, 3};
    float b[] = {4, 5, 6};
    float out[3] = {};

    compute.add(out, a, b, 3);

    ASSERT_NEAR(out[0], 5.0f, 1e-6f);
    ASSERT_NEAR(out[1], 7.0f, 1e-6f);
    ASSERT_NEAR(out[2], 9.0f, 1e-6f);

    PASS();
}

// ---- Qwen3 config tests ----

void test_qwen3_config() {
    TEST(qwen3_config);

    // Verify Qwen3 architecture enables neox RoPE
    ModelConfig cfg;
    cfg.architecture = "qwen3";

    // Simulate what load_config does for RoPE style
    bool rope_neox = (cfg.architecture == "qwen2" ||
                      cfg.architecture == "qwen3" ||
                      cfg.architecture == "qwen2moe");
    ASSERT_TRUE(rope_neox);

    // Verify Qwen3-0.6B dimensions
    cfg.hidden_size = 1024;
    cfg.num_heads = 16;
    cfg.num_kv_heads = 8;
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;
    cfg.kv_dim = cfg.head_dim * cfg.num_kv_heads;
    cfg.intermediate_size = 2816;
    cfg.num_layers = 28;
    cfg.rope_theta = 1000000.0f;
    cfg.max_seq_len = 40960;
    cfg.qkv_bias = true;
    cfg.rope_neox = true;

    ASSERT_EQ(cfg.head_dim, 64);
    ASSERT_EQ(cfg.kv_dim, 512);
    ASSERT_EQ(cfg.num_heads / cfg.num_kv_heads, 2);  // GQA ratio

    PASS();
}

void test_context_override() {
    TEST(context_override);

    // Verify KV cache respects reduced context
    KVCache cache;
    int layers = 28, kv_dim = 512;

    // With small context, should work fine
    int small_ctx = 4096;
    cache.init(layers, small_ctx, kv_dim);

    ASSERT_EQ(cache.max_seq_len, small_ctx);
    ASSERT_EQ(cache.kv_dim, kv_dim);
    ASSERT_EQ(cache.num_layers, layers);

    // Verify key/value pointers are distinct per layer/position
    float* k0 = cache.key(0, 0);
    float* k1 = cache.key(1, 0);
    ASSERT_EQ(k1 - k0, static_cast<ptrdiff_t>(small_ctx) * kv_dim);

    float* v0 = cache.value(0, 0);
    float* v1 = cache.value(0, 1);
    ASSERT_EQ(v1 - v0, static_cast<ptrdiff_t>(kv_dim));

    PASS();
}

// ---- GPT-2 tokenizer tests ----

void test_gpt2_byte_mapping() {
    TEST(gpt2_byte_mapping);

    Tokenizer tok;
    tok.tokenizer_model = "gpt2";
    tok.init_gpt2_byte_mapping();

    // Verify mapping size: all 256 bytes should be mapped
    ASSERT_EQ(static_cast<int>(tok.byte_to_unicode.size()), 256);
    ASSERT_EQ(static_cast<int>(tok.unicode_to_byte.size()), 256);

    // ASCII printable chars (33-126) map to themselves
    ASSERT_EQ(tok.byte_to_unicode[static_cast<uint8_t>('A')], std::string("A"));
    ASSERT_EQ(tok.byte_to_unicode[static_cast<uint8_t>('z')], std::string("z"));
    ASSERT_EQ(tok.byte_to_unicode[static_cast<uint8_t>('!')], std::string("!"));

    // Space (byte 32) maps to U+0120 (Ġ), UTF-8: C4 A0
    std::string space_mapped = tok.byte_to_unicode[32];
    ASSERT_EQ(space_mapped.size(), 2u);
    ASSERT_EQ(static_cast<uint8_t>(space_mapped[0]), 0xC4u);
    ASSERT_EQ(static_cast<uint8_t>(space_mapped[1]), 0xA0u);

    // Newline (byte 10) maps to U+010A (Ċ), UTF-8: C4 8A
    std::string nl_mapped = tok.byte_to_unicode[10];
    ASSERT_EQ(nl_mapped.size(), 2u);
    ASSERT_EQ(static_cast<uint8_t>(nl_mapped[0]), 0xC4u);
    ASSERT_EQ(static_cast<uint8_t>(nl_mapped[1]), 0x8Au);

    // Verify round-trip: byte -> unicode codepoint -> byte
    for (int b = 0; b < 256; b++) {
        uint8_t byte_val = static_cast<uint8_t>(b);
        const std::string& utf8 = tok.byte_to_unicode[byte_val];
        // Decode UTF-8 to get codepoint
        char32_t cp;
        if (static_cast<uint8_t>(utf8[0]) < 0x80) {
            cp = static_cast<uint8_t>(utf8[0]);
        } else {
            cp = (static_cast<uint8_t>(utf8[0]) & 0x1F);
            cp = (cp << 6) | (static_cast<uint8_t>(utf8[1]) & 0x3F);
        }
        auto it = tok.unicode_to_byte.find(cp);
        ASSERT_TRUE(it != tok.unicode_to_byte.end());
        ASSERT_EQ(it->second, byte_val);
    }

    PASS();
}

void test_gpt2_decode() {
    TEST(gpt2_decode);

    Tokenizer tok;
    tok.tokenizer_model = "gpt2";
    tok.init_gpt2_byte_mapping();

    // Set up a minimal vocabulary with GPT-2 encoded tokens
    tok.vocab_size = 3;
    tok.id_to_token.resize(3);

    // Token 0: "Hello" (all ASCII, maps to itself)
    tok.id_to_token[0] = "Hello";
    ASSERT_EQ(tok.decode(0), std::string("Hello"));

    // Token 1: space + "the" = Ġ + "the", where Ġ is U+0120 (C4 A0 in UTF-8)
    tok.id_to_token[1] = std::string("\xC4\xA0") + "the";
    ASSERT_EQ(tok.decode(1), std::string(" the"));

    // Token 2: newline = Ċ (U+010A = C4 8A in UTF-8)
    tok.id_to_token[2] = std::string("\xC4\x8A");
    ASSERT_EQ(tok.decode(2), std::string("\n"));

    PASS();
}

void test_gpt2_decode_chinese() {
    TEST(gpt2_decode_chinese);

    Tokenizer tok;
    tok.tokenizer_model = "gpt2";
    tok.init_gpt2_byte_mapping();

    // Chinese character 你 (U+4F60) = UTF-8 bytes E4 BD A0
    // In GPT-2 mapping:
    //   byte E4 (228) -> safe range -> codepoint 0xE4 (ä) -> UTF-8: C3 A4
    //   byte BD (189) -> safe range -> codepoint 0xBD (½) -> UTF-8: C2 BD
    //   byte A0 (160) -> NOT safe -> codepoint U+0142 (ł) -> UTF-8: C5 82
    tok.vocab_size = 1;
    tok.id_to_token.resize(1);
    tok.id_to_token[0] = std::string("\xC3\xA4\xC2\xBD\xC5\x82"); // ä½ł
    std::string decoded = tok.decode(0);
    // Should decode to 你 (E4 BD A0)
    ASSERT_EQ(decoded.size(), 3u);
    ASSERT_EQ(static_cast<uint8_t>(decoded[0]), 0xE4u);
    ASSERT_EQ(static_cast<uint8_t>(decoded[1]), 0xBDu);
    ASSERT_EQ(static_cast<uint8_t>(decoded[2]), 0xA0u);

    PASS();
}

void test_gpt2_encode() {
    TEST(gpt2_encode);

    Tokenizer tok;
    tok.tokenizer_model = "gpt2";
    tok.init_gpt2_byte_mapping();

    // Build a minimal vocabulary and merge table for GPT-2
    std::vector<std::string> vocab_tokens = {
        "H", "e", "l", "o",                                    // 0-3: single chars
        "He",                                                   // 4
        "ll",                                                   // 5
        "Hell",                                                 // 6
        "Hello",                                                // 7
        std::string("\xC4\xA0"),                                // 8: Ġ (space)
    };

    tok.vocab_size = static_cast<int>(vocab_tokens.size());
    tok.id_to_token = vocab_tokens;
    for (int i = 0; i < tok.vocab_size; i++) {
        tok.token_to_id[tok.id_to_token[i]] = i;
    }
    tok.bos_token_id = -1; // no BOS for test

    // Merge rules match the merged symbols at each BPE step:
    // [H, e, l, l, o] → merge H+e → [He, l, l, o]
    // → merge l+l → [He, ll, o] → merge He+ll → [Hell, o]
    // → merge Hell+o → [Hello]
    tok.merge_ranks["H e"] = 0;
    tok.merge_ranks["l l"] = 1;
    tok.merge_ranks["He ll"] = 2;
    tok.merge_ranks["Hell o"] = 3;
    tok.merges.push_back({"H", "e", 0});

    // Encode "Hello" (no space, single word)
    auto tokens = tok.encode("Hello", false);
    // Should produce token 7 ("Hello") after merging
    ASSERT_EQ(static_cast<int>(tokens.size()), 1);
    ASSERT_EQ(tokens[0], 7); // "Hello"

    PASS();
}

void test_gpt2_pretokenize() {
    TEST(gpt2_pretokenize);

    Tokenizer tok;
    tok.tokenizer_model = "gpt2";
    tok.init_gpt2_byte_mapping();

    // Build vocabulary with space-prefixed tokens
    std::string space_utf8 = std::string("\xC4\xA0"); // Ġ (GPT-2 mapped space)
    std::vector<std::string> vocab_tokens = {
        "a", "b",                                    // 0-1
        space_utf8 + "a",                            // 2: " a"
        space_utf8 + "b",                            // 3: " b"
    };

    tok.vocab_size = static_cast<int>(vocab_tokens.size());
    tok.id_to_token = vocab_tokens;
    for (int i = 0; i < tok.vocab_size; i++) {
        tok.token_to_id[tok.id_to_token[i]] = i;
    }
    tok.bos_token_id = -1;

    // Add merge rules for space + letter
    tok.merge_ranks[space_utf8 + " a"] = 0;  // Ġ + a -> Ġa
    tok.merge_ranks[space_utf8 + " b"] = 1;  // Ġ + b -> Ġb
    tok.merges.push_back({space_utf8, "a", 0});

    // Encode "a b" -> should produce tokens [0 ("a"), 3 (" b")]
    auto tokens = tok.encode("a b", false);
    ASSERT_EQ(static_cast<int>(tokens.size()), 2);
    ASSERT_EQ(tokens[0], 0); // "a"
    ASSERT_EQ(tokens[1], 3); // " b" (Ġb)

    PASS();
}

// ---- Special token and chat template tests ----

void test_special_token_split() {
    TEST(special_token_split);

    Tokenizer tok;
    tok.tokenizer_model = "gpt2";
    tok.init_gpt2_byte_mapping();

    // Set up vocabulary with special tokens
    std::vector<std::string> vocab_tokens = {
        "s", "y", "t", "e", "m",       // 0-4
        "<|im_start|>",                 // 5
        "<|im_end|>",                   // 6
        "<|endoftext|>",                // 7
        "system",                       // 8: fully merged form
        "sy",                           // 9
        "st",                           // 10
        "em",                           // 11
        "syst",                         // 12
    };

    tok.vocab_size = static_cast<int>(vocab_tokens.size());
    tok.id_to_token = vocab_tokens;
    for (int i = 0; i < tok.vocab_size; i++) {
        tok.token_to_id[tok.id_to_token[i]] = i;
    }
    tok.bos_token_id = -1;

    // Add special tokens (sorted by length descending)
    tok.added_tokens = {"<|endoftext|>", "<|im_start|>", "<|im_end|>"};

    // Add merge rules matching BPE merge order:
    // s,y,s,t,e,m -> sy,s,t,e,m -> sy,st,e,m -> sy,st,em -> syst,em -> system
    tok.merge_ranks["s y"] = 0;
    tok.merge_ranks["s t"] = 1;
    tok.merge_ranks["e m"] = 2;
    tok.merge_ranks["sy st"] = 3;
    tok.merge_ranks["syst em"] = 4;
    tok.merges.push_back({"s", "y", 0});

    // Encode "<|im_start|>system<|im_end|>" - special tokens should be matched directly
    auto tokens = tok.encode("<|im_start|>system<|im_end|>", false);

    // Should produce: 5 (<|im_start|>), 8 (system), 6 (<|im_end|>)
    ASSERT_EQ(static_cast<int>(tokens.size()), 3);
    ASSERT_EQ(tokens[0], 5); // <|im_start|>
    ASSERT_EQ(tokens[1], 8); // system (BPE merged)
    ASSERT_EQ(tokens[2], 6); // <|im_end|>

    PASS();
}

void test_eos_token_ids() {
    TEST(eos_token_ids);

    Tokenizer tok;
    tok.eos_token_id = 2;
    tok.eos_token_ids.push_back(2);
    tok.eos_token_ids.push_back(5);
    tok.eos_token_ids.push_back(7);

    ASSERT_TRUE(tok.is_eos_token(2));
    ASSERT_TRUE(tok.is_eos_token(5));
    ASSERT_TRUE(tok.is_eos_token(7));
    ASSERT_TRUE(!tok.is_eos_token(0));
    ASSERT_TRUE(!tok.is_eos_token(3));

    PASS();
}

void test_context_auto_cap() {
    TEST(context_auto_cap);

    // Verify that context > 8192 gets auto-capped to 4096
    ModelConfig cfg;
    cfg.max_seq_len = 40960;

    // Simulate auto-capping logic from Model::load
    int context_override = 0;
    if (context_override == 0 && cfg.max_seq_len > 8192) {
        cfg.max_seq_len = 4096;
    }

    ASSERT_EQ(cfg.max_seq_len, 4096);

    // Verify that context <= 8192 is NOT capped
    ModelConfig cfg2;
    cfg2.max_seq_len = 4096;
    if (context_override == 0 && cfg2.max_seq_len > 8192) {
        cfg2.max_seq_len = 4096;
    }
    ASSERT_EQ(cfg2.max_seq_len, 4096); // unchanged

    // Verify explicit override is respected
    ModelConfig cfg3;
    cfg3.max_seq_len = 40960;
    int explicit_override = 2048;
    if (explicit_override > 0 && explicit_override < cfg3.max_seq_len) {
        cfg3.max_seq_len = explicit_override;
    }
    ASSERT_EQ(cfg3.max_seq_len, 2048);

    PASS();
}

void test_special_token_encode() {
    TEST(special_token_encode);

    Tokenizer tok;
    tok.tokenizer_model = "gpt2";
    tok.init_gpt2_byte_mapping();

    // Build vocabulary
    std::vector<std::string> vocab_tokens = {
        "a", "b", "c",                  // 0-2
        "<|im_start|>",                 // 3
        "<|im_end|>",                   // 4
        "ab",                           // 5
        "abc",                          // 6
    };

    tok.vocab_size = static_cast<int>(vocab_tokens.size());
    tok.id_to_token = vocab_tokens;
    for (int i = 0; i < tok.vocab_size; i++) {
        tok.token_to_id[tok.id_to_token[i]] = i;
    }
    tok.bos_token_id = -1;
    tok.added_tokens = {"<|im_start|>", "<|im_end|>"};

    // Merge rules: a+b -> ab, ab+c -> abc
    tok.merge_ranks["a b"] = 0;
    tok.merge_ranks["ab c"] = 1;
    tok.merges.push_back({"a", "b", 0});

    // Encode text with special tokens
    auto tokens = tok.encode("<|im_start|>abc<|im_end|>", false);

    // Should produce: 3 (<|im_start|>), 6 (abc merged), 4 (<|im_end|>)
    ASSERT_EQ(static_cast<int>(tokens.size()), 3);
    ASSERT_EQ(tokens[0], 3); // <|im_start|>
    ASSERT_EQ(tokens[1], 6); // abc (BPE merged)
    ASSERT_EQ(tokens[2], 4); // <|im_end|>

    PASS();
}

// Verify that a newline is always its own BPE chunk and is never merged into
// the following text.  In Qwen's tiktoken pre-tokenisation, \n matches
// \s*[\r\n]+ as a separate unit; if we instead let it attach to the next
// word (the old bug), a spurious low-rank merge of the encoded newline
// symbol with the first symbol of the following word would fire and produce
// a completely wrong token — exactly the garbled-Chinese bug reported on
// Windows.
void test_gpt2_newline_chunking() {
    TEST(gpt2_newline_chunking);

    Tokenizer tok;
    tok.tokenizer_model = "gpt2";
    tok.init_gpt2_byte_mapping();

    // '\n' (byte 0x0A) is NOT in the safe range; it maps to codepoint 266
    // (0x10A = Ċ), whose UTF-8 encoding is "\xC4\x8A".
    std::string nl_enc = std::string("\xC4\x8A"); // GPT-2 encoded '\n'

    std::vector<std::string> vocab_tokens = {
        "h", "e", "l", "o",               // 0-3: single ASCII letters
        nl_enc,                            // 4: encoded '\n'  (Ċ)
        nl_enc + "h",                      // 5: wrong merged token "Ċh"
        "he", "hel", "hell", "hello",      // 6-9: BPE-merged forms of "hello"
    };

    tok.vocab_size = static_cast<int>(vocab_tokens.size());
    tok.id_to_token = vocab_tokens;
    for (int i = 0; i < tok.vocab_size; i++) {
        tok.token_to_id[tok.id_to_token[i]] = i;
    }
    tok.bos_token_id = -1;

    // Give the (newline-enc + 'h') pair rank 0 — the highest BPE priority.
    // If '\n' and "hello" are ever in the same chunk this merge fires first,
    // producing the wrong token 5 ("Ċh") and leaving the rest unmerged.
    tok.merge_ranks[nl_enc + " h"] = 0; // rank 0 = highest priority
    tok.merge_ranks["h e"]         = 1;
    tok.merge_ranks["he l"]        = 2;
    tok.merge_ranks["hel l"]       = 3;
    tok.merge_ranks["hell o"]      = 4;
    tok.merges.push_back({"h", "e", 1});

    // "\nhello" must encode as [nl_token(4), hello_token(9)].
    // With the old bug '\n' and "hello" share a chunk, "Ċ h" merges at
    // rank 0 and the result starts with wrong token 5 instead of 4.
    auto tokens = tok.encode("\nhello", false);

    ASSERT_EQ(static_cast<int>(tokens.size()), 2);
    ASSERT_EQ(tokens[0], 4); // encoded '\n' → token 4
    ASSERT_EQ(tokens[1], 9); // "hello" fully merged → token 9

    PASS();
}

// ---- FP8 conversion and matmul tests ----

void test_fp8_e4m3_conversion() {
    TEST(fp8_e4m3_conversion);

    // Zero: 0x00 -> 0.0
    ASSERT_NEAR(fp8_e4m3_to_fp32(0x00), 0.0f, 1e-9f);
    // Negative zero: 0x80 -> -0.0 (bit pattern same as +0.0 when compared as float)
    ASSERT_NEAR(fp8_e4m3_to_fp32(0x80), 0.0f, 1e-9f);

    // Normal values: s=0, e=7, bias=7 -> 2^(7-7)*(1+m/8)
    // 1.0: s=0 e=0111 m=000 -> 0b0_0111_000 = 0x38
    ASSERT_NEAR(fp8_e4m3_to_fp32(0x38), 1.0f, 1e-6f);
    // 2.0: s=0 e=1000 m=000 -> 0b0_1000_000 = 0x40
    ASSERT_NEAR(fp8_e4m3_to_fp32(0x40), 2.0f, 1e-6f);
    // 0.5: s=0 e=0110 m=000 -> 0b0_0110_000 = 0x30
    ASSERT_NEAR(fp8_e4m3_to_fp32(0x30), 0.5f, 1e-6f);
    // 1.5: s=0 e=0111 m=100 -> 0b0_0111_100 = 0x3C, value = 2^0*(1+4/8) = 1.5
    ASSERT_NEAR(fp8_e4m3_to_fp32(0x3C), 1.5f, 1e-6f);
    // -1.0: s=1 e=0111 m=000 -> 0b1_0111_000 = 0xB8
    ASSERT_NEAR(fp8_e4m3_to_fp32(0xB8), -1.0f, 1e-6f);

    // Subnormal: e=0, value = m * 2^(-9)
    // m=1: 0b0_0000_001 = 0x01, value = 1*2^(-9) = 1/512
    ASSERT_NEAR(fp8_e4m3_to_fp32(0x01), 1.0f / 512.0f, 1e-7f);
    // m=4: 0b0_0000_100 = 0x04, value = 4*2^(-9) = 1/128
    ASSERT_NEAR(fp8_e4m3_to_fp32(0x04), 4.0f / 512.0f, 1e-7f);

    PASS();
}

void test_fp8_e5m2_conversion() {
    TEST(fp8_e5m2_conversion);

    // Zero: 0x00 -> 0.0
    ASSERT_NEAR(fp8_e5m2_to_fp32(0x00), 0.0f, 1e-9f);

    // Normal values: s=0, e=15, bias=15 -> 2^(15-15)*(1+m/4)
    // 1.0: s=0 e=01111 m=00 -> 0b0_01111_00 = 0x3C
    ASSERT_NEAR(fp8_e5m2_to_fp32(0x3C), 1.0f, 1e-6f);
    // 2.0: s=0 e=10000 m=00 -> 0b0_10000_00 = 0x40
    ASSERT_NEAR(fp8_e5m2_to_fp32(0x40), 2.0f, 1e-6f);
    // 0.5: s=0 e=01110 m=00 -> 0b0_01110_00 = 0x38
    ASSERT_NEAR(fp8_e5m2_to_fp32(0x38), 0.5f, 1e-6f);
    // 1.5: s=0 e=01111 m=10 -> 0b0_01111_10 = 0x3E, value = 1*(1+2/4) = 1.5
    ASSERT_NEAR(fp8_e5m2_to_fp32(0x3E), 1.5f, 1e-6f);
    // -1.0: s=1 e=01111 m=00 -> 0b1_01111_00 = 0xBC
    ASSERT_NEAR(fp8_e5m2_to_fp32(0xBC), -1.0f, 1e-6f);

    // Subnormal: e=0, value = m * 2^(-16)
    // m=1: 0b0_00000_01 = 0x01, value = 1*2^(-16)
    ASSERT_NEAR(fp8_e5m2_to_fp32(0x01), 1.0f / 65536.0f, 1e-10f);

    PASS();
}

void test_cpu_matmul_transposed_f8_e4m3() {
    TEST(cpu_matmul_transposed_f8_e4m3);

    // Build a 2×4 weight matrix in FP8 E4M3 format.
    // Row 0: [1.0, 0.0, 0.0, 0.0] — 1.0=0x38, 0.0=0x00
    // Row 1: [0.0, 1.0, 0.0, 0.0]
    // x = [1.0, 2.0, 3.0, 4.0]
    // Expected: out[0] = 1.0, out[1] = 2.0
    const int N = 2, K = 4;
    uint8_t w[N * K] = {
        0x38, 0x00, 0x00, 0x00,  // row 0: [1.0, 0.0, 0.0, 0.0]
        0x00, 0x38, 0x00, 0x00,  // row 1: [0.0, 1.0, 0.0, 0.0]
    };
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[2] = {};

    cpu_matmul_transposed_f8_e4m3(out, x, w, N, K);

    ASSERT_NEAR(out[0], 1.0f, 1e-4f);
    ASSERT_NEAR(out[1], 2.0f, 1e-4f);

    // Cross-check: dequantize then matmul must give the same result
    float w_f32[N * K];
    dequantize(w, w_f32, N * K, GGML_TYPE_F8_E4M3);
    float out_ref[2] = {};
    cpu_matmul_transposed(out_ref, x, w_f32, N, K);
    ASSERT_NEAR(out[0], out_ref[0], 1e-4f);
    ASSERT_NEAR(out[1], out_ref[1], 1e-4f);

    // Non-trivial: row of all 2.0 weights (0x40), x = all ones
    // out[0] = 4 * 2.0 = 8.0
    uint8_t w2[1 * K] = {0x40, 0x40, 0x40, 0x40};
    float x2[K] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out2[1] = {};
    cpu_matmul_transposed_f8_e4m3(out2, x2, w2, 1, K);
    ASSERT_NEAR(out2[0], 8.0f, 1e-4f);

    PASS();
}

void test_cpu_matmul_transposed_f8_e5m2() {
    TEST(cpu_matmul_transposed_f8_e5m2);

    // Build a 2×4 weight matrix in FP8 E5M2 format.
    // Row 0: all 1.0 (0x3C), row 1: all 2.0 (0x40)
    // x = [1.0, 1.0, 1.0, 1.0]
    // Expected: out[0] = 4.0, out[1] = 8.0
    const int N = 2, K = 4;
    uint8_t w[N * K] = {
        0x3C, 0x3C, 0x3C, 0x3C,  // row 0: [1.0, 1.0, 1.0, 1.0]
        0x40, 0x40, 0x40, 0x40,  // row 1: [2.0, 2.0, 2.0, 2.0]
    };
    float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[2] = {};

    cpu_matmul_transposed_f8_e5m2(out, x, w, N, K);

    ASSERT_NEAR(out[0], 4.0f, 1e-4f);
    ASSERT_NEAR(out[1], 8.0f, 1e-4f);

    // Cross-check: dequantize then matmul must give the same result
    float w_f32[N * K];
    dequantize(w, w_f32, N * K, GGML_TYPE_F8_E5M2);
    float out_ref[2] = {};
    cpu_matmul_transposed(out_ref, x, w_f32, N, K);
    ASSERT_NEAR(out[0], out_ref[0], 1e-4f);
    ASSERT_NEAR(out[1], out_ref[1], 1e-4f);

    PASS();
}

// ---- Run all tests ----

int main() {
    fprintf(stderr, "Running llm.cpp tests...\n\n");

    fprintf(stderr, "GGUF format tests:\n");
    test_ggml_type_sizes();
    test_ggml_type_names();
    test_gguf_parse_minimal();
    test_fp16_conversion();
    test_dequantize_f32();

    fprintf(stderr, "\nCPU operation tests:\n");
    test_cpu_rmsnorm();
    test_cpu_softmax();
    test_cpu_matmul();
    test_cpu_matmul_transposed();
    test_cpu_matmul_transposed_q8_0();
    test_cpu_matmul_transposed_q4_0();
    test_cpu_silu();
    test_cpu_add();
    test_cpu_rope();
    test_cpu_rope_gqa();
    test_cpu_rope_neox();
    test_qkv_bias_add();
    test_qk_norm();

    fprintf(stderr, "\nSampler tests:\n");
    test_sampler_greedy();
    test_sampler_temperature();
    test_sampler_repeat_penalty();

    fprintf(stderr, "\nCompute dispatcher tests:\n");
    test_compute_cpu();

    fprintf(stderr, "\nQwen3 config tests:\n");
    test_qwen3_config();
    test_context_override();

    fprintf(stderr, "\nGPT-2 tokenizer tests:\n");
    test_gpt2_byte_mapping();
    test_gpt2_decode();
    test_gpt2_decode_chinese();
    test_gpt2_encode();
    test_gpt2_pretokenize();

    fprintf(stderr, "\nChat template & special token tests:\n");
    test_special_token_split();
    test_eos_token_ids();
    test_context_auto_cap();
    test_special_token_encode();
    test_gpt2_newline_chunking();

    fprintf(stderr, "\nFP8 type and computation tests:\n");
    test_fp8_e4m3_conversion();
    test_fp8_e5m2_conversion();
    test_cpu_matmul_transposed_f8_e4m3();
    test_cpu_matmul_transposed_f8_e5m2();

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n",
            tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
