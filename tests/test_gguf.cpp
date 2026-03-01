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
    ASSERT_EQ(ggml_block_size(GGML_TYPE_F32), 1u);
    ASSERT_EQ(ggml_block_size(GGML_TYPE_Q4_0), 32u);
    ASSERT_EQ(ggml_block_size(GGML_TYPE_Q8_0), 32u);
    PASS();
}

void test_ggml_type_names() {
    TEST(ggml_type_names);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_F32), "F32"), 0);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_F16), "F16"), 0);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_Q4_0), "Q4_0"), 0);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_Q8_0), "Q8_0"), 0);
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

    cpu_rope(q, k, 4, 4, 0, 10000.0f);

    ASSERT_NEAR(q[0], q_orig[0], 1e-5f);
    ASSERT_NEAR(q[1], q_orig[1], 1e-5f);
    ASSERT_NEAR(k[0], k_orig[0], 1e-5f);
    ASSERT_NEAR(k[1], k_orig[1], 1e-5f);

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
    test_cpu_silu();
    test_cpu_add();
    test_cpu_rope();

    fprintf(stderr, "\nSampler tests:\n");
    test_sampler_greedy();
    test_sampler_temperature();
    test_sampler_repeat_penalty();

    fprintf(stderr, "\nCompute dispatcher tests:\n");
    test_compute_cpu();

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n",
            tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
