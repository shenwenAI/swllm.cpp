// GPU (CUDA) tests for llm.cpp kernel operations.
// Compiled only when LLM_CUDA=ON.  Each test verifies that the CUDA kernel
// produces results consistent with the reference CPU implementation.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tensor.h"

static int tests_passed = 0;
static int tests_failed = 0;
static bool gpu_available = false;

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

// ---- GPU kernel tests ----

void test_gpu_add() {
    TEST(gpu_add);
    if (!gpu_available) { fprintf(stderr, "SKIP (no GPU)\n"); return; }

    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float out[4] = {};

    cuda_add(out, a, b, 4);

    ASSERT_NEAR(out[0], 6.0f,  1e-5f);
    ASSERT_NEAR(out[1], 8.0f,  1e-5f);
    ASSERT_NEAR(out[2], 10.0f, 1e-5f);
    ASSERT_NEAR(out[3], 12.0f, 1e-5f);

    PASS();
}

void test_gpu_rmsnorm() {
    TEST(gpu_rmsnorm);
    if (!gpu_available) { fprintf(stderr, "SKIP (no GPU)\n"); return; }

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float gpu_out[4] = {};
    float cpu_out[4] = {};

    cuda_rmsnorm(gpu_out, x, w, 4, 1e-5f);
    cpu_rmsnorm(cpu_out, x, w, 4, 1e-5f);

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(gpu_out[i], cpu_out[i], 1e-4f);
    }

    PASS();
}

void test_gpu_softmax() {
    TEST(gpu_softmax);
    if (!gpu_available) { fprintf(stderr, "SKIP (no GPU)\n"); return; }

    float gpu_x[] = {1.0f, 2.0f, 3.0f};
    float cpu_x[] = {1.0f, 2.0f, 3.0f};

    cuda_softmax(gpu_x, 3);
    cpu_softmax(cpu_x, 3);

    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(gpu_x[i], cpu_x[i], 1e-5f);
    }

    // Results should sum to 1
    float sum = gpu_x[0] + gpu_x[1] + gpu_x[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f);

    PASS();
}

void test_gpu_silu_mul() {
    TEST(gpu_silu_mul);
    if (!gpu_available) { fprintf(stderr, "SKIP (no GPU)\n"); return; }

    float gate[] = {0.0f, 1.0f, -1.0f, 2.0f};
    float up[]   = {1.0f, 1.0f,  1.0f, 1.0f};
    float gpu_out[4] = {};
    float cpu_out[4] = {};

    cuda_silu_elementwise_mul(gpu_out, gate, up, 4);
    cpu_silu_elementwise_mul(cpu_out, gate, up, 4);

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(gpu_out[i], cpu_out[i], 1e-5f);
    }

    PASS();
}

void test_gpu_matmul() {
    TEST(gpu_matmul);
    if (!gpu_available) { fprintf(stderr, "SKIP (no GPU)\n"); return; }

    // 2x3 * 3x2 = 2x2
    float a[] = {1, 2, 3, 4, 5, 6};
    float b[] = {7, 8, 9, 10, 11, 12};
    float gpu_out[4] = {};
    float cpu_out[4] = {};

    cuda_matmul(gpu_out, a, b, 2, 2, 3);
    cpu_matmul(cpu_out, a, b, 2, 2, 3);

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(gpu_out[i], cpu_out[i], 1e-2f);
    }

    PASS();
}

void test_gpu_matmul_transposed() {
    TEST(gpu_matmul_transposed);
    if (!gpu_available) { fprintf(stderr, "SKIP (no GPU)\n"); return; }

    // x = [1, 2, 3], w (N×K) = [[1,2,3],[4,5,6]]
    // out = x * w^T = [14, 32]
    float x[] = {1.0f, 2.0f, 3.0f};
    float w[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float gpu_out[2] = {};
    float cpu_out[2] = {};

    cuda_matmul_transposed_weight(gpu_out, x, w, 2, 3);
    cpu_matmul_transposed(cpu_out, x, w, 2, 3);

    for (int i = 0; i < 2; i++) {
        ASSERT_NEAR(gpu_out[i], cpu_out[i], 1e-3f);
    }

    PASS();
}

void test_gpu_rope_identity() {
    TEST(gpu_rope_identity);
    if (!gpu_available) { fprintf(stderr, "SKIP (no GPU)\n"); return; }

    // At position 0 RoPE should be the identity transform
    float q[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float k[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float q_orig[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float k_orig[] = {5.0f, 6.0f, 7.0f, 8.0f};

    cuda_rope(q, k, 4, 4, 4, 0, 10000.0f);

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(q[i], q_orig[i], 1e-5f);
        ASSERT_NEAR(k[i], k_orig[i], 1e-5f);
    }

    PASS();
}

void test_gpu_rope_vs_cpu() {
    TEST(gpu_rope_vs_cpu);
    if (!gpu_available) { fprintf(stderr, "SKIP (no GPU)\n"); return; }

    float gpu_q[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float gpu_k[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float cpu_q[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float cpu_k[] = {1.0f, 0.0f, 0.0f, 0.0f};

    cuda_rope(gpu_q, gpu_k, 4, 4, 4, 1, 10000.0f);
    cpu_rope(cpu_q, cpu_k, 4, 4, 4, 1, 10000.0f);

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(gpu_q[i], cpu_q[i], 1e-5f);
        ASSERT_NEAR(gpu_k[i], cpu_k[i], 1e-5f);
    }

    PASS();
}

// ---- Compute dispatcher tests ----

void test_compute_cuda_add() {
    TEST(compute_cuda_add);
    if (!gpu_available) { fprintf(stderr, "SKIP (no GPU)\n"); return; }

    Compute compute(Backend::CUDA);

    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float out[3] = {};

    compute.add(out, a, b, 3);

    ASSERT_NEAR(out[0], 5.0f, 1e-5f);
    ASSERT_NEAR(out[1], 7.0f, 1e-5f);
    ASSERT_NEAR(out[2], 9.0f, 1e-5f);

    PASS();
}

void test_compute_cuda_rmsnorm() {
    TEST(compute_cuda_rmsnorm);
    if (!gpu_available) { fprintf(stderr, "SKIP (no GPU)\n"); return; }

    Compute compute(Backend::CUDA);

    float x[] = {3.0f, 4.0f};
    float w[] = {1.0f, 1.0f};
    float out[2] = {};

    compute.rmsnorm(out, x, w, 2, 1e-5f);

    // RMS = sqrt((9+16)/2) = sqrt(12.5)
    float rms = sqrtf((9.0f + 16.0f) / 2.0f + 1e-5f);
    ASSERT_NEAR(out[0], 3.0f / rms, 1e-4f);
    ASSERT_NEAR(out[1], 4.0f / rms, 1e-4f);

    PASS();
}

void test_compute_cuda_matmul() {
    TEST(compute_cuda_matmul);
    if (!gpu_available) { fprintf(stderr, "SKIP (no GPU)\n"); return; }

    Compute compute(Backend::CUDA);

    // x = [1, 2, 3], w (N×K) = [[1,2,3],[4,5,6]]
    float x[] = {1.0f, 2.0f, 3.0f};
    float w[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float out[2] = {};

    compute.matmul_transposed(out, x, w, 2, 3);

    ASSERT_NEAR(out[0], 14.0f, 1e-3f);
    ASSERT_NEAR(out[1], 32.0f, 1e-3f);

    PASS();
}

// ---- Run all tests ----

int main() {
    fprintf(stderr, "Running GPU tests...\n\n");

    gpu_available = cuda_check_gpu();
    if (!gpu_available) {
        fprintf(stderr, "No CUDA GPU detected - all GPU tests will be skipped.\n\n");
    }

    fprintf(stderr, "GPU kernel tests:\n");
    test_gpu_add();
    test_gpu_rmsnorm();
    test_gpu_softmax();
    test_gpu_silu_mul();
    test_gpu_matmul();
    test_gpu_matmul_transposed();
    test_gpu_rope_identity();
    test_gpu_rope_vs_cpu();

    fprintf(stderr, "\nCompute dispatcher (CUDA backend) tests:\n");
    test_compute_cuda_add();
    test_compute_cuda_rmsnorm();
    test_compute_cuda_matmul();

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n",
            tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
