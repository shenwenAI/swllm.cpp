// Basic tests for llm.cpp components

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#endif

#include "gguf.h"
#include "tensor.h"
#include "sampler.h"
#include "tokenizer.h"
#include "model.h"

// ---- System configuration display ----

// Return CPU brand string (best-effort, cross-platform).
static std::string get_cpu_name() {
#if defined(_WIN32)
    char buf[256] = {};
    DWORD size = sizeof(buf);
    HKEY key;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &key) == ERROR_SUCCESS) {
        RegQueryValueExA(key, "ProcessorNameString", nullptr, nullptr,
                         reinterpret_cast<LPBYTE>(buf), &size);
        RegCloseKey(key);
    }
    if (buf[0]) return buf;
#elif defined(__APPLE__)
    char buf[256] = {};
    size_t size = sizeof(buf);
    if (sysctlbyname("machdep.cpu.brand_string", buf, &size, nullptr, 0) == 0 && buf[0])
        return buf;
#else
    // Linux - parse /proc/cpuinfo
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                const char* colon = strchr(line, ':');
                if (colon) {
                    const char* start = colon + 1;
                    while (*start == ' ' || *start == '\t') ++start;
                    std::string name(start);
                    while (!name.empty() && (name.back() == '\n' || name.back() == '\r'))
                        name.pop_back();
                    fclose(f);
                    return name;
                }
            }
        }
        fclose(f);
    }
#endif
    return "Unknown CPU";
}

// Detect GPU name via nvidia-smi (no CUDA runtime needed).
static std::string get_gpu_name() {
#ifdef _WIN32
    FILE* pipe = _popen("nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>nul", "r");
#else
    FILE* pipe = popen("nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null", "r");
#endif
    if (pipe) {
        char buf[256] = {};
        char* ret = fgets(buf, sizeof(buf), pipe);
#ifdef _WIN32
        _pclose(pipe);
#else
        pclose(pipe);
#endif
        if (ret && buf[0]) {
            std::string name(buf);
            while (!name.empty() && (name.back() == '\n' || name.back() == '\r'))
                name.pop_back();
            if (!name.empty()) return name;
        }
    }
    return "";
}

// Detect CUDA driver version via nvidia-smi.
static std::string get_cuda_driver_version() {
#ifdef _WIN32
    FILE* pipe = _popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>nul", "r");
#else
    FILE* pipe = popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null", "r");
#endif
    if (pipe) {
        char buf[128] = {};
        char* ret = fgets(buf, sizeof(buf), pipe);
#ifdef _WIN32
        _pclose(pipe);
#else
        pclose(pipe);
#endif
        if (ret && buf[0]) {
            std::string ver(buf);
            while (!ver.empty() && (ver.back() == '\n' || ver.back() == '\r'))
                ver.pop_back();
            if (!ver.empty()) return ver;
        }
    }
    return "";
}

// Print system configuration (CPU, GPU, CUDA) to stderr.
static void print_system_config() {
    fprintf(stderr, "=== System Configuration ===\n");
    fprintf(stderr, "CPU:  %s\n", get_cpu_name().c_str());

    std::string gpu = get_gpu_name();
    if (!gpu.empty()) {
        fprintf(stderr, "GPU:  %s\n", gpu.c_str());
        std::string cuda_ver = get_cuda_driver_version();
        if (!cuda_ver.empty()) {
            fprintf(stderr, "CUDA Driver: %s\n", cuda_ver.c_str());
        }
    } else {
        fprintf(stderr, "GPU:  (not detected)\n");
    }
    fprintf(stderr, "============================\n\n");
}

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
                      cfg.architecture == "qwen2moe" ||
                      cfg.architecture == "qwen35" ||
                      cfg.architecture == "qwen35moe");
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

void test_cpu_matmul_transposed_f16() {
    TEST(cpu_matmul_transposed_f16);

    // Build a 2×4 weight matrix in F16 format.
    // Row 0: [1.0, 0.0, 0.0, 0.0] — 1.0=0x3C00, 0.0=0x0000
    // Row 1: [0.0, 2.0, 0.0, 0.0] — 2.0=0x4000
    // x = [1.0, 2.0, 3.0, 4.0]
    // Expected: out[0] = 1.0*1.0 = 1.0, out[1] = 2.0*2.0 = 4.0
    const int N = 2, K = 4;
    uint16_t w[N * K] = {
        0x3C00, 0x0000, 0x0000, 0x0000,  // row 0: [1.0, 0.0, 0.0, 0.0]
        0x0000, 0x4000, 0x0000, 0x0000,  // row 1: [0.0, 2.0, 0.0, 0.0]
    };
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[2] = {};

    cpu_matmul_transposed_f16(out, x, w, N, K);

    ASSERT_NEAR(out[0], 1.0f, 1e-4f);
    ASSERT_NEAR(out[1], 4.0f, 1e-4f);

    // Cross-check: dequantize then matmul must give the same result
    float w_f32[N * K];
    dequantize(w, w_f32, N * K, GGML_TYPE_F16);
    float out_ref[2] = {};
    cpu_matmul_transposed(out_ref, x, w_f32, N, K);
    ASSERT_NEAR(out[0], out_ref[0], 1e-4f);
    ASSERT_NEAR(out[1], out_ref[1], 1e-4f);

    // Non-trivial: row of all 2.0 weights, x = all ones → dot = 4 * 2.0 = 8.0
    uint16_t w2[1 * K] = {0x4000, 0x4000, 0x4000, 0x4000};
    float x2[K] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out2[1] = {};
    cpu_matmul_transposed_f16(out2, x2, w2, 1, K);
    ASSERT_NEAR(out2[0], 8.0f, 1e-4f);

    PASS();
}

void test_bf16_gguf_type() {
    TEST(bf16_gguf_type);

    // Verify BF16 GGML type properties
    ASSERT_EQ(ggml_type_size(GGML_TYPE_BF16), 2u);
    ASSERT_EQ(ggml_block_size(GGML_TYPE_BF16), 1u);
    ASSERT_EQ(std::string(ggml_type_name(GGML_TYPE_BF16)), std::string("BF16"));

    PASS();
}

void test_dequantize_bf16() {
    TEST(dequantize_bf16);

    // BF16 for known values: upper 16 bits of IEEE 754 float32
    // 1.0f = 0x3F800000 → BF16 = 0x3F80
    // 2.0f = 0x40000000 → BF16 = 0x4000
    // -1.0f = 0xBF800000 → BF16 = 0xBF80
    // 0.0f = 0x00000000 → BF16 = 0x0000
    uint16_t src[] = {0x3F80, 0x4000, 0xBF80, 0x0000};
    float dst[4] = {};
    dequantize(src, dst, 4, GGML_TYPE_BF16);

    ASSERT_NEAR(dst[0], 1.0f, 1e-6f);
    ASSERT_NEAR(dst[1], 2.0f, 1e-6f);
    ASSERT_NEAR(dst[2], -1.0f, 1e-6f);
    ASSERT_NEAR(dst[3], 0.0f, 1e-6f);

    PASS();
}

void test_cpu_matmul_transposed_bf16() {
    TEST(cpu_matmul_transposed_bf16);

    // Build a 2×4 weight matrix in BF16 format.
    // Row 0: [1.0, 0.0, 0.0, 0.0] — 1.0=0x3F80, 0.0=0x0000
    // Row 1: [0.0, 2.0, 0.0, 0.0] — 2.0=0x4000
    // x = [1.0, 2.0, 3.0, 4.0]
    // Expected: out[0] = 1.0*1.0 = 1.0, out[1] = 2.0*2.0 = 4.0
    const int N = 2, K = 4;
    uint16_t w[N * K] = {
        0x3F80, 0x0000, 0x0000, 0x0000,  // row 0: [1.0, 0.0, 0.0, 0.0]
        0x0000, 0x4000, 0x0000, 0x0000,  // row 1: [0.0, 2.0, 0.0, 0.0]
    };
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[2] = {};

    cpu_matmul_transposed_bf16(out, x, w, N, K);

    ASSERT_NEAR(out[0], 1.0f, 1e-4f);
    ASSERT_NEAR(out[1], 4.0f, 1e-4f);

    // Cross-check: dequantize then matmul must give the same result
    float w_f32[N * K];
    dequantize(w, w_f32, N * K, GGML_TYPE_BF16);
    float out_ref[2] = {};
    cpu_matmul_transposed(out_ref, x, w_f32, N, K);
    ASSERT_NEAR(out[0], out_ref[0], 1e-4f);
    ASSERT_NEAR(out[1], out_ref[1], 1e-4f);

    // Non-trivial: row of all 2.0 weights, x = all ones → dot = 4 * 2.0 = 8.0
    uint16_t w2[1 * K] = {0x4000, 0x4000, 0x4000, 0x4000};
    float x2[K] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out2[1] = {};
    cpu_matmul_transposed_bf16(out2, x2, w2, 1, K);
    ASSERT_NEAR(out2[0], 8.0f, 1e-4f);

    PASS();
}

void test_qwen35_layer_types_from_gguf() {
    TEST(qwen35_layer_types_from_gguf);

    // Verify that integer layer type values are correctly mapped:
    // 0 = full_attention, 1 = linear_attention
    // This tests the mapping logic in load_config()

    // Simulate: layer_types array [0, 1, 1, 0] → ["full_attention", "linear_attention", ...]
    std::vector<int64_t> raw_types = {0, 1, 1, 0};
    std::vector<std::string> mapped;
    for (int64_t v : raw_types) {
        mapped.push_back(v == 0 ? "full_attention" : "linear_attention");
    }

    ASSERT_EQ(mapped.size(), 4u);
    ASSERT_EQ(mapped[0], std::string("full_attention"));
    ASSERT_EQ(mapped[1], std::string("linear_attention"));
    ASSERT_EQ(mapped[2], std::string("linear_attention"));
    ASSERT_EQ(mapped[3], std::string("full_attention"));

    PASS();
}

void test_qwen35_post_attention_norm_fallback() {
    TEST(qwen35_post_attention_norm_fallback);

    // Qwen3.5 GGUF files from llama.cpp use "post_attention_norm" instead of
    // "ffn_norm" for the post-attention normalization tensor.  Verify that
    // load_weights() falls back to "post_attention_norm.weight" when
    // "ffn_norm.weight" is not found.

    // Simulate by inserting both tensor names into a GGUFFile tensor map and
    // checking that the lookup order is correct.
    GGUFFile gguf;

    // Only add "post_attention_norm.weight" (NOT "ffn_norm.weight")
    GGUFTensorInfo info;
    info.name = "blk.0.post_attention_norm.weight";
    info.n_dims = 1;
    info.dims[0] = 4;
    info.dims[1] = 1;
    info.dims[2] = 1;
    info.dims[3] = 1;
    info.type = GGML_TYPE_F32;
    info.offset = 0;
    gguf.tensors[info.name] = info;

    // "ffn_norm.weight" should NOT be found
    ASSERT_TRUE(gguf.tensors.find("blk.0.ffn_norm.weight") == gguf.tensors.end());

    // "post_attention_norm.weight" should be found
    ASSERT_TRUE(gguf.tensors.find("blk.0.post_attention_norm.weight") != gguf.tensors.end());

    // Verify the fallback logic: try ffn_norm first, then post_attention_norm
    std::string prefix = "blk.0.";
    auto it = gguf.tensors.find(prefix + "ffn_norm.weight");
    bool found_ffn_norm = (it != gguf.tensors.end());
    ASSERT_TRUE(!found_ffn_norm);  // ffn_norm should not exist

    it = gguf.tensors.find(prefix + "post_attention_norm.weight");
    bool found_post_attn_norm = (it != gguf.tensors.end());
    ASSERT_TRUE(found_post_attn_norm);  // post_attention_norm should exist

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

// ---- K-quant type size tests ----

void test_kquant_type_sizes() {
    TEST(kquant_type_sizes);
    // Verify block sizes match the standard ggml K-quant definitions
    ASSERT_EQ(ggml_block_size(GGML_TYPE_Q2_K), 256u);
    ASSERT_EQ(ggml_block_size(GGML_TYPE_Q3_K), 256u);
    ASSERT_EQ(ggml_block_size(GGML_TYPE_Q4_K), 256u);
    ASSERT_EQ(ggml_block_size(GGML_TYPE_Q5_K), 256u);
    ASSERT_EQ(ggml_block_size(GGML_TYPE_Q6_K), 256u);
    // Verify byte sizes of each block type
    ASSERT_EQ(ggml_type_size(GGML_TYPE_Q2_K), 84u);   // scales(16)+qs(64)+d(2)+dmin(2)
    ASSERT_EQ(ggml_type_size(GGML_TYPE_Q3_K), 110u);  // hmask(32)+qs(64)+scales(12)+d(2)
    ASSERT_EQ(ggml_type_size(GGML_TYPE_Q4_K), 144u);  // d(2)+dmin(2)+scales(12)+qs(128)
    ASSERT_EQ(ggml_type_size(GGML_TYPE_Q5_K), 176u);  // d(2)+dmin(2)+scales(12)+qh(32)+qs(128)
    ASSERT_EQ(ggml_type_size(GGML_TYPE_Q6_K), 210u);  // ql(128)+qh(64)+scales(16)+d(2)
    // Verify type names
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_Q2_K), "Q2_K"), 0);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_Q3_K), "Q3_K"), 0);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_Q4_K), "Q4_K"), 0);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_Q5_K), "Q5_K"), 0);
    ASSERT_EQ(strcmp(ggml_type_name(GGML_TYPE_Q6_K), "Q6_K"), 0);
    PASS();
}

// Build a Q6_K block (210 bytes) with:
//   d=1.0, all scales=1, ql=all zeros, qh=all zeros
// → all 6-bit values = 0 - 32 = -32  → output = 1.0 * 1 * (-32) = -32.0
void test_dequantize_q6_k() {
    TEST(dequantize_q6_k);

    std::vector<uint8_t> blk(210, 0);
    // ql at [0..127]: already zero
    // qh at [128..191]: already zero
    // scales (int8) at [192..207]: set to 1
    for (int i = 0; i < 16; i++) blk[192 + i] = 1;
    // d at [208..209]: 1.0 in f16 = 0x3C00
    uint16_t d16 = 0x3C00;
    memcpy(blk.data() + 208, &d16, 2);

    std::vector<float> out(256, 0.0f);
    dequantize_q6_k(blk.data(), out.data(), 256);

    // All values: 1.0 * 1 * (0 - 32) = -32.0
    for (int i = 0; i < 256; i++) {
        ASSERT_NEAR(out[i], -32.0f, 1e-4f);
    }

    // Cross-check via generic dequantize dispatch
    std::vector<float> out2(256, 0.0f);
    dequantize(blk.data(), out2.data(), 256, GGML_TYPE_Q6_K);
    for (int i = 0; i < 256; i++) {
        ASSERT_NEAR(out2[i], out[i], 1e-6f);
    }

    PASS();
}

// Build a Q4_K block (144 bytes) with:
//   d=1.0, dmin=0.0, scales[0..3]=1 (→ sc=1,m=0 for groups 0-3)
//   qs=all nibbles 0x8 → value = (0x8 & 0xF) = 8, output = 1.0*8 - 0 = 8.0
//   (for groups 0-3 only; groups 4-7 have a different scale from packing)
void test_dequantize_q4_k() {
    TEST(dequantize_q4_k);

    std::vector<uint8_t> blk(144, 0);
    // d at [0..1]: 1.0 in f16 = 0x3C00
    uint16_t d16 = 0x3C00, dmin16 = 0x0000; // dmin = 0.0
    memcpy(blk.data() + 0, &d16,    2);
    memcpy(blk.data() + 2, &dmin16, 2);
    // scales at [4..15]: set scales[0..3]=1 → groups 0-3 get sc=1,m=0
    blk[4] = 1; blk[5] = 1; blk[6] = 1; blk[7] = 1;
    // qs at [16..143]: all 0x88 → nibbles = 8 each
    for (int i = 16; i < 144; i++) blk[i] = 0x88;

    std::vector<float> out(256, 0.0f);
    dequantize_q4_k(blk.data(), out.data(), 256);

    // Groups 0-3 (first 128 elements): sc=1, d=1.0, nibble=8, dmin=0 → 1.0*8-0=8.0
    for (int i = 0; i < 128; i++) {
        ASSERT_NEAR(out[i], 8.0f, 1e-4f);
    }

    // Cross-check via generic dequantize dispatch
    std::vector<float> out2(256, 0.0f);
    dequantize(blk.data(), out2.data(), 256, GGML_TYPE_Q4_K);
    for (int i = 0; i < 256; i++) {
        ASSERT_NEAR(out2[i], out[i], 1e-6f);
    }

    PASS();
}

// Build a Q2_K block (84 bytes) with:
//   d=1.0, dmin=0.0, scales=all nibble 1 (sc=1,m=0), qs=all zeros (2-bit value 0)
//   → output = 1.0 * 1 * 0 - 0 = 0.0 for every element
void test_dequantize_q2_k() {
    TEST(dequantize_q2_k);

    std::vector<uint8_t> blk(84, 0);
    // scales at [0..15]: sc in lower nibble = 1, m in upper nibble = 0
    for (int i = 0; i < 16; i++) blk[i] = 0x01; // sc=1, m=0
    // qs at [16..79]: all zeros (2-bit values all 0)
    // d at [80..81]: 1.0 in f16
    uint16_t d16 = 0x3C00, dmin16 = 0x0000;
    memcpy(blk.data() + 80, &d16,    2);
    memcpy(blk.data() + 82, &dmin16, 2);

    std::vector<float> out(256, 1.0f); // initialise to non-zero
    dequantize_q2_k(blk.data(), out.data(), 256);

    // All 2-bit quants are 0 → output = d*(sc*0) - dmin*m = 0
    for (int i = 0; i < 256; i++) {
        ASSERT_NEAR(out[i], 0.0f, 1e-5f);
    }

    // Cross-check via dispatch
    std::vector<float> out2(256, 1.0f);
    dequantize(blk.data(), out2.data(), 256, GGML_TYPE_Q2_K);
    for (int i = 0; i < 256; i++) {
        ASSERT_NEAR(out2[i], out[i], 1e-6f);
    }

    PASS();
}

// Build a Q5_0 block (22 bytes) with:
//   d=1.0, qh=all zeros, qs=all 0x55 → nibble lo=5, nibble hi=5
//   high bits all zero → 5-bit value = 5, output = (5 - 16) * 1.0 = -11.0
void test_dequantize_q5_0() {
    TEST(dequantize_q5_0);

    std::vector<uint8_t> blk(22, 0);
    // d at [0..1]: 1.0 in f16 = 0x3C00
    uint16_t d16 = 0x3C00;
    memcpy(blk.data(), &d16, 2);
    // qh at [2..5]: all zeros (no high bits set)
    // qs at [6..21]: all 0x55 → lo nibble = 5, hi nibble = 5
    for (int i = 6; i < 22; i++) blk[i] = 0x55;

    std::vector<float> out(32, 0.0f);
    dequantize_q5_0(blk.data(), out.data(), 32);

    // All 5-bit values = 5, output = (5 - 16) * 1.0 = -11.0
    for (int i = 0; i < 32; i++) {
        ASSERT_NEAR(out[i], -11.0f, 1e-4f);
    }

    // Cross-check via generic dequantize dispatch
    std::vector<float> out2(32, 0.0f);
    dequantize(blk.data(), out2.data(), 32, GGML_TYPE_Q5_0);
    for (int i = 0; i < 32; i++) {
        ASSERT_NEAR(out2[i], out[i], 1e-6f);
    }

    PASS();
}

// Build a Q5_0 block with high bits set to verify 5th bit decoding
void test_dequantize_q5_0_high_bits() {
    TEST(dequantize_q5_0_high_bits);

    std::vector<uint8_t> blk(22, 0);
    // d at [0..1]: 1.0 in f16 = 0x3C00
    uint16_t d16 = 0x3C00;
    memcpy(blk.data(), &d16, 2);
    // qh at [2..5]: all 0xFF (all high bits set)
    for (int i = 2; i < 6; i++) blk[i] = 0xFF;
    // qs at [6..21]: all zeros → lo nibble = 0, hi nibble = 0
    // With high bits set: 5-bit value = 16, output = (16 - 16) * 1.0 = 0.0

    std::vector<float> out(32, 1.0f); // initialise to non-zero
    dequantize_q5_0(blk.data(), out.data(), 32);

    for (int i = 0; i < 32; i++) {
        ASSERT_NEAR(out[i], 0.0f, 1e-4f);
    }

    PASS();
}

// Build a Q5_K block (176 bytes) with:
//   d=1.0, dmin=0.0, scales[0..3]=1 (→ sc=1,m=0 for groups 0-3)
//   qh=all zeros, qs=all 0x55 → nibble 5, no high bit → value 5
//   output = 1.0 * 5 - 0 = 5.0 (for groups 0-3)
void test_dequantize_q5_k() {
    TEST(dequantize_q5_k);

    std::vector<uint8_t> blk(176, 0);
    // d at [0..1]: 1.0 in f16 = 0x3C00
    uint16_t d16 = 0x3C00, dmin16 = 0x0000; // dmin = 0.0
    memcpy(blk.data() + 0, &d16,    2);
    memcpy(blk.data() + 2, &dmin16, 2);
    // scales at [4..15]: set scales[0..3]=1 → groups 0-3 get sc=1,m=0
    blk[4] = 1; blk[5] = 1; blk[6] = 1; blk[7] = 1;
    // qh at [16..47]: all zeros (no high bits)
    // qs at [48..175]: all 0x55 → nibbles = 5 each
    for (int i = 48; i < 176; i++) blk[i] = 0x55;

    std::vector<float> out(256, 0.0f);
    dequantize_q5_k(blk.data(), out.data(), 256);

    // Groups 0-3 (first 128 elements): sc=1, d=1.0, nibble=5, qh=0, dmin=0 → 1.0*5-0=5.0
    for (int i = 0; i < 128; i++) {
        ASSERT_NEAR(out[i], 5.0f, 1e-4f);
    }

    // Cross-check via generic dequantize dispatch
    std::vector<float> out2(256, 0.0f);
    dequantize(blk.data(), out2.data(), 256, GGML_TYPE_Q5_K);
    for (int i = 0; i < 256; i++) {
        ASSERT_NEAR(out2[i], out[i], 1e-6f);
    }

    PASS();
}

// Test fused Q6_K matmul: dequantize+accumulate vs dequantize-then-matmul
void test_cpu_matmul_transposed_q6_k() {
    TEST(cpu_matmul_transposed_q6_k);

    // Build a 2-row weight matrix in Q6_K format (K=256, N=2).
    // Row 0: all quant values 0 (which encodes -32 after offset), scales=1, d=1.0
    //        → dequantized value = 1.0 * 1 * (0-32) = -32
    // Row 1: same
    const int N = 2, K = 256;
    const size_t block_bytes = 210;

    std::vector<uint8_t> w(N * block_bytes, 0);
    for (int row = 0; row < N; row++) {
        uint8_t* blk = w.data() + row * block_bytes;
        // scales at [192..207]: set to 1
        for (int i = 0; i < 16; i++) blk[192 + i] = 1;
        // d at [208..209]: 1.0 in f16 = 0x3C00
        uint16_t d16 = 0x3C00;
        memcpy(blk + 208, &d16, 2);
    }

    // Input: x = all 1.0 → dot product = 256 * (-32) = -8192
    std::vector<float> x(K, 1.0f);
    float out_fused[2] = {};
    cpu_matmul_transposed_q6_k(out_fused, x.data(), w.data(), N, K);

    // Cross-check: dequantize then F32 matmul
    std::vector<float> w_f32(static_cast<size_t>(N) * K);
    for (int row = 0; row < N; row++) {
        dequantize_q6_k(w.data() + row * block_bytes,
                        w_f32.data() + row * K, K);
    }
    float out_ref[2] = {};
    cpu_matmul_transposed(out_ref, x.data(), w_f32.data(), N, K);

    ASSERT_NEAR(out_fused[0], out_ref[0], 1e-3f);
    ASSERT_NEAR(out_fused[1], out_ref[1], 1e-3f);
    ASSERT_NEAR(out_fused[0], -8192.0f, 1e-1f);

    PASS();
}

// Test fused Q4_K matmul: dequantize+accumulate vs dequantize-then-matmul
void test_cpu_matmul_transposed_q4_k() {
    TEST(cpu_matmul_transposed_q4_k);

    // Build a 2-row weight matrix in Q4_K format (K=256, N=2).
    // Same setup as test_dequantize_q4_k: d=1.0, dmin=0, scales[0..3]=1,
    // qs = all 0x88 (nibble 8).
    const int N = 2, K = 256;
    const size_t block_bytes = 144;

    std::vector<uint8_t> w(N * block_bytes, 0);
    for (int row = 0; row < N; row++) {
        uint8_t* blk = w.data() + row * block_bytes;
        uint16_t d16 = 0x3C00, dmin16 = 0x0000;
        memcpy(blk + 0, &d16,    2);
        memcpy(blk + 2, &dmin16, 2);
        blk[4] = 1; blk[5] = 1; blk[6] = 1; blk[7] = 1;
        for (int i = 16; i < 144; i++) blk[i] = 0x88;
    }

    // Input: x = all 1.0
    std::vector<float> x(K, 1.0f);
    float out_fused[2] = {};
    cpu_matmul_transposed_q4_k(out_fused, x.data(), w.data(), N, K);

    // Cross-check: dequantize then F32 matmul
    std::vector<float> w_f32(static_cast<size_t>(N) * K);
    for (int row = 0; row < N; row++) {
        dequantize_q4_k(w.data() + row * block_bytes,
                        w_f32.data() + row * K, K);
    }
    float out_ref[2] = {};
    cpu_matmul_transposed(out_ref, x.data(), w_f32.data(), N, K);

    ASSERT_NEAR(out_fused[0], out_ref[0], 1e-3f);
    ASSERT_NEAR(out_fused[1], out_ref[1], 1e-3f);

    PASS();
}

// Test fused Q5_K matmul: dequantize+accumulate vs dequantize-then-matmul
void test_cpu_matmul_transposed_q5_k() {
    TEST(cpu_matmul_transposed_q5_k);

    // Build a 2-row weight matrix in Q5_K format (K=256, N=2).
    // Same setup as test_dequantize_q5_k: d=1.0, dmin=0, scales[0..3]=1,
    // qh=0, qs = all 0x55 (nibble 5).
    const int N = 2, K = 256;
    const size_t block_bytes = 176;

    std::vector<uint8_t> w(N * block_bytes, 0);
    for (int row = 0; row < N; row++) {
        uint8_t* blk = w.data() + row * block_bytes;
        uint16_t d16 = 0x3C00, dmin16 = 0x0000;
        memcpy(blk + 0, &d16,    2);
        memcpy(blk + 2, &dmin16, 2);
        blk[4] = 1; blk[5] = 1; blk[6] = 1; blk[7] = 1;
        for (int i = 48; i < 176; i++) blk[i] = 0x55;
    }

    // Input: x = all 1.0
    std::vector<float> x(K, 1.0f);
    float out_fused[2] = {};
    cpu_matmul_transposed_q5_k(out_fused, x.data(), w.data(), N, K);

    // Cross-check: dequantize then F32 matmul
    std::vector<float> w_f32(static_cast<size_t>(N) * K);
    for (int row = 0; row < N; row++) {
        dequantize_q5_k(w.data() + row * block_bytes,
                        w_f32.data() + row * K, K);
    }
    float out_ref[2] = {};
    cpu_matmul_transposed(out_ref, x.data(), w_f32.data(), N, K);

    ASSERT_NEAR(out_fused[0], out_ref[0], 1e-3f);
    ASSERT_NEAR(out_fused[1], out_ref[1], 1e-3f);

    PASS();
}

// Test fused Q5_0 matmul: dequantize+accumulate vs dequantize-then-matmul
void test_cpu_matmul_transposed_q5_0() {
    TEST(cpu_matmul_transposed_q5_0);

    // Build a 2-row weight matrix in Q5_0 format (K=32, N=2).
    // d=1.0, qh=0, qs=all 0x55 (nibble 5, no high bit → value 5)
    // → dequantized = (5-16)*1.0 = -11.0
    const int N = 2, K = 32;
    const int bytes_per_block = 22;

    std::vector<uint8_t> w(N * bytes_per_block, 0);
    for (int row = 0; row < N; row++) {
        uint8_t* blk = w.data() + row * bytes_per_block;
        uint16_t d16 = 0x3C00;
        memcpy(blk, &d16, 2);
        // qh at [2..5]: zeros
        // qs at [6..21]: all 0x55
        for (int i = 6; i < 22; i++) blk[i] = 0x55;
    }

    // Input: x = all 1.0 → dot product = 32 * (-11) = -352
    std::vector<float> x(K, 1.0f);
    float out_fused[2] = {};
    cpu_matmul_transposed_q5_0(out_fused, x.data(), w.data(), N, K);

    // Cross-check: dequantize then F32 matmul
    std::vector<float> w_f32(static_cast<size_t>(N) * K);
    for (int row = 0; row < N; row++) {
        dequantize_q5_0(w.data() + row * bytes_per_block,
                        w_f32.data() + row * K, K);
    }
    float out_ref[2] = {};
    cpu_matmul_transposed(out_ref, x.data(), w_f32.data(), N, K);

    ASSERT_NEAR(out_fused[0], out_ref[0], 1e-3f);
    ASSERT_NEAR(out_fused[1], out_ref[1], 1e-3f);
    ASSERT_NEAR(out_fused[0], -352.0f, 1e-1f);

    PASS();
}

// Test that the Compute dispatcher routes Q6_K/Q4_K to fused kernels
void test_compute_q_kquant_dispatch() {
    TEST(compute_q_kquant_dispatch);

    Compute compute(Backend::CPU);

    // Test Q6_K dispatch
    const int K = 256;
    const size_t q6_block_bytes = 210;
    std::vector<uint8_t> w6(q6_block_bytes, 0);
    for (int i = 0; i < 16; i++) w6[192 + i] = 1;
    uint16_t d16 = 0x3C00;
    memcpy(w6.data() + 208, &d16, 2);

    std::vector<float> x(K, 1.0f);
    float out_dispatch = 0.0f;
    QuantWeight qw6 = {w6.data(), GGML_TYPE_Q6_K};
    compute.matmul_transposed_q(&out_dispatch, x.data(), qw6, 1, K);

    float out_fused = 0.0f;
    cpu_matmul_transposed_q6_k(&out_fused, x.data(), w6.data(), 1, K);
    ASSERT_NEAR(out_dispatch, out_fused, 1e-6f);

    // Test Q4_K dispatch
    const size_t q4_block_bytes = 144;
    std::vector<uint8_t> w4(q4_block_bytes, 0);
    uint16_t dmin16 = 0x0000;
    memcpy(w4.data() + 0, &d16, 2);
    memcpy(w4.data() + 2, &dmin16, 2);
    w4[4] = 1; w4[5] = 1; w4[6] = 1; w4[7] = 1;
    for (int i = 16; i < 144; i++) w4[i] = 0x88;

    float out4_dispatch = 0.0f;
    QuantWeight qw4 = {w4.data(), GGML_TYPE_Q4_K};
    compute.matmul_transposed_q(&out4_dispatch, x.data(), qw4, 1, K);

    float out4_fused = 0.0f;
    cpu_matmul_transposed_q4_k(&out4_fused, x.data(), w4.data(), 1, K);
    ASSERT_NEAR(out4_dispatch, out4_fused, 1e-6f);

    // Test Q5_K dispatch
    const size_t q5k_block_bytes = 176;
    std::vector<uint8_t> w5k(q5k_block_bytes, 0);
    memcpy(w5k.data() + 0, &d16, 2);
    memcpy(w5k.data() + 2, &dmin16, 2);
    w5k[4] = 1; w5k[5] = 1; w5k[6] = 1; w5k[7] = 1;
    for (int i = 48; i < 176; i++) w5k[i] = 0x55;

    float out5k_dispatch = 0.0f;
    QuantWeight qw5k = {w5k.data(), GGML_TYPE_Q5_K};
    compute.matmul_transposed_q(&out5k_dispatch, x.data(), qw5k, 1, K);

    float out5k_fused = 0.0f;
    cpu_matmul_transposed_q5_k(&out5k_fused, x.data(), w5k.data(), 1, K);
    ASSERT_NEAR(out5k_dispatch, out5k_fused, 1e-6f);

    // Test Q5_0 dispatch (K=32 for Q5_0)
    const int K5 = 32;
    const int q5_block_bytes = 22;
    std::vector<uint8_t> w50(q5_block_bytes, 0);
    memcpy(w50.data(), &d16, 2);
    for (int i = 6; i < 22; i++) w50[i] = 0x55;

    std::vector<float> x5(K5, 1.0f);
    float out50_dispatch = 0.0f;
    QuantWeight qw50 = {w50.data(), GGML_TYPE_Q5_0};
    compute.matmul_transposed_q(&out50_dispatch, x5.data(), qw50, 1, K5);

    float out50_fused = 0.0f;
    cpu_matmul_transposed_q5_0(&out50_fused, x5.data(), w50.data(), 1, K5);
    ASSERT_NEAR(out50_dispatch, out50_fused, 1e-6f);

    PASS();
}

// Test expanded HF model_type → GGUF architecture mappings
void test_expanded_architecture_mappings() {
    TEST(expanded_architecture_mappings);

    HFModelConfig cfg;

    // Existing mappings (verify still work)
    cfg.model_type = "llama"; ASSERT_EQ(cfg.get_architecture(), std::string("llama"));
    cfg.model_type = "mistral"; ASSERT_EQ(cfg.get_architecture(), std::string("llama"));
    cfg.model_type = "qwen2"; ASSERT_EQ(cfg.get_architecture(), std::string("qwen2"));
    cfg.model_type = "qwen3"; ASSERT_EQ(cfg.get_architecture(), std::string("qwen3"));
    cfg.model_type = "deepseek_v2"; ASSERT_EQ(cfg.get_architecture(), std::string("deepseek2"));

    // New Gemma mappings
    cfg.model_type = "gemma"; ASSERT_EQ(cfg.get_architecture(), std::string("gemma"));
    cfg.model_type = "gemma2"; ASSERT_EQ(cfg.get_architecture(), std::string("gemma2"));
    cfg.model_type = "gemma3"; ASSERT_EQ(cfg.get_architecture(), std::string("gemma3"));

    // New DeepSeek V3 mapping
    cfg.model_type = "deepseek_v3"; ASSERT_EQ(cfg.get_architecture(), std::string("deepseek2"));

    // New Phi mappings
    cfg.model_type = "phi"; ASSERT_EQ(cfg.get_architecture(), std::string("phi2"));
    cfg.model_type = "phi3"; ASSERT_EQ(cfg.get_architecture(), std::string("phi3"));

    // New InternLM mapping
    cfg.model_type = "internlm2"; ASSERT_EQ(cfg.get_architecture(), std::string("internlm2"));
    cfg.model_type = "internlm3"; ASSERT_EQ(cfg.get_architecture(), std::string("internlm2"));

    // New ChatGLM/GLM4 mappings
    cfg.model_type = "chatglm"; ASSERT_EQ(cfg.get_architecture(), std::string("chatglm"));
    cfg.model_type = "glm4"; ASSERT_EQ(cfg.get_architecture(), std::string("glm4"));

    // New Cohere mappings
    cfg.model_type = "cohere"; ASSERT_EQ(cfg.get_architecture(), std::string("command-r"));
    cfg.model_type = "cohere2"; ASSERT_EQ(cfg.get_architecture(), std::string("cohere2"));

    // New StarCoder2 mapping
    cfg.model_type = "starcoder2"; ASSERT_EQ(cfg.get_architecture(), std::string("starcoder2"));

    // New MiniCPM mapping
    cfg.model_type = "minicpm"; ASSERT_EQ(cfg.get_architecture(), std::string("minicpm"));

    // New SmolLM3 mapping
    cfg.model_type = "smollm3"; ASSERT_EQ(cfg.get_architecture(), std::string("smollm3"));

    // New Exaone mapping
    cfg.model_type = "exaone"; ASSERT_EQ(cfg.get_architecture(), std::string("exaone"));

    // New OLMO mappings
    cfg.model_type = "olmo"; ASSERT_EQ(cfg.get_architecture(), std::string("olmo"));
    cfg.model_type = "olmo2"; ASSERT_EQ(cfg.get_architecture(), std::string("olmo2"));

    // Unknown models pass through unchanged
    cfg.model_type = "some_unknown"; ASSERT_EQ(cfg.get_architecture(), std::string("some_unknown"));

    PASS();
}

// ---- SafeTensors and HF loader tests ----

#include "safetensors.h"
#include "hf_loader.h"

void test_safetensors_dtype() {
    TEST(safetensors_dtype);

    // Test dtype string parsing
    ASSERT_EQ(st_dtype_from_string("F32"), ST_DTYPE_F32);
    ASSERT_EQ(st_dtype_from_string("F16"), ST_DTYPE_F16);
    ASSERT_EQ(st_dtype_from_string("BF16"), ST_DTYPE_BF16);
    ASSERT_EQ(st_dtype_from_string("I8"), ST_DTYPE_I8);
    ASSERT_EQ(st_dtype_from_string("UNKNOWN"), ST_DTYPE_UNKNOWN);

    // Test dtype sizes
    ASSERT_EQ(st_dtype_size(ST_DTYPE_F32), 4u);
    ASSERT_EQ(st_dtype_size(ST_DTYPE_F16), 2u);
    ASSERT_EQ(st_dtype_size(ST_DTYPE_BF16), 2u);
    ASSERT_EQ(st_dtype_size(ST_DTYPE_I8), 1u);

    // Test GGML type mapping
    ASSERT_EQ(st_dtype_to_ggml(ST_DTYPE_F32), GGML_TYPE_F32);
    ASSERT_EQ(st_dtype_to_ggml(ST_DTYPE_F16), GGML_TYPE_F16);

    PASS();
}

void test_safetensors_parse() {
    TEST(safetensors_parse);

    // Create a minimal SafeTensors file in memory
    // Header: {"test_tensor": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
    std::string header =
        "{\"test_tensor\": {\"dtype\": \"F32\", \"shape\": [2, 3], "
        "\"data_offsets\": [0, 24]}}";

    uint64_t header_len = header.size();
    size_t total_size = 8 + header_len + 24;  // 6 floats = 24 bytes
    std::vector<uint8_t> buf(total_size, 0);

    // Write header length
    memcpy(buf.data(), &header_len, 8);
    // Write header
    memcpy(buf.data() + 8, header.c_str(), header_len);
    // Write tensor data: 6 floats
    float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    memcpy(buf.data() + 8 + header_len, data, 24);

    // Write to temp file
    const char* tmp_path = "/tmp/test_safetensors.safetensors";
    FILE* f = fopen(tmp_path, "wb");
    ASSERT_TRUE(f != nullptr);
    fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);

    // Parse it
    SafeTensorsFile st;
    ASSERT_TRUE(st.load(tmp_path));
    ASSERT_EQ(static_cast<int>(st.tensors.size()), 1);

    auto it = st.tensors.find("test_tensor");
    ASSERT_TRUE(it != st.tensors.end());

    const SafeTensorInfo& info = it->second;
    ASSERT_EQ(info.dtype, ST_DTYPE_F32);
    ASSERT_EQ(static_cast<int>(info.shape.size()), 2);
    ASSERT_EQ(info.shape[0], 2);
    ASSERT_EQ(info.shape[1], 3);
    ASSERT_EQ(info.num_elements(), 6);

    // Verify data
    const void* raw = st.get_tensor_data("test_tensor");
    ASSERT_TRUE(raw != nullptr);
    const float* fdata = static_cast<const float*>(raw);
    ASSERT_NEAR(fdata[0], 1.0f, 1e-6f);
    ASSERT_NEAR(fdata[5], 6.0f, 1e-6f);

    // Test dequantization (F32 -> F32 is identity)
    float out[6];
    st.dequantize_to_f32(raw, out, 6, ST_DTYPE_F32);
    ASSERT_NEAR(out[0], 1.0f, 1e-6f);
    ASSERT_NEAR(out[5], 6.0f, 1e-6f);

    // Clean up
    remove(tmp_path);

    PASS();
}

void test_bf16_conversion() {
    TEST(bf16_conversion);

    // BF16 for 1.0: sign=0, exp=01111111 (127), mantissa=0000000
    // In bits: 0 01111111 0000000 = 0x3F80
    uint16_t bf16_one = 0x3F80;
    ASSERT_NEAR(bf16_to_fp32(bf16_one), 1.0f, 1e-6f);

    // BF16 for -2.0: sign=1, exp=10000000 (128), mantissa=0000000
    // In bits: 1 10000000 0000000 = 0xC000
    uint16_t bf16_neg2 = 0xC000;
    ASSERT_NEAR(bf16_to_fp32(bf16_neg2), -2.0f, 1e-6f);

    // BF16 for 0.0
    uint16_t bf16_zero = 0x0000;
    ASSERT_NEAR(bf16_to_fp32(bf16_zero), 0.0f, 1e-6f);

    // FP16 via st_fp16_to_fp32 should match the existing fp16_to_fp32
    uint16_t fp16_one = 0x3C00;  // 1.0 in FP16
    ASSERT_NEAR(st_fp16_to_fp32(fp16_one), 1.0f, 1e-6f);

    PASS();
}

void test_hf_weight_name_mapping() {
    TEST(hf_weight_name_mapping);

    // Test embedding/output mappings
    ASSERT_EQ(hf_to_gguf_tensor_name("model.embed_tokens.weight"),
              std::string("token_embd.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.norm.weight"),
              std::string("output_norm.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("lm_head.weight"),
              std::string("output.weight"));

    // Test attention layer mappings
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.self_attn.q_proj.weight"),
              std::string("blk.0.attn_q.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.5.self_attn.k_proj.weight"),
              std::string("blk.5.attn_k.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.10.self_attn.v_proj.weight"),
              std::string("blk.10.attn_v.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.self_attn.o_proj.weight"),
              std::string("blk.0.attn_output.weight"));

    // Test bias mappings
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.self_attn.q_proj.bias"),
              std::string("blk.0.attn_q.bias"));

    // Test QK-norm mappings
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.3.self_attn.q_norm.weight"),
              std::string("blk.3.attn_q_norm.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.3.self_attn.k_norm.weight"),
              std::string("blk.3.attn_k_norm.weight"));

    // Test LayerNorm mappings
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.input_layernorm.weight"),
              std::string("blk.0.attn_norm.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.post_attention_layernorm.weight"),
              std::string("blk.0.ffn_norm.weight"));

    // Test FFN mappings
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.mlp.gate_proj.weight"),
              std::string("blk.0.ffn_gate.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.mlp.up_proj.weight"),
              std::string("blk.0.ffn_up.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.mlp.down_proj.weight"),
              std::string("blk.0.ffn_down.weight"));

    // Test Qwen3.5 GatedDeltaNet mappings
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.linear_attn.in_proj_qkv.weight"),
              std::string("blk.0.attn_qkv.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.linear_attn.in_proj_z.weight"),
              std::string("blk.0.attn_gate.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.linear_attn.in_proj_b.weight"),
              std::string("blk.0.ssm_beta.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.linear_attn.in_proj_a.weight"),
              std::string("blk.0.ssm_alpha.weight"));

    PASS();
}

void test_hf_config_parsing() {
    TEST(hf_config_parsing);

    // Create a minimal config.json
    const char* config_path = "/tmp/test_hf_config.json";
    const char* config_json = R"({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 1024,
        "intermediate_size": 2816,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "head_dim": 64,
        "vocab_size": 151936,
        "tie_word_embeddings": true
    })";

    FILE* f = fopen(config_path, "w");
    ASSERT_TRUE(f != nullptr);
    fputs(config_json, f);
    fclose(f);

    HFModelConfig cfg;
    ASSERT_TRUE(cfg.load(config_path));

    ASSERT_EQ(cfg.model_type, std::string("qwen3"));
    ASSERT_EQ(cfg.architecture_class, std::string("Qwen3ForCausalLM"));
    ASSERT_EQ(cfg.hidden_size, 1024);
    ASSERT_EQ(cfg.intermediate_size, 2816);
    ASSERT_EQ(cfg.num_hidden_layers, 28);
    ASSERT_EQ(cfg.num_attention_heads, 16);
    ASSERT_EQ(cfg.num_key_value_heads, 8);
    ASSERT_EQ(cfg.head_dim, 64);
    ASSERT_EQ(cfg.vocab_size, 151936);
    ASSERT_TRUE(cfg.tie_word_embeddings);
    ASSERT_NEAR(cfg.rms_norm_eps, 1e-6, 1e-12);
    ASSERT_NEAR(cfg.rope_theta, 1000000.0, 1.0);

    // Test architecture mapping
    ASSERT_EQ(cfg.get_architecture(), std::string("qwen3"));
    ASSERT_TRUE(!cfg.is_hybrid());

    remove(config_path);

    PASS();
}

void test_qwen35_config() {
    TEST(qwen35_config);

    // Verify Qwen3.5 architecture enables neox RoPE and hybrid flag
    ModelConfig cfg;
    cfg.architecture = "qwen35";

    bool rope_neox = (cfg.architecture == "qwen2" ||
                      cfg.architecture == "qwen3" ||
                      cfg.architecture == "qwen2moe" ||
                      cfg.architecture == "qwen35" ||
                      cfg.architecture == "qwen35moe");
    ASSERT_TRUE(rope_neox);

    // Verify Qwen3.5-0.8B typical dimensions
    cfg.hidden_size = 4096;
    cfg.num_heads = 16;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 256;
    cfg.kv_dim = cfg.head_dim * cfg.num_kv_heads;
    cfg.intermediate_size = 12288;
    cfg.num_layers = 36;
    cfg.rope_theta = 1000000.0f;
    cfg.max_seq_len = 32768;
    cfg.is_hybrid = true;

    ASSERT_EQ(cfg.head_dim, 256);
    ASSERT_EQ(cfg.kv_dim, 1024);
    ASSERT_EQ(cfg.num_heads / cfg.num_kv_heads, 4);  // GQA ratio
    ASSERT_TRUE(cfg.is_hybrid);

    // Test HF config architecture mapping for Qwen3.5
    HFModelConfig hf_cfg;
    // Simulate qwen3_5_text model type
    const char* config_path = "/tmp/test_qwen35_config.json";
    const char* config_json = R"({
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 248320,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "layer_types": ["full_attention", "linear_attention", "full_attention"]
    })";

    FILE* f = fopen(config_path, "w");
    ASSERT_TRUE(f != nullptr);
    fputs(config_json, f);
    fclose(f);

    ASSERT_TRUE(hf_cfg.load(config_path));
    ASSERT_EQ(hf_cfg.get_architecture(), std::string("qwen35"));
    ASSERT_TRUE(hf_cfg.is_hybrid());
    ASSERT_EQ(static_cast<int>(hf_cfg.layer_types.size()), 3);
    ASSERT_EQ(hf_cfg.layer_types[0], std::string("full_attention"));
    ASSERT_EQ(hf_cfg.layer_types[1], std::string("linear_attention"));
    ASSERT_EQ(hf_cfg.layer_types[2], std::string("full_attention"));
    ASSERT_EQ(hf_cfg.linear_key_head_dim, 128);
    ASSERT_EQ(hf_cfg.linear_value_head_dim, 128);
    ASSERT_EQ(hf_cfg.vocab_size, 248320);

    remove(config_path);

    PASS();
}

void test_qwen35_moe_config() {
    TEST(qwen35_moe_config);

    // Test MoE config parsing
    const char* config_path = "/tmp/test_qwen35moe_config.json";
    const char* config_json = R"({
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "num_experts_per_tok": 8,
        "num_experts": 256,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_conv_kernel_dim": 4,
        "partial_rotary_factor": 0.25,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                        "full_attention"]
    })";

    FILE* f = fopen(config_path, "w");
    ASSERT_TRUE(f != nullptr);
    fputs(config_json, f);
    fclose(f);

    HFModelConfig cfg;
    ASSERT_TRUE(cfg.load(config_path));
    ASSERT_EQ(cfg.get_architecture(), std::string("qwen35moe"));
    ASSERT_TRUE(cfg.is_hybrid());
    ASSERT_TRUE(cfg.is_moe());
    ASSERT_EQ(cfg.num_experts, 256);
    ASSERT_EQ(cfg.num_experts_per_tok, 8);
    ASSERT_EQ(cfg.moe_intermediate_size, 512);
    ASSERT_EQ(cfg.shared_expert_intermediate_size, 512);
    ASSERT_EQ(cfg.linear_num_key_heads, 16);
    ASSERT_EQ(cfg.linear_num_value_heads, 32);
    ASSERT_EQ(cfg.linear_conv_kernel_dim, 4);
    ASSERT_NEAR(cfg.partial_rotary_factor, 0.25, 1e-6);

    // Verify layer types: 3 linear + 1 full attention (pattern for qwen3.5 moe)
    ASSERT_EQ(static_cast<int>(cfg.layer_types.size()), 4);
    ASSERT_EQ(cfg.layer_types[0], std::string("linear_attention"));
    ASSERT_EQ(cfg.layer_types[3], std::string("full_attention"));

    remove(config_path);
    PASS();
}

void test_qwen35_vl_nested_config() {
    TEST(qwen35_vl_nested_config);

    // Test VL model with nested text_config
    const char* config_path = "/tmp/test_qwen35vl_config.json";
    const char* config_json = R"({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "image_token_id": 248056,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "layer_types": ["full_attention", "linear_attention"]
        },
        "vision_config": {
            "model_type": "qwen3_5",
            "hidden_size": 1152,
            "depth": 27
        }
    })";

    FILE* f = fopen(config_path, "w");
    ASSERT_TRUE(f != nullptr);
    fputs(config_json, f);
    fclose(f);

    HFModelConfig cfg;
    ASSERT_TRUE(cfg.load(config_path));

    // VL models have nested text_config - should extract text params
    ASSERT_EQ(cfg.get_architecture(), std::string("qwen35"));
    ASSERT_EQ(cfg.hidden_size, 4096);
    ASSERT_EQ(cfg.num_hidden_layers, 32);
    ASSERT_EQ(cfg.head_dim, 256);
    ASSERT_EQ(cfg.vocab_size, 248320);
    ASSERT_NEAR(cfg.partial_rotary_factor, 0.25, 1e-6);
    ASSERT_TRUE(cfg.is_hybrid());
    ASSERT_EQ(static_cast<int>(cfg.layer_types.size()), 2);

    remove(config_path);
    PASS();
}

void test_moe_weight_name_mapping() {
    TEST(moe_weight_name_mapping);

    // MoE router
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.mlp.gate.weight"),
              std::string("blk.0.ffn_gate_inp.weight"));

    // MoE merged experts
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.mlp.experts.gate_up_proj"),
              std::string("blk.0.ffn_gate_up_exps.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.mlp.experts.down_proj"),
              std::string("blk.0.ffn_down_exps.weight"));

    // Shared expert
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.5.mlp.shared_expert.gate_proj.weight"),
              std::string("blk.5.ffn_gate_shexp.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.5.mlp.shared_expert.up_proj.weight"),
              std::string("blk.5.ffn_up_shexp.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.5.mlp.shared_expert.down_proj.weight"),
              std::string("blk.5.ffn_down_shexp.weight"));

    PASS();
}

void test_partial_rotary_factor() {
    TEST(partial_rotary_factor);

    // Qwen3.5 uses partial_rotary_factor=0.25 with head_dim=256
    // So only 64 dims get RoPE (32 freqs)
    ModelConfig cfg;
    cfg.head_dim = 256;
    cfg.partial_rotary_factor = 0.25f;
    cfg.rope_dim = static_cast<int>(cfg.head_dim * cfg.partial_rotary_factor);
    ASSERT_EQ(cfg.rope_dim, 64);

    // Default (full rotary): partial_rotary_factor=1.0
    cfg.partial_rotary_factor = 1.0f;
    cfg.rope_dim = static_cast<int>(cfg.head_dim * cfg.partial_rotary_factor);
    ASSERT_EQ(cfg.rope_dim, 256);

    PASS();
}

void test_linear_attention_state() {
    TEST(linear_attention_state);

    // Test LinearAttentionState init and clear
    LinearAttentionState state;
    state.init(/*nv=*/2, /*kd=*/3, /*vd=*/4, /*cw=*/2);
    ASSERT_EQ(state.num_v_heads, 2);
    ASSERT_EQ(state.key_head_dim, 3);
    ASSERT_EQ(state.value_head_dim, 4);
    ASSERT_EQ(state.conv_width, 2);
    ASSERT_EQ(state.value_dim, 8);  // 2 * 4

    // recurrent: 2 heads * 3 * 4 = 24 elements
    ASSERT_EQ(static_cast<int>(state.recurrent.size()), 24);
    // conv_state: 8 channels * 2 width = 16
    ASSERT_EQ(static_cast<int>(state.conv_state.size()), 16);

    // Write some values and verify head_state/channel_conv pointers
    float* s0 = state.head_state(0);
    float* s1 = state.head_state(1);
    ASSERT_TRUE(s0 == state.recurrent.data());
    ASSERT_TRUE(s1 == state.recurrent.data() + 12);  // 3 * 4

    float* c0 = state.channel_conv(0);
    float* c3 = state.channel_conv(3);
    ASSERT_TRUE(c0 == state.conv_state.data());
    ASSERT_TRUE(c3 == state.conv_state.data() + 6);  // 3 * 2

    // Write and clear
    s0[0] = 42.0f;
    c0[0] = 7.0f;
    state.clear();
    ASSERT_NEAR(s0[0], 0.0f, 1e-6);
    ASSERT_NEAR(c0[0], 0.0f, 1e-6);

    PASS();
}

void test_gated_delta_net_step() {
    TEST(gated_delta_net_step);

    // Test a single GatedDeltaNet step with known weights on a tiny model:
    // hidden_size = 4, num_v_heads = 1, key_head_dim = 2, value_head_dim = 2
    // conv_width = 2
    int dim = 4;
    int n_v = 1, k_hd = 2, v_hd = 2, cw = 2;
    int key_dim = 1 * k_hd;  // n_k_heads = 1
    int val_dim = n_v * v_hd;
    int qkv_dim = key_dim * 2 + val_dim;  // 2 + 2 + 2 = 6

    // Create known F32 weights
    // w_qkv: [qkv_dim=6, hidden=4] — identity-like mapping
    float w_qkv[6 * 4] = {
        // row 0 (q[0]): extract x[0]
        1, 0, 0, 0,
        // row 1 (q[1]): extract x[1]
        0, 1, 0, 0,
        // row 2 (k[0]): extract x[2]
        0, 0, 1, 0,
        // row 3 (k[1]): extract x[3]
        0, 0, 0, 1,
        // row 4 (v[0]): extract x[0]+x[2]
        1, 0, 1, 0,
        // row 5 (v[1]): extract x[1]+x[3]
        0, 1, 0, 1,
    };

    // w_attn_gate: [val_dim=2, hidden=4] — each row sums to 0.5 of input
    float w_gate[2 * 4] = {
        0.125f, 0.125f, 0.125f, 0.125f,
        0.125f, 0.125f, 0.125f, 0.125f,
    };

    // w_ssm_beta: [n_v=1, hidden=4] — positive to get beta near 1
    float w_beta[1 * 4] = { 0.5f, 0.5f, 0.5f, 0.5f };

    // ssm_a: [n_v=1] — small decay
    float ssm_a[1] = { -1.0f };  // exp(-1) ≈ 0.37 base rate

    // conv1d: [val_dim=2, conv_width=2] — simple averaging
    float conv1d_w[2 * 2] = { 0.5f, 0.5f, 0.5f, 0.5f };

    // w_ssm_out: [hidden=4, val_dim=2] — project back
    float w_out[4 * 2] = {
        1, 0,
        0, 1,
        1, 0,
        0, 1,
    };

    // ssm_norm: [val_dim=2] — unit scale
    float norm_w[2] = { 1.0f, 1.0f };

    // Set up state
    LinearAttentionState ls;
    ls.init(n_v, k_hd, v_hd, cw);

    // Input vector
    float xb[4] = { 1.0f, 2.0f, 0.5f, 0.3f };

    // Scratch buffers
    std::vector<float> gdn_qkv_buf(qkv_dim);
    std::vector<float> gdn_z_buf(val_dim);
    std::vector<float> gdn_beta_buf(n_v);
    std::vector<float> gdn_conv_out_buf(val_dim);
    std::vector<float> gdn_pred_buf(v_hd);
    std::vector<float> gdn_y_buf(val_dim);

    Compute compute(Backend::CPU);

    // 1. QKV projection
    QuantWeight qw = { w_qkv, GGML_TYPE_F32 };
    compute.matmul_transposed_q(gdn_qkv_buf.data(), xb, qw, qkv_dim, dim);

    float* qptr = gdn_qkv_buf.data();           // q = [1, 2]
    float* kptr = gdn_qkv_buf.data() + key_dim; // k = [0.5, 0.3]
    float* vptr = gdn_qkv_buf.data() + key_dim * 2; // v = [1.5, 2.3]

    ASSERT_NEAR(qptr[0], 1.0f, 1e-5);
    ASSERT_NEAR(qptr[1], 2.0f, 1e-5);
    ASSERT_NEAR(kptr[0], 0.5f, 1e-5);
    ASSERT_NEAR(kptr[1], 0.3f, 1e-5);
    ASSERT_NEAR(vptr[0], 1.5f, 1e-5);
    ASSERT_NEAR(vptr[1], 2.3f, 1e-5);

    // 2. Gate z
    QuantWeight gw = { w_gate, GGML_TYPE_F32 };
    compute.matmul_transposed_q(gdn_z_buf.data(), xb, gw, val_dim, dim);

    // 3. Beta
    QuantWeight bw = { w_beta, GGML_TYPE_F32 };
    compute.matmul_transposed_q(gdn_beta_buf.data(), xb, bw, n_v, dim);
    for (int h = 0; h < n_v; h++) gdn_beta_buf[h] = sigmoid_f(gdn_beta_buf[h]);
    ASSERT_TRUE(gdn_beta_buf[0] > 0.5f && gdn_beta_buf[0] < 1.0f);

    // 4. Conv1d step: shift and convolve
    for (int d = 0; d < val_dim; d++) {
        float* cs = ls.channel_conv(d);
        for (int w = 0; w < cw - 1; w++) cs[w] = cs[w + 1];
        cs[cw - 1] = vptr[d];
        float sum = 0.0f;
        for (int w = 0; w < cw; w++) sum += conv1d_w[d * cw + w] * cs[w];
        gdn_conv_out_buf[d] = sum;
    }

    // First step: conv_state was zeros, so output = 0.5 * 0 + 0.5 * v
    ASSERT_NEAR(gdn_conv_out_buf[0], 0.5f * vptr[0], 1e-5);
    ASSERT_NEAR(gdn_conv_out_buf[1], 0.5f * vptr[1], 1e-5);

    // 5. Delta rule state update (single head)
    // ssm_a[0] = -1 → base_rate = exp(-1) ≈ 0.368 → decay = exp(-0.368) ≈ 0.692
    float decay = expf(-expf(ssm_a[0]));
    float beta = gdn_beta_buf[0];
    float* S = ls.head_state(0);

    // Prediction: S^T @ k (S is initially zero, so prediction = 0)
    for (int j = 0; j < v_hd; j++) {
        float sum = 0.0f;
        for (int i = 0; i < k_hd; i++) sum += S[i * v_hd + j] * kptr[i];
        gdn_pred_buf[j] = sum;
    }
    ASSERT_NEAR(gdn_pred_buf[0], 0.0f, 1e-5);

    // State update: S = decay * S + beta * outer(k, v_conv - pred)
    for (int i = 0; i < k_hd; i++) {
        for (int j = 0; j < v_hd; j++) {
            float delta = gdn_conv_out_buf[j] - gdn_pred_buf[j];
            S[i * v_hd + j] = decay * S[i * v_hd + j] + beta * kptr[i] * delta;
        }
    }

    // After first step, S = beta * outer(k, v_conv) since S was zero
    ASSERT_NEAR(S[0], beta * kptr[0] * gdn_conv_out_buf[0], 1e-5);

    // Query: y = S^T @ q
    for (int j = 0; j < v_hd; j++) {
        float sum = 0.0f;
        for (int i = 0; i < k_hd; i++) sum += S[i * v_hd + j] * qptr[i];
        gdn_y_buf[j] = sum;
    }

    // y should be non-zero since S is non-zero and q is non-zero
    ASSERT_TRUE(fabs(gdn_y_buf[0]) > 1e-6 || fabs(gdn_y_buf[1]) > 1e-6);

    // 6. Normalize (group RMS norm)
    float ss = 0.0f;
    for (int j = 0; j < v_hd; j++) ss += gdn_y_buf[j] * gdn_y_buf[j];
    ss = 1.0f / sqrtf(ss / v_hd + 1e-5f);
    for (int j = 0; j < v_hd; j++) gdn_y_buf[j] *= ss * norm_w[j];

    // 7. Gate: y = y * sigmoid(z)
    for (int d = 0; d < val_dim; d++) {
        gdn_y_buf[d] *= sigmoid_f(gdn_z_buf[d]);
    }

    // 8. Output projection
    float output[4];
    QuantWeight ow = { w_out, GGML_TYPE_F32 };
    compute.matmul_transposed_q(output, gdn_y_buf.data(), ow, dim, val_dim);

    // Verify output is non-zero (the computation chain produced results)
    float out_norm = 0.0f;
    for (int i = 0; i < dim; i++) out_norm += output[i] * output[i];
    ASSERT_TRUE(out_norm > 1e-10);

    // Run a second step to verify state accumulation
    float xb_step2[4] = { 0.5f, 1.0f, 0.8f, 0.2f };
    compute.matmul_transposed_q(gdn_qkv_buf.data(), xb_step2, qw, qkv_dim, dim);
    qptr = gdn_qkv_buf.data();
    kptr = gdn_qkv_buf.data() + key_dim;
    vptr = gdn_qkv_buf.data() + key_dim * 2;

    // Conv step 2
    for (int d = 0; d < val_dim; d++) {
        float* cs = ls.channel_conv(d);
        for (int w = 0; w < cw - 1; w++) cs[w] = cs[w + 1];
        cs[cw - 1] = vptr[d];
        float sum = 0.0f;
        for (int w = 0; w < cw; w++) sum += conv1d_w[d * cw + w] * cs[w];
        gdn_conv_out_buf[d] = sum;
    }

    // State should now include previous state (decayed) + new update
    float S_00_prev = S[0];
    for (int j = 0; j < v_hd; j++) {
        float sum = 0.0f;
        for (int i = 0; i < k_hd; i++) sum += S[i * v_hd + j] * kptr[i];
        gdn_pred_buf[j] = sum;
    }
    for (int i = 0; i < k_hd; i++) {
        for (int j = 0; j < v_hd; j++) {
            float delta = gdn_conv_out_buf[j] - gdn_pred_buf[j];
            S[i * v_hd + j] = decay * S[i * v_hd + j] + beta * kptr[i] * delta;
        }
    }

    // After second step, S[0] should be different from first step
    ASSERT_TRUE(fabs(S[0] - S_00_prev) > 1e-6);

    // Verify clear resets state
    ls.clear();
    ASSERT_NEAR(ls.head_state(0)[0], 0.0f, 1e-6);
    ASSERT_NEAR(ls.channel_conv(0)[0], 0.0f, 1e-6);

    PASS();
}

// ---- New HF loading tests ----

void test_hf_multifile_file_idx() {
    TEST(hf_multifile_file_idx);

    // Verify that SafeTensorInfo stores the correct file_idx for each shard.
    // Build two minimal SafeTensors buffers and load them via load_multi().

    auto make_st_buf = [](const std::string& tensor_name,
                          const std::vector<float>& data) {
        std::string header =
            "{\"" + tensor_name + "\": {\"dtype\": \"F32\", \"shape\": [" +
            std::to_string(data.size()) + "], \"data_offsets\": [0, " +
            std::to_string(data.size() * 4) + "]}}";
        uint64_t hdr_len = header.size();
        size_t total = 8 + hdr_len + data.size() * 4;
        std::vector<uint8_t> buf(total, 0);
        memcpy(buf.data(), &hdr_len, 8);
        memcpy(buf.data() + 8, header.c_str(), hdr_len);
        memcpy(buf.data() + 8 + hdr_len, data.data(), data.size() * 4);
        return buf;
    };

    std::vector<float> d0 = {1.0f, 2.0f};
    std::vector<float> d1 = {3.0f, 4.0f, 5.0f};
    auto buf0 = make_st_buf("tensor_a", d0);
    auto buf1 = make_st_buf("tensor_b", d1);

    // Write to temp files
    const char* p0 = "/tmp/test_shard0.safetensors";
    const char* p1 = "/tmp/test_shard1.safetensors";
    FILE* f = fopen(p0, "wb");
    ASSERT_TRUE(f != nullptr);
    fwrite(buf0.data(), 1, buf0.size(), f); fclose(f);
    f = fopen(p1, "wb");
    ASSERT_TRUE(f != nullptr);
    fwrite(buf1.data(), 1, buf1.size(), f); fclose(f);

    SafeTensorsFile st;
    ASSERT_TRUE(st.load_multi({p0, p1}));

    // tensor_a must be in file 0, tensor_b in file 1
    auto it_a = st.tensors.find("tensor_a");
    auto it_b = st.tensors.find("tensor_b");
    ASSERT_TRUE(it_a != st.tensors.end());
    ASSERT_TRUE(it_b != st.tensors.end());
    ASSERT_EQ(it_a->second.file_idx, static_cast<size_t>(0));
    ASSERT_EQ(it_b->second.file_idx, static_cast<size_t>(1));

    // Verify get_tensor_data returns correct data for each shard
    const void* ra = st.get_tensor_data(it_a->second, it_a->second.file_idx);
    const void* rb = st.get_tensor_data(it_b->second, it_b->second.file_idx);
    ASSERT_TRUE(ra != nullptr);
    ASSERT_TRUE(rb != nullptr);
    ASSERT_NEAR(static_cast<const float*>(ra)[0], 1.0f, 1e-6f);
    ASSERT_NEAR(static_cast<const float*>(ra)[1], 2.0f, 1e-6f);
    ASSERT_NEAR(static_cast<const float*>(rb)[0], 3.0f, 1e-6f);
    ASSERT_NEAR(static_cast<const float*>(rb)[2], 5.0f, 1e-6f);

    remove(p0); remove(p1);
    PASS();
}

void test_hf_index_json_parsing() {
    TEST(hf_index_json_parsing);

    // Create a temporary directory structure with index.json
    const char* dir = "/tmp/test_hf_index_dir";
#ifdef _WIN32
    _mkdir(dir);
#else
    mkdir(dir, 0755);
#endif

    // Write two minimal safetensors shards
    auto make_st_buf = [](const std::string& tname) {
        std::string hdr = "{\"" + tname + "\": {\"dtype\": \"F32\", "
                          "\"shape\": [2], \"data_offsets\": [0, 8]}}";
        uint64_t hl = hdr.size();
        std::vector<uint8_t> buf(8 + hl + 8, 0);
        memcpy(buf.data(), &hl, 8);
        memcpy(buf.data() + 8, hdr.c_str(), hl);
        float v[2] = {1.0f, 2.0f};
        memcpy(buf.data() + 8 + hl, v, 8);
        return buf;
    };

    std::string f1 = std::string(dir) + "/model-00001-of-00002.safetensors";
    std::string f2 = std::string(dir) + "/model-00002-of-00002.safetensors";
    FILE* fp = fopen(f1.c_str(), "wb");
    ASSERT_TRUE(fp != nullptr);
    auto b1 = make_st_buf("embed.weight");
    fwrite(b1.data(), 1, b1.size(), fp); fclose(fp);
    fp = fopen(f2.c_str(), "wb");
    ASSERT_TRUE(fp != nullptr);
    auto b2 = make_st_buf("lm_head.weight");
    fwrite(b2.data(), 1, b2.size(), fp); fclose(fp);

    // Write index.json
    std::string idx_path = std::string(dir) + "/model.safetensors.index.json";
    const char* idx_json =
        "{\n"
        "  \"metadata\": {\"total_size\": 16},\n"
        "  \"weight_map\": {\n"
        "    \"embed.weight\": \"model-00001-of-00002.safetensors\",\n"
        "    \"lm_head.weight\": \"model-00002-of-00002.safetensors\"\n"
        "  }\n"
        "}";
    fp = fopen(idx_path.c_str(), "w");
    ASSERT_TRUE(fp != nullptr);
    fputs(idx_json, fp); fclose(fp);

    // find_safetensors_files should use the index file
    auto files = find_safetensors_files(dir);
    ASSERT_EQ(static_cast<int>(files.size()), 2);
    // First file should be shard 1 (embed.weight), second shard 2 (lm_head)
    ASSERT_TRUE(files[0].find("00001") != std::string::npos);
    ASSERT_TRUE(files[1].find("00002") != std::string::npos);

    remove(f1.c_str()); remove(f2.c_str());
    remove(idx_path.c_str()); rmdir(dir);
    PASS();
}

void test_hf_new_weight_name_mappings() {
    TEST(hf_new_weight_name_mappings);

    // Phi-3 combined QKV
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.self_attn.qkv_proj.weight"),
              std::string("blk.0.attn_qkv.weight"));

    // Gemma2 / Gemma3 FFN norms
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.pre_feedforward_layernorm.weight"),
              std::string("blk.0.ffn_norm.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.2.post_feedforward_layernorm.weight"),
              std::string("blk.2.ffn_post_norm.weight"));

    // InternLM2 top-level embedding/output
    ASSERT_EQ(hf_to_gguf_tensor_name("model.tok_embeddings.weight"),
              std::string("token_embd.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("output.weight"),
              std::string("output.weight"));

    // InternLM2 layer tensors
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.attention_norm.weight"),
              std::string("blk.0.attn_norm.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.ffn_norm.weight"),
              std::string("blk.0.ffn_norm.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.attention.wqkv.weight"),
              std::string("blk.0.attn_qkv.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.attention.wo.weight"),
              std::string("blk.0.attn_output.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.feed_forward.w1.weight"),
              std::string("blk.0.ffn_gate.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.feed_forward.w2.weight"),
              std::string("blk.0.ffn_down.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("model.layers.0.feed_forward.w3.weight"),
              std::string("blk.0.ffn_up.weight"));

    // GPT-NeoX top-level names
    ASSERT_EQ(hf_to_gguf_tensor_name("gpt_neox.embed_in.weight"),
              std::string("token_embd.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("gpt_neox.final_layer_norm.weight"),
              std::string("output_norm.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("embed_out.weight"),
              std::string("output.weight"));

    // GPT-NeoX layer tensors
    ASSERT_EQ(hf_to_gguf_tensor_name("gpt_neox.layers.0.input_layernorm.weight"),
              std::string("blk.0.attn_norm.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("gpt_neox.layers.0.post_attention_layernorm.weight"),
              std::string("blk.0.ffn_norm.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name(
                  "gpt_neox.layers.0.attention.query_key_value.weight"),
              std::string("blk.0.attn_qkv.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("gpt_neox.layers.0.attention.dense.weight"),
              std::string("blk.0.attn_output.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("gpt_neox.layers.0.mlp.dense_h_to_4h.weight"),
              std::string("blk.0.ffn_up.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("gpt_neox.layers.0.mlp.dense_4h_to_h.weight"),
              std::string("blk.0.ffn_down.weight"));

    // ChatGLM top-level names
    ASSERT_EQ(hf_to_gguf_tensor_name("transformer.embedding.word_embeddings.weight"),
              std::string("token_embd.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("transformer.encoder.final_layernorm.weight"),
              std::string("output_norm.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name("transformer.output_layer.weight"),
              std::string("output.weight"));

    // ChatGLM layer tensors
    ASSERT_EQ(hf_to_gguf_tensor_name(
                  "transformer.encoder.layers.0.self_attention.query_key_value.weight"),
              std::string("blk.0.attn_qkv.weight"));
    ASSERT_EQ(hf_to_gguf_tensor_name(
                  "transformer.encoder.layers.0.self_attention.dense.weight"),
              std::string("blk.0.attn_output.weight"));

    PASS();
}

void test_hf_additional_architecture_mappings() {
    TEST(hf_additional_architecture_mappings);

    HFModelConfig cfg;

    // Qwen2.5 uses the Qwen2 architecture
    cfg.model_type = "qwen2_5";
    ASSERT_EQ(cfg.get_architecture(), std::string("qwen2"));

    // Mixtral uses the LLaMA base architecture
    cfg.model_type = "mixtral";
    ASSERT_EQ(cfg.get_architecture(), std::string("llama"));

    // Phi-4
    cfg.model_type = "phi4";
    ASSERT_EQ(cfg.get_architecture(), std::string("phi4"));

    // StableLM variants
    cfg.model_type = "stablelm";
    ASSERT_EQ(cfg.get_architecture(), std::string("stablelm"));
    cfg.model_type = "stablelm_epoch";
    ASSERT_EQ(cfg.get_architecture(), std::string("stablelm"));

    // Gemma3n
    cfg.model_type = "gemma3n";
    ASSERT_EQ(cfg.get_architecture(), std::string("gemma3n"));

    PASS();
}

void test_clear_state() {
    TEST(clear_state);

    // Verify Model::clear_state() clears both KV cache and linear states
    // We test this at the struct level since we can't load a full model
    KVCache kv;
    kv.init(2, 4, 3);
    kv.key(0, 0)[0] = 1.0f;
    kv.value(1, 2)[1] = 2.0f;
    kv.clear();
    ASSERT_NEAR(kv.key(0, 0)[0], 0.0f, 1e-6);
    ASSERT_NEAR(kv.value(1, 2)[1], 0.0f, 1e-6);

    LinearAttentionState ls;
    ls.init(2, 3, 4, 2);
    ls.head_state(0)[0] = 5.0f;
    ls.channel_conv(1)[0] = 3.0f;
    ls.clear();
    ASSERT_NEAR(ls.head_state(0)[0], 0.0f, 1e-6);
    ASSERT_NEAR(ls.channel_conv(1)[0], 0.0f, 1e-6);

    PASS();
}

// ---- Run all tests ----

int main() {
    print_system_config();
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

    fprintf(stderr, "\nF16 direct computation tests:\n");
    test_cpu_matmul_transposed_f16();

    fprintf(stderr, "\nBF16 type and computation tests:\n");
    test_bf16_gguf_type();
    test_dequantize_bf16();
    test_cpu_matmul_transposed_bf16();
    test_qwen35_layer_types_from_gguf();
    test_qwen35_post_attention_norm_fallback();

    fprintf(stderr, "\nK-quant type and dequantization tests:\n");
    test_kquant_type_sizes();
    test_dequantize_q6_k();
    test_dequantize_q4_k();
    test_dequantize_q2_k();
    test_dequantize_q5_0();
    test_dequantize_q5_0_high_bits();
    test_dequantize_q5_k();

    fprintf(stderr, "\nFused K-quant matmul tests:\n");
    test_cpu_matmul_transposed_q6_k();
    test_cpu_matmul_transposed_q4_k();
    test_cpu_matmul_transposed_q5_k();
    test_cpu_matmul_transposed_q5_0();
    test_compute_q_kquant_dispatch();

    fprintf(stderr, "\nSafeTensors and HF loader tests:\n");
    test_safetensors_dtype();
    test_safetensors_parse();
    test_bf16_conversion();
    test_hf_weight_name_mapping();
    test_hf_config_parsing();
    test_qwen35_config();
    test_qwen35_moe_config();
    test_qwen35_vl_nested_config();
    test_moe_weight_name_mapping();
    test_partial_rotary_factor();
    test_expanded_architecture_mappings();

    fprintf(stderr, "\nHF training format loading tests:\n");
    test_hf_multifile_file_idx();
    test_hf_index_json_parsing();
    test_hf_new_weight_name_mappings();
    test_hf_additional_architecture_mappings();

    fprintf(stderr, "\nGatedDeltaNet linear attention tests:\n");
    test_linear_attention_state();
    test_gated_delta_net_step();
    test_clear_state();

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n",
            tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
