#ifdef LLM_USE_CUDA

// CUDA GPU kernels for LLM inference operations.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>

#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(1); \
    } \
} while(0)

// ---- GPU availability check ----

bool cuda_check_gpu() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA-capable GPU detected.\n");
        return false;
    }

    int device_id = 0;
    cudaDeviceProp prop;
    if (cudaGetDevice(&device_id) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        fprintf(stderr, "Failed to query CUDA device properties.\n");
        return false;
    }
    size_t free_mem = 0, total_mem_query = 0;
    bool mem_info_ok = (cudaMemGetInfo(&free_mem, &total_mem_query) == cudaSuccess);
    if (mem_info_ok) {
        fprintf(stderr, "GPU: %s (compute %d.%d, %.0f MB total, %.0f MB free, %d SMs)\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / (1024.0 * 1024.0),
                free_mem / (1024.0 * 1024.0),
                prop.multiProcessorCount);
    } else {
        fprintf(stderr, "GPU: %s (compute %d.%d, %.0f MB, %d SMs)\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / (1024.0 * 1024.0),
                prop.multiProcessorCount);
    }
    return true;
}

static cublasHandle_t cublas_handle = nullptr;

static void ensure_cublas() {
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }
}

// ---- Matrix multiplication using cuBLAS ----

void cuda_matmul(float* out, const float* a, const float* b, int M, int N, int K) {
    ensure_cublas();

    float *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;
    // cuBLAS uses column-major, so we compute B^T * A^T = (A*B)^T
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, d_b, N, d_a, K, &beta, d_out, N);

    CUDA_CHECK(cudaMemcpy(out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

// Compute out[1×N] = x[1×K] × W^T where W is row-major [N×K].
// This is the transposed-weight matmul used for all projection layers.
// cuBLAS CUBLAS_OP_T on W (passed as column-major [K×N]) produces W^T [N×K].
void cuda_matmul_transposed_weight(float* out, const float* x,
                                   const float* w, int N, int K) {
    ensure_cublas();

    float *d_x, *d_w, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, (size_t)N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, x, K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f, beta = 0.0f;
    // W is row-major [N×K]. Passing to cuBLAS (column-major) with lda=K
    // makes cuBLAS interpret it as a column-major [K×N] matrix (= W^T).
    // CUBLAS_OP_T then transposes it back to [N×K], so the sgemm computes
    // the column-major result d_out[N×1] = W_cm[K×N]^T * d_x[K×1] = W * x^T,
    // which corresponds to the desired row-major out[1×N] = x * W^T.
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, 1, K, &alpha, d_w, K, d_x, K, &beta, d_out, N);

    CUDA_CHECK(cudaMemcpy(out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_out);
}

// ---- RMS Norm kernel ----

__global__ void rmsnorm_kernel(float* out, const float* x, const float* w, int n, float eps) {
    // Single block reduction for RMS computation
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += x[i] * x[i];
    }
    shared[tid] = sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    float scale = rsqrtf(shared[0] / n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = x[i] * scale * w[i];
    }
}

void cuda_rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    float *d_out, *d_x, *d_w;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w, n * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    rmsnorm_kernel<<<1, threads, threads * sizeof(float)>>>(d_out, d_x, d_w, n, eps);

    CUDA_CHECK(cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    cudaFree(d_x);
    cudaFree(d_w);
}

// ---- Softmax kernel ----

__global__ void softmax_kernel(float* x, int n) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;

    // Find max
    float max_val = -INFINITY;
    for (int i = tid; i < n; i += blockDim.x) {
        if (x[i] > max_val) max_val = x[i];
    }
    shared[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && shared[tid + s] > shared[tid])
            shared[tid] = shared[tid + s];
        __syncthreads();
    }
    max_val = shared[0];

    // Exp and sum
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    sum = shared[0];

    for (int i = tid; i < n; i += blockDim.x) {
        x[i] /= sum;
    }
}

void cuda_softmax(float* x, int n) {
    float* d_x;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    softmax_kernel<<<1, threads, threads * sizeof(float)>>>(d_x, n);

    CUDA_CHECK(cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_x);
}

// ---- SiLU element-wise multiply kernel ----

__global__ void silu_mul_kernel(float* out, const float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

void cuda_silu_elementwise_mul(float* out, const float* gate, const float* up, int n) {
    float *d_out, *d_gate, *d_up;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_gate, gate, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, up, n * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads>>>(d_out, d_gate, d_up, n);

    CUDA_CHECK(cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    cudaFree(d_gate);
    cudaFree(d_up);
}

// ---- RoPE kernel ----

__global__ void rope_kernel(float* data, int dim, int head_dim, int pos, float theta) {
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < dim) {
        int head_offset = i % head_dim;
        float freq = 1.0f / powf(theta, (float)head_offset / head_dim);
        float angle = pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        float d0 = data[i], d1 = data[i + 1];
        data[i]     = d0 * cos_val - d1 * sin_val;
        data[i + 1] = d0 * sin_val + d1 * cos_val;
    }
}

void cuda_rope(float* q, float* k, int q_dim, int k_dim, int head_dim, int pos, float theta) {
    float *d_q, *d_k;
    CUDA_CHECK(cudaMalloc(&d_q, q_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, k_dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_q, q, q_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k, k_dim * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int q_blocks = (q_dim / 2 + threads - 1) / threads;
    int k_blocks = (k_dim / 2 + threads - 1) / threads;
    rope_kernel<<<q_blocks, threads>>>(d_q, q_dim, head_dim, pos, theta);
    rope_kernel<<<k_blocks, threads>>>(d_k, k_dim, head_dim, pos, theta);

    CUDA_CHECK(cudaMemcpy(q, d_q, q_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k, d_k, k_dim * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_q);
    cudaFree(d_k);
}

// ---- Element-wise add kernel ----

__global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

void cuda_add(float* out, const float* a, const float* b, int n) {
    float *d_out, *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_out, d_a, d_b, n);

    CUDA_CHECK(cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    cudaFree(d_a);
    cudaFree(d_b);
}

#endif // LLM_USE_CUDA
