#ifdef LLM_USE_CUDA

// CUDA GPU kernels for LLM inference operations.
// Optimized for high-throughput inference with persistent memory management.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <memory>

#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(1); \
    } \
} while(0)

// ---- GPU memory pool for efficient allocation ----

class CUDAMemoryPool {
public:
    static CUDAMemoryPool& instance() {
        static CUDAMemoryPool pool;
        return pool;
    }

    void* allocate(size_t size) {
        // Round up to 256-byte alignment
        size_t aligned_size = ((size + 255) / 256) * 256;
        
        // Try to find a suitable free block
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (it->second >= aligned_size) {
                void* ptr = it->first;
                free_blocks_.erase(it);
                used_blocks_[ptr] = aligned_size;
                return ptr;
            }
        }
        
        // Allocate new block
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, aligned_size));
        used_blocks_[ptr] = aligned_size;
        return ptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        auto it = used_blocks_.find(ptr);
        if (it != used_blocks_.end()) {
            free_blocks_[ptr] = it->second;
            used_blocks_.erase(it);
        }
    }

    void clear() {
        for (auto& block : used_blocks_) {
            cudaFree(block.first);
        }
        used_blocks_.clear();
        free_blocks_.clear();
    }

private:
    CUDAMemoryPool() = default;
    ~CUDAMemoryPool() { clear(); }
    
    std::unordered_map<void*, size_t> used_blocks_;
    std::unordered_map<void*, size_t> free_blocks_;
};

// Helper functions for pooled memory allocation
inline void* cuda_malloc_pooled(size_t size) {
    return CUDAMemoryPool::instance().allocate(size);
}

inline void cuda_free_pooled(void* ptr) {
    CUDAMemoryPool::instance().deallocate(ptr);
}

// ---- Persistent weight storage on GPU ----

struct GPUWeightCache {
    void* d_data = nullptr;
    size_t size = 0;
    bool is_cached = false;
    
    void cache(const void* h_data, size_t bytes) {
        if (is_cached && size == bytes) {
            CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
        } else {
            release();
            size = bytes;
            CUDA_CHECK(cudaMalloc(&d_data, bytes));
            CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
            is_cached = true;
        }
    }
    
    void release() {
        if (d_data) {
            cudaFree(d_data);
            d_data = nullptr;
        }
        size = 0;
        is_cached = false;
    }
};

// Global weight cache (keyed by tensor pointer)
static std::unordered_map<const void*, GPUWeightCache> g_weight_cache;

inline void* get_cached_weight(const void* h_ptr, size_t bytes) {
    auto& cache = g_weight_cache[h_ptr];
    if (!cache.is_cached || cache.size != bytes) {
        cache.cache(h_ptr, bytes);
    } else {
        // Update cache in case weights changed
        CUDA_CHECK(cudaMemcpy(cache.d_data, h_ptr, bytes, cudaMemcpyHostToDevice));
    }
    return cache.d_data;
}

// ---- CUDA streams for asynchronous execution ----

static cudaStream_t compute_stream = nullptr;

static void ensure_stream() {
    if (!compute_stream) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
    }
}

// ---- GPU availability check ----

// Print information about all available CUDA devices to stderr.
// Returns true if at least one CUDA-capable device is present.
bool cuda_print_gpu_info() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "GPU:  (none detected)\n");
        return false;
    }

    for (int id = 0; id < device_count; id++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, id) != cudaSuccess) {
            fprintf(stderr, "GPU %d: (failed to query device properties)\n", id);
            continue;
        }
        fprintf(stderr, "GPU %d: %s (compute %d.%d, %.0f MB, %d SMs)\n",
                id, prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / (1024.0 * 1024.0),
                prop.multiProcessorCount);
    }
    fprintf(stderr, "CUDA: %d.%d\n",
            CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    return true;
}

// Quiet GPU availability check (no output). Returns true if a CUDA GPU is present.
bool cuda_check_gpu() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

static cublasHandle_t cublas_handle = nullptr;

static void ensure_cublas() {
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
        cublasSetStream(cublas_handle, compute_stream);
    }
}

// ---- Optimized matrix multiplication using cuBLAS with persistent memory ----

// Compute out[1×N] = x[1×K] × W^T where W is row-major [N×K].
// Uses persistent GPU memory for weights to avoid repeated transfers.
void cuda_matmul_transposed_weight(float* out, const float* x,
                                   const float* w, int N, int K) {
    ensure_cublas();
    ensure_stream();

    float *d_x, *d_w, *d_out;
    size_t x_size = K * sizeof(float);
    size_t w_size = (size_t)N * K * sizeof(float);
    size_t out_size = N * sizeof(float);

    // Allocate from memory pool (faster than cudaMalloc)
    d_x = static_cast<float*>(cuda_malloc_pooled(x_size));
    d_w = static_cast<float*>(cuda_malloc_pooled(w_size));
    d_out = static_cast<float*>(cuda_malloc_pooled(out_size));

    // Copy input vector to GPU
    CUDA_CHECK(cudaMemcpyAsync(d_x, x, x_size, cudaMemcpyHostToDevice, compute_stream));
    
    // Use cached weight pointer if available, otherwise copy
    auto it = g_weight_cache.find(w);
    if (it != g_weight_cache.end() && it->second.is_cached) {
        d_w = static_cast<float*>(it->second.d_data);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_w, w, w_size, cudaMemcpyHostToDevice, compute_stream));
        g_weight_cache[w].cache(w, w_size);
    }

    float alpha = 1.0f, beta = 0.0f;
    // cuBLAS uses column-major, so we compute B^T * A^T = (A*B)^T
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, 1, K, &alpha, d_w, K, d_x, K, &beta, d_out, N);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpyAsync(out, d_out, out_size, cudaMemcpyDeviceToHost, compute_stream));
    
    // Synchronize to ensure completion
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    // Return memory to pool
    cuda_free_pooled(d_x);
    cuda_free_pooled(d_out);
}

// ---- RMS Norm kernel with persistent memory ----

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
    ensure_stream();
    
    float *d_out, *d_x, *d_w;
    size_t buf_size = n * sizeof(float);
    
    // Use pooled memory allocation
    d_out = static_cast<float*>(cuda_malloc_pooled(buf_size));
    d_x = static_cast<float*>(cuda_malloc_pooled(buf_size));
    d_w = static_cast<float*>(cuda_malloc_pooled(buf_size));

    CUDA_CHECK(cudaMemcpyAsync(d_x, x, buf_size, cudaMemcpyHostToDevice, compute_stream));
    
    // Check for cached weights
    auto it = g_weight_cache.find(w);
    if (it != g_weight_cache.end() && it->second.is_cached) {
        d_w = static_cast<float*>(it->second.d_data);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_w, w, buf_size, cudaMemcpyHostToDevice, compute_stream));
        g_weight_cache[w].cache(w, buf_size);
    }

    int threads = 256;
    rmsnorm_kernel<<<1, threads, threads * sizeof(float)>>>(d_out, d_x, d_w, n, eps);

    CUDA_CHECK(cudaMemcpyAsync(out, d_out, buf_size, cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    
    cuda_free_pooled(d_out);
    cuda_free_pooled(d_x);
}

// ---- Softmax kernel with persistent memory ----

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
    ensure_stream();
    
    float* d_x;
    size_t buf_size = n * sizeof(float);
    d_x = static_cast<float*>(cuda_malloc_pooled(buf_size));

    CUDA_CHECK(cudaMemcpyAsync(d_x, x, buf_size, cudaMemcpyHostToDevice, compute_stream));

    int threads = 256;
    softmax_kernel<<<1, threads, threads * sizeof(float)>>>(d_x, n);

    CUDA_CHECK(cudaMemcpyAsync(x, d_x, buf_size, cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    cuda_free_pooled(d_x);
}

// ---- SiLU element-wise multiply kernel with persistent memory ----

__global__ void silu_mul_kernel(float* out, const float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

void cuda_silu_elementwise_mul(float* out, const float* gate, const float* up, int n) {
    ensure_stream();
    
    float *d_out, *d_gate, *d_up;
    size_t buf_size = n * sizeof(float);
    
    d_out = static_cast<float*>(cuda_malloc_pooled(buf_size));
    d_gate = static_cast<float*>(cuda_malloc_pooled(buf_size));
    d_up = static_cast<float*>(cuda_malloc_pooled(buf_size));

    CUDA_CHECK(cudaMemcpyAsync(d_gate, gate, buf_size, cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_up, up, buf_size, cudaMemcpyHostToDevice, compute_stream));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads>>>(d_out, d_gate, d_up, n);

    CUDA_CHECK(cudaMemcpyAsync(out, d_out, buf_size, cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    
    cuda_free_pooled(d_out);
    cuda_free_pooled(d_gate);
    cuda_free_pooled(d_up);
}

// ---- RoPE kernel with persistent memory ----

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
    ensure_stream();
    
    float *d_q, *d_k;
    size_t q_size = q_dim * sizeof(float);
    size_t k_size = k_dim * sizeof(float);
    
    d_q = static_cast<float*>(cuda_malloc_pooled(q_size));
    d_k = static_cast<float*>(cuda_malloc_pooled(k_size));

    CUDA_CHECK(cudaMemcpyAsync(d_q, q, q_size, cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_k, k, k_size, cudaMemcpyHostToDevice, compute_stream));

    int threads = 256;
    int q_blocks = (q_dim / 2 + threads - 1) / threads;
    int k_blocks = (k_dim / 2 + threads - 1) / threads;
    rope_kernel<<<q_blocks, threads>>>(d_q, q_dim, head_dim, pos, theta);
    rope_kernel<<<k_blocks, threads>>>(d_k, k_dim, head_dim, pos, theta);

    CUDA_CHECK(cudaMemcpyAsync(q, d_q, q_size, cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(k, d_k, k_size, cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    
    cuda_free_pooled(d_q);
    cuda_free_pooled(d_k);
}

// ---- Element-wise add kernel with persistent memory ----

__global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

void cuda_add(float* out, const float* a, const float* b, int n) {
    ensure_stream();
    
    float *d_out, *d_a, *d_b;
    size_t buf_size = n * sizeof(float);
    
    d_out = static_cast<float*>(cuda_malloc_pooled(buf_size));
    d_a = static_cast<float*>(cuda_malloc_pooled(buf_size));
    d_b = static_cast<float*>(cuda_malloc_pooled(buf_size));

    CUDA_CHECK(cudaMemcpyAsync(d_a, a, buf_size, cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, b, buf_size, cudaMemcpyHostToDevice, compute_stream));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_out, d_a, d_b, n);

    CUDA_CHECK(cudaMemcpyAsync(out, d_out, buf_size, cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    
    cuda_free_pooled(d_out);
    cuda_free_pooled(d_a);
    cuda_free_pooled(d_b);
}

#endif // LLM_USE_CUDA
