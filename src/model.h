#ifndef LLM_MODEL_H
#define LLM_MODEL_H

// LLaMA-style transformer model for inference.
// Loads weights from GGUF files or HuggingFace SafeTensors and runs forward
// pass on CPU or GPU.  Supports standard transformer and hybrid architectures
// (Qwen3.5 GatedDeltaNet + attention).

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gguf.h"
#include "hf_loader.h"
#include "tensor.h"
#include "tokenizer.h"

// ---- Model configuration ----

struct ModelConfig {
    std::string architecture;
    int vocab_size = 0;
    int hidden_size = 0;        // embedding dimension
    int intermediate_size = 0;  // feed-forward hidden size
    int num_layers = 0;
    int num_heads = 0;
    int num_kv_heads = 0;       // for GQA (grouped query attention)
    int max_seq_len = 2048;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int head_dim = 0;           // derived: hidden_size / num_heads
    int kv_dim = 0;             // derived: head_dim * num_kv_heads
    bool qkv_bias = false;      // Qwen3-style QKV bias
    bool rope_neox = false;     // neox-style (halved) RoPE for Qwen models
    float partial_rotary_factor = 1.0f;  // Qwen3.5: 0.25
    int rope_dim = 0;           // derived: head_dim * partial_rotary_factor

    // Qwen3.5 hybrid architecture
    bool is_hybrid = false;     // true for Qwen3.5 (mix of attention + GatedDeltaNet)
    std::vector<std::string> layer_types;  // "full_attention" or "linear_attention"
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    int linear_num_key_heads = 0;
    int linear_num_value_heads = 0;
    int linear_conv_kernel_dim = 0;

    // MoE (Mixture of Experts) parameters
    bool is_moe = false;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;
};

// ---- Transformer weights ----

struct LayerWeights {
    float* attn_norm = nullptr;     // [hidden_size]
    QuantWeight wq;                 // [num_heads * head_dim, hidden_size]
    QuantWeight wk;                 // [num_kv_heads * head_dim, hidden_size]
    QuantWeight wv;                 // [num_kv_heads * head_dim, hidden_size]
    QuantWeight wo;                 // [hidden_size, num_heads * head_dim]
    float* bq = nullptr;            // [num_heads * head_dim] (optional, Qwen3)
    float* bk = nullptr;            // [num_kv_heads * head_dim] (optional, Qwen3)
    float* bv = nullptr;            // [num_kv_heads * head_dim] (optional, Qwen3)
    float* attn_q_norm = nullptr;   // [head_dim] (optional, Qwen3 QK-norm)
    float* attn_k_norm = nullptr;   // [head_dim] (optional, Qwen3 QK-norm)
    float* ffn_norm = nullptr;      // [hidden_size]
    QuantWeight w_gate;             // [intermediate_size, hidden_size]
    QuantWeight w_up;               // [intermediate_size, hidden_size]
    QuantWeight w_down;             // [hidden_size, intermediate_size]

    // Qwen3.5 GatedDeltaNet (linear attention) layer weights
    std::string layer_type;         // "full_attention" or "linear_attention"
    QuantWeight w_qkv;              // [key_dim*2 + value_dim, hidden_size] (fused QKV)
    QuantWeight w_attn_gate;        // [value_dim, hidden_size] (gate z projection)
    QuantWeight w_ssm_beta;         // [num_v_heads, hidden_size]
    QuantWeight w_ssm_alpha;        // [num_v_heads, hidden_size]
    float* ssm_a = nullptr;         // [num_v_heads] (decay parameter)
    QuantWeight w_ssm_conv1d;       // [value_dim, 1, conv_width]
    QuantWeight w_ssm_dt;           // [num_v_heads, hidden_size]
    QuantWeight w_ssm_out;          // [hidden_size, value_dim]
    float* ssm_norm = nullptr;      // [value_dim]
};

struct ModelWeights {
    float* token_embd = nullptr;    // [vocab_size, hidden_size] (kept as F32 for embedding lookup)
    float* output_norm = nullptr;   // [hidden_size]
    QuantWeight output;             // [vocab_size, hidden_size]
    std::vector<LayerWeights> layers;
};

// ---- KV Cache ----

struct KVCache {
    std::vector<float> key_cache;   // [num_layers, max_seq_len, kv_dim]
    std::vector<float> value_cache; // [num_layers, max_seq_len, kv_dim]
    int max_seq_len = 0;
    int kv_dim = 0;
    int num_layers = 0;

    void init(int layers, int seq_len, int dim) {
        num_layers = layers;
        max_seq_len = seq_len;
        kv_dim = dim;
        key_cache.resize(static_cast<size_t>(layers) * seq_len * dim, 0.0f);
        value_cache.resize(static_cast<size_t>(layers) * seq_len * dim, 0.0f);
    }

    float* key(int layer, int pos) {
        return key_cache.data() + (static_cast<size_t>(layer) * max_seq_len + pos) * kv_dim;
    }

    float* value(int layer, int pos) {
        return value_cache.data() + (static_cast<size_t>(layer) * max_seq_len + pos) * kv_dim;
    }

    void clear() {
        std::fill(key_cache.begin(), key_cache.end(), 0.0f);
        std::fill(value_cache.begin(), value_cache.end(), 0.0f);
    }
};

// ---- Transformer model ----

class Model {
public:
    ModelConfig config;
    ModelWeights weights;
    KVCache kv_cache;
    Tokenizer tokenizer;
    Compute compute;
    GGUFFile gguf;

    // HuggingFace format support
    SafeTensorsFile safetensors;
    HFModelConfig hf_config;
    bool is_hf_model = false;

    // Dequantized weight storage
    std::vector<std::vector<float>> weight_storage;

    // Scratch buffers for inference
    std::vector<float> x;       // [hidden_size] current hidden state
    std::vector<float> xb;      // [hidden_size] buffer
    std::vector<float> xb2;     // [hidden_size] buffer
    std::vector<float> q;       // [num_heads * head_dim]
    std::vector<float> k;       // [num_kv_heads * head_dim]
    std::vector<float> v;       // [num_kv_heads * head_dim]
    std::vector<float> att;     // [num_heads, max_seq_len]
    std::vector<float> hb;      // [intermediate_size]
    std::vector<float> hb2;     // [intermediate_size]
    std::vector<float> logits;  // [vocab_size]
    std::vector<float> rope_freqs; // [head_dim/2] precomputed RoPE frequencies

    explicit Model(Backend backend) : compute(backend) {}

    bool load(const std::string& model_path, int context_override = 0) {
        fprintf(stderr, "Loading model from: %s\n", model_path.c_str());

        // Auto-detect format: directory = HuggingFace, file = GGUF
        if (is_directory(model_path)) {
            return load_from_hf(model_path, context_override);
        }

        if (!gguf.load(model_path)) {
            return false;
        }

        fprintf(stderr, "GGUF version: %u, tensors: %lu, metadata keys: %lu\n",
                gguf.version,
                static_cast<unsigned long>(gguf.tensor_count),
                static_cast<unsigned long>(gguf.metadata_kv_count));

        // Read model configuration from metadata
        if (!load_config()) return false;

        // Apply context length override
        apply_context_override(context_override);

        // Load tokenizer
        if (!tokenizer.load_from_gguf(gguf)) return false;

        // Allocate and load weights
        if (!load_weights()) return false;

        // Allocate inference buffers
        alloc_buffers();

        print_model_info();
        return true;
    }

    // Run forward pass for a single token at position pos
    float* forward(int token, int pos) {
        int dim = config.hidden_size;
        int kv_dim = config.kv_dim;
        int head_dim = config.head_dim;
        int num_heads = config.num_heads;
        int num_kv_heads = config.num_kv_heads;
        int kv_mul = num_heads / num_kv_heads; // GQA multiplier

        // Token embedding lookup
        memcpy(x.data(), weights.token_embd + token * dim, dim * sizeof(float));

        // Process each transformer layer
        for (int l = 0; l < config.num_layers; l++) {
            const LayerWeights& lw = weights.layers[l];

            // Attention norm (applies to both full and linear attention layers)
            compute.rmsnorm(xb.data(), x.data(), lw.attn_norm, dim,
                            config.rms_norm_eps);

            // Determine if this is a standard attention or linear attention layer
            bool is_full_attention = (lw.layer_type != "linear_attention") ||
                                     lw.wq.valid();

            if (is_full_attention) {
                // Standard transformer attention (or fallback for linear layers with Q/K/V)

                // QKV projections
                compute.matmul_transposed_q(q.data(), xb.data(), lw.wq,
                                            num_heads * head_dim, dim);
                compute.matmul_transposed_q(k.data(), xb.data(), lw.wk,
                                            num_kv_heads * head_dim, dim);
                compute.matmul_transposed_q(v.data(), xb.data(), lw.wv,
                                            num_kv_heads * head_dim, dim);

                // Add QKV bias (Qwen3-style models)
                if (lw.bq) compute.add(q.data(), q.data(), lw.bq, num_heads * head_dim);
                if (lw.bk) compute.add(k.data(), k.data(), lw.bk, kv_dim);
                if (lw.bv) compute.add(v.data(), v.data(), lw.bv, kv_dim);

                // Apply QK-norm: per-head RMSNorm on Q and K (Qwen3-style)
                if (lw.attn_q_norm) {
                    #ifdef LLM_USE_OPENMP
                    #pragma omp parallel for
                    #endif
                    for (int h = 0; h < num_heads; h++) {
                        compute.rmsnorm(q.data() + h * head_dim,
                                        q.data() + h * head_dim,
                                        lw.attn_q_norm, head_dim,
                                        config.rms_norm_eps);
                    }
                }
                if (lw.attn_k_norm) {
                    #ifdef LLM_USE_OPENMP
                    #pragma omp parallel for
                    #endif
                    for (int h = 0; h < num_kv_heads; h++) {
                        compute.rmsnorm(k.data() + h * head_dim,
                                        k.data() + h * head_dim,
                                        lw.attn_k_norm, head_dim,
                                        config.rms_norm_eps);
                    }
                }

                // Apply RoPE (separate Q and K dims for GQA correctness)
                compute.rope(q.data(), k.data(), num_heads * head_dim,
                             num_kv_heads * head_dim, head_dim,
                             pos, config.rope_theta, rope_freqs.data(), config.rope_neox);

                // Store K, V in cache
                memcpy(kv_cache.key(l, pos), k.data(), kv_dim * sizeof(float));
                memcpy(kv_cache.value(l, pos), v.data(), kv_dim * sizeof(float));

                // Multi-head attention
                #ifdef LLM_USE_OPENMP
                #pragma omp parallel for
                #endif
                for (int h = 0; h < num_heads; h++) {
                    float* q_h = q.data() + h * head_dim;
                    float* att_h = att.data() + h * config.max_seq_len;
                    int kv_h = h / kv_mul; // GQA: map query head to kv head

                    // Compute attention scores: Q * K^T / sqrt(head_dim)
                    for (int t = 0; t <= pos; t++) {
                        float* k_t = kv_cache.key(l, t) + kv_h * head_dim;
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            score += q_h[d] * k_t[d];
                        }
                        att_h[t] = score / sqrtf(static_cast<float>(head_dim));
                    }

                    // Softmax over attention scores
                    compute.softmax(att_h, pos + 1);

                    // Weighted sum of values
                    float* out_h = xb2.data() + h * head_dim;
                    memset(out_h, 0, head_dim * sizeof(float));
                    for (int t = 0; t <= pos; t++) {
                        float* v_t = kv_cache.value(l, t) + kv_h * head_dim;
                        float w = att_h[t];
                        for (int d = 0; d < head_dim; d++) {
                            out_h[d] += w * v_t[d];
                        }
                    }
                }

                // Output projection
                compute.matmul_transposed_q(xb.data(), xb2.data(), lw.wo, dim,
                                            num_heads * head_dim);

                // Residual connection
                compute.add(x.data(), x.data(), xb.data(), dim);
            }
            // else: Qwen3.5 GatedDeltaNet linear attention layer without Q/K/V
            //       weights - skip attention (identity residual) until full
            //       GatedDeltaNet implementation is added.

            // FFN norm
            compute.rmsnorm(xb.data(), x.data(), lw.ffn_norm, dim,
                            config.rms_norm_eps);

            // Feed-forward network (SwiGLU)
            compute.matmul_transposed_q(hb.data(), xb.data(), lw.w_gate,
                                        config.intermediate_size, dim);
            compute.matmul_transposed_q(hb2.data(), xb.data(), lw.w_up,
                                        config.intermediate_size, dim);
            compute.silu_mul(hb.data(), hb.data(), hb2.data(),
                             config.intermediate_size);
            compute.matmul_transposed_q(xb.data(), hb.data(), lw.w_down,
                                        dim, config.intermediate_size);

            // Residual connection
            compute.add(x.data(), x.data(), xb.data(), dim);
        }

        // Final norm
        compute.rmsnorm(x.data(), x.data(), weights.output_norm, dim,
                        config.rms_norm_eps);

        // Output projection to vocab
        compute.matmul_transposed_q(logits.data(), x.data(), weights.output,
                                    config.vocab_size, dim);

        return logits.data();
    }

    double count_parameters() const {
        double n = 0;
        n += static_cast<double>(config.vocab_size) * config.hidden_size; // token_embd
        n += config.hidden_size; // output_norm
        n += static_cast<double>(config.vocab_size) * config.hidden_size; // output

        for (int l = 0; l < config.num_layers; l++) {
            n += config.hidden_size; // attn_norm
            n += static_cast<double>(config.num_heads) * config.head_dim * config.hidden_size; // wq
            n += static_cast<double>(config.num_kv_heads) * config.head_dim * config.hidden_size; // wk
            n += static_cast<double>(config.num_kv_heads) * config.head_dim * config.hidden_size; // wv
            n += static_cast<double>(config.hidden_size) * config.num_heads * config.head_dim; // wo
            if (config.qkv_bias) {
                n += config.num_heads * config.head_dim; // bq
                n += config.kv_dim; // bk
                n += config.kv_dim; // bv
            }
            n += config.hidden_size; // ffn_norm
            n += static_cast<double>(config.intermediate_size) * config.hidden_size; // w_gate
            n += static_cast<double>(config.intermediate_size) * config.hidden_size; // w_up
            n += static_cast<double>(config.hidden_size) * config.intermediate_size; // w_down
        }
        return n;
    }

    // Returns total weight storage in bytes using the actual quantization types
    // recorded in the GGUF file.  More accurate than count_parameters()*4
    // when the model uses Q4_0 or Q8_0 weights.
    double count_weight_bytes() const {
        double bytes = 0;
        for (const auto& kv : gguf.tensors) {
            const auto& info = kv.second;
            size_t n = info.num_elements();
            size_t block_size = ggml_block_size(info.type);
            size_t type_size  = ggml_type_size(info.type);
            bytes += static_cast<double>(n / block_size) * type_size;
        }
        return bytes;
    }

private:
    bool load_config() {
        config.architecture = gguf.get_str("general.architecture", "llama");
        std::string arch = config.architecture;

        config.vocab_size = static_cast<int>(
            gguf.get_i64(arch + ".vocab_size",
                         gguf.get_i64("tokenizer.ggml.tokens",
                                      static_cast<int64_t>(tokenizer.vocab_size))));
        config.hidden_size = static_cast<int>(
            gguf.get_i64(arch + ".embedding_length", 0));
        config.intermediate_size = static_cast<int>(
            gguf.get_i64(arch + ".feed_forward_length", 0));
        config.num_layers = static_cast<int>(
            gguf.get_i64(arch + ".block_count", 0));
        config.num_heads = static_cast<int>(
            gguf.get_i64(arch + ".attention.head_count", 0));
        config.num_kv_heads = static_cast<int>(
            gguf.get_i64(arch + ".attention.head_count_kv", config.num_heads));
        config.max_seq_len = static_cast<int>(
            gguf.get_i64(arch + ".context_length", 2048));
        config.rms_norm_eps = static_cast<float>(
            gguf.get_f64(arch + ".attention.layer_norm_rms_epsilon", 1e-5));
        config.rope_theta = static_cast<float>(
            gguf.get_f64(arch + ".rope.freq_base", 10000.0));

        if (config.hidden_size == 0 || config.num_layers == 0 || config.num_heads == 0) {
            fprintf(stderr, "Error: incomplete model config (hidden=%d, layers=%d, heads=%d)\n",
                    config.hidden_size, config.num_layers, config.num_heads);
            return false;
        }

        config.head_dim = config.hidden_size / config.num_heads;

        // Override head_dim if explicitly specified (e.g. Qwen models)
        int explicit_head_dim = static_cast<int>(
            gguf.get_i64(arch + ".attention.key_length", 0));
        if (explicit_head_dim > 0) {
            config.head_dim = explicit_head_dim;
        }

        config.kv_dim = config.head_dim * config.num_kv_heads;

        // Use neox-style (halved) RoPE for Qwen models
        if (arch == "qwen2" || arch == "qwen3" || arch == "qwen2moe" ||
            arch == "qwen35" || arch == "qwen35moe") {
            config.rope_neox = true;
        }

        // Mark hybrid architecture for Qwen3.5
        if (arch == "qwen35" || arch == "qwen35moe") {
            config.is_hybrid = true;
        }

        // Get vocab size from tokenizer if not in model metadata
        if (config.vocab_size == 0) {
            auto it = gguf.metadata.find("tokenizer.ggml.tokens");
            if (it != gguf.metadata.end()) {
                config.vocab_size = static_cast<int>(it->second.arr.size());
            }
        }

        return true;
    }

    // Load and dequantize a tensor from GGUF
    float* load_tensor(const std::string& name, int64_t expected_elements = -1,
                       bool optional = false) {
        auto it = gguf.tensors.find(name);
        if (it == gguf.tensors.end()) {
            if (!optional) {
                fprintf(stderr, "Warning: tensor '%s' not found\n", name.c_str());
            }
            return nullptr;
        }

        const GGUFTensorInfo& info = it->second;
        int64_t n = static_cast<int64_t>(info.num_elements());

        if (expected_elements > 0 && n != expected_elements) {
            fprintf(stderr, "Warning: tensor '%s' has %ld elements, expected %ld\n",
                    name.c_str(), static_cast<long>(n), static_cast<long>(expected_elements));
        }

        // Allocate storage and dequantize
        weight_storage.emplace_back(n);
        float* data = weight_storage.back().data();

        const void* raw = gguf.get_tensor_data(name);
        if (!raw) {
            fprintf(stderr, "Error: cannot get data for tensor '%s'\n", name.c_str());
            return nullptr;
        }

        dequantize(raw, data, n, info.type);
        return data;
    }

    // Load a weight tensor keeping it in its native quantization format.
    // For Q4_0/Q8_0/F16/F32: returns a pointer directly into the GGUF file
    // buffer (no copy, no allocation).  F16 weights are kept in their compact
    // format and consumed by the fused cpu_matmul_transposed_f16 kernel.
    // This is used for large projection matrices to preserve memory bandwidth.
    QuantWeight load_tensor_raw(const std::string& name,
                                int64_t expected_elements = -1,
                                bool optional = false) {
        auto it = gguf.tensors.find(name);
        if (it == gguf.tensors.end()) {
            if (!optional) {
                fprintf(stderr, "Warning: tensor '%s' not found\n", name.c_str());
            }
            return {};
        }

        const GGUFTensorInfo& info = it->second;
        int64_t n = static_cast<int64_t>(info.num_elements());

        if (expected_elements > 0 && n != expected_elements) {
            fprintf(stderr, "Warning: tensor '%s' has %ld elements, expected %ld\n",
                    name.c_str(), static_cast<long>(n), static_cast<long>(expected_elements));
        }

        const void* raw = gguf.get_tensor_data(name);
        if (!raw) {
            fprintf(stderr, "Error: cannot get data for tensor '%s'\n", name.c_str());
            return {};
        }

        // F32, F16 and integer-quantized types: use the raw GGUF pointer directly.
        // gguf.owned_data (or the mmap region) owns this memory and lives as
        // long as the Model object does.
        return {raw, info.type};
    }

    bool load_weights() {
        int dim = config.hidden_size;
        int kv_dim = config.kv_dim;
        int ffn = config.intermediate_size;
        int n_heads = config.num_heads;
        int head_dim = config.head_dim;

        weights.token_embd = load_tensor("token_embd.weight",
            static_cast<int64_t>(config.vocab_size) * dim);
        weights.output_norm = load_tensor("output_norm.weight", dim);
        weights.output = load_tensor_raw("output.weight",
            static_cast<int64_t>(config.vocab_size) * dim, true);

        // If output weights are not present, use token embeddings (weight tying)
        if (!weights.output.valid()) {
            weights.output = {weights.token_embd, GGML_TYPE_F32};
        }

        if (!weights.token_embd || !weights.output_norm) {
            fprintf(stderr, "Error: missing essential model weights\n");
            return false;
        }

        weights.layers.resize(config.num_layers);
        for (int l = 0; l < config.num_layers; l++) {
            std::string prefix = "blk." + std::to_string(l) + ".";

            // Determine layer type for hybrid architectures
            if (l < static_cast<int>(config.layer_types.size())) {
                weights.layers[l].layer_type = config.layer_types[l];
            }

            weights.layers[l].attn_norm = load_tensor(prefix + "attn_norm.weight", dim);

            // For Qwen3.5 linear attention layers, try loading GatedDeltaNet weights
            bool is_linear = weights.layers[l].layer_type == "linear_attention";
            if (is_linear) {
                // Try to load GatedDeltaNet-specific weights
                weights.layers[l].w_qkv = load_tensor_raw(prefix + "attn_qkv.weight", -1, true);
                weights.layers[l].w_attn_gate = load_tensor_raw(prefix + "attn_gate.weight", -1, true);
                weights.layers[l].w_ssm_beta = load_tensor_raw(prefix + "ssm_beta.weight", -1, true);
                weights.layers[l].w_ssm_alpha = load_tensor_raw(prefix + "ssm_alpha.weight", -1, true);
                weights.layers[l].ssm_a = load_tensor(prefix + "ssm_a", -1, true);
                weights.layers[l].w_ssm_conv1d = load_tensor_raw(prefix + "ssm_conv1d.weight", -1, true);
                weights.layers[l].w_ssm_dt = load_tensor_raw(prefix + "ssm_dt.weight", -1, true);
                weights.layers[l].w_ssm_out = load_tensor_raw(prefix + "ssm_out.weight", -1, true);
                weights.layers[l].ssm_norm = load_tensor(prefix + "ssm_norm.weight", -1, true);
            }

            // Load standard attention weights (may exist even for linear attention
            // layers if converted from QKV format)
            weights.layers[l].wq = load_tensor_raw(prefix + "attn_q.weight",
                static_cast<int64_t>(n_heads) * head_dim * dim, !is_linear);
            weights.layers[l].wk = load_tensor_raw(prefix + "attn_k.weight",
                static_cast<int64_t>(kv_dim) * dim, !is_linear);
            weights.layers[l].wv = load_tensor_raw(prefix + "attn_v.weight",
                static_cast<int64_t>(kv_dim) * dim, !is_linear);
            weights.layers[l].wo = load_tensor_raw(prefix + "attn_output.weight",
                static_cast<int64_t>(dim) * n_heads * head_dim, !is_linear);

            // Load QKV biases (optional, used by Qwen3-style models)
            weights.layers[l].bq = load_tensor(prefix + "attn_q.bias",
                static_cast<int64_t>(n_heads) * head_dim, true);
            weights.layers[l].bk = load_tensor(prefix + "attn_k.bias",
                static_cast<int64_t>(kv_dim), true);
            weights.layers[l].bv = load_tensor(prefix + "attn_v.bias",
                static_cast<int64_t>(kv_dim), true);

            if (l == 0 && weights.layers[l].bq) {
                config.qkv_bias = true;
            }

            // Load QK-norm weights (optional, used by Qwen3-style models)
            weights.layers[l].attn_q_norm = load_tensor(prefix + "attn_q_norm.weight",
                static_cast<int64_t>(head_dim), true);
            weights.layers[l].attn_k_norm = load_tensor(prefix + "attn_k_norm.weight",
                static_cast<int64_t>(head_dim), true);

            weights.layers[l].ffn_norm = load_tensor(prefix + "ffn_norm.weight", dim);
            weights.layers[l].w_gate = load_tensor_raw(prefix + "ffn_gate.weight",
                static_cast<int64_t>(ffn) * dim);
            weights.layers[l].w_up = load_tensor_raw(prefix + "ffn_up.weight",
                static_cast<int64_t>(ffn) * dim);
            weights.layers[l].w_down = load_tensor_raw(prefix + "ffn_down.weight",
                static_cast<int64_t>(dim) * ffn);

            // For standard attention layers, validate essential weights
            if (!is_linear) {
                if (!weights.layers[l].attn_norm ||
                    !weights.layers[l].wq.valid() ||
                    !weights.layers[l].wk.valid() ||
                    !weights.layers[l].wv.valid() ||
                    !weights.layers[l].wo.valid() ||
                    !weights.layers[l].ffn_norm ||
                    !weights.layers[l].w_gate.valid() ||
                    !weights.layers[l].w_up.valid() ||
                    !weights.layers[l].w_down.valid()) {
                    fprintf(stderr, "Error: missing weights for layer %d\n", l);
                    return false;
                }
            }

            fprintf(stderr, "\r  Loading layer %d/%d%s...",
                    l + 1, config.num_layers,
                    is_linear ? " (linear)" : "");
        }
        fprintf(stderr, "\n");

        // Initialize KV cache
        kv_cache.init(config.num_layers, config.max_seq_len, kv_dim);

        return true;
    }

    void alloc_buffers() {
        int dim = config.hidden_size;
        x.resize(dim);
        xb.resize(dim);
        xb2.resize(dim);
        q.resize(config.num_heads * config.head_dim);
        k.resize(config.kv_dim);
        v.resize(config.kv_dim);
        att.resize(static_cast<size_t>(config.num_heads) * config.max_seq_len);
        hb.resize(config.intermediate_size);
        hb2.resize(config.intermediate_size);
        logits.resize(config.vocab_size);

        // Precompute RoPE frequency table: freqs[i] = 1 / theta^(2i / rope_dim)
        // Depends only on head_dim, rope_dim and rope_theta (all fixed at load
        // time), so computing once here avoids a heap allocation on every
        // forward() call.  When partial_rotary_factor < 1.0 (e.g. Qwen3.5
        // uses 0.25), only the first rope_dim positions get RoPE.
        int rope_dim = config.rope_dim > 0 ? config.rope_dim : config.head_dim;
        int half_rope = rope_dim / 2;
        rope_freqs.resize(half_rope);
        for (int i = 0; i < half_rope; i++) {
            rope_freqs[i] = 1.0f / powf(config.rope_theta,
                                        static_cast<float>(2 * i) / rope_dim);
        }
    }

    void apply_context_override(int context_override) {
        if (context_override > 0 && context_override < config.max_seq_len) {
            fprintf(stderr, "Context length capped: %d -> %d\n",
                    config.max_seq_len, context_override);
            config.max_seq_len = context_override;
        } else if (context_override == 0 && config.max_seq_len > 8192) {
            fprintf(stderr, "Context length auto-capped: %d -> %d (use -c to override)\n",
                    config.max_seq_len, 4096);
            config.max_seq_len = 4096;
        }
    }

    void print_model_info() {
        fprintf(stderr, "Model loaded successfully!\n");
        fprintf(stderr, "  Architecture: %s\n", config.architecture.c_str());
        fprintf(stderr, "  Parameters: %.1fM\n", count_parameters() / 1e6);
        fprintf(stderr, "  Layers: %d, Heads: %d, KV Heads: %d\n",
                config.num_layers, config.num_heads, config.num_kv_heads);
        fprintf(stderr, "  Hidden: %d, FFN: %d, Vocab: %d\n",
                config.hidden_size, config.intermediate_size, config.vocab_size);
        fprintf(stderr, "  Context: %d, RoPE theta: %.0f\n",
                config.max_seq_len, config.rope_theta);

        double kv_cache_mb = 2.0 * config.num_layers * config.max_seq_len
                             * config.kv_dim * sizeof(float) / (1024.0 * 1024.0);
        double weights_mb = count_weight_bytes() / (1024.0 * 1024.0);
        fprintf(stderr, "  Memory estimate: weights=%.0fMB, kv_cache=%.0fMB\n",
                weights_mb, kv_cache_mb);

        if (config.qkv_bias) {
            fprintf(stderr, "  QKV bias: enabled (Qwen3-style)\n");
        }
        if (config.is_hybrid) {
            fprintf(stderr, "  Hybrid: yes (Qwen3.5 attention + GatedDeltaNet)\n");
        }
        if (config.is_moe) {
            fprintf(stderr, "  MoE: %d experts, %d active per token\n",
                    config.num_experts, config.num_experts_per_tok);
        }
        if (config.partial_rotary_factor < 1.0f) {
            fprintf(stderr, "  Partial rotary: %.0f%% (%d/%d dims)\n",
                    config.partial_rotary_factor * 100.0f,
                    config.rope_dim, config.head_dim);
        }
        if (is_hf_model) {
            fprintf(stderr, "  Format: HuggingFace SafeTensors (direct loading)\n");
        }
        fprintf(stderr, "  Backend: %s\n", backend_name(compute.backend));
    }

    // ---- HuggingFace model loading ----

    bool load_from_hf(const std::string& dir_path, int context_override) {
        is_hf_model = true;
        fprintf(stderr, "Detected HuggingFace model directory\n");

        // Load config.json
        std::string config_path = dir_path + "/config.json";
        if (!file_exists(config_path)) {
            fprintf(stderr, "Error: config.json not found in '%s'\n", dir_path.c_str());
            return false;
        }

        if (!hf_config.load(config_path)) {
            fprintf(stderr, "Error: failed to parse config.json\n");
            return false;
        }

        // Map HF config to ModelConfig
        config.architecture = hf_config.get_architecture();
        config.vocab_size = hf_config.vocab_size;
        config.hidden_size = hf_config.hidden_size;
        config.intermediate_size = hf_config.intermediate_size;
        config.num_layers = hf_config.num_hidden_layers;
        config.num_heads = hf_config.num_attention_heads;
        config.num_kv_heads = hf_config.num_key_value_heads;
        config.max_seq_len = hf_config.max_position_embeddings;
        config.rms_norm_eps = static_cast<float>(hf_config.rms_norm_eps);
        config.rope_theta = static_cast<float>(hf_config.rope_theta);
        config.head_dim = hf_config.head_dim;
        config.kv_dim = config.head_dim * config.num_kv_heads;
        config.is_hybrid = hf_config.is_hybrid();
        config.layer_types = hf_config.layer_types;
        config.linear_key_head_dim = hf_config.linear_key_head_dim;
        config.linear_value_head_dim = hf_config.linear_value_head_dim;
        config.linear_num_key_heads = hf_config.linear_num_key_heads;
        config.linear_num_value_heads = hf_config.linear_num_value_heads;
        config.linear_conv_kernel_dim = hf_config.linear_conv_kernel_dim;
        config.partial_rotary_factor = static_cast<float>(hf_config.partial_rotary_factor);
        config.rope_dim = static_cast<int>(config.head_dim * config.partial_rotary_factor);
        if (config.rope_dim == 0) config.rope_dim = config.head_dim;

        // MoE parameters
        config.is_moe = hf_config.is_moe();
        config.num_experts = hf_config.num_experts;
        config.num_experts_per_tok = hf_config.num_experts_per_tok;
        config.moe_intermediate_size = hf_config.moe_intermediate_size;
        config.shared_expert_intermediate_size = hf_config.shared_expert_intermediate_size;

        // Set RoPE style for Qwen models
        std::string arch = config.architecture;
        if (arch == "qwen2" || arch == "qwen3" || arch == "qwen2moe" ||
            arch == "qwen35" || arch == "qwen35moe") {
            config.rope_neox = true;
        }

        fprintf(stderr, "HF model type: %s -> architecture: %s\n",
                hf_config.model_type.c_str(), config.architecture.c_str());

        // Apply context override
        apply_context_override(context_override);

        // Load tokenizer
        if (!load_hf_tokenizer(dir_path)) {
            fprintf(stderr, "Error: failed to load tokenizer\n");
            return false;
        }

        // Load SafeTensors weights
        if (!load_hf_weights(dir_path)) {
            fprintf(stderr, "Error: failed to load weights\n");
            return false;
        }

        // Allocate inference buffers
        alloc_buffers();

        print_model_info();
        return true;
    }

    bool load_hf_tokenizer(const std::string& dir_path) {
        HFTokenizerData hf_tok;
        std::string tok_path = dir_path + "/tokenizer.json";

        if (!file_exists(tok_path)) {
            fprintf(stderr, "Error: tokenizer.json not found in '%s'\n",
                    dir_path.c_str());
            return false;
        }

        if (!hf_tok.load_tokenizer_json(tok_path)) {
            return false;
        }

        // Load config for special token IDs
        std::string tok_config_path = dir_path + "/tokenizer_config.json";
        if (file_exists(tok_config_path)) {
            hf_tok.load_tokenizer_config(tok_config_path);
        }

        // Map to our Tokenizer struct
        tokenizer.vocab_size = static_cast<int>(hf_tok.vocab.size());
        tokenizer.id_to_token = hf_tok.vocab;
        tokenizer.token_to_id = hf_tok.token_to_id;
        tokenizer.tokenizer_model = "gpt2";  // HF BPE tokenizers use GPT-2 style

        // Set special token IDs
        if (hf_tok.bos_id >= 0) tokenizer.bos_token_id = hf_tok.bos_id;
        if (hf_tok.eos_id >= 0) tokenizer.eos_token_id = hf_tok.eos_id;
        tokenizer.eos_token_ids = hf_tok.eos_ids;

        // Load merges
        for (size_t i = 0; i < hf_tok.merges.size(); i++) {
            Tokenizer::BPEMerge merge;
            merge.left = hf_tok.merges[i].first;
            merge.right = hf_tok.merges[i].second;
            merge.rank = static_cast<int>(i);
            tokenizer.merges.push_back(merge);
            tokenizer.merge_ranks[merge.left + " " + merge.right] = merge.rank;
        }

        // Populate scores (uniform for BPE)
        tokenizer.token_scores.resize(tokenizer.vocab_size, 0.0f);

        // Set up added/special tokens
        for (const auto& at : hf_tok.added_tokens) {
            tokenizer.added_tokens.push_back(at.first);
        }
        // Sort by length descending for greedy matching
        std::sort(tokenizer.added_tokens.begin(), tokenizer.added_tokens.end(),
                  [](const std::string& a, const std::string& b) {
                      return a.size() > b.size();
                  });

        // Initialize GPT-2 byte mapping
        tokenizer.init_gpt2_byte_mapping();

        // Update vocab_size in config if needed
        if (config.vocab_size == 0 || config.vocab_size < tokenizer.vocab_size) {
            config.vocab_size = tokenizer.vocab_size;
        }

        fprintf(stderr, "Tokenizer loaded: %d tokens, %zu merges\n",
                tokenizer.vocab_size, tokenizer.merges.size());
        return true;
    }

    bool load_hf_weights(const std::string& dir_path) {
        // Find SafeTensors files
        auto st_files = find_safetensors_files(dir_path);
        if (st_files.empty()) {
            fprintf(stderr, "Error: no SafeTensors files found in '%s'\n",
                    dir_path.c_str());
            return false;
        }

        fprintf(stderr, "Loading %zu SafeTensors file(s)...\n", st_files.size());

        // Load all SafeTensors files
        if (st_files.size() == 1) {
            if (!safetensors.load(st_files[0])) return false;
        } else {
            if (!safetensors.load_multi(st_files)) return false;
        }

        fprintf(stderr, "  Found %zu tensors\n", safetensors.tensors.size());

        // Build a mapping from GGUF names to HF tensor info
        std::unordered_map<std::string, std::pair<std::string, size_t>> gguf_to_hf;
        for (const auto& kv : safetensors.tensors) {
            std::string gguf_name = hf_to_gguf_tensor_name(kv.first);
            gguf_to_hf[gguf_name] = {kv.first, 0};  // file_idx=0 for single file
        }

        // Load embedding and output weights
        weights.token_embd = load_hf_tensor("token_embd.weight", gguf_to_hf);
        weights.output_norm = load_hf_tensor("output_norm.weight", gguf_to_hf);

        // Output/lm_head - may use weight tying
        float* output_w = load_hf_tensor("output.weight", gguf_to_hf, true);
        if (output_w) {
            weights.output = {output_w, GGML_TYPE_F32};
        } else if (hf_config.tie_word_embeddings && weights.token_embd) {
            weights.output = {weights.token_embd, GGML_TYPE_F32};
        }

        if (!weights.token_embd || !weights.output_norm || !weights.output.valid()) {
            fprintf(stderr, "Error: missing essential HF model weights\n");
            return false;
        }

        // Load layer weights
        weights.layers.resize(config.num_layers);
        for (int l = 0; l < config.num_layers; l++) {
            std::string gguf_prefix = "blk." + std::to_string(l) + ".";

            if (l < static_cast<int>(config.layer_types.size())) {
                weights.layers[l].layer_type = config.layer_types[l];
            }

            bool is_linear = weights.layers[l].layer_type == "linear_attention";

            weights.layers[l].attn_norm = load_hf_tensor(
                gguf_prefix + "attn_norm.weight", gguf_to_hf);

            if (!is_linear) {
                // Standard attention layer
                float* wq = load_hf_tensor(gguf_prefix + "attn_q.weight", gguf_to_hf);
                if (wq) weights.layers[l].wq = {wq, GGML_TYPE_F32};

                float* wk = load_hf_tensor(gguf_prefix + "attn_k.weight", gguf_to_hf);
                if (wk) weights.layers[l].wk = {wk, GGML_TYPE_F32};

                float* wv = load_hf_tensor(gguf_prefix + "attn_v.weight", gguf_to_hf);
                if (wv) weights.layers[l].wv = {wv, GGML_TYPE_F32};

                float* wo = load_hf_tensor(gguf_prefix + "attn_output.weight", gguf_to_hf);
                if (wo) weights.layers[l].wo = {wo, GGML_TYPE_F32};
            } else {
                // GatedDeltaNet linear attention layer
                float* wqkv = load_hf_tensor(gguf_prefix + "attn_qkv.weight",
                                              gguf_to_hf, true);
                if (wqkv) weights.layers[l].w_qkv = {wqkv, GGML_TYPE_F32};

                float* wgate = load_hf_tensor(gguf_prefix + "attn_gate.weight",
                                               gguf_to_hf, true);
                if (wgate) weights.layers[l].w_attn_gate = {wgate, GGML_TYPE_F32};
            }

            // Biases (optional)
            weights.layers[l].bq = load_hf_tensor(
                gguf_prefix + "attn_q.bias", gguf_to_hf, true);
            weights.layers[l].bk = load_hf_tensor(
                gguf_prefix + "attn_k.bias", gguf_to_hf, true);
            weights.layers[l].bv = load_hf_tensor(
                gguf_prefix + "attn_v.bias", gguf_to_hf, true);
            if (l == 0 && weights.layers[l].bq) config.qkv_bias = true;

            // QK-norm weights (optional)
            weights.layers[l].attn_q_norm = load_hf_tensor(
                gguf_prefix + "attn_q_norm.weight", gguf_to_hf, true);
            weights.layers[l].attn_k_norm = load_hf_tensor(
                gguf_prefix + "attn_k_norm.weight", gguf_to_hf, true);

            // FFN weights
            weights.layers[l].ffn_norm = load_hf_tensor(
                gguf_prefix + "ffn_norm.weight", gguf_to_hf);

            float* wg = load_hf_tensor(gguf_prefix + "ffn_gate.weight", gguf_to_hf);
            if (wg) weights.layers[l].w_gate = {wg, GGML_TYPE_F32};

            float* wu = load_hf_tensor(gguf_prefix + "ffn_up.weight", gguf_to_hf);
            if (wu) weights.layers[l].w_up = {wu, GGML_TYPE_F32};

            float* wd = load_hf_tensor(gguf_prefix + "ffn_down.weight", gguf_to_hf);
            if (wd) weights.layers[l].w_down = {wd, GGML_TYPE_F32};

            // Validate essential weights for standard attention layers
            if (!is_linear) {
                if (!weights.layers[l].attn_norm ||
                    !weights.layers[l].wq.valid() ||
                    !weights.layers[l].wk.valid() ||
                    !weights.layers[l].wv.valid() ||
                    !weights.layers[l].wo.valid() ||
                    !weights.layers[l].ffn_norm ||
                    !weights.layers[l].w_gate.valid() ||
                    !weights.layers[l].w_up.valid() ||
                    !weights.layers[l].w_down.valid()) {
                    fprintf(stderr, "Error: missing HF weights for layer %d\n", l);
                    return false;
                }
            }

            fprintf(stderr, "\r  Loading layer %d/%d%s...",
                    l + 1, config.num_layers,
                    is_linear ? " (linear)" : "");
        }
        fprintf(stderr, "\n");

        // Initialize KV cache
        kv_cache.init(config.num_layers, config.max_seq_len, config.kv_dim);

        return true;
    }

    // Load a tensor from SafeTensors, dequantizing to F32
    float* load_hf_tensor(
        const std::string& gguf_name,
        const std::unordered_map<std::string, std::pair<std::string, size_t>>& name_map,
        bool optional = false)
    {
        auto it = name_map.find(gguf_name);
        if (it == name_map.end()) {
            if (!optional) {
                fprintf(stderr, "Warning: HF tensor for '%s' not found\n",
                        gguf_name.c_str());
            }
            return nullptr;
        }

        const std::string& hf_name = it->second.first;
        size_t file_idx = it->second.second;

        auto st_it = safetensors.tensors.find(hf_name);
        if (st_it == safetensors.tensors.end()) return nullptr;

        const SafeTensorInfo& info = st_it->second;
        int64_t n = info.num_elements();

        // Get raw data pointer
        const void* raw = safetensors.get_tensor_data(info, file_idx);
        if (!raw) {
            if (!optional) {
                fprintf(stderr, "Error: cannot get data for HF tensor '%s'\n",
                        hf_name.c_str());
            }
            return nullptr;
        }

        // Allocate and dequantize to F32
        weight_storage.emplace_back(n);
        float* data = weight_storage.back().data();
        safetensors.dequantize_to_f32(raw, data, n, info.dtype);
        return data;
    }
};

#endif // LLM_MODEL_H
