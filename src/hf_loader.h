#ifndef LLM_HF_LOADER_H
#define LLM_HF_LOADER_H

// HuggingFace model directory loader.
// Reads config.json, tokenizer.json, and SafeTensors weight files from a
// HuggingFace model directory (e.g., downloaded via `huggingface-cli download`).
//
// This enables direct inference from HF model checkpoints without
// requiring conversion to GGUF format first.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define access _access
#define F_OK 0
#else
#include <unistd.h>
#endif
#include <sys/stat.h>

#include "safetensors.h"

// ---- Minimal JSON value reader for config.json ----

// Read entire file into string
static inline std::string read_file_to_string(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return "";
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

// Check if a path exists and is a directory
static inline bool is_directory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

// Check if a file exists
static inline bool file_exists(const std::string& path) {
    return access(path.c_str(), F_OK) == 0;
}

// Simple JSON string value extraction
static inline std::string hf_json_get_str(const std::string& json,
                                           const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return "";
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos) return "";
    size_t q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return "";
    size_t q2 = q1 + 1;
    while (q2 < json.size()) {
        if (json[q2] == '\\') { q2 += 2; continue; }
        if (json[q2] == '"')  break;
        q2++;
    }
    return json.substr(q1 + 1, q2 - q1 - 1);
}

// Simple JSON integer extraction
static inline int64_t hf_json_get_int(const std::string& json,
                                       const std::string& key,
                                       int64_t default_val = 0) {
    std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return default_val;
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos) return default_val;
    size_t vpos = colon + 1;
    while (vpos < json.size() && (json[vpos] == ' ' || json[vpos] == '\t' ||
                                   json[vpos] == '\n' || json[vpos] == '\r'))
        vpos++;
    if (vpos >= json.size()) return default_val;
    char* end = nullptr;
    long long v = strtoll(json.c_str() + vpos, &end, 10);
    return (end > json.c_str() + vpos) ? static_cast<int64_t>(v) : default_val;
}

// Simple JSON float extraction
static inline double hf_json_get_float(const std::string& json,
                                        const std::string& key,
                                        double default_val = 0.0) {
    std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return default_val;
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos) return default_val;
    size_t vpos = colon + 1;
    while (vpos < json.size() && (json[vpos] == ' ' || json[vpos] == '\t' ||
                                   json[vpos] == '\n' || json[vpos] == '\r'))
        vpos++;
    if (vpos >= json.size()) return default_val;
    char* end = nullptr;
    double v = strtod(json.c_str() + vpos, &end);
    return (end > json.c_str() + vpos) ? v : default_val;
}

// Simple JSON bool extraction
static inline bool hf_json_get_bool(const std::string& json,
                                     const std::string& key,
                                     bool default_val = false) {
    std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return default_val;
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos) return default_val;
    size_t vpos = colon + 1;
    while (vpos < json.size() && (json[vpos] == ' ' || json[vpos] == '\t' ||
                                   json[vpos] == '\n' || json[vpos] == '\r'))
        vpos++;
    if (vpos >= json.size()) return default_val;
    if (json.compare(vpos, 4, "true")  == 0) return true;
    if (json.compare(vpos, 5, "false") == 0) return false;
    return default_val;
}

// Extract rope_theta from nested structures
// Handles both top-level "rope_theta" and nested "rope_parameters.rope_theta"
static inline double hf_get_rope_theta(const std::string& json, double default_val) {
    double v = hf_json_get_float(json, "rope_theta", 0.0);
    if (v > 0.0) return v;
    // Try inside rope_parameters or rope_scaling
    v = hf_json_get_float(json, "rope_theta", default_val);
    return v;
}

// ---- HuggingFace model configuration ----

struct HFModelConfig {
    std::string model_type;          // e.g. "qwen3", "qwen3_5_text", "llama"
    std::string architecture_class;  // e.g. "Qwen3ForCausalLM"
    int vocab_size = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int max_position_embeddings = 2048;
    double rms_norm_eps = 1e-6;
    double rope_theta = 10000.0;
    int head_dim = 0;
    bool tie_word_embeddings = false;
    double partial_rotary_factor = 1.0;  // Qwen3.5: 0.25 (64/256)

    // Qwen3.5-specific
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    int linear_num_key_heads = 0;
    int linear_num_value_heads = 0;
    int linear_conv_kernel_dim = 0;
    std::vector<std::string> layer_types;  // "full_attention" or "linear_attention"

    // MoE-specific
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;
    int num_experts_per_tok = 0;
    int num_experts = 0;

    bool load(const std::string& config_path) {
        std::string json = read_file_to_string(config_path);
        if (json.empty()) {
            fprintf(stderr, "HF: cannot read config file '%s'\n", config_path.c_str());
            return false;
        }

        model_type = hf_json_get_str(json, "model_type");

        // For VL (vision-language) models, the text model config is nested
        // inside a "text_config" sub-object. Detect and extract it.
        std::string text_json = json;
        if (model_type == "qwen3_5" || model_type == "qwen3_5_moe") {
            std::string nested = extract_text_config(json);
            if (!nested.empty()) {
                text_json = nested;
                // Re-read model_type from nested config
                std::string nested_type = hf_json_get_str(text_json, "model_type");
                if (!nested_type.empty()) model_type = nested_type;
            }
        }

        // Parse from text config (may be nested or top-level)
        vocab_size = static_cast<int>(hf_json_get_int(text_json, "vocab_size", 0));
        hidden_size = static_cast<int>(hf_json_get_int(text_json, "hidden_size", 0));
        intermediate_size = static_cast<int>(hf_json_get_int(text_json, "intermediate_size", 0));
        num_hidden_layers = static_cast<int>(hf_json_get_int(text_json, "num_hidden_layers", 0));
        num_attention_heads = static_cast<int>(hf_json_get_int(text_json, "num_attention_heads", 0));
        num_key_value_heads = static_cast<int>(hf_json_get_int(text_json, "num_key_value_heads",
                                                                num_attention_heads));
        max_position_embeddings = static_cast<int>(hf_json_get_int(text_json,
                                                    "max_position_embeddings", 2048));
        rms_norm_eps = hf_json_get_float(text_json, "rms_norm_eps", 1e-6);
        rope_theta = hf_get_rope_theta(text_json, 10000.0);
        head_dim = static_cast<int>(hf_json_get_int(text_json, "head_dim", 0));
        tie_word_embeddings = hf_json_get_bool(text_json, "tie_word_embeddings",
                                                hf_json_get_bool(json, "tie_word_embeddings", false));
        partial_rotary_factor = hf_json_get_float(text_json, "partial_rotary_factor", 1.0);

        // Qwen3.5-specific
        linear_key_head_dim = static_cast<int>(
            hf_json_get_int(text_json, "linear_key_head_dim", 0));
        linear_value_head_dim = static_cast<int>(
            hf_json_get_int(text_json, "linear_value_head_dim", 0));
        linear_num_key_heads = static_cast<int>(
            hf_json_get_int(text_json, "linear_num_key_heads", 0));
        linear_num_value_heads = static_cast<int>(
            hf_json_get_int(text_json, "linear_num_value_heads", 0));
        linear_conv_kernel_dim = static_cast<int>(
            hf_json_get_int(text_json, "linear_conv_kernel_dim", 0));

        // MoE-specific
        moe_intermediate_size = static_cast<int>(
            hf_json_get_int(text_json, "moe_intermediate_size", 0));
        shared_expert_intermediate_size = static_cast<int>(
            hf_json_get_int(text_json, "shared_expert_intermediate_size", 0));
        num_experts_per_tok = static_cast<int>(
            hf_json_get_int(text_json, "num_experts_per_tok", 0));
        num_experts = static_cast<int>(
            hf_json_get_int(text_json, "num_experts", 0));

        // Parse layer_types array
        parse_layer_types(text_json);

        // Extract architecture class from "architectures" array
        size_t arch_pos = json.find("\"architectures\"");
        if (arch_pos != std::string::npos) {
            size_t bracket = json.find('[', arch_pos);
            if (bracket != std::string::npos) {
                size_t q1 = json.find('"', bracket + 1);
                if (q1 != std::string::npos) {
                    size_t q2 = json.find('"', q1 + 1);
                    if (q2 != std::string::npos) {
                        architecture_class = json.substr(q1 + 1, q2 - q1 - 1);
                    }
                }
            }
        }

        if (head_dim == 0 && num_attention_heads > 0) {
            head_dim = hidden_size / num_attention_heads;
        }

        return hidden_size > 0 && num_hidden_layers > 0 && num_attention_heads > 0;
    }

    // Map HF model_type to GGUF architecture string.
    // HF uses underscores in compound types (e.g., "qwen3_5_text") while GGUF
    // uses concatenated names (e.g., "qwen35") per the llama.cpp convention.
    std::string get_architecture() const {
        if (model_type == "qwen3_5_text" || model_type == "qwen3_5") return "qwen35";
        if (model_type == "qwen3_5_moe_text" || model_type == "qwen3_5_moe") return "qwen35moe";
        if (model_type == "qwen3") return "qwen3";
        if (model_type == "qwen2") return "qwen2";
        if (model_type == "qwen2_moe") return "qwen2moe";
        if (model_type == "llama") return "llama";
        if (model_type == "mistral") return "llama";
        if (model_type == "deepseek_v2") return "deepseek2";
        return model_type;
    }

    // Check if this is a hybrid model (mix of attention + linear attention)
    bool is_hybrid() const {
        return !layer_types.empty();
    }

    // Check if this is a MoE model
    bool is_moe() const {
        return num_experts > 0;
    }

    // Get the layer type for a given layer index
    std::string get_layer_type(int layer_idx) const {
        if (layer_types.empty() || layer_idx >= static_cast<int>(layer_types.size())) {
            return "full_attention";
        }
        return layer_types[layer_idx];
    }

private:
    void parse_layer_types(const std::string& json) {
        size_t pos = json.find("\"layer_types\"");
        if (pos == std::string::npos) return;
        size_t bracket = json.find('[', pos);
        if (bracket == std::string::npos) return;
        size_t end_bracket = json.find(']', bracket);
        if (end_bracket == std::string::npos) return;

        std::string arr = json.substr(bracket + 1, end_bracket - bracket - 1);
        size_t p = 0;
        while (p < arr.size()) {
            size_t q1 = arr.find('"', p);
            if (q1 == std::string::npos) break;
            size_t q2 = arr.find('"', q1 + 1);
            if (q2 == std::string::npos) break;
            layer_types.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
            p = q2 + 1;
        }
    }

    // Extract nested "text_config" object from VL model config.json
    // VL models wrap the text model config in: { "text_config": { ... }, "vision_config": {...} }
    std::string extract_text_config(const std::string& json) {
        size_t pos = json.find("\"text_config\"");
        if (pos == std::string::npos) return "";
        size_t colon = json.find(':', pos + 13);
        if (colon == std::string::npos) return "";
        size_t brace = json.find('{', colon);
        if (brace == std::string::npos) return "";

        // Find matching closing brace
        int depth = 1;
        size_t p = brace + 1;
        while (p < json.size() && depth > 0) {
            if (json[p] == '{') depth++;
            else if (json[p] == '}') depth--;
            else if (json[p] == '"') {
                p++;
                while (p < json.size() && json[p] != '"') {
                    if (json[p] == '\\') p++;
                    p++;
                }
            }
            p++;
        }
        if (depth != 0) return "";
        return json.substr(brace, p - brace);
    }
};

// ---- HuggingFace weight name mapping ----

// Map HuggingFace PyTorch weight names to GGUF-style tensor names
static inline std::string hf_to_gguf_tensor_name(const std::string& hf_name) {
    // Embedding and output layers
    if (hf_name == "model.embed_tokens.weight") return "token_embd.weight";
    if (hf_name == "model.norm.weight") return "output_norm.weight";
    if (hf_name == "lm_head.weight") return "output.weight";

    // Layer-specific tensors: model.layers.{N}.xxx -> blk.{N}.xxx
    if (hf_name.find("model.layers.") == 0) {
        // Extract layer number
        size_t dot1 = hf_name.find('.', 13);  // after "model.layers."
        if (dot1 == std::string::npos) return hf_name;
        std::string layer_num = hf_name.substr(13, dot1 - 13);
        std::string rest = hf_name.substr(dot1 + 1);

        std::string prefix = "blk." + layer_num + ".";

        // Attention layers
        if (rest == "self_attn.q_proj.weight") return prefix + "attn_q.weight";
        if (rest == "self_attn.k_proj.weight") return prefix + "attn_k.weight";
        if (rest == "self_attn.v_proj.weight") return prefix + "attn_v.weight";
        if (rest == "self_attn.o_proj.weight") return prefix + "attn_output.weight";
        if (rest == "self_attn.q_proj.bias")   return prefix + "attn_q.bias";
        if (rest == "self_attn.k_proj.bias")   return prefix + "attn_k.bias";
        if (rest == "self_attn.v_proj.bias")   return prefix + "attn_v.bias";
        if (rest == "self_attn.q_norm.weight") return prefix + "attn_q_norm.weight";
        if (rest == "self_attn.k_norm.weight") return prefix + "attn_k_norm.weight";

        // LayerNorm
        if (rest == "input_layernorm.weight")           return prefix + "attn_norm.weight";
        if (rest == "post_attention_layernorm.weight")   return prefix + "ffn_norm.weight";

        // MLP / FFN (dense models)
        if (rest == "mlp.gate_proj.weight") return prefix + "ffn_gate.weight";
        if (rest == "mlp.up_proj.weight")   return prefix + "ffn_up.weight";
        if (rest == "mlp.down_proj.weight") return prefix + "ffn_down.weight";

        // MoE FFN (Qwen3.5-MoE)
        // Router
        if (rest == "mlp.gate.weight") return prefix + "ffn_gate_inp.weight";
        // Merged experts
        if (rest == "mlp.experts.gate_up_proj") return prefix + "ffn_gate_up_exps.weight";
        if (rest == "mlp.experts.down_proj")    return prefix + "ffn_down_exps.weight";
        // Individual experts: mlp.experts.{E}.gate_proj.weight
        if (rest.find("mlp.experts.") == 0 && rest.find(".gate_proj.weight") != std::string::npos) {
            return prefix + rest;  // pass through for per-expert weights
        }
        if (rest.find("mlp.experts.") == 0 && rest.find(".up_proj.weight") != std::string::npos) {
            return prefix + rest;
        }
        if (rest.find("mlp.experts.") == 0 && rest.find(".down_proj.weight") != std::string::npos) {
            return prefix + rest;
        }
        // Shared expert
        if (rest == "mlp.shared_expert.gate_proj.weight") return prefix + "ffn_gate_shexp.weight";
        if (rest == "mlp.shared_expert.up_proj.weight")   return prefix + "ffn_up_shexp.weight";
        if (rest == "mlp.shared_expert.down_proj.weight") return prefix + "ffn_down_shexp.weight";

        // Qwen3.5 GatedDeltaNet (linear attention) layers
        if (rest == "linear_attn.in_proj_qkv.weight") return prefix + "attn_qkv.weight";
        if (rest == "linear_attn.in_proj_z.weight")    return prefix + "attn_gate.weight";
        if (rest == "linear_attn.in_proj_b.weight")    return prefix + "ssm_beta.weight";
        if (rest == "linear_attn.in_proj_a.weight")    return prefix + "ssm_alpha.weight";
        if (rest == "linear_attn.a_param")             return prefix + "ssm_a";
        if (rest == "linear_attn.conv1d.weight")       return prefix + "ssm_conv1d.weight";
        if (rest == "linear_attn.dt_proj.weight")      return prefix + "ssm_dt.weight";
        if (rest == "linear_attn.out_proj.weight")     return prefix + "ssm_out.weight";
        if (rest == "linear_attn.norm.weight")         return prefix + "ssm_norm.weight";

        // Return as-is if no mapping found
        return prefix + rest;
    }

    return hf_name;
}

// ---- HuggingFace tokenizer loading ----

struct HFTokenizerData {
    std::vector<std::string> vocab;          // id -> token string
    std::vector<float> scores;               // optional token scores
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::pair<std::string,std::string>> merges;
    std::string model_type;                  // "BPE", "Unigram", etc.
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;
    std::vector<std::pair<std::string, int>> added_tokens;
    std::vector<int> eos_ids;

    bool load_tokenizer_json(const std::string& path) {
        std::string json = read_file_to_string(path);
        if (json.empty()) {
            fprintf(stderr, "HF: cannot read tokenizer '%s'\n", path.c_str());
            return false;
        }

        // Extract model type
        model_type = hf_json_get_str(json, "type");

        // Parse vocab from "model.vocab" section
        parse_vocab(json);

        // Parse merges from "model.merges" section
        parse_merges(json);

        // Parse added_tokens
        parse_added_tokens(json);

        return !vocab.empty();
    }

    // Load tokenizer config to get special token IDs
    void load_tokenizer_config(const std::string& path) {
        std::string json = read_file_to_string(path);
        if (json.empty()) return;

        // Look for eos_token, bos_token
        // Format can be: "bos_token": "<|endoftext|>" or nested object
        std::string bos_str = hf_json_get_str(json, "bos_token");
        std::string eos_str = hf_json_get_str(json, "eos_token");

        if (!bos_str.empty()) {
            auto it = token_to_id.find(bos_str);
            if (it != token_to_id.end()) bos_id = it->second;
        }
        if (!eos_str.empty()) {
            auto it = token_to_id.find(eos_str);
            if (it != token_to_id.end()) {
                eos_id = it->second;
                eos_ids.push_back(eos_id);
            }
        }

        // Look for additional EOS tokens like <|im_end|>, <|endoftext|>
        for (const auto& at : added_tokens) {
            if (at.first == "<|im_end|>" || at.first == "<|endoftext|>") {
                bool already_in = false;
                for (int id : eos_ids) {
                    if (id == at.second) { already_in = true; break; }
                }
                if (!already_in) eos_ids.push_back(at.second);
            }
        }
    }

private:
    void parse_vocab(const std::string& json) {
        // Find "vocab" object inside "model" object
        size_t model_pos = json.find("\"model\"");
        if (model_pos == std::string::npos) return;
        size_t vocab_pos = json.find("\"vocab\"", model_pos);
        if (vocab_pos == std::string::npos) return;

        // Find the opening brace of the vocab object
        size_t brace = json.find('{', vocab_pos + 7);
        if (brace == std::string::npos) return;

        // Parse key-value pairs: "token_string": id
        size_t pos = brace + 1;
        int max_id = -1;
        std::unordered_map<int, std::string> id_to_tok;

        while (pos < json.size()) {
            // Skip whitespace
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                                         json[pos] == '\n' || json[pos] == '\r' ||
                                         json[pos] == ','))
                pos++;
            if (pos >= json.size() || json[pos] == '}') break;

            // Read token string
            if (json[pos] != '"') break;
            pos++;
            std::string token;
            while (pos < json.size() && json[pos] != '"') {
                if (json[pos] == '\\' && pos + 1 < json.size()) {
                    pos++;
                    switch (json[pos]) {
                        case 'n': token += '\n'; break;
                        case 't': token += '\t'; break;
                        case 'r': token += '\r'; break;
                        case '"': token += '"'; break;
                        case '\\': token += '\\'; break;
                        case 'u': {
                            // Unicode escape \uXXXX
                            if (pos + 4 < json.size()) {
                                std::string hex = json.substr(pos + 1, 4);
                                unsigned int cp = 0;
                                for (char c : hex) {
                                    cp <<= 4;
                                    if (c >= '0' && c <= '9') cp |= (c - '0');
                                    else if (c >= 'a' && c <= 'f') cp |= (c - 'a' + 10);
                                    else if (c >= 'A' && c <= 'F') cp |= (c - 'A' + 10);
                                }
                                // Encode as UTF-8
                                if (cp < 0x80) {
                                    token += static_cast<char>(cp);
                                } else if (cp < 0x800) {
                                    token += static_cast<char>(0xC0 | (cp >> 6));
                                    token += static_cast<char>(0x80 | (cp & 0x3F));
                                } else {
                                    token += static_cast<char>(0xE0 | (cp >> 12));
                                    token += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                    token += static_cast<char>(0x80 | (cp & 0x3F));
                                }
                                pos += 4;
                            }
                            break;
                        }
                        default: token += json[pos]; break;
                    }
                } else {
                    token += json[pos];
                }
                pos++;
            }
            if (pos < json.size()) pos++;  // skip closing quote

            // Skip colon
            while (pos < json.size() && json[pos] != ':') pos++;
            if (pos < json.size()) pos++;

            // Read ID
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t'))
                pos++;
            char* end = nullptr;
            int id = static_cast<int>(strtol(json.c_str() + pos, &end, 10));
            if (end > json.c_str() + pos) {
                id_to_tok[id] = token;
                token_to_id[token] = id;
                if (id > max_id) max_id = id;
                pos = static_cast<size_t>(end - json.c_str());
            }
        }

        // Build vocab vector
        if (max_id >= 0) {
            vocab.resize(max_id + 1);
            for (const auto& kv : id_to_tok) {
                vocab[kv.first] = kv.second;
            }
        }
    }

    void parse_merges(const std::string& json) {
        // Find "merges" array inside "model" object
        size_t model_pos = json.find("\"model\"");
        if (model_pos == std::string::npos) return;
        size_t merges_pos = json.find("\"merges\"", model_pos);
        if (merges_pos == std::string::npos) return;

        size_t bracket = json.find('[', merges_pos);
        if (bracket == std::string::npos) return;

        size_t pos = bracket + 1;
        while (pos < json.size()) {
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                                         json[pos] == '\n' || json[pos] == '\r' ||
                                         json[pos] == ','))
                pos++;
            if (pos >= json.size() || json[pos] == ']') break;

            if (json[pos] != '"') break;
            pos++;
            std::string merge_str;
            while (pos < json.size() && json[pos] != '"') {
                if (json[pos] == '\\' && pos + 1 < json.size()) {
                    pos++;
                    merge_str += json[pos];
                } else {
                    merge_str += json[pos];
                }
                pos++;
            }
            if (pos < json.size()) pos++;

            // Split merge string by space: "token1 token2"
            size_t sp = merge_str.find(' ');
            if (sp != std::string::npos) {
                merges.push_back({merge_str.substr(0, sp),
                                  merge_str.substr(sp + 1)});
            }
        }
    }

    void parse_added_tokens(const std::string& json) {
        size_t pos = json.find("\"added_tokens\"");
        if (pos == std::string::npos) return;
        size_t bracket = json.find('[', pos);
        if (bracket == std::string::npos) return;

        pos = bracket + 1;
        int depth = 1;
        while (pos < json.size() && depth > 0) {
            if (json[pos] == '[') { depth++; pos++; continue; }
            if (json[pos] == ']') { depth--; pos++; continue; }
            if (json[pos] == '{') {
                // Parse added token object
                size_t obj_start = pos;
                int obj_depth = 1;
                pos++;
                while (pos < json.size() && obj_depth > 0) {
                    if (json[pos] == '{') obj_depth++;
                    else if (json[pos] == '}') obj_depth--;
                    else if (json[pos] == '"') {
                        pos++;
                        while (pos < json.size() && json[pos] != '"') {
                            if (json[pos] == '\\') pos++;
                            pos++;
                        }
                    }
                    pos++;
                }
                std::string obj = json.substr(obj_start, pos - obj_start);

                std::string content = hf_json_get_str(obj, "content");
                int id = static_cast<int>(hf_json_get_int(obj, "id", -1));
                if (!content.empty() && id >= 0) {
                    added_tokens.push_back({content, id});
                    // Add to token_to_id if not already there
                    if (token_to_id.find(content) == token_to_id.end()) {
                        token_to_id[content] = id;
                    }
                    // Extend vocab if needed
                    if (id >= static_cast<int>(vocab.size())) {
                        vocab.resize(id + 1);
                    }
                    vocab[id] = content;
                }
            } else {
                pos++;
            }
        }
    }
};

// ---- Find SafeTensors files in a directory ----

static inline std::vector<std::string> find_safetensors_files(const std::string& dir) {
    std::vector<std::string> files;

    // Try single file first
    std::string single = dir + "/model.safetensors";
    if (file_exists(single)) {
        files.push_back(single);
        return files;
    }

    // Try sharded files: model-00001-of-NNNNN.safetensors
    for (int i = 1; i < 1000; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "/model-%05d-of-", i);
        std::string prefix = dir + buf;

        // Try to find any matching file
        bool found = false;
        for (int total = i; total < 1000; total++) {
            snprintf(buf, sizeof(buf), "%05d.safetensors", total);
            std::string path = prefix + buf;
            if (file_exists(path)) {
                files.push_back(path);
                found = true;
                break;
            }
        }
        if (!found) break;
    }

    return files;
}

#endif // LLM_HF_LOADER_H
