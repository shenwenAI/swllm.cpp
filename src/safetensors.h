#ifndef LLM_SAFETENSORS_H
#define LLM_SAFETENSORS_H

// SafeTensors file format parser for loading HuggingFace model files directly.
// Reference: https://huggingface.co/docs/safetensors
//
// SafeTensors format:
//   [8 bytes] uint64 LE header length
//   [header_length bytes] JSON header mapping tensor_name -> {dtype, shape, data_offsets}
//   [remaining bytes] raw tensor data

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "gguf.h"  // for GGMLType enum and helpers

// ---- SafeTensors dtype ----

enum SafeTensorsDType {
    ST_DTYPE_F64,
    ST_DTYPE_F32,
    ST_DTYPE_F16,
    ST_DTYPE_BF16,
    ST_DTYPE_I64,
    ST_DTYPE_I32,
    ST_DTYPE_I16,
    ST_DTYPE_I8,
    ST_DTYPE_U8,
    ST_DTYPE_BOOL,
    ST_DTYPE_UNKNOWN,
};

inline SafeTensorsDType st_dtype_from_string(const std::string& s) {
    if (s == "F64")  return ST_DTYPE_F64;
    if (s == "F32")  return ST_DTYPE_F32;
    if (s == "F16")  return ST_DTYPE_F16;
    if (s == "BF16") return ST_DTYPE_BF16;
    if (s == "I64")  return ST_DTYPE_I64;
    if (s == "I32")  return ST_DTYPE_I32;
    if (s == "I16")  return ST_DTYPE_I16;
    if (s == "I8")   return ST_DTYPE_I8;
    if (s == "U8")   return ST_DTYPE_U8;
    if (s == "BOOL") return ST_DTYPE_BOOL;
    return ST_DTYPE_UNKNOWN;
}

inline const char* st_dtype_name(SafeTensorsDType dtype) {
    switch (dtype) {
        case ST_DTYPE_F64:  return "F64";
        case ST_DTYPE_F32:  return "F32";
        case ST_DTYPE_F16:  return "F16";
        case ST_DTYPE_BF16: return "BF16";
        case ST_DTYPE_I64:  return "I64";
        case ST_DTYPE_I32:  return "I32";
        case ST_DTYPE_I16:  return "I16";
        case ST_DTYPE_I8:   return "I8";
        case ST_DTYPE_U8:   return "U8";
        case ST_DTYPE_BOOL: return "BOOL";
        default:            return "UNKNOWN";
    }
}

inline size_t st_dtype_size(SafeTensorsDType dtype) {
    switch (dtype) {
        case ST_DTYPE_F64:  return 8;
        case ST_DTYPE_F32:  return 4;
        case ST_DTYPE_F16:  return 2;
        case ST_DTYPE_BF16: return 2;
        case ST_DTYPE_I64:  return 8;
        case ST_DTYPE_I32:  return 4;
        case ST_DTYPE_I16:  return 2;
        case ST_DTYPE_I8:   return 1;
        case ST_DTYPE_U8:   return 1;
        case ST_DTYPE_BOOL: return 1;
        default:            return 0;
    }
}

// Map SafeTensors dtype to GGMLType for compatibility
inline GGMLType st_dtype_to_ggml(SafeTensorsDType dtype) {
    switch (dtype) {
        case ST_DTYPE_F32:  return GGML_TYPE_F32;
        case ST_DTYPE_F16:  return GGML_TYPE_F16;
        case ST_DTYPE_BF16: return GGML_TYPE_F16;  // will need BF16->F32 conversion
        case ST_DTYPE_I8:   return GGML_TYPE_I8;
        case ST_DTYPE_I16:  return GGML_TYPE_I16;
        case ST_DTYPE_I32:  return GGML_TYPE_I32;
        case ST_DTYPE_I64:  return GGML_TYPE_I64;
        case ST_DTYPE_F64:  return GGML_TYPE_F64;
        default:            return GGML_TYPE_F32;
    }
}

// ---- SafeTensors tensor info ----

struct SafeTensorInfo {
    std::string name;
    SafeTensorsDType dtype;
    std::vector<int64_t> shape;
    uint64_t data_start;  // offset from start of data section
    uint64_t data_end;
    size_t file_idx = 0;  // which shard file this tensor belongs to

    int64_t num_elements() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    uint64_t num_bytes() const {
        return data_end - data_start;
    }
};

// ---- BFloat16 and Float16 conversion ----
// bf16_to_fp32() is defined in tensor.h (canonical location for type conversions).
// When safetensors.h is used standalone (e.g. in tests), provide a fallback.
#ifndef LLM_TENSOR_H
inline float bf16_to_fp32(uint16_t bf) {
    uint32_t f32_bits = static_cast<uint32_t>(bf) << 16;
    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}
#endif

inline float st_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            exponent = 1;
            while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3FF;
            result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        result = sign | 0x7F800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float f;
    memcpy(&f, &result, sizeof(float));
    return f;
}

// ---- Minimal JSON parser for SafeTensors header ----

class SafeTensorsFile {
public:
    std::map<std::string, SafeTensorInfo> tensors;
    std::map<std::string, std::string> metadata;  // __metadata__ section

    // File data
    const uint8_t* data_ptr = nullptr;  // pointer to tensor data section
    uint64_t header_size = 0;
    std::vector<uint8_t> owned_data;
    size_t file_size = 0;

    // For multi-file models
    struct FileEntry {
        std::vector<uint8_t> data;
        const uint8_t* data_ptr;
        uint64_t header_size;
    };
    std::vector<FileEntry> file_entries;

    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "SafeTensors: cannot open '%s'\n", path.c_str());
            return false;
        }

        fseek(f, 0, SEEK_END);
        file_size = static_cast<size_t>(ftell(f));
        fseek(f, 0, SEEK_SET);

        if (file_size < 8) {
            fprintf(stderr, "SafeTensors: file too small\n");
            fclose(f);
            return false;
        }

        owned_data.resize(file_size);
        if (fread(owned_data.data(), 1, file_size, f) != file_size) {
            fprintf(stderr, "SafeTensors: failed to read '%s'\n", path.c_str());
            fclose(f);
            return false;
        }
        fclose(f);

        return parse_buffer(owned_data.data(), file_size, 0);
    }

    // Load multiple safetensors files (for sharded models)
    bool load_multi(const std::vector<std::string>& paths) {
        for (const auto& path : paths) {
            FILE* f = fopen(path.c_str(), "rb");
            if (!f) {
                fprintf(stderr, "SafeTensors: cannot open '%s'\n", path.c_str());
                return false;
            }

            fseek(f, 0, SEEK_END);
            size_t fsize = static_cast<size_t>(ftell(f));
            fseek(f, 0, SEEK_SET);

            FileEntry entry;
            entry.data.resize(fsize);
            if (fread(entry.data.data(), 1, fsize, f) != fsize) {
                fprintf(stderr, "SafeTensors: failed to read '%s'\n", path.c_str());
                fclose(f);
                return false;
            }
            fclose(f);

            size_t file_idx = file_entries.size();
            entry.data_ptr = entry.data.data();
            entry.header_size = 0;
            file_entries.push_back(std::move(entry));

            if (!parse_buffer(file_entries.back().data_ptr,
                             file_entries.back().data.size(),
                             file_idx)) {
                return false;
            }
        }
        return true;
    }

    // Get raw tensor data pointer
    const void* get_tensor_data(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) return nullptr;
        const auto& info = it->second;

        if (file_entries.empty()) {
            // Single file - validate bounds
            uint64_t data_offset = 8 + header_size + info.data_start;
            if (data_offset + info.num_bytes() > owned_data.size()) {
                fprintf(stderr, "SafeTensors: tensor '%s' data out of bounds\n",
                        name.c_str());
                return nullptr;
            }
            return owned_data.data() + data_offset;
        } else {
            return nullptr;  // Use get_tensor_data_with_info instead
        }
    }

    // Get tensor data with full info
    const void* get_tensor_data(const SafeTensorInfo& info, size_t file_idx) const {
        if (file_entries.empty()) {
            uint64_t data_offset = 8 + header_size + info.data_start;
            if (data_offset + info.num_bytes() > owned_data.size()) {
                return nullptr;
            }
            return owned_data.data() + data_offset;
        }
        if (file_idx < file_entries.size()) {
            const auto& entry = file_entries[file_idx];
            uint64_t data_offset = 8 + entry.header_size + info.data_start;
            if (data_offset + info.num_bytes() > entry.data.size()) {
                return nullptr;
            }
            return entry.data_ptr + data_offset;
        }
        return nullptr;
    }

    // Dequantize tensor data to F32
    // Handles F32, F16, BF16 → F32 conversion
    void dequantize_to_f32(const void* src, float* dst, int64_t n_elements,
                           SafeTensorsDType dtype) const {
        switch (dtype) {
            case ST_DTYPE_F32:
                memcpy(dst, src, n_elements * sizeof(float));
                break;
            case ST_DTYPE_F16: {
                const uint16_t* src16 = static_cast<const uint16_t*>(src);
                for (int64_t i = 0; i < n_elements; i++) {
                    dst[i] = st_fp16_to_fp32(src16[i]);
                }
                break;
            }
            case ST_DTYPE_BF16: {
                const uint16_t* src16 = static_cast<const uint16_t*>(src);
                for (int64_t i = 0; i < n_elements; i++) {
                    dst[i] = bf16_to_fp32(src16[i]);
                }
                break;
            }
            default:
                fprintf(stderr, "SafeTensors: unsupported dtype %s for dequantization\n",
                        st_dtype_name(dtype));
                memset(dst, 0, n_elements * sizeof(float));
                break;
        }
    }

private:
    // Minimal JSON string extraction helpers
    static size_t skip_whitespace(const char* json, size_t pos, size_t len) {
        while (pos < len && (json[pos] == ' ' || json[pos] == '\t' ||
                             json[pos] == '\n' || json[pos] == '\r'))
            pos++;
        return pos;
    }

    // Read a JSON string starting at pos (which must point to '"')
    // Returns the string content and advances pos past the closing '"'
    static std::string read_json_string(const char* json, size_t& pos, size_t len) {
        if (pos >= len || json[pos] != '"') return "";
        pos++;  // skip opening quote
        std::string result;
        while (pos < len && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < len) {
                pos++;
                switch (json[pos]) {
                    case '"':  result += '"';  break;
                    case '\\': result += '\\'; break;
                    case 'n':  result += '\n'; break;
                    case 't':  result += '\t'; break;
                    case 'r':  result += '\r'; break;
                    default:   result += json[pos]; break;
                }
            } else {
                result += json[pos];
            }
            pos++;
        }
        if (pos < len) pos++;  // skip closing quote
        return result;
    }

    // Read a JSON number (integer)
    static int64_t read_json_int(const char* json, size_t& pos, size_t len) {
        size_t start = pos;
        if (pos < len && json[pos] == '-') pos++;
        while (pos < len && json[pos] >= '0' && json[pos] <= '9') pos++;
        return atoll(std::string(json + start, pos - start).c_str());
    }

    // Skip a JSON value (string, number, object, array, bool, null)
    static void skip_json_value(const char* json, size_t& pos, size_t len) {
        pos = skip_whitespace(json, pos, len);
        if (pos >= len) return;

        if (json[pos] == '"') {
            read_json_string(json, pos, len);
        } else if (json[pos] == '{') {
            int depth = 1;
            pos++;
            while (pos < len && depth > 0) {
                if (json[pos] == '{') depth++;
                else if (json[pos] == '}') depth--;
                else if (json[pos] == '"') {
                    pos++;
                    while (pos < len && json[pos] != '"') {
                        if (json[pos] == '\\') pos++;
                        pos++;
                    }
                }
                pos++;
            }
        } else if (json[pos] == '[') {
            int depth = 1;
            pos++;
            while (pos < len && depth > 0) {
                if (json[pos] == '[') depth++;
                else if (json[pos] == ']') depth--;
                else if (json[pos] == '"') {
                    pos++;
                    while (pos < len && json[pos] != '"') {
                        if (json[pos] == '\\') pos++;
                        pos++;
                    }
                }
                pos++;
            }
        } else {
            // number, bool, null
            while (pos < len && json[pos] != ',' && json[pos] != '}' &&
                   json[pos] != ']' && json[pos] != ' ' && json[pos] != '\n')
                pos++;
        }
    }

    // Parse a tensor entry: {"dtype": "F16", "shape": [128, 64], "data_offsets": [0, 16384]}
    SafeTensorInfo parse_tensor_entry(const std::string& name, const char* json,
                                      size_t& pos, size_t len) {
        SafeTensorInfo info;
        info.name = name;
        info.dtype = ST_DTYPE_UNKNOWN;
        info.data_start = 0;
        info.data_end = 0;

        // Expect '{'
        pos = skip_whitespace(json, pos, len);
        if (pos >= len || json[pos] != '{') return info;
        pos++;

        while (pos < len) {
            pos = skip_whitespace(json, pos, len);
            if (pos >= len || json[pos] == '}') { pos++; break; }
            if (json[pos] == ',') { pos++; continue; }

            std::string key = read_json_string(json, pos, len);
            pos = skip_whitespace(json, pos, len);
            if (pos < len && json[pos] == ':') pos++;
            pos = skip_whitespace(json, pos, len);

            if (key == "dtype") {
                std::string dtype_str = read_json_string(json, pos, len);
                info.dtype = st_dtype_from_string(dtype_str);
            } else if (key == "shape") {
                // Parse array of ints
                if (pos < len && json[pos] == '[') {
                    pos++;
                    while (pos < len && json[pos] != ']') {
                        pos = skip_whitespace(json, pos, len);
                        if (json[pos] == ']') break;
                        if (json[pos] == ',') { pos++; continue; }
                        info.shape.push_back(read_json_int(json, pos, len));
                    }
                    if (pos < len) pos++;  // skip ']'
                }
            } else if (key == "data_offsets") {
                // Parse [start, end]
                if (pos < len && json[pos] == '[') {
                    pos++;
                    pos = skip_whitespace(json, pos, len);
                    info.data_start = static_cast<uint64_t>(read_json_int(json, pos, len));
                    pos = skip_whitespace(json, pos, len);
                    if (pos < len && json[pos] == ',') pos++;
                    pos = skip_whitespace(json, pos, len);
                    info.data_end = static_cast<uint64_t>(read_json_int(json, pos, len));
                    pos = skip_whitespace(json, pos, len);
                    if (pos < len && json[pos] == ']') pos++;
                }
            } else {
                skip_json_value(json, pos, len);
            }
        }
        return info;
    }

    bool parse_buffer(const uint8_t* buf, size_t buf_size, size_t file_idx) {
        if (buf_size < 8) return false;

        // Read header length
        uint64_t hdr_len;
        memcpy(&hdr_len, buf, 8);

        if (8 + hdr_len > buf_size) {
            fprintf(stderr, "SafeTensors: header too large (%lu bytes)\n",
                    static_cast<unsigned long>(hdr_len));
            return false;
        }

        if (file_idx == 0 && file_entries.empty()) {
            header_size = hdr_len;
            data_ptr = buf + 8 + hdr_len;
        } else if (file_idx < file_entries.size()) {
            file_entries[file_idx].header_size = hdr_len;
            file_entries[file_idx].data_ptr = buf;
        }

        // Parse JSON header
        const char* json = reinterpret_cast<const char*>(buf + 8);
        size_t json_len = static_cast<size_t>(hdr_len);
        size_t pos = 0;

        pos = skip_whitespace(json, pos, json_len);
        if (pos >= json_len || json[pos] != '{') {
            fprintf(stderr, "SafeTensors: invalid JSON header\n");
            return false;
        }
        pos++;

        while (pos < json_len) {
            pos = skip_whitespace(json, pos, json_len);
            if (pos >= json_len || json[pos] == '}') break;
            if (json[pos] == ',') { pos++; continue; }

            std::string key = read_json_string(json, pos, json_len);
            pos = skip_whitespace(json, pos, json_len);
            if (pos < json_len && json[pos] == ':') pos++;
            pos = skip_whitespace(json, pos, json_len);

            if (key == "__metadata__") {
                // Parse metadata object
                if (pos < json_len && json[pos] == '{') {
                    pos++;
                    while (pos < json_len && json[pos] != '}') {
                        pos = skip_whitespace(json, pos, json_len);
                        if (json[pos] == '}') break;
                        if (json[pos] == ',') { pos++; continue; }
                        std::string mk = read_json_string(json, pos, json_len);
                        pos = skip_whitespace(json, pos, json_len);
                        if (pos < json_len && json[pos] == ':') pos++;
                        pos = skip_whitespace(json, pos, json_len);
                        std::string mv = read_json_string(json, pos, json_len);
                        metadata[mk] = mv;
                    }
                    if (pos < json_len) pos++;  // skip '}'
                } else {
                    skip_json_value(json, pos, json_len);
                }
            } else {
                // Tensor entry
                SafeTensorInfo info = parse_tensor_entry(key, json, pos, json_len);
                info.file_idx = file_idx;
                tensors[key] = info;
            }
        }

        return true;
    }
};

#endif // LLM_SAFETENSORS_H
