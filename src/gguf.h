#ifndef LLM_GGUF_H
#define LLM_GGUF_H

// GGUF file format parser for loading GGML-based model files.
// Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// ---- GGUF / GGML type definitions ----

enum GGMLType : uint32_t {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_F8_E4M3 = 37,  // FP8: 1 sign + 4 exponent + 3 mantissa (bias=7, no Inf)
    GGML_TYPE_F8_E5M2 = 38,  // FP8: 1 sign + 5 exponent + 2 mantissa (bias=15)
};

enum GGUFMetadataValueType : uint32_t {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// Returns the byte size of a single element for a given GGML type.
// For quantized types this returns the block size.
inline size_t ggml_type_size(GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q4_0: return 2 + 16;   // block of 32: 1 f16 scale + 16 bytes of nibbles
        case GGML_TYPE_Q4_1: return 2 + 2 + 16;
        case GGML_TYPE_Q5_0: return 2 + 4 + 16;
        case GGML_TYPE_Q5_1: return 2 + 2 + 4 + 16;
        case GGML_TYPE_Q8_0: return 2 + 32;   // block of 32: 1 f16 scale + 32 bytes
        case GGML_TYPE_Q8_1: return 4 + 4 + 32;
        case GGML_TYPE_I8:       return 1;
        case GGML_TYPE_I16:      return 2;
        case GGML_TYPE_I32:      return 4;
        case GGML_TYPE_I64:      return 8;
        case GGML_TYPE_F64:      return 8;
        case GGML_TYPE_F8_E4M3:  return 1;  // 1 byte per element
        case GGML_TYPE_F8_E5M2:  return 1;  // 1 byte per element
        default: return 0;
    }
}

// Returns the block size (number of elements per quantization block).
inline size_t ggml_block_size(GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
        case GGML_TYPE_F64:
        case GGML_TYPE_F8_E4M3:
        case GGML_TYPE_F8_E5M2:
            return 1;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
            return 32;
        default:
            return 1;
    }
}

inline const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_I8:   return "I8";
        case GGML_TYPE_I16:  return "I16";
        case GGML_TYPE_I32:  return "I32";
        case GGML_TYPE_I64:  return "I64";
        case GGML_TYPE_F64:  return "F64";
        case GGML_TYPE_F8_E4M3: return "F8_E4M3";
        case GGML_TYPE_F8_E5M2: return "F8_E5M2";
        default: return "UNKNOWN";
    }
}

// ---- GGUF metadata value ----

struct GGUFString {
    uint64_t len;
    std::string str;
};

struct GGUFValue {
    GGUFMetadataValueType type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        bool     b;
        uint64_t u64;
        int64_t  i64;
        double   f64;
    };
    std::string str;
    std::vector<GGUFValue> arr;
    GGUFMetadataValueType arr_type;

    int64_t to_int() const {
        switch (type) {
            case GGUF_TYPE_UINT8:   return u8;
            case GGUF_TYPE_INT8:    return i8;
            case GGUF_TYPE_UINT16:  return u16;
            case GGUF_TYPE_INT16:   return i16;
            case GGUF_TYPE_UINT32:  return u32;
            case GGUF_TYPE_INT32:   return i32;
            case GGUF_TYPE_UINT64:  return static_cast<int64_t>(u64);
            case GGUF_TYPE_INT64:   return i64;
            case GGUF_TYPE_BOOL:    return b ? 1 : 0;
            default: return 0;
        }
    }

    double to_float() const {
        if (type == GGUF_TYPE_FLOAT32) return f32;
        if (type == GGUF_TYPE_FLOAT64) return f64;
        return static_cast<double>(to_int());
    }

    const std::string& to_string() const { return str; }
};

// ---- GGUF tensor info ----

struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t dims[4];
    GGMLType type;
    uint64_t offset;  // offset from start of tensor data section

    uint64_t num_elements() const {
        uint64_t n = 1;
        for (uint32_t i = 0; i < n_dims; i++) n *= dims[i];
        return n;
    }

    uint64_t num_bytes() const {
        uint64_t n = num_elements();
        size_t bs = ggml_block_size(type);
        size_t ts = ggml_type_size(type);
        return (n / bs) * ts;
    }
};

// ---- GGUF file ----

class GGUFFile {
public:
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_kv_count = 0;
    uint64_t alignment = 32;
    uint64_t data_offset = 0;

    std::map<std::string, GGUFValue> metadata;
    std::map<std::string, GGUFTensorInfo> tensors;

    // Mapped file data (points into memory-mapped or loaded file)
    const uint8_t* file_data = nullptr;
    size_t file_size = 0;
    std::vector<uint8_t> owned_data;  // owns the data if loaded into memory

    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "Error: cannot open file '%s'\n", path.c_str());
            return false;
        }

        fseek(f, 0, SEEK_END);
        file_size = static_cast<size_t>(ftell(f));
        fseek(f, 0, SEEK_SET);

        owned_data.resize(file_size);
        if (fread(owned_data.data(), 1, file_size, f) != file_size) {
            fprintf(stderr, "Error: failed to read file '%s'\n", path.c_str());
            fclose(f);
            return false;
        }
        fclose(f);
        file_data = owned_data.data();

        return parse();
    }

    // Get pointer to tensor data in the loaded file
    const void* get_tensor_data(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) return nullptr;
        return file_data + data_offset + it->second.offset;
    }

    // Check if a metadata key exists
    bool has_metadata(const std::string& key) const {
        return metadata.find(key) != metadata.end();
    }

    // Get metadata as integer
    int64_t get_i64(const std::string& key, int64_t default_val = 0) const {
        auto it = metadata.find(key);
        if (it == metadata.end()) return default_val;
        return it->second.to_int();
    }

    // Get metadata as float
    double get_f64(const std::string& key, double default_val = 0.0) const {
        auto it = metadata.find(key);
        if (it == metadata.end()) return default_val;
        return it->second.to_float();
    }

    // Get metadata as string
    std::string get_str(const std::string& key, const std::string& default_val = "") const {
        auto it = metadata.find(key);
        if (it == metadata.end()) return default_val;
        return it->second.to_string();
    }

private:
    size_t pos_ = 0;

    template<typename T>
    T read() {
        if (pos_ + sizeof(T) > file_size) {
            throw std::runtime_error("unexpected end of file");
        }
        T val;
        memcpy(&val, file_data + pos_, sizeof(T));
        pos_ += sizeof(T);
        return val;
    }

    std::string read_string() {
        uint64_t len = read<uint64_t>();
        if (pos_ + len > file_size) {
            throw std::runtime_error("unexpected end of file reading string");
        }
        std::string s(reinterpret_cast<const char*>(file_data + pos_), len);
        pos_ += len;
        return s;
    }

    GGUFValue read_value(GGUFMetadataValueType type) {
        GGUFValue val;
        val.type = type;
        switch (type) {
            case GGUF_TYPE_UINT8:   val.u8  = read<uint8_t>();  break;
            case GGUF_TYPE_INT8:    val.i8  = read<int8_t>();   break;
            case GGUF_TYPE_UINT16:  val.u16 = read<uint16_t>(); break;
            case GGUF_TYPE_INT16:   val.i16 = read<int16_t>();  break;
            case GGUF_TYPE_UINT32:  val.u32 = read<uint32_t>(); break;
            case GGUF_TYPE_INT32:   val.i32 = read<int32_t>();  break;
            case GGUF_TYPE_FLOAT32: val.f32 = read<float>();    break;
            case GGUF_TYPE_BOOL:    val.b   = read<uint8_t>() != 0; break;
            case GGUF_TYPE_STRING:  val.str = read_string();    break;
            case GGUF_TYPE_UINT64:  val.u64 = read<uint64_t>(); break;
            case GGUF_TYPE_INT64:   val.i64 = read<int64_t>();  break;
            case GGUF_TYPE_FLOAT64: val.f64 = read<double>();   break;
            case GGUF_TYPE_ARRAY: {
                val.arr_type = static_cast<GGUFMetadataValueType>(read<uint32_t>());
                uint64_t arr_len = read<uint64_t>();
                val.arr.reserve(static_cast<size_t>(arr_len));
                for (uint64_t i = 0; i < arr_len; i++) {
                    val.arr.push_back(read_value(val.arr_type));
                }
                break;
            }
        }
        return val;
    }

    bool parse() {
        pos_ = 0;
        try {
            // Magic number: "GGUF"
            uint32_t magic = read<uint32_t>();
            // "GGUF" in ASCII: 0x47='G', 0x47='G', 0x55='U', 0x46='F'
            // As little-endian uint32: 0x46554747
            if (magic != 0x46554747u) {
                fprintf(stderr, "Error: not a GGUF file (magic: 0x%08X)\n", magic);
                return false;
            }

            version = read<uint32_t>();
            if (version < 2 || version > 3) {
                fprintf(stderr, "Error: unsupported GGUF version %u\n", version);
                return false;
            }

            tensor_count = read<uint64_t>();
            metadata_kv_count = read<uint64_t>();

            // Read metadata
            for (uint64_t i = 0; i < metadata_kv_count; i++) {
                std::string key = read_string();
                auto vtype = static_cast<GGUFMetadataValueType>(read<uint32_t>());
                GGUFValue val = read_value(vtype);
                if (key == "general.alignment") {
                    alignment = static_cast<uint64_t>(val.to_int());
                }
                metadata[key] = std::move(val);
            }

            // Read tensor infos
            for (uint64_t i = 0; i < tensor_count; i++) {
                GGUFTensorInfo info;
                info.name = read_string();
                info.n_dims = read<uint32_t>();
                for (uint32_t d = 0; d < info.n_dims; d++) {
                    info.dims[d] = read<uint64_t>();
                }
                for (uint32_t d = info.n_dims; d < 4; d++) {
                    info.dims[d] = 1;
                }
                info.type = static_cast<GGMLType>(read<uint32_t>());
                info.offset = read<uint64_t>();
                tensors[info.name] = info;
            }

            // Calculate data offset (aligned)
            data_offset = pos_;
            if (alignment > 0) {
                data_offset = ((data_offset + alignment - 1) / alignment) * alignment;
            }

            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "Error parsing GGUF: %s\n", e.what());
            return false;
        }
    }
};

#endif // LLM_GGUF_H
