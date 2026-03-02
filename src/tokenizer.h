#ifndef LLM_TOKENIZER_H
#define LLM_TOKENIZER_H

// BPE tokenizer that reads vocabulary from a GGUF model file.

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"

class Tokenizer {
public:
    int vocab_size = 0;
    int bos_token_id = 1;
    int eos_token_id = 2;
    int pad_token_id = -1;

    std::vector<std::string> id_to_token;
    std::vector<float> token_scores;
    std::unordered_map<std::string, int> token_to_id;

    // Merge info for BPE
    struct BPEMerge {
        std::string left;
        std::string right;
        int rank;
    };
    std::vector<BPEMerge> merges;
    std::unordered_map<std::string, int> merge_ranks;

    // Tokenizer model type ("gpt2", "llama", etc.)
    std::string tokenizer_model;

    // GPT-2 byte-to-unicode mapping (for gpt2-style tokenizers like Qwen)
    std::unordered_map<char32_t, uint8_t> unicode_to_byte;
    std::unordered_map<uint8_t, std::string> byte_to_unicode;

    bool load_from_gguf(const GGUFFile& gguf) {
        // Read vocabulary tokens
        auto tokens_it = gguf.metadata.find("tokenizer.ggml.tokens");
        if (tokens_it == gguf.metadata.end()) {
            fprintf(stderr, "Error: tokenizer.ggml.tokens not found in GGUF\n");
            return false;
        }

        const auto& tokens_val = tokens_it->second;
        vocab_size = static_cast<int>(tokens_val.arr.size());
        id_to_token.resize(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            id_to_token[i] = tokens_val.arr[i].str;
            token_to_id[id_to_token[i]] = i;
        }

        // Read token scores (optional)
        auto scores_it = gguf.metadata.find("tokenizer.ggml.scores");
        if (scores_it != gguf.metadata.end()) {
            token_scores.resize(vocab_size);
            for (int i = 0; i < vocab_size && i < static_cast<int>(scores_it->second.arr.size()); i++) {
                token_scores[i] = scores_it->second.arr[i].f32;
            }
        }

        // Read BPE merges (optional)
        auto merges_it = gguf.metadata.find("tokenizer.ggml.merges");
        if (merges_it != gguf.metadata.end()) {
            int rank = 0;
            for (const auto& m : merges_it->second.arr) {
                auto space_pos = m.str.find(' ');
                if (space_pos != std::string::npos) {
                    BPEMerge merge;
                    merge.left = m.str.substr(0, space_pos);
                    merge.right = m.str.substr(space_pos + 1);
                    merge.rank = rank;
                    merge_ranks[merge.left + " " + merge.right] = rank;
                    merges.push_back(merge);
                    rank++;
                }
            }
        }

        // Read special token IDs
        bos_token_id = static_cast<int>(gguf.get_i64("tokenizer.ggml.bos_token_id", 1));
        eos_token_id = static_cast<int>(gguf.get_i64("tokenizer.ggml.eos_token_id", 2));
        pad_token_id = static_cast<int>(gguf.get_i64("tokenizer.ggml.padding_token_id", -1));

        // Read tokenizer model type
        tokenizer_model = gguf.get_str("tokenizer.ggml.model", "");

        // Initialize GPT-2 byte mapping if needed (used by Qwen, GPT-2, etc.)
        if (tokenizer_model == "gpt2") {
            init_gpt2_byte_mapping();
        }

        fprintf(stderr, "Tokenizer: vocab_size=%d, bos=%d, eos=%d\n",
                vocab_size, bos_token_id, eos_token_id);
        return true;
    }

    // Simple encode: byte-level BPE encoding
    std::vector<int> encode(const std::string& text, bool add_bos = true) const {
        std::vector<int> tokens;
        if (add_bos) {
            tokens.push_back(bos_token_id);
        }

        if (!merges.empty()) {
            // BPE tokenization
            bpe_encode(text, tokens);
        } else {
            // Fallback: sentencepiece-style greedy longest match
            sp_encode(text, tokens);
        }

        return tokens;
    }

    // Decode a single token ID to string
    std::string decode(int token_id) const {
        if (token_id < 0 || token_id >= vocab_size) return "";

        const std::string& token = id_to_token[token_id];

        // GPT-2 style: reverse byte-to-unicode mapping
        if (tokenizer_model == "gpt2") {
            return decode_gpt2(token);
        }

        // Handle sentencepiece space character (▁ = U+2581)
        std::string result;
        size_t i = 0;
        while (i < token.size()) {
            // UTF-8 encoded ▁ is 0xE2 0x96 0x81
            if (i + 2 < token.size() &&
                static_cast<uint8_t>(token[i]) == 0xE2 &&
                static_cast<uint8_t>(token[i + 1]) == 0x96 &&
                static_cast<uint8_t>(token[i + 2]) == 0x81) {
                result += ' ';
                i += 3;
            } else if (token[i] == '<' && token.find('>', i) != std::string::npos) {
                // Handle special tokens like <s>, </s>, <0x0A>, etc.
                size_t end = token.find('>', i);
                std::string special = token.substr(i, end - i + 1);
                if (special.size() == 6 && special[1] == '0' && special[2] == 'x') {
                    // Byte token like <0x0A>
                    int byte_val = 0;
                    for (size_t j = 3; j < 5; j++) {
                        byte_val <<= 4;
                        char c = special[j];
                        if (c >= '0' && c <= '9') byte_val += c - '0';
                        else if (c >= 'A' && c <= 'F') byte_val += c - 'A' + 10;
                        else if (c >= 'a' && c <= 'f') byte_val += c - 'a' + 10;
                    }
                    result += static_cast<char>(byte_val);
                }
                // Skip other special tokens in output
                i = end + 1;
            } else {
                result += token[i];
                i++;
            }
        }
        return result;
    }

    // Initialize GPT-2 byte-to-unicode mapping tables
    void init_gpt2_byte_mapping() {
        // Track which bytes are in the "safe" ranges (map to themselves)
        bool safe[256] = {};
        for (int b = 0x21; b <= 0x7E; b++) safe[b] = true;  // '!' to '~'
        for (int b = 0xA1; b <= 0xAC; b++) safe[b] = true;  // '¡' to '¬'
        for (int b = 0xAE; b <= 0xFF; b++) safe[b] = true;  // '®' to 'ÿ'

        // Build mapping: safe bytes map to same codepoint,
        // remaining 68 bytes map to U+0100 through U+0143
        int n = 0;
        for (int b = 0; b < 256; b++) {
            int codepoint = safe[b] ? b : (256 + n++);

            unicode_to_byte[static_cast<char32_t>(codepoint)] = static_cast<uint8_t>(b);

            // Convert codepoint to UTF-8 string
            std::string utf8;
            if (codepoint < 0x80) {
                utf8 += static_cast<char>(codepoint);
            } else {
                // All GPT-2 mapped codepoints are < 0x200, so 2-byte UTF-8
                utf8 += static_cast<char>(0xC0 | (codepoint >> 6));
                utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
            }
            byte_to_unicode[static_cast<uint8_t>(b)] = utf8;
        }
    }

private:
    // SentencePiece-style greedy longest match encoding
    void sp_encode(const std::string& text, std::vector<int>& tokens) const {
        size_t i = 0;
        while (i < text.size()) {
            int best_len = 0;
            int best_id = -1;

            // Try longest match first
            for (int len = std::min(static_cast<int>(text.size() - i), 64); len > 0; len--) {
                std::string candidate = text.substr(i, len);
                auto it = token_to_id.find(candidate);
                if (it != token_to_id.end()) {
                    best_len = len;
                    best_id = it->second;
                    break;
                }
                // Also try with sentencepiece space prefix for word starts
                if (i == 0 || text[i - 1] == ' ') {
                    std::string sp_candidate = "\xe2\x96\x81" + candidate;
                    auto sp_it = token_to_id.find(sp_candidate);
                    if (sp_it != token_to_id.end()) {
                        best_len = len;
                        best_id = sp_it->second;
                        break;
                    }
                }
            }

            if (best_id >= 0) {
                tokens.push_back(best_id);
                i += best_len;
            } else {
                // Unknown byte - try to find byte-level fallback
                std::string byte_token = "<0x";
                uint8_t byte_val = static_cast<uint8_t>(text[i]);
                const char hex[] = "0123456789ABCDEF";
                byte_token += hex[byte_val >> 4];
                byte_token += hex[byte_val & 0xF];
                byte_token += '>';
                auto it = token_to_id.find(byte_token);
                if (it != token_to_id.end()) {
                    tokens.push_back(it->second);
                } else {
                    // Skip unknown bytes
                    fprintf(stderr, "Warning: unknown byte 0x%02X at position %zu in input\n",
                            byte_val, i);
                }
                i++;
            }
        }
    }

    // BPE encoding
    void bpe_encode(const std::string& text, std::vector<int>& tokens) const {
        if (tokenizer_model == "gpt2") {
            bpe_encode_gpt2(text, tokens);
            return;
        }

        // Standard BPE: split into initial UTF-8 characters
        std::vector<std::string> symbols;
        for (size_t i = 0; i < text.size();) {
            // Handle UTF-8 multi-byte chars
            int len = 1;
            uint8_t c = static_cast<uint8_t>(text[i]);
            if (c >= 0xF0) len = 4;
            else if (c >= 0xE0) len = 3;
            else if (c >= 0xC0) len = 2;
            len = std::min(len, static_cast<int>(text.size() - i));
            symbols.push_back(text.substr(i, len));
            i += len;
        }

        // Iteratively merge pairs with lowest rank
        while (symbols.size() > 1) {
            int best_rank = INT32_MAX;
            int best_idx = -1;
            for (size_t i = 0; i + 1 < symbols.size(); i++) {
                std::string pair = symbols[i] + " " + symbols[i + 1];
                auto it = merge_ranks.find(pair);
                if (it != merge_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = static_cast<int>(i);
                }
            }
            if (best_idx < 0) break;
            symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
            symbols.erase(symbols.begin() + best_idx + 1);
        }

        // Map merged symbols to token IDs
        for (const auto& sym : symbols) {
            auto it = token_to_id.find(sym);
            if (it != token_to_id.end()) {
                tokens.push_back(it->second);
            } else {
                // Fallback to byte-level
                for (uint8_t byte_val : sym) {
                    std::string byte_token = "<0x";
                    const char hex[] = "0123456789ABCDEF";
                    byte_token += hex[byte_val >> 4];
                    byte_token += hex[byte_val & 0xF];
                    byte_token += '>';
                    auto bit = token_to_id.find(byte_token);
                    if (bit != token_to_id.end()) {
                        tokens.push_back(bit->second);
                    }
                }
            }
        }
    }

    // GPT-2 style BPE encoding with byte-to-unicode mapping
    void bpe_encode_gpt2(const std::string& text, std::vector<int>& tokens) const {
        // Pre-tokenize: split at whitespace boundaries, keeping space
        // attached to the beginning of the following word
        std::vector<std::string> chunks;
        std::string current;
        for (size_t i = 0; i < text.size(); i++) {
            uint8_t c = static_cast<uint8_t>(text[i]);
            bool is_ws = (c == ' ' || c == '\n' || c == '\r' || c == '\t');
            if (is_ws && !current.empty()) {
                chunks.push_back(current);
                current.clear();
            }
            current += text[i];
        }
        if (!current.empty()) chunks.push_back(current);

        // Process each chunk independently
        for (const auto& chunk : chunks) {
            // Convert each byte to its GPT-2 unicode character
            std::vector<std::string> symbols;
            for (size_t i = 0; i < chunk.size(); i++) {
                uint8_t b = static_cast<uint8_t>(chunk[i]);
                auto it = byte_to_unicode.find(b);
                if (it != byte_to_unicode.end()) {
                    symbols.push_back(it->second);
                }
            }

            // BPE merge loop
            while (symbols.size() > 1) {
                int best_rank = INT32_MAX;
                int best_idx = -1;
                for (size_t j = 0; j + 1 < symbols.size(); j++) {
                    std::string pair = symbols[j] + " " + symbols[j + 1];
                    auto it = merge_ranks.find(pair);
                    if (it != merge_ranks.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = static_cast<int>(j);
                    }
                }
                if (best_idx < 0) break;
                symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
                symbols.erase(symbols.begin() + best_idx + 1);
            }

            // Map merged symbols to token IDs
            for (const auto& sym : symbols) {
                auto it = token_to_id.find(sym);
                if (it != token_to_id.end()) {
                    tokens.push_back(it->second);
                }
            }
        }
    }

    // Decode GPT-2 style token: reverse byte-to-unicode mapping
    std::string decode_gpt2(const std::string& token) const {
        std::string result;
        size_t i = 0;
        while (i < token.size()) {
            // Decode UTF-8 to Unicode codepoint
            char32_t cp;
            int len;
            uint8_t c = static_cast<uint8_t>(token[i]);
            if (c < 0x80) {
                cp = c; len = 1;
            } else if (c < 0xE0) {
                if (i + 1 >= token.size()) { result += token[i]; i++; continue; }
                cp = (c & 0x1F);
                cp = (cp << 6) | (static_cast<uint8_t>(token[i + 1]) & 0x3F);
                len = 2;
            } else if (c < 0xF0) {
                if (i + 2 >= token.size()) { result += token[i]; i++; continue; }
                cp = (c & 0x0F);
                cp = (cp << 6) | (static_cast<uint8_t>(token[i + 1]) & 0x3F);
                cp = (cp << 6) | (static_cast<uint8_t>(token[i + 2]) & 0x3F);
                len = 3;
            } else {
                if (i + 3 >= token.size()) { result += token[i]; i++; continue; }
                cp = (c & 0x07);
                cp = (cp << 6) | (static_cast<uint8_t>(token[i + 1]) & 0x3F);
                cp = (cp << 6) | (static_cast<uint8_t>(token[i + 2]) & 0x3F);
                cp = (cp << 6) | (static_cast<uint8_t>(token[i + 3]) & 0x3F);
                len = 4;
            }

            auto it = unicode_to_byte.find(cp);
            if (it != unicode_to_byte.end()) {
                result += static_cast<char>(it->second);
            } else {
                // Pass through unmapped characters
                result += token.substr(i, len);
            }
            i += len;
        }
        return result;
    }
};

#endif // LLM_TOKENIZER_H
