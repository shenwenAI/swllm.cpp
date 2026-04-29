#ifndef LLM_SAMPLER_H
#define LLM_SAMPLER_H

// Token sampling strategies: greedy, temperature, top-k, top-p (nucleus).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

struct SamplerConfig {
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.9f;
    float repeat_penalty = 1.2f;  // Increased from 1.1 for small models
    int no_repeat_ngram_size = 3; // Prevent repeating n-grams (critical for small models)
    uint64_t seed = 0;  // 0 = random seed
};

class Sampler {
public:
    SamplerConfig config;
    std::mt19937 rng;

    explicit Sampler(const SamplerConfig& cfg) : config(cfg) {
        if (config.seed == 0) {
            std::random_device rd;
            rng.seed(rd());
        } else {
            rng.seed(static_cast<unsigned>(config.seed));
        }
    }

    // Sample a token from logits array of size vocab_size
    int sample(const float* logits, int vocab_size,
               const std::vector<int>& recent_tokens = {},
               const std::vector<int>& all_generated_tokens = {}) {
        // Copy logits
        std::vector<float> probs(logits, logits + vocab_size);

        // Apply repetition penalty (token-level)
        if (config.repeat_penalty != 1.0f && !recent_tokens.empty()) {
            for (int tok : recent_tokens) {
                if (tok >= 0 && tok < vocab_size) {
                    if (probs[tok] > 0) {
                        probs[tok] /= config.repeat_penalty;
                    } else {
                        probs[tok] *= config.repeat_penalty;
                    }
                }
            }
        }

        // Apply no-repeat-ngram penalty (block complete n-grams from repeating)
        if (config.no_repeat_ngram_size > 0 && all_generated_tokens.size() >= static_cast<size_t>(config.no_repeat_ngram_size - 1)) {
            // Build the current (n-1)-gram context
            int n = config.no_repeat_ngram_size;
            size_t ctx_start = all_generated_tokens.size() - (n - 1);
            
            // Find all tokens that would complete a repeated n-gram
            for (size_t i = 0; i <= all_generated_tokens.size() - static_cast<size_t>(n - 1); i++) {
                bool match = true;
                for (int j = 0; j < n - 1; j++) {
                    if (all_generated_tokens[ctx_start + j] != all_generated_tokens[i + j]) {
                        match = false;
                        break;
                    }
                }
                if (match && i + n - 1 < all_generated_tokens.size()) {
                    // This token would complete a repeated n-gram, suppress it
                    int forbidden_token = all_generated_tokens[i + n - 1];
                    if (forbidden_token >= 0 && forbidden_token < vocab_size) {
                        probs[forbidden_token] = -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }

        // Greedy if temperature <= 0
        if (config.temperature <= 0.0f) {
            return static_cast<int>(
                std::max_element(probs.begin(), probs.end()) - probs.begin());
        }

        // Apply temperature
        for (int i = 0; i < vocab_size; i++) {
            probs[i] /= config.temperature;
        }

        // Create sorted index for top-k / top-p
        struct TokenProb {
            int id;
            float logit;
        };
        std::vector<TokenProb> candidates(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            candidates[i] = {i, probs[i]};
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const TokenProb& a, const TokenProb& b) {
                      return a.logit > b.logit;
                  });

        // Top-K filtering
        int k = config.top_k > 0 ? std::min(config.top_k, vocab_size) : vocab_size;
        candidates.resize(k);

        // Softmax over remaining candidates
        float max_logit = candidates[0].logit;
        float sum = 0.0f;
        for (auto& c : candidates) {
            c.logit = expf(c.logit - max_logit);
            sum += c.logit;
        }
        for (auto& c : candidates) {
            c.logit /= sum;
        }

        // Top-P (nucleus) filtering
        if (config.top_p < 1.0f) {
            float cumsum = 0.0f;
            int cutoff = static_cast<int>(candidates.size());
            for (int i = 0; i < static_cast<int>(candidates.size()); i++) {
                cumsum += candidates[i].logit;
                if (cumsum >= config.top_p) {
                    cutoff = i + 1;
                    break;
                }
            }
            candidates.resize(cutoff);

            // Renormalize
            sum = 0.0f;
            for (auto& c : candidates) sum += c.logit;
            for (auto& c : candidates) c.logit /= sum;
        }

        // Sample from distribution
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        float cumsum = 0.0f;
        for (const auto& c : candidates) {
            cumsum += c.logit;
            if (r <= cumsum) {
                return c.id;
            }
        }
        return candidates.back().id;
    }
};

#endif // LLM_SAMPLER_H
