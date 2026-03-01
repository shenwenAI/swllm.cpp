// llm.cpp - A lightweight C++ LLM inference engine
//
// Supports loading GGUF models and running text generation on CPU or GPU.
// Usage: ./llm -m <model_path> [options] -p "prompt"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "model.h"
#include "sampler.h"

struct RunConfig {
    std::string model_path;
    std::string prompt = "Hello";
    int max_tokens = 256;
    bool use_gpu = false;
    int num_threads = 0;  // 0 = auto
    SamplerConfig sampler;
    bool interactive = false;
    bool show_info = false;
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "llm.cpp - Lightweight LLM inference engine\n"
        "\n"
        "Usage: %s -m <model.gguf> [options] [-p \"prompt\"]\n"
        "\n"
        "Required:\n"
        "  -m, --model <path>       Path to GGUF model file\n"
        "\n"
        "Generation options:\n"
        "  -p, --prompt <text>      Input prompt (default: \"Hello\")\n"
        "  -n, --max-tokens <N>     Max tokens to generate (default: 256)\n"
        "  -t, --temperature <F>    Sampling temperature (default: 0.8)\n"
        "  --top-k <N>              Top-K sampling (default: 40)\n"
        "  --top-p <F>              Top-P nucleus sampling (default: 0.9)\n"
        "  --repeat-penalty <F>     Repetition penalty (default: 1.1)\n"
        "  --seed <N>               Random seed (default: random)\n"
        "\n"
        "Backend options:\n"
        "  --gpu                    Use GPU (CUDA) backend\n"
        "  --cpu                    Use CPU backend (default)\n"
        "  --threads <N>            Number of CPU threads (default: auto)\n"
        "\n"
        "Other:\n"
        "  -i, --interactive        Interactive chat mode\n"
        "  --info                   Show model info and exit\n"
        "  -h, --help               Show this help\n"
        "\n"
        "Examples:\n"
        "  %s -m model.gguf -p \"Once upon a time\"\n"
        "  %s -m model.gguf --gpu -n 512 -p \"Explain quantum computing\"\n"
        "  %s -m model.gguf -i\n",
        prog, prog, prog, prog);
}

static bool parse_args(int argc, char** argv, RunConfig& cfg) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            cfg.prompt = argv[++i];
        } else if ((arg == "-n" || arg == "--max-tokens") && i + 1 < argc) {
            cfg.max_tokens = atoi(argv[++i]);
        } else if ((arg == "-t" || arg == "--temperature") && i + 1 < argc) {
            cfg.sampler.temperature = static_cast<float>(atof(argv[++i]));
        } else if (arg == "--top-k" && i + 1 < argc) {
            cfg.sampler.top_k = atoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            cfg.sampler.top_p = static_cast<float>(atof(argv[++i]));
        } else if (arg == "--repeat-penalty" && i + 1 < argc) {
            cfg.sampler.repeat_penalty = static_cast<float>(atof(argv[++i]));
        } else if (arg == "--seed" && i + 1 < argc) {
            cfg.sampler.seed = static_cast<uint64_t>(atoll(argv[++i]));
        } else if (arg == "--gpu") {
            cfg.use_gpu = true;
        } else if (arg == "--cpu") {
            cfg.use_gpu = false;
        } else if (arg == "--threads" && i + 1 < argc) {
            cfg.num_threads = atoi(argv[++i]);
        } else if (arg == "-i" || arg == "--interactive") {
            cfg.interactive = true;
        } else if (arg == "--info") {
            cfg.show_info = true;
        } else if (arg == "-h" || arg == "--help") {
            return false;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    if (cfg.model_path.empty()) {
        fprintf(stderr, "Error: model path is required (-m <path>)\n\n");
        return false;
    }

    return true;
}

static void generate(Model& model, Sampler& sampler, const std::string& prompt,
                     int max_tokens) {
    // Encode prompt
    std::vector<int> tokens = model.tokenizer.encode(prompt, true);

    fprintf(stderr, "Prompt tokens: %zu\n", tokens.size());
    fprintf(stderr, "Generating up to %d tokens...\n\n", max_tokens);

    auto start = std::chrono::high_resolution_clock::now();
    int total_tokens = 0;
    int prompt_tokens = static_cast<int>(tokens.size());

    // Track recent tokens for repetition penalty
    std::vector<int> recent_tokens;
    int recent_window = 64;

    // Process prompt tokens (prefill)
    for (int i = 0; i < prompt_tokens; i++) {
        model.forward(tokens[i], i);
    }

    auto prefill_end = std::chrono::high_resolution_clock::now();

    // Generate tokens
    int pos = prompt_tokens;
    int next_token = -1;
    for (int i = 0; i < max_tokens; i++) {
        float* logits_ptr;
        if (i == 0) {
            // First generated token uses logits from last prompt token
            logits_ptr = model.logits.data();
        } else {
            logits_ptr = model.forward(next_token, pos - 1);
        }

        // Sample next token
        next_token = sampler.sample(logits_ptr, model.config.vocab_size, recent_tokens);

        // Check for EOS
        if (next_token == model.tokenizer.eos_token_id) {
            break;
        }

        // Track for repetition penalty
        recent_tokens.push_back(next_token);
        if (static_cast<int>(recent_tokens.size()) > recent_window) {
            recent_tokens.erase(recent_tokens.begin());
        }

        // Decode and print
        std::string token_str = model.tokenizer.decode(next_token);
        printf("%s", token_str.c_str());
        fflush(stdout);

        pos++;
        total_tokens++;
    }
    printf("\n");

    auto end = std::chrono::high_resolution_clock::now();
    double prefill_time = std::chrono::duration<double>(prefill_end - start).count();
    double total_time = std::chrono::duration<double>(end - start).count();
    double gen_time = total_time - prefill_time;

    fprintf(stderr, "\n--- Stats ---\n");
    fprintf(stderr, "Prompt tokens:     %d\n", prompt_tokens);
    fprintf(stderr, "Generated tokens:  %d\n", total_tokens);
    if (prefill_time > 0 && prompt_tokens > 0) {
        fprintf(stderr, "Prefill speed:     %.1f tokens/s\n",
                prompt_tokens / prefill_time);
    }
    if (gen_time > 0 && total_tokens > 0) {
        fprintf(stderr, "Generation speed:  %.1f tokens/s\n",
                total_tokens / gen_time);
    }
    fprintf(stderr, "Total time:        %.2f s\n", total_time);
}

static void interactive_mode(Model& model, Sampler& sampler, int max_tokens) {
    fprintf(stderr, "Interactive mode. Type your prompt and press Enter.\n");
    fprintf(stderr, "Type 'quit' or 'exit' to stop.\n\n");

    char line[4096];
    while (true) {
        printf("> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) break;

        // Remove trailing newline
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }

        if (len == 0) continue;
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) break;

        // Reset KV cache for new conversation
        model.kv_cache.clear();

        generate(model, sampler, line, max_tokens);
        printf("\n");
    }
}

int main(int argc, char** argv) {
    RunConfig cfg;

    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    // Check GPU support
    Backend backend = Backend::CPU;
    if (cfg.use_gpu) {
#ifdef LLM_USE_CUDA
        backend = Backend::CUDA;
        fprintf(stderr, "Using CUDA GPU backend\n");
#else
        fprintf(stderr, "Warning: GPU support not compiled. "
                "Rebuild with -DLLM_CUDA=ON. Falling back to CPU.\n");
#endif
    }

    // Set thread count
#ifdef LLM_USE_OPENMP
    if (cfg.num_threads > 0) {
        omp_set_num_threads(cfg.num_threads);
    }
    fprintf(stderr, "CPU threads: %d\n", omp_get_max_threads());
#endif

    // Load model
    Model model(backend);
    if (!model.load(cfg.model_path)) {
        fprintf(stderr, "Failed to load model from: %s\n", cfg.model_path.c_str());
        return 1;
    }

    if (cfg.show_info) {
        return 0;
    }

    // Create sampler
    Sampler sampler(cfg.sampler);

    if (cfg.interactive) {
        interactive_mode(model, sampler, cfg.max_tokens);
    } else {
        generate(model, sampler, cfg.prompt, cfg.max_tokens);
    }

    return 0;
}
