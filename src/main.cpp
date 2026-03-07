// llm.cpp - A lightweight C++ LLM inference engine
// Supports loading GGUF models and running text generation on CPU or GPU.
// Quantization support: F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q6_K (fused), Q2_K-Q5_K (fallback), FP8

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#else
#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#endif

#include "model.h"
#include "sampler.h"
#include "server.h"

#ifdef LLM_USE_CUDA
// Forward declaration – defined in cuda_kernels.cu
bool cuda_print_gpu_info();
#endif

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
                    // Strip trailing newline
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

// Print CPU and GPU hardware information to stderr.
static void print_hardware_info() {
    fprintf(stderr, "CPU:  %s\n", get_cpu_name().c_str());
#ifdef LLM_USE_CUDA
    cuda_print_gpu_info();
#else
    fprintf(stderr, "GPU:  (not available - rebuild with -DLLM_CUDA=ON to enable)\n");
#endif
}

struct RunConfig {
    std::string model_path;
    std::string prompt = "Hello";
    std::string system_prompt = "You are a helpful assistant.";
    int max_tokens = 256;
    int context_len = 0;  // 0 = use model default
    bool use_gpu = false;
    int num_threads = 0;  // 0 = auto
    SamplerConfig sampler;
    bool interactive = false;
    bool show_info = false;
    bool no_chat_template = false;  // disable auto chat template
    bool no_thinking = false;       // hide <think>...</think> blocks
    bool server_mode = false;       // run as HTTP API server
    int server_port = 8080;
    std::string api_key;            // optional Bearer token for HTTP server
    std::string model_name;         // display name reported by /v1/models
    bool upnp = false;              // attempt UPnP IGD port mapping
};

static void print_usage(const char* prog) {
    fprintf(stderr,
        "llm.cpp - Lightweight LLM inference engine\n"
        "\n"
        "Usage: %s -m <model> [options] [-p \"prompt\"]\n"
        "\n"
        "Required:\n"
        "  -m, --model <path>       Path to GGUF model file or HuggingFace model directory\n"
        "\n"
        "Generation options:\n"
        "  -p, --prompt <text>      Input prompt (default: \"Hello\")\n"
        "  -s, --system <text>      System prompt (default: \"You are a helpful assistant.\")\n"
        "  -n, --max-tokens <N>     Max tokens to generate (default: 256)\n"
        "  -c, --context <N>        Max context length (default: model's value)\n"
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
        "Server options:\n"
        "  --server                 Run as OpenAI-compatible HTTP API server\n"
        "  --port <N>               HTTP server port (default: 8080)\n"
        "  --api-key <key>          Require Bearer token for server requests\n"
        "  --model-name <name>      Model name reported by /v1/models (default: llm.cpp)\n"
        "  --upnp                   Auto-map port via UPnP IGD (for external access)\n"
        "\n"
        "Other:\n"
        "  -i, --interactive        Interactive chat mode\n"
        "  --no-chat-template       Disable automatic chat template\n"
        "  --no-thinking            Hide <think>...</think> blocks in output\n"
        "  --info                   Show model info and exit\n"
        "  -h, --help               Show this help\n"
        "\n"
        "Supported model formats:\n"
        "  - GGUF files (.gguf) - quantized format (F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q6_K/FP8)\n"
        "  - HuggingFace directories (config.json + *.safetensors) - direct F32/F16/BF16 inference\n"
        "\n"
        "Supported architectures:\n"
        "  LLaMA, Mistral, Qwen (2/3/3.5), DeepSeek, Gemma (1/2/3), Phi (2/3),\n"
        "  InternLM2, ChatGLM/GLM4, Cohere/Command-R, StarCoder2, MiniCPM, SmolLM3,\n"
        "  Exaone, Nemotron, Falcon-H1, OLMO, GPT-NeoX and compatible models\n"
        "\n"
        "Examples:\n"
        "  %s -m model.gguf -p \"Once upon a time\"\n"
        "  %s -m model.gguf --gpu -n 512 -p \"Explain quantum computing\"\n"
        "  %s -m ./Qwen3.5-0.8B/ -p \"Hello\"  # load HuggingFace model directly\n"
        "  %s -m model.gguf -i\n"
        "  %s -m model.gguf --server --port 8080\n",
        prog, prog, prog, prog, prog, prog);
}

static bool parse_args(int argc, char** argv, RunConfig& cfg) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            cfg.prompt = argv[++i];
        } else if ((arg == "-s" || arg == "--system") && i + 1 < argc) {
            cfg.system_prompt = argv[++i];
        } else if ((arg == "-n" || arg == "--max-tokens") && i + 1 < argc) {
            cfg.max_tokens = atoi(argv[++i]);
        } else if ((arg == "-c" || arg == "--context") && i + 1 < argc) {
            cfg.context_len = atoi(argv[++i]);
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
        } else if (arg == "--no-chat-template") {
            cfg.no_chat_template = true;
        } else if (arg == "--no-thinking") {
            cfg.no_thinking = true;
        } else if (arg == "--info") {
            cfg.show_info = true;
        } else if (arg == "--server") {
            cfg.server_mode = true;
        } else if (arg == "--port" && i + 1 < argc) {
            cfg.server_port = atoi(argv[++i]);
        } else if (arg == "--api-key" && i + 1 < argc) {
            cfg.api_key = argv[++i];
        } else if (arg == "--model-name" && i + 1 < argc) {
            cfg.model_name = argv[++i];
        } else if (arg == "--upnp") {
            cfg.upnp = true;
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

// Detect if model supports ChatML format and apply template
static bool has_chatml_support(const Model& model) {
    return model.tokenizer.token_to_id.count("<|im_start|>") &&
           model.tokenizer.token_to_id.count("<|im_end|>");
}

static std::string apply_chat_template(const Model& model, const std::string& prompt,
                                       const std::string& system_prompt) {
    // ChatML format: system message + user prompt + assistant turn
    if (has_chatml_support(model)) {
        return "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
               "<|im_start|>user\n" + prompt + "<|im_end|>\n"
               "<|im_start|>assistant\n";
    }
    return prompt;
}

static void generate(Model& model, Sampler& sampler, const std::string& prompt,
                     int max_tokens, const std::string& system_prompt = "",
                     bool use_chat_template = false,
                     bool no_thinking = false) {
    // Optionally apply chat template
    std::string final_prompt = prompt;
    bool template_applied = false;
    if (use_chat_template) {
        final_prompt = apply_chat_template(model, prompt, system_prompt);
        template_applied = (final_prompt != prompt);
        if (template_applied) {
            fprintf(stderr, "Chat template applied (ChatML format)\n");
        }
    }

    // Encode prompt
    // Skip BOS when chat template is applied: the template's own special tokens
    // (e.g., <|im_start|>) serve as the proper start markers.
    std::vector<int> tokens = model.tokenizer.encode(final_prompt, !template_applied);

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
    bool inside_think = false;          // currently inside <think> block
    std::string output_buf;             // buffer to detect think tags
    const std::string think_open = "<think>";
    const std::string think_close = "</think>";
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
        if (model.tokenizer.is_eos_token(next_token)) {
            break;
        }

        // Track for repetition penalty
        recent_tokens.push_back(next_token);
        if (static_cast<int>(recent_tokens.size()) > recent_window) {
            recent_tokens.erase(recent_tokens.begin());
        }

        // Decode and print (with optional thinking filter)
        std::string token_str = model.tokenizer.decode(next_token);
        if (no_thinking) {
            output_buf += token_str;
            // Process buffer for <think> / </think> tags
            while (true) {
                if (inside_think) {
                    auto close_pos = output_buf.find(think_close);
                    if (close_pos != std::string::npos) {
                        output_buf.erase(0, close_pos + think_close.size());
                        inside_think = false;
                        continue;
                    }
                    // Keep only a tail that could be a partial </think>
                    if (output_buf.size() >= think_close.size()) {
                        output_buf.erase(0, output_buf.size() - (think_close.size() - 1));
                    }
                    break;
                } else {
                    auto open_pos = output_buf.find(think_open);
                    if (open_pos != std::string::npos) {
                        // Flush text before the tag
                        if (open_pos > 0) {
                            printf("%s", output_buf.substr(0, open_pos).c_str());
                            fflush(stdout);
                        }
                        output_buf.erase(0, open_pos + think_open.size());
                        inside_think = true;
                        continue;
                    }
                    // Flush safe portion (keep tail that could be partial <think>)
                    if (output_buf.size() >= think_open.size()) {
                        size_t safe = output_buf.size() - (think_open.size() - 1);
                        printf("%s", output_buf.substr(0, safe).c_str());
                        fflush(stdout);
                        output_buf.erase(0, safe);
                    }
                    break;
                }
            }
        } else {
            printf("%s", token_str.c_str());
            fflush(stdout);
        }

        pos++;
        total_tokens++;
    }
    // Flush any remaining buffered output
    if (no_thinking && !inside_think && !output_buf.empty()) {
        printf("%s", output_buf.c_str());
        fflush(stdout);
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

static void interactive_mode(Model& model, Sampler& sampler, int max_tokens,
                            const std::string& system_prompt = "",
                            bool use_chat_template = false,
                            bool no_thinking = false) {
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

        // Reset state for new conversation
        model.clear_state();

        generate(model, sampler, line, max_tokens, system_prompt,
                 use_chat_template, no_thinking);
        printf("\n");
    }
}

int main(int argc, char** argv) {
#ifdef _WIN32
    // Enable UTF-8 console I/O on Windows
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    // Re-encode command-line arguments from their native UTF-16 representation
    // to UTF-8.  On Windows, argv is transcoded from the process's Unicode
    // command line using the current ANSI code page (e.g. GBK on Chinese
    // systems), which corrupts non-ASCII characters such as Chinese text.
    // CommandLineToArgvW retrieves the original UTF-16 arguments directly,
    // giving us a reliable UTF-8 conversion regardless of the active code page.
    std::vector<std::string> utf8_arg_storage;
    std::vector<char*>       utf8_argv_storage;
    {
        int wargc = 0;
        LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
        if (wargv && wargc == argc) {
            utf8_arg_storage.reserve(wargc);
            for (int i = 0; i < wargc; i++) {
                // n includes the null terminator; n - 1 is the actual string length
                int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                            nullptr, 0, nullptr, nullptr);
                std::string s(n > 0 ? static_cast<size_t>(n - 1) : 0u, '\0');
                if (n > 1) {
                    WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                        &s[0], n - 1, nullptr, nullptr);
                }
                utf8_arg_storage.push_back(std::move(s));
            }
            LocalFree(wargv);
        }
        utf8_argv_storage.reserve(utf8_arg_storage.size());
        for (auto& s : utf8_arg_storage) utf8_argv_storage.push_back(&s[0]);
        if (static_cast<int>(utf8_argv_storage.size()) == argc) {
            argv = utf8_argv_storage.data();
        }
    }
#endif

    RunConfig cfg;

    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    // Show CPU and GPU hardware information
    print_hardware_info();

    // Check GPU support
    Backend backend = Backend::CPU;
    if (cfg.use_gpu) {
#ifdef LLM_USE_CUDA
        if (cuda_check_gpu()) {
            backend = Backend::CUDA;
            fprintf(stderr, "Using CUDA GPU backend\n");
        } else {
            fprintf(stderr, "Warning: No usable GPU found. Falling back to CPU.\n");
        }
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
    if (!model.load(cfg.model_path, cfg.context_len)) {
        fprintf(stderr, "Failed to load model from: %s\n", cfg.model_path.c_str());
        return 1;
    }

    if (cfg.show_info) {
        return 0;
    }

    // Create sampler
    Sampler sampler(cfg.sampler);

    // Auto-detect chat template support
    bool use_chat_template = !cfg.no_chat_template && has_chatml_support(model);
    if (use_chat_template) {
        fprintf(stderr, "Chat template: auto-detected (ChatML)\n");
    }

    if (cfg.server_mode) {
        ServerConfig srv;
        srv.port          = cfg.server_port;
        srv.system_prompt = cfg.system_prompt;
        srv.api_key       = cfg.api_key;
        srv.model_name    = cfg.model_name.empty() ? "llm.cpp" : cfg.model_name;
        srv.upnp          = cfg.upnp;
        run_server(model, sampler, srv);
        return 0;
    }

    if (cfg.interactive) {
        interactive_mode(model, sampler, cfg.max_tokens, cfg.system_prompt,
                        use_chat_template, cfg.no_thinking);
    } else {
        generate(model, sampler, cfg.prompt, cfg.max_tokens, cfg.system_prompt,
                use_chat_template, cfg.no_thinking);
    }

    return 0;
}
