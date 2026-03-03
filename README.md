# llm.cpp

A lightweight C++ LLM inference engine, similar to [llama.cpp](https://github.com/ggerganov/llama.cpp). Loads GGUF model files and runs text generation on **CPU** (with OpenMP) or **GPU** (with CUDA).
You can sponsor us on https://shenwen.578388.xyz/payus.html Thank you.
## Features

- **GGUF model loading** – parses the standard GGUF format used by the llama.cpp ecosystem
- **LLaMA-style transformer** – full inference with RoPE, GQA, SwiGLU, RMS normalization
- **CPU backend** – optimized with OpenMP multi-threading
- **GPU backend** – CUDA kernels with cuBLAS matrix multiplication; all installed GPUs are enumerated at startup
- **Broad quantization support** – F32, F16, FP8 (E4M3/E5M2), Q4_0, Q8_0 fused kernels; Q2_K, Q3_K, Q4_K, Q5_K, Q6_K dequantization fallback (covers the most popular GGUF model variants)
- **Sampling** – temperature, top-k, top-p (nucleus), repetition penalty
- **Interactive mode** – chat-style conversation loop
- **Customizable system prompt** – set via `-s` / `--system`
- **OpenAI-compatible HTTP API** – run as a local server (`--server`) that OpenWebUI and other frontends can connect to via `/v1/chat/completions`
- **Custom model path** – load any GGUF model via `-m <path>`

## Build

### Requirements

- C++17 compiler (GCC 7+ or Clang 5+)
- CMake 3.16+
- (Optional) OpenMP for CPU parallelization
- (Optional) CUDA toolkit for GPU support

### CPU only (default)

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### With GPU (CUDA)

```bash
mkdir build && cd build
cmake .. -DLLM_CUDA=ON
cmake --build .
```

### Options

| CMake Option  | Default | Description                  |
|---------------|---------|------------------------------|
| `LLM_CUDA`   | OFF     | Enable CUDA GPU support      |
| `LLM_OPENMP` | ON      | Enable OpenMP parallelization|

## Usage

```bash
# Basic generation
./llm -m /path/to/model.gguf -p "Once upon a time"

# With GPU acceleration
./llm -m model.gguf --gpu -p "Explain quantum computing"

# Adjust generation parameters
./llm -m model.gguf -n 512 -t 0.7 --top-k 50 --top-p 0.95 -p "Hello"

# Interactive chat mode
./llm -m model.gguf -i

# Custom system prompt
./llm -m model.gguf -s "You are a pirate." -p "Tell me about ships"

# Show model info
./llm -m model.gguf --info

# Set CPU threads
./llm -m model.gguf --threads 8 -p "Hello world"

# Run as OpenAI-compatible HTTP server (connect OpenWebUI to http://localhost:8080/v1)
./llm -m model.gguf --server --port 8080
```

### Command-line Options

```
Required:
  -m, --model <path>       Path to GGUF model file

Generation:
  -p, --prompt <text>      Input prompt (default: "Hello")
  -s, --system <text>      System prompt (default: "You are a helpful assistant.")
  -n, --max-tokens <N>     Max tokens to generate (default: 256)
  -c, --context <N>        Max context length (default: model's value, auto-capped to 4096)
  -t, --temperature <F>    Sampling temperature (default: 0.8)
  --top-k <N>              Top-K sampling (default: 40)
  --top-p <F>              Top-P nucleus sampling (default: 0.9)
  --repeat-penalty <F>     Repetition penalty (default: 1.1)
  --seed <N>               Random seed (default: random)

Backend:
  --gpu                    Use GPU (CUDA) backend
  --cpu                    Use CPU backend (default)
  --threads <N>            Number of CPU threads (default: auto)

Server:
  --server                 Run as OpenAI-compatible HTTP API server
  --port <N>               HTTP server port (default: 8080)

Other:
  -i, --interactive        Interactive chat mode
  --no-chat-template       Disable automatic chat template
  --no-thinking            Hide <think>...</think> blocks in output
  --info                   Show model info and exit
  -h, --help               Show help
```

### Chat Template

For chat models (e.g., Qwen3, Qwen2) that include `<|im_start|>` and `<|im_end|>` tokens,
a ChatML-style template is automatically applied to prompts. This wraps your input in the
proper format so the model generates coherent responses. Use `--no-chat-template` to disable.
The system prompt (default: "You are a helpful assistant.") can be customized with `-s`.

### OpenAI-compatible HTTP Server

Start the server with `--server` (optionally `--port <N>`):

```bash
./llm -m model.gguf --server --port 8080
```

The following endpoints are available:

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Generate a chat completion |

Connect **OpenWebUI** by adding a new OpenAI-compatible connection with base URL
`http://localhost:8080/v1`. Both non-streaming and streaming (`"stream": true`) responses
are supported.

Example `curl` request:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm.cpp",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

## Quantization Support

| Format | Load | Inference kernel |
|--------|------|-----------------|
| F32    | ✓ | direct |
| F16    | ✓ | fused (1-pass) |
| FP8 E4M3 / E5M2 | ✓ | fused (1-pass) |
| Q4_0   | ✓ | fused (1-pass) |
| Q8_0   | ✓ | fused (1-pass) |
| Q2_K   | ✓ | dequantize fallback |
| Q3_K   | ✓ | dequantize fallback |
| Q4_K   | ✓ | dequantize fallback |
| Q5_K   | ✓ | dequantize fallback |
| Q6_K   | ✓ | dequantize fallback |

The most widely distributed GGUF models use Q4_K_M (Q4_K) or Q5_K_M (Q5_K), both of
which are now supported via the dequantize fallback path.

## Compatible Models

Any GGUF model with LLaMA-style architecture works, including:

- Deepseek / Deepseek v3.1
- Qwen / Qwen3 / Qwen3.5
- Minimax / Minimax2.5
- And other LLaMA-compatible models

Download models from [Hugging Face](https://huggingface.co/models?search=gguf) in GGUF format.

## Project Structure

```
src/
  main.cpp          CLI entry point and generation loop
  gguf.h            GGUF file format parser
  tensor.h          Tensor operations (CPU + GPU dispatch)
  model.h           LLaMA transformer model
  tokenizer.h       BPE tokenizer
  sampler.h         Token sampling strategies
  server.h          Minimal OpenAI-compatible HTTP server
  cuda_kernels.cu   CUDA GPU kernels
tests/
  test_gguf.cpp     Unit tests
CMakeLists.txt      Build configuration
```

## Running Tests

```bash
cd build
cmake --build .
ctest
# or directly:
./test_gguf
```

## License

See [LICENSE](LICENSE) for details.
