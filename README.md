---

# swllm.cpp

A lightweight, high-performance LLM (Large Language Model) inference engine written in C++ with CUDA GPU acceleration. Designed for efficient local deployment of LLMs with minimal dependencies.

[![][license-shield]][license]
[![][stars-shield]][stars]
[![][cuda-shield]][cuda]

[license-shield]: https://img.shields.io/github/license/shenwenAI/swllm.cpp
[stars-shield]: https://img.shields.io/github/stars/shenwenAI/swllm.cpp
[cuda-shield]: https://img.shields.io/badge/CUDA-Supported-green
[license]: LICENSE
[stars]: https://github.com/shenwenAI/swllm.cpp
[cuda]: https://developer.nvidia.com/cuda-toolkit

## Features

- **Pure C++ Implementation** — No Python dependencies required for inference
- **CUDA GPU Acceleration** — High-performance GPU kernels for fast inference
- **Single Header Design** — Header-only library for easy integration
- **Multiple Model Formats** — Support for GGUF and HuggingFace SafeTensors
- **Extensive Quantization** — F32, F16, FP8, Q4/Q5/Q6/Q8 variants
- **OpenAI-Compatible API** — Drop-in replacement for OpenAI endpoints
- **Chat Templates** — Automatic ChatML-style templates for popular models
- **Streaming Responses** — Real-time token streaming support
- **Interactive Mode** — Conversational REPL for testing

## Supported Models

| Model Family | Variants | Architecture |
|--------------|----------|--------------|
| **Qwen / Qwen3** | 0.8B, 2B, 4B, 9B, 27B | GatedDeltaNet + Attention |
| **Qwen3.5 MoE** | 35B-A3B, 122B-A10B, 397B-A17B | Mixture of Experts |
| **Deepseek** | Deepseek v3.1 | LLaMA-compatible |
| **Minimax** | Minimax2.5 | LLaMA-compatible |
| **Generic LLaMA** | Any GGUF-compatible | Standard LLaMA |

> **Note:** Any LLaMA-architecture model in GGUF format is supported.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/shenwenAI/swllm.cpp.git
cd swllm.cpp

# Build with CUDA support
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Basic Usage

```bash
# Simple inference
./swllm -m model.gguf -p "Hello, world!"

# With custom parameters
./swllm -m model.gguf -p "Explain quantum computing:" -n 256 -t 8

# Interactive chat mode
./swllm -m model.gguf -i

# Load from HuggingFace directory
./swllm -m ./Qwen3-0.6B/ -p "Hello, world!"
```

### Start HTTP Server

```bash
# Start OpenAI-compatible API server
./swllm -m model.gguf --server --port 8080

# Query the API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m <path>` | Model file path (GGUF or HF directory) | Required |
| `-p <text>` | Prompt for generation | None |
| `-n <num>` | Maximum tokens to generate | -1 (infinite) |
| `-t <num>` | Number of CPU threads | 4 |
| `-i` | Interactive chat mode | Disabled |
| `-v` | Verbose output | Disabled |
| `-s <text>` | System prompt | None |
| `--no-chat-template` | Disable automatic chat template | Enabled |
| `--server` | Start HTTP API server | Disabled |
| `--port <num>` | Server port | 8080 |

## Quantization Formats

| Format | Memory Reduction | Quality | Speed |
|--------|------------------|--------|-------|
| **F32** | 0% | Highest | Baseline |
| **F16** | 50% | High | Fast |
| **FP8 E4M3/E5M2** | 75% | High | Fast |
| **Q8_0** | 75% | Very High | Fast |
| **Q6_K** | 60% | Excellent | Medium |
| **Q5_K** | 65% | Excellent | Medium |
| **Q4_K** | 70% | Good | Fast |
| **Q3_K** | 77% | Good | Medium |
| **Q2_K** | 80% | Fair | Medium |

## HTTP API

### Endpoints

#### `GET /v1/models`

List available models.

```bash
curl http://localhost:8080/v1/models
```

#### `POST /v1/chat/completions`

Generate chat completion.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-name",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "stream": false,
    "max_tokens": 256
  }'
```

#### Streaming Response

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-name",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

## Project Structure

```
swllm.cpp/
├── src/
│   ├── main.cpp           # CLI entry point
│   ├── gguf.h             # GGUF file format parser
│   ├── safetensors.h      # SafeTensors parser
│   ├── hf_loader.h        # HuggingFace directory loader
│   ├── tensor.h           # Tensor operations (CPU/GPU)
│   ├── model.h            # LLaMA transformer model
│   ├── tokenizer.h        # BPE tokenizer
│   ├── sampler.h          # Token sampling strategies
│   ├── server.h           # HTTP API server
│   └── cuda_kernels.cu    # CUDA GPU kernels
├── tests/
│   └── test_gguf.cpp      # Unit tests
├── CMakeLists.txt         # Build configuration
└── README.md              # This file
```

## Requirements

- **C++ Compiler** with C++17 support
- **CUDA Toolkit** 11.0+ (for GPU acceleration)
- **CMake** 3.15+
- **cuBLAS** (optional, for optimized matrix operations)

## Getting Models

Download GGUF models from:

- [TheBloke's GGUF Collection](https://huggingface.co/models?search=gguf)
- [LM Studio](https://lmstudio.ai/models)
- [Ollama](https://ollama.ai/library)

Example with Qwen3:

```bash
# Download a quantized model
# Visit: https://huggingface.co/models?search=gguf+qwen3

# Or use the HuggingFace hub
huggingface-cli download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf
```

## Performance Tips

1. **GPU Memory** — Ensure your GPU has enough VRAM for the model
2. **CPU Threads** — Use `-t` to match your CPU core count
3. **Quantization** — Q4_K offers the best balance of speed and quality
4. **Batch Size** — For server mode, consider batching requests

## License

This project is licensed under the [GPL-3.0 License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [ggml](https://github.com/ggerganov/ggml) — Inspirational tensor computation library
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Pioneer in efficient LLM inference
- [HuggingFace](https://huggingface.co/) — Model formats and ecosystem
