<!-- Language Selector -->
<div align="center">

[🇺🇸 English](README.md) | [🇨🇳 简体中文](README_zh_CN.md)

</div>

# swllm.cpp

A lightweight, high-performance LLM (Large Language Model) inference engine written in C++ with multi-platform GPU acceleration (CUDA, ROCm, Intel GPU) and CPU-only support. Designed for efficient local deployment of LLMs with minimal dependencies.

[![][license-shield]][license]
[![][stars-shield]][stars]
[![][cuda-shield]][cuda]
[![][rocm-shield]][rocm]
[![][intel-shield]][intel]
[![][cpu-shield]][cpu]

[license-shield]: https://img.shields.io/github/license/shenwenAI/swllm.cpp
[stars-shield]: https://img.shields.io/github/stars/shenwenAI/swllm.cpp
[cuda-shield]: https://img.shields.io/badge/CUDA-Supported-green
[rocm-shield]: https://img.shields.io/badge/ROCm-Supported-orange
[intel-shield]: https://img.shields.io/badge/Intel_GPU-Supported-blue
[cpu-shield]: https://img.shields.io/badge/CPU-Only-gray
[license]: LICENSE
[stars]: https://github.com/shenwenAI/swllm.cpp
[cuda]: https://developer.nvidia.com/cuda-toolkit
[rocm]: https://www.amd.com/en/products/graphics/workstations/amd-rocm.html
[intel]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpcpp-compiler.html
[cpu]: #

## Features

- **Pure C++ Implementation** — No Python dependencies required for inference
- **Multi-Platform GPU Acceleration** — CUDA (NVIDIA), ROCm (AMD), Intel GPU (Arc, iGPU)
- **CPU-Only Mode** — Optimized AVX2/AVX512 CPU inference without GPU
- **Single Header Design** — Header-only library for easy integration
- **Multiple Model Formats** — Direct HuggingFace format & GGUF support (all quantization types)
- **Complete GGUF Quantization** — F32, F16, FP8, Q2/Q3/Q4/Q5/Q6/Q8 all variants (K_M/K_S/K_L/IQ_XS/IQ_XM/IQ_XL)
- **OpenAI-Compatible API** — Drop-in replacement for OpenAI endpoints
- **Chat Templates** — Automatic ChatML-style templates for popular models
- **Streaming Responses** — Real-time token streaming support
- **Interactive Mode** — Conversational REPL for testing

## Supported Models

### Latest Models (2024-2025)

| Model Family | Variants | Architecture | Format Support |
|--------------|----------|--------------|----------------|
| **Qwen3.5** | 7B, 14B, 32B | GatedDeltaNet + Attention | HF + GGUF |
| **Qwen3.5 MoE** | 35B-A3B, 122B-A10B | Mixture of Experts | HF + GGUF |
| **Qwen3** | 0.6B, 1.7B, 4B, 8B, 14B, 32B | Standard Transformer | HF + GGUF |
| **MiniMax 2.5** | MiniMax-Text-01 | LLaMA-compatible | HF + GGUF |
| **Kimi 2.5** | Kimi-k1.5, Kimi-Dev | Long-context Transformer | HF + GGUF |
| **DeepSeek V3** | DeepSeek-V3, V3.1 | MoE + Multi-token | HF + GGUF |

### Mainstream Models

| Model Family | Variants | Architecture | Notes |
|--------------|----------|--------------|-------|
| **LLaMA 3/3.1** | 8B, 70B, 405B | Standard Transformer | Full support |
| **Mistral** | 7B, Mixtral 8x7B | GQA + MoE | Full support |
| **Gemma 2/3** | 2B, 9B, 27B | Google's LLaMA variant | Full support |
| **Phi 3/4** | Phi-3-mini, Phi-4 | Microsoft small models | Full support |
| **Yi 1.5** | 6B, 9B, 34B | LLaMA-compatible | Full support |
| **Baichuan 2** | 7B, 13B | Chinese LLM | Full support |
| **InternLM 2.5** | 7B, 20B | Shanghai AI Lab | Full support |
| **GLM-4** | GLM-Edge, GLM-Air | Zhipu AI | Full support |
| **Command-R+** | Cohere R+ | Enterprise RAG | Full support |
| **Falcon H1** | Falcon-H1-7B | TII UAE | Full support |
| **StableLM 2** | 1.6B, 12B | Stability AI | Full support |
| **OLMo 2** | 7B, 13B | AI2 Open Model | Full support |

### Custom Architecture Support

The framework supports **custom model architectures** through:

1. **Automatic Detection**: Unknown `model_type` in `config.json` automatically falls back to generic transformer architecture
2. **Plugin Registration**: Register custom architectures via `register_architecture()` API
3. **Weight Mapping**: Flexible tensor name mapping for custom weight formats
4. **Configuration Override**: Manual config parameter override for non-standard models

```cpp
// Example: Register custom architecture
ModelConfig custom_config;
custom_config.architecture = "my_custom_arch";
custom_config.hidden_size = 4096;
custom_config.num_heads = 32;
// ... set other parameters
model.register_custom_config(custom_config);
```

> **Note:** Any LLaMA-architecture model in GGUF format is supported. Direct HuggingFace format models (`.safetensors` + `tokenizer.json` + `config.json`) are also supported without conversion. Custom models with standard transformer structure can be loaded even without explicit architecture registration.

## Quick Start

### Installation

#### Build with CUDA (NVIDIA GPU)

```bash
# Clone the repository
git clone https://github.com/shenwenAI/swllm.cpp.git
cd swllm.cpp

# Build with CUDA support
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=CUDA
make -j$(nproc)
```

#### Build with ROCm (AMD GPU)

```bash
# Clone the repository
git clone https://github.com/shenwenAI/swllm.cpp.git
cd swllm.cpp

# Build with ROCm support
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=ROCm \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
make -j$(nproc)
```

#### Build with Intel GPU (Arc, iGPU)

```bash
# Clone the repository
git clone https://github.com/shenwenAI/swllm.cpp.git
cd swllm.cpp

# Source Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Build with Intel GPU support (SYCL/DPC++)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=SYCL \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER=icx
make -j$(nproc)
```

#### Build for CPU-Only (No GPU Required)

```bash
# Clone the repository
git clone https://github.com/shenwenAI/swllm.cpp.git
cd swllm.cpp

# Build with CPU-only support (AVX2/AVX512 optimized)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=CPU
make -j$(nproc)
```

> **Note:** For best CPU performance, ensure your compiler supports AVX2 or AVX512 instructions. The build will auto-detect and enable appropriate instruction sets.

### Basic Usage

```bash
# Simple inference with GGUF model
./swllm -m model.gguf -p "Hello, world!"

# Load HuggingFace format model directly (no conversion needed)
./swllm -m ./Qwen3-0.6B/ -p "Hello, world!"

# With custom parameters
./swllm -m model.gguf -p "Explain quantum computing:" -n 256 -t 8

# Interactive chat mode
./swllm -m model.gguf -i

# CPU-only mode (when built without GPU support)
./swllm -m model.gguf -p "Hello!" -t 16

# Use any GGUF quantization format (Q2_K to Q8_0, IQ2_XS to IQ4_XL)
./swllm -m model-Q4_K_M.gguf -p "Test"
./swllm -m model-IQ2_XXS.gguf -p "Test"
./swllm -m model-Q8_0.gguf -p "Test"
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

swllm.cpp supports **all GGUF quantization types**, including standard formats, K-Means variants, and the IQ series. Choose the right format based on your hardware constraints and quality requirements.

### Hardware Compatibility Guide

| Target Hardware | Recommended Format | Quality Level |
|-----------------|-------------------|---------------|
| **High-End GPU** (RTX 4090, RX 7900 XTX, Arc A770) | Q8_0 / BF16 | Maximum |
| **Mid-Range GPU** (RTX 4070, RX 7800 XT, Arc A750) | Q6_K / Q5_K_M | Excellent |
| **Entry GPU** (RTX 3060 12GB, RX 7600, Arc A580) | Q4_K_M | Good (Recommended) |
| **Low VRAM GPU** (8GB or less) | Q3_K_M / Q3_K_S | Fair |
| **Ultra-Low VRAM** (4-6GB) | Q2_K | Minimum Viable |
| **CPU-Only** (System RAM) | Any format | Depends on format |

> **Note:** Memory usage varies by model size. Actual memory usage may vary by ~10-15%.

### Complete GGUF Quantization Table

#### 2-bit Quantization

| Format | Quality | Speed | Best For |
|--------|---------|-------|----------|
| **Q2_K** | Fair | Medium | Ultra-low VRAM GPUs (4-6GB) |

#### 3-bit Quantization

| Format | Quality | Speed | Best For |
|--------|---------|-------|----------|
| **Q3_K_S** | Good | Medium | Low VRAM GPUs (6-8GB) |
| **Q3_K_M** | Good | Medium | Low VRAM GPUs (8GB) |
| **Q3_K_L** | Very Good | Medium | Entry GPUs (8-10GB) |

#### 4-bit Quantization (Most Popular)

| Format | Quality | Speed | Best For |
|--------|---------|-------|----------|
| **Q4_K_S** | Good | Fast | Entry GPUs (8GB) |
| **Q4_K_M** | **Very Good** | Fast | **Most Users (8-12GB)** ⭐ |

#### 5-bit Quantization

| Format | Quality | Speed | Best For |
|--------|---------|-------|----------|
| **Q5_K_S** | Very Good | Medium | Mid-Range GPUs (10-12GB) |
| **Q5_K_M** | **Excellent** | Medium | **Quality-focused (12GB+)** ⭐ |

#### 6-bit Quantization

| Format | Quality | Speed | Best For |
|--------|---------|-------|----------|
| **Q6_K** | **Excellent** | Medium | High-quality inference (12-16GB) |

#### 8-bit Quantization

| Format | Quality | Speed | Best For |
|--------|---------|-------|----------|
| **Q8_0** | **Near-Lossless** | Fast | High-end GPUs (16-24GB) |

#### Brain Floating Point (BF16)

| Format | Quality | Speed | Best For |
|--------|---------|-------|----------|
| **BF16** | **Maximum** | Fast | Research/Development (24GB+) |

#### Integer Quantization (IQ Series - Advanced)

The IQ series provides improved quality at ultra-low bitrates using advanced quantization techniques.

| Format | Quality | Speed | Best For |
|--------|---------|-------|----------|
| **IQ2_XXS** | Fair | Slow | Extreme memory constraints |
| **IQ2_XS** | Fair | Slow | Extreme memory constraints |
| **IQ2_XXL** | Good | Medium | Low VRAM alternative to Q2_K |
| **IQ3_XXS** | Good | Medium | Alternative to Q3_K_S |
| **IQ3_XS** | Good | Medium | Alternative to Q3_K_M |
| **IQ3_XXL** | Very Good | Medium | Alternative to Q3_K_L |
| **IQ4_XS** | Very Good | Fast | Alternative to Q4_K_S |
| **IQ4_XL** | Excellent | Fast | Alternative to Q4_K_M |

### Quick Selection Guide

```
🎯 Most Users: Q4_K_M (best balance of quality/speed/memory)
🏆 Quality Focus: Q5_K_M or Q6_K (excellent quality with reasonable size)
💾 Memory Constrained: Q3_K_M or IQ3_XS (good quality in minimal space)
🔬 Maximum Quality: Q8_0 or BF16 (near-lossless, requires high VRAM)
⚡ Ultra-Low Memory: IQ2_XXS or Q2_K (minimum viable quality)
```

> **Tip:** For most use cases, **Q4_K_M** offers the best balance of quality, speed, and memory usage. For maximum quality with reduced size, use **Q5_K_M** or **Q6_K**. Test multiple formats to find the optimal choice for your specific hardware and use case.

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

### Core Requirements

- **C++ Compiler** with C++17 support (GCC 8+, Clang 7+, MSVC 2019+)
- **CMake** 3.15+

### GPU Backend Options (Choose One)

#### NVIDIA CUDA
- **CUDA Toolkit** 11.0+
- **cuBLAS** (optional, for optimized matrix operations)
- NVIDIA GPU with compute capability 6.0+

#### AMD ROCm
- **ROCm** 5.0+ 
- **HIP SDK**
- AMD GPU (RDNA2/RDNA3 or CDNA architecture recommended)
- Linux OS required

#### Intel GPU (SYCL/DPC++)
- **Intel oneAPI Base Toolkit** 2023+
- **Intel oneAPI DPC++ Compiler**
- Intel Arc GPU or Intel integrated graphics (Iris Xe, UHD)
- Windows/Linux supported

### CPU-Only Mode

No special requirements - works on any x86_64 CPU with AVX2 support. AVX512 will be auto-detected and enabled if available.

## Getting Models

### GGUF Models

Download GGUF models from:

- [TheBloke's GGUF Collection](https://huggingface.co/models?search=gguf)
- [LM Studio](https://lmstudio.ai/models)
- [Ollama](https://ollama.ai/library)
- [HuggingFace GGUF Search](https://huggingface.co/models?search=gguf)

Example with Qwen3:

```bash
# Download a quantized model
# Visit: https://huggingface.co/models?search=gguf+qwen3

# Or use the HuggingFace hub
huggingface-cli download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf
```

### HuggingFace Format Models (Direct Support)

Use HuggingFace models directly without conversion:

```bash
# Download a HuggingFace model
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B

# Run directly (requires model.safetensors, tokenizer.json, config.json)
./swllm -m ./Qwen3-0.6B/ -p "Hello!"
```

Supported HuggingFace formats:
- `.safetensors` weight files
- `tokenizer.json` for tokenization
- `config.json` for model configuration
- `generation_config.json` (optional)

## Performance Tips

### GPU Optimization

#### NVIDIA GPU (Maximum Performance)

For **best NVIDIA GPU performance**, swllm.cpp includes specialized optimizations:

1. **Tensor Core Acceleration** — Automatic FP16/FP8 Tensor Core utilization on RTX 20/30/40 series
2. **Memory Coalescing** — Optimized memory access patterns for maximum bandwidth
3. **Kernel Fusion** — Fused attention and MLP kernels reduce memory transfers
4. **Multi-Stream Execution** — Overlap computation and data transfer
5. **Persistent Kernel** — Reduce kernel launch overhead for long sequences
6. **Flash Attention** — O(1) memory complexity attention implementation
7. **Paged KV Cache** — Efficient memory management for long context

**Recommended Settings:**
```bash
# Enable all optimizations (default for CUDA builds)
./swllm -m model.gguf -p "Hello" --gpu-layers 999

# For maximum throughput on datacenter GPUs (A100/H100)
./swllm -m model.gguf --server --port 8080 --batch-size 32
```

**Supported Architectures:**
- **Ampere** (RTX 30xx, A100): TF32/FP16 Tensor Cores
- **Ada Lovelace** (RTX 40xx): FP8 Tensor Cores + Sparse acceleration
- **Hopper** (H100): Transformer Engine + FP8 mixed precision
- **Volta/Turing** (V100, RTX 20xx): FP16 Tensor Cores

#### AMD GPU (ROCm)

1. **Matrix Core Support** — Leverage CDNA/RDNA matrix cores for FP16/INT8
2. **Wavefront Optimization** — Optimal wavefront size for GCN/RDNA architectures
3. **HBM2/HBM3 Bandwidth** — Maximize high-bandwidth memory utilization
4. **Async Compute** — Overlap compute and memory operations

**Recommended Settings:**
```bash
# Build with ROCm optimizations
cmake .. -DGPU_BACKEND=ROCm -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++

# Run with optimal thread block size
./swllm -m model.gguf -p "Hello" --gpu-layers 999
```

**Supported Architectures:**
- **RDNA2** (RX 6000 series): Matrix Core acceleration
- **RDNA3** (RX 7000 series): Enhanced AI accelerators
- **CDNA2/3** (MI200/MI300): Datacenter-grade performance

#### Intel GPU (Arc, iGPU)

1. **Xe Matrix Extensions** — XMX acceleration for AI workloads
2. **Subgroup Operations** — Efficient SIMD execution
3. **Local Memory** — Optimize data reuse in L1 cache
4. **Unified Memory** — Zero-copy between CPU and GPU

**Recommended Settings:**
```bash
# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Build with SYCL optimizations
cmake .. -DGPU_BACKEND=SYCL -DCMAKE_CXX_COMPILER=icpx

# Run with device selection
./swllm -m model.gguf -p "Hello" --device gpu
```

**Supported Hardware:**
- **Arc A-Series** (A770, A750, A580): Full XMX support
- **Iris Xe** (11th/12th gen Intel): Integrated graphics acceleration
- **UHD Graphics** (Recent Intel): Basic GPU offload

### CPU Optimization

1. **CPU Threads** — Use `-t` to match your physical CPU core count (not hyperthreads)
2. **Quantization** — Q4_K_M offers the best balance of speed and quality
3. **AVX512** — Enable AVX512 in BIOS if available for 2x CPU performance
4. **Memory Bandwidth** — Dual/quad-channel RAM improves CPU inference speed

### Model Selection

- **High-quality**: Q5_K_M or Q6_K
- **Balanced**: Q4_K_M (recommended)
- **Low-memory**: Q3_K_M or IQ3_XS
- **Ultra-low-memory**: IQ2_XXS

## License

This project is licensed under the [GPL-3.0 License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Connect With Us

Stay updated with the latest developments, model releases, and community discussions:

<div align="center">

[![GitHub][github-badge]][github-link]
[![Hugging Face][hf-badge]][hf-link]
[![Twitter][twitter-badge]][twitter-link]

[github-badge]: https://img.shields.io/badge/GitHub-shenwenAI-181717?style=for-the-badge&logo=github
[github-link]: https://github.com/shenwenAI
[hf-badge]: https://img.shields.io/badge/Hugging_Face-shenwenAI-fcd022?style=for-the-badge&logo=huggingface
[hf-link]: https://huggingface.co/shenwenAI
[twitter-badge]: https://img.shields.io/badge/Twitter-@shenwenai-1DA1F2?style=for-the-badge&logo=twitter
[twitter-link]: https://x.com/shenwenai

</div>

## Acknowledgments

- [ggml](https://github.com/ggerganov/ggml) — Inspirational tensor computation library
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Pioneer in efficient LLM inference
- [HuggingFace](https://huggingface.co/) — Model formats and ecosystem
- [AMD ROCm](https://www.amd.com/en/products/graphics/workstations/amd-rocm.html) — AMD GPU computing platform
- [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) — Cross-architecture programming framework
- [SYCL/DPC++](https://www.khronos.org/sycl/) — Open standard for heterogeneous computing
