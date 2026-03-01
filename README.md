# llm.cpp

A lightweight C++ LLM inference engine, similar to [llama.cpp](https://github.com/ggerganov/llama.cpp). Loads GGUF model files and runs text generation on **CPU** (with OpenMP) or **GPU** (with CUDA).

## Features

- **GGUF model loading** – parses the standard GGUF format used by the llama.cpp ecosystem
- **LLaMA-style transformer** – full inference with RoPE, GQA, SwiGLU, RMS normalization
- **CPU backend** – optimized with OpenMP multi-threading
- **GPU backend** – CUDA kernels with cuBLAS matrix multiplication
- **Quantization support** – F32, F16, Q4_0, Q8_0 dequantization
- **Sampling** – temperature, top-k, top-p (nucleus), repetition penalty
- **Interactive mode** – chat-style conversation loop
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

# Show model info
./llm -m model.gguf --info

# Set CPU threads
./llm -m model.gguf --threads 8 -p "Hello world"
```

### Command-line Options

```
Required:
  -m, --model <path>       Path to GGUF model file

Generation:
  -p, --prompt <text>      Input prompt (default: "Hello")
  -n, --max-tokens <N>     Max tokens to generate (default: 256)
  -t, --temperature <F>    Sampling temperature (default: 0.8)
  --top-k <N>              Top-K sampling (default: 40)
  --top-p <F>              Top-P nucleus sampling (default: 0.9)
  --repeat-penalty <F>     Repetition penalty (default: 1.1)
  --seed <N>               Random seed (default: random)

Backend:
  --gpu                    Use GPU (CUDA) backend
  --cpu                    Use CPU backend (default)
  --threads <N>            Number of CPU threads (default: auto)

Other:
  -i, --interactive        Interactive chat mode
  --info                   Show model info and exit
  -h, --help               Show help
```

## Compatible Models

Any GGUF model with LLaMA-style architecture works, including:

- LLaMA / LLaMA 2 / LLaMA 3
- Mistral / Mixtral
- Qwen / Qwen2
- TinyLlama
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
