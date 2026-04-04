<!-- Language Selector -->
<div align="center">

[🇺🇸 English](README.md) | [🇨🇳 简体中文](README_zh_CN.md)

</div>

# swllm.cpp

一个用 C++ 编写的高性能轻量级大语言模型（LLM）推理引擎，支持多平台 GPU 加速（CUDA、ROCm、Intel GPU）和纯 CPU 模式。专为高效的本地 LLM 部署而设计，依赖极少。

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

## 特性

- **纯 C++ 实现** — 推理无需 Python 依赖
- **多平台 GPU 加速** — CUDA (NVIDIA)、ROCm (AMD)、Intel GPU (Arc、核显)
- **纯 CPU 模式** — 优化的 AVX2/AVX512 CPU 推理，无需 GPU
- **单头文件设计** — 头文件库，易于集成
- **多种模型格式** — 直接支持 HuggingFace 格式和 GGUF（所有量化类型）
- **完整 GGUF 量化支持** — F32、F16、FP8、Q2/Q3/Q4/Q5/Q6/Q8 所有变体（K_M/K_S/K_L/IQ_XS/IQ_XM/IQ_XL）
- **OpenAI 兼容 API** — 可直接替代 OpenAI 端点
- **聊天模板** — 自动为流行模型应用 ChatML 风格模板
- **流式响应** — 支持实时 token 流输出
- **交互模式** — 对话式 REPL 测试

## 支持的模型

### 最新模型（2024-2025）

| 模型系列 | 变体 | 架构 | 格式支持 |
|----------|------|------|----------|
| **Qwen3.5** | 7B, 14B, 32B | GatedDeltaNet + Attention | HF + GGUF |
| **Qwen3.5 MoE** | 35B-A3B, 122B-A10B | 混合专家 | HF + GGUF |
| **Qwen3** | 0.6B, 1.7B, 4B, 8B, 14B, 32B | 标准 Transformer | HF + GGUF |
| **MiniMax 2.5** | MiniMax-Text-01 | LLaMA 兼容 | HF + GGUF |
| **Kimi 2.5** | Kimi-k1.5, Kimi-Dev | 长上下文 Transformer | HF + GGUF |
| **DeepSeek V3** | DeepSeek-V3, V3.1 | MoE + 多 token | HF + GGUF |

### 主流模型

| 模型系列 | 变体 | 架构 | 备注 |
|----------|------|------|------|
| **LLaMA 3/3.1** | 8B, 70B, 405B | 标准 Transformer | 完整支持 |
| **Mistral** | 7B, Mixtral 8x7B | GQA + MoE | 完整支持 |
| **Gemma 2/3** | 2B, 9B, 27B | Google 的 LLaMA 变体 | 完整支持 |
| **Phi 3/4** | Phi-3-mini, Phi-4 | 微软小模型 | 完整支持 |
| **Yi 1.5** | 6B, 9B, 34B | LLaMA 兼容 | 完整支持 |
| **Baichuan 2** | 7B, 13B | 中文 LLM | 完整支持 |
| **InternLM 2.5** | 7B, 20B | 上海 AI Lab | 完整支持 |
| **GLM-4** | GLM-Edge, GLM-Air | 智谱 AI | 完整支持 |
| **Command-R+** | Cohere R+ | 企业 RAG | 完整支持 |
| **Falcon H1** | Falcon-H1-7B | TII UAE | 完整支持 |
| **StableLM 2** | 1.6B, 12B | Stability AI | 完整支持 |
| **OLMo 2** | 7B, 13B | AI2 开放模型 | 完整支持 |

### 自定义架构支持

本框架支持**自定义模型架构**：

1. **自动检测**：`config.json` 中未知的 `model_type` 会自动回退到通用 Transformer 架构
2. **插件注册**：通过 `register_architecture()` API 注册自定义架构
3. **权重映射**：灵活的张量名称映射，支持自定义权重格式
4. **配置覆盖**：手动覆盖配置参数以支持非标准模型

```cpp
// 示例：注册自定义架构
ModelConfig custom_config;
custom_config.architecture = "my_custom_arch";
custom_config.hidden_size = 4096;
custom_config.num_heads = 32;
// ... 设置其他参数
model.register_custom_config(custom_config);
```

> **注意：** GGUF 格式的任何 LLaMA 架构模型都受支持。HuggingFace 格式模型（`.safetensors` + `tokenizer.json` + `config.json`）无需转换即可直接使用。具有标准 Transformer 结构的自定义模型即使没有显式架构注册也可以加载。

---

## 🚀 快速开始 - 编译教程

### 选择你的操作系统和硬件

<details open>
<summary><strong>🪟 Windows 用户</strong></summary>

#### Windows + NVIDIA GPU (CUDA)

**前置要求：**
- Windows 10/11 (64 位)
- Visual Studio 2019 或更高版本（需安装 C++ 桌面开发组件）
- CUDA Toolkit 11.0+ （从 [NVIDIA 官网](https://developer.nvidia.com/cuda-toolkit) 下载）
- CMake 3.15+ （从 [CMake 官网](https://cmake.org/download/) 下载）

**编译步骤：**

1. **安装 Visual Studio**
   - 下载 [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/)
   - 安装时勾选"使用 C++ 的桌面开发"

2. **安装 CUDA Toolkit**
   - 访问 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
   - 下载并安装适合你显卡的版本（推荐 CUDA 12.x）
   - 安装完成后重启电脑

3. **验证 CUDA 安装**
   ```cmd
   nvcc --version
   ```

4. **克隆项目**
   ```cmd
   git clone https://github.com/shenwenAI/swllm.cpp.git
   cd swllm.cpp
   ```

5. **创建构建目录并编译**
   ```cmd
   mkdir build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=CUDA -G "Visual Studio 17 2022" -A x64
   cmake --build . --config Release -j 8
   ```

6. **运行测试**
   ```cmd
   Release\swllm.exe -m path\to\model.gguf -p "你好，世界！"
   ```

#### Windows + AMD GPU (ROCm)

> ⚠️ **注意：** ROCm 在 Windows 上的支持有限，建议使用 WSL2 或 Linux 以获得最佳 AMD GPU 支持。

**替代方案：使用 WSL2**

1. 安装 [WSL2](https://learn.microsoft.com/zh-cn/windows/wsl/install)
2. 安装 Ubuntu 22.04 LTS
3. 按照下方 [Linux + AMD GPU](#linux--amd-gpu-rocm) 的说明操作

#### Windows + Intel GPU (Arc/核显)

**前置要求：**
- Windows 10/11
- Visual Studio 2019+
- Intel oneAPI Base Toolkit 2023+

**编译步骤：**

1. **安装 Intel oneAPI**
   - 下载 [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
   - 安装时确保包含 DPC++ 编译器

2. **设置环境变量**
   ```cmd
   call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
   ```

3. **克隆并编译**
   ```cmd
   git clone https://github.com/shenwenAI/swllm.cpp.git
   cd swllm.cpp
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=SYCL ^
       -DCMAKE_CXX_COMPILER=icx ^
       -DCMAKE_C_COMPILER=icx
   cmake --build . --config Release -j 8
   ```

#### Windows + CPU Only

**最简单的编译方式：**

```cmd
git clone https://github.com/shenwenAI/swllm.cpp.git
cd swllm.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=CPU -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release -j 8
```

</details>

<details>
<summary><strong>🍎 macOS (Apple Silicon / Intel)</strong></summary>

#### macOS + Apple Silicon (M1/M2/M3)

macOS 上使用 Metal GPU 加速或 CPU 加速。

**前置要求：**
- macOS 11.0+ (Big Sur 或更高)
- Xcode Command Line Tools
- CMake 3.15+

**编译步骤：**

1. **安装 Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

2. **安装 CMake**
   ```bash
   brew install cmake
   ```

3. **克隆项目**
   ```bash
   git clone https://github.com/shenwenAI/swllm.cpp.git
   cd swllm.cpp
   ```

4. **编译（Metal GPU 加速）**
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=METAL
   make -j$(sysctl -n hw.ncpu)
   ```

5. **运行测试**
   ```bash
   ./swllm -m model.gguf -p "你好，世界！"
   ```

#### macOS + Intel CPU

仅 CPU 模式（无 GPU 加速）：

```bash
git clone https://github.com/shenwenAI/swllm.cpp.git
cd swllm.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=CPU
make -j$(sysctl -n hw.ncpu)
```

</details>

<details>
<summary><strong>🐧 Linux 用户</strong></summary>

#### Linux + NVIDIA GPU (CUDA)

**前置要求：**
- Ubuntu 20.04+ / Debian 11+ / CentOS 8+ / Fedora 35+
- GCC 8+ 或 Clang 10+
- CUDA Toolkit 11.0+
- CMake 3.15+

**编译步骤：**

1. **安装 CUDA Toolkit**

   **Ubuntu/Debian:**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-0
   ```

   **CentOS/RHEL:**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
   sudo cp cuda-rhel8.repo /etc/yum.repos.d/
   sudo yum install -y cuda-toolkit-12-0
   ```

2. **验证 CUDA 安装**
   ```bash
   nvcc --version
   nvidia-smi
   ```

3. **安装编译工具**
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential cmake git
   ```

4. **克隆并编译**
   ```bash
   git clone https://github.com/shenwenAI/swllm.cpp.git
   cd swllm.cpp
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=CUDA
   make -j$(nproc)
   ```

5. **运行测试**
   ```bash
   ./swllm -m model.gguf -p "你好，世界！"
   ```

#### Linux + AMD GPU (ROCm)

**前置要求：**
- Ubuntu 22.04 / RHEL 9 / SLES 15
- AMD GPU (RDNA2/RDNA3 或 CDNA 架构)
- ROCm 5.0+
- CMake 3.15+

**编译步骤：**

1. **安装 ROCm**

   **Ubuntu 22.04:**
   ```bash
   wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.2.60002-1_all.deb
   sudo apt install ./amdgpu-install_6.0.2.60002-1_all.deb
   sudo amdgpu-install --usecase=hip,rocm
   ```

   **RHEL 9:**
   ```bash
   sudo dnf config-manager --add-repo https://repo.radeon.com/amdgpu/6.0.2/rhel/9.2/main/x86_64/
   sudo dnf install amdgpu-dkms rocm-hip-sdk
   ```

2. **验证 ROCm 安装**
   ```bash
   rocminfo
   hipinfo
   ```

3. **克隆并编译**
   ```bash
   git clone https://github.com/shenwenAI/swllm.cpp.git
   cd swllm.cpp
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=ROCm \
       -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
       -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
   make -j$(nproc)
   ```

#### Linux + Intel GPU (Arc/核显)

**前置要求：**
- Ubuntu 22.04+ / Fedora 38+
- Intel Arc GPU 或 Iris Xe 核显
- Intel oneAPI 2023+

**编译步骤：**

1. **安装 Intel oneAPI**
   ```bash
   wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
   sudo apt-key add *.PUB
   echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
   sudo apt update
   sudo apt install intel-basekit intel-hpckit
   ```

2. **设置环境**
   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

3. **克隆并编译**
   ```bash
   git clone https://github.com/shenwenAI/swllm.cpp.git
   cd swllm.cpp
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=SYCL \
       -DCMAKE_CXX_COMPILER=icpx \
       -DCMAKE_C_COMPILER=icx
   make -j$(nproc)
   ```

#### Linux + CPU Only

**最简单的编译方式：**

```bash
git clone https://github.com/shenwenAI/swllm.cpp.git
cd swllm.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=CPU
make -j$(nproc)
```

</details>

---

## 基本使用

```bash
# 使用 GGUF 模型进行简单推理
./swllm -m model.gguf -p "你好，世界！"

# 直接加载 HuggingFace 格式模型（无需转换）
./swllm -m ./Qwen3-0.6B/ -p "你好，世界！"

# 自定义参数
./swllm -m model.gguf -p "解释量子计算：" -n 256 -t 8

# 交互式聊天模式
./swllm -m model.gguf -i

# CPU 模式（当编译时未启用 GPU 支持）
./swllm -m model.gguf -p "你好！" -t 16

# 使用任意 GGUF 量化格式（Q2_K 到 Q8_0，IQ2_XXS 到 IQ4_XL）
./swllm -m model-Q4_K_M.gguf -p "测试"
./swllm -m model-IQ2_XXS.gguf -p "测试"
./swllm -m model-Q8_0.gguf -p "测试"
```

### 启动 HTTP 服务器

```bash
# 启动 OpenAI 兼容的 API 服务器
./swllm -m model.gguf --server --port 8080

# 查询 API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "你好！"}]
  }'
```

---

## 命令行选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `-m <path>` | 模型文件路径（GGUF 或 HF 目录） | 必需 |
| `-p <text>` | 生成提示词 | 无 |
| `-n <num>` | 最大生成 token 数 | -1 (无限) |
| `-t <num>` | CPU 线程数 | 4 |
| `-i` | 交互式聊天模式 | 禁用 |
| `-v` | 详细输出 | 禁用 |
| `-s <text>` | 系统提示词 | 无 |
| `--no-chat-template` | 禁用自动聊天模板 | 启用 |
| `--server` | 启动 HTTP API 服务器 | 禁用 |
| `--port <num>` | 服务器端口 | 8080 |

---

## 量化格式

swllm.cpp 支持**所有 GGUF 量化类型**，包括标准格式、K-Means 变体和 IQ 系列。根据你的硬件限制和质量需求选择合适的格式。

### 硬件兼容性指南

| 目标硬件 | 推荐格式 | 质量等级 |
|----------|----------|----------|
| **高端 GPU** (RTX 4090, RX 7900 XTX, Arc A770) | Q8_0 / BF16 | 最高 |
| **中端 GPU** (RTX 4070, RX 7800 XT, Arc A750) | Q6_K / Q5_K_M | 优秀 |
| **入门 GPU** (RTX 3060 12GB, RX 7600, Arc A580) | Q4_K_M | 良好（推荐）⭐ |
| **低显存 GPU** (8GB 或更少) | Q3_K_M / Q3_K_S | 一般 |
| **超低显存** (4-6GB) | Q2_K | 最低可用 |
| **纯 CPU** (系统内存) | 任意格式 | 取决于格式 |

> **注意：** 内存使用量因模型大小而异。实际内存使用量可能有约 10-15% 的偏差。

### 快速选择指南

```
🎯 大多数用户：Q4_K_M（质量/速度/内存的最佳平衡）
🏆 质量优先：Q5_K_M 或 Q6_K（优秀质量，合理大小）
💾 内存受限：Q3_K_M 或 IQ3_XS（最小空间内的良好质量）
🔬 最高质量：Q8_0 或 BF16（近乎无损，需要高显存）
⚡ 超低内存：IQ2_XXS 或 Q2_K（最低可用质量）
```

> **提示：** 对于大多数用例，**Q4_K_M** 提供了质量、速度和内存使用的最佳平衡。对于追求最大质量且减小体积的用户，使用 **Q5_K_M** 或 **Q6_K**。测试多种格式以找到最适合你特定硬件和用例的选择。

---

## 获取模型

### GGUF 模型

从以下来源下载 GGUF 模型：

- [TheBloke 的 GGUF 合集](https://huggingface.co/models?search=gguf)
- [LM Studio](https://lmstudio.ai/models)
- [Ollama](https://ollama.ai/library)
- [HuggingFace GGUF 搜索](https://huggingface.co/models?search=gguf)

Qwen3 示例：

```bash
# 下载量化模型
# 访问：https://huggingface.co/models?search=gguf+qwen3

# 或使用 HuggingFace hub
huggingface-cli download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf
```

### HuggingFace 格式模型（直接支持）

直接使用 HuggingFace 模型，无需转换：

```bash
# 下载 HuggingFace 模型
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B

# 直接运行（需要 model.safetensors、tokenizer.json、config.json）
./swllm -m ./Qwen3-0.6B/ -p "你好！"
```

支持的 HuggingFace 格式：
- `.safetensors` 权重文件
- `tokenizer.json` 用于分词
- `config.json` 用于模型配置
- `generation_config.json`（可选）

---

## 项目结构

```
swllm.cpp/
├── src/
│   ├── main.cpp           # CLI 入口点
│   ├── gguf.h             # GGUF 文件格式解析器
│   ├── safetensors.h      # SafeTensors 解析器
│   ├── hf_loader.h        # HuggingFace 目录加载器
│   ├── tensor.h           # 张量操作（CPU/GPU）
│   ├── model.h            # LLaMA Transformer 模型
│   ├── tokenizer.h        # BPE 分词器
│   ├── sampler.h          # Token 采样策略
│   ├── server.h           # HTTP API 服务器
│   └── cuda_kernels.cu    # CUDA GPU 内核
├── tests/
│   └── test_gguf.cpp      # 单元测试
├── CMakeLists.txt         # 构建配置
└── README.md              # 本文件
```

---

## 要求

### 核心要求

- **C++ 编译器** 支持 C++17（GCC 8+、Clang 7+、MSVC 2019+）
- **CMake** 3.15+

### GPU 后端选项（选择一个）

#### NVIDIA CUDA
- **CUDA Toolkit** 11.0+
- **cuBLAS**（可选，用于优化的矩阵运算）
- 计算能力 6.0+ 的 NVIDIA GPU

#### AMD ROCm
- **ROCm** 5.0+
- **HIP SDK**
- AMD GPU（推荐 RDNA2/RDNA3 或 CDNA 架构）
- 需要 Linux 操作系统

#### Intel GPU (SYCL/DPC++)
- **Intel oneAPI Base Toolkit** 2023+
- **Intel oneAPI DPC++ 编译器**
- Intel Arc GPU 或 Intel 集成显卡（Iris Xe、UHD）
- 支持 Windows/Linux

### 纯 CPU 模式

无特殊要求 - 适用于任何支持 AVX2 的 x86_64 CPU。如果可用，将自动检测并启用 AVX512。

---

## 许可证

本项目采用 [LICENSE](LICENSE) 许可证。

---

## 贡献

欢迎贡献代码、报告问题或提出建议！请查看我们的 [贡献指南](CONTRIBUTING.md)。

---

## 关注我们

获取最新开发动态、模型发布和社区讨论：

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

---

## 致谢

感谢以下开源项目：

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF 格式和量化灵感
- [transformers](https://github.com/huggingface/transformers) - 模型架构参考
- [ggml](https://github.com/ggerganov/ggml) - 张量运算库

---

<div align="center">

**Made with ❤️ by the swllm.cpp team**

[返回顶部](#swllmcpp)

</div>
