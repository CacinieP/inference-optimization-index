<div align="center">

# 推理优化学习索引

**LLM 服务端推理 · 端侧与通用模型推理**：仓库、书籍、课程、论文 —— 链接逐条实测可访问

[![License: MIT](https://img.shields.io/badge/LICENSE-MIT-14262b?style=flat-square)](./LICENSE)
[![Links](https://img.shields.io/badge/LINKS-CHECKED%202026--10--25-69c8bd?style=flat-square)](./scripts/check-links.py)
[![PRs](https://img.shields.io/badge/PRs-WELCOME-e9795f?style=flat-square)](./CONTRIBUTING.md)
[![links](https://github.com/CacinieP/inference-optimization-index/actions/workflows/links.yml/badge.svg)](https://github.com/CacinieP/inference-optimization-index/actions/workflows/links.yml)

</div>

> **一句话**：先把「为什么慢」算清楚，再把一个引擎跑起来，最后才读源码。
>
> 这不是链接堆砌。每一条都标了「为什么看」；动不了手的资料（纯 PPT、无代码、长期停更）不收录。

---

## 30 秒选路

| 你的处境 | 从这里开始 | 预计投入 |
|---|---|---|
| 只会调 API，想把推理系统搞明白 | [time-to-first-token](https://github.com/patchy631/time-to-first-token) 10 周路线 + 本文 §1-A | 4–6 周 |
| 要给公司做部署/降本决策 | §1-C 引擎对比 + §7 指标 | 1 周建立判断力 |
| 准备推理方向面试 | [llm_interview_note](https://github.com/wdndev/llm_interview_note) + §1-D CUDA 基本功 | 2–3 周 |
| 做手机 / 车机 / 嵌入式 / 浏览器 | §2 端侧矩阵 + [efficientml.ai](https://efficientml.ai/) | 6–8 周 |
| 想发论文或做研究 | §5 论文主线 + [AISystem](https://github.com/Infrasys-AI/AISystem) | 持续 |

## 目录

- [一、服务端 LLM 推理](#一服务端-llm-推理)
- [二、端侧与通用模型推理](#二端侧与通用模型推理)
- [三、书](#三书)
- [四、课程](#四课程)
- [五、论文主线](#五论文主线14-篇)
- [六、落地计划](#六落地计划)
- [七、指标与压测](#七指标与压测)
- [技术地图（docs/）](#技术地图docs)
- [收录标准](#收录标准)

---

## 一、服务端 LLM 推理

### A. 路线与清单（先建地图，别急着读代码）

| 资源 | 规模 | 为什么看 |
|---|---|---|
| [patchy631/time-to-first-token](https://github.com/patchy631/time-to-first-token) | ★ 1.0k | **10 周 × 每天 30 分钟**的推理服务路线图，含 vLLM/SGLang/量化/投机解码的周计划与实验 |
| [amitshekhariitbhu/llm-internals](https://github.com/amitshekhariitbhu/llm-internals) | ★ 1.8k | 从 tokenization → attention → 推理优化逐级推进，解释最不绕 |
| [wdndev/llm_interview_note](https://github.com/wdndev/llm_interview_note) | ★ 15.3k | 中文面试笔记，推理优化章节完整，适合当速查手册 |
| [Infrasys-AI/AISystem](https://github.com/Infrasys-AI/AISystem) | ★ 18.0k | 中文 AI 全栈：AI 芯片 → 编译器 → 推理框架 → 训练框架，建立「系统视角」必读 |
| [xlite-dev/Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference) | ★ 5.5k | 论文 + 代码清单，按技术点按图索骥 |
| [cypggs/llm-inference-mastery](https://github.com/cypggs/llm-inference-mastery) | ★ 4 | 中文十周实机笔记：手算 KV Cache 容量再与 vLLM 实测对比，工程直觉很好 |
| [depixels/Advanced-LLM-Systems-Learning](https://github.com/depixels/Advanced-LLM-Systems-Learning) | ★ 16 | 训练/推理/RLHF 的系统学习框架 + 节奏表 |

### B. 从零手写一个引擎（性价比最高的一步）

| 资源 | 规模 | 为什么看 |
|---|---|---|
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | ★ 16.1k | **首选**。千行级复现 vLLM 核心：PagedAttention、连续批处理、KV 管理。读完能讲清调度器 |
| [sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang) | ★ 5.4k | 官方精简版 SGLang：RadixCache、chunked prefill、overlap 调度 |
| [karpathy/llm.c](https://github.com/karpathy/llm.c) | ★ 31.2k | 纯 C/CUDA 跑 LLM，看清算子层现实 |
| [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) | ★ 23.3k | 编译路线（TVM 系），换硬件后端时的正解 |
| [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) | ★ 33.8k | 极简框架 + 自研编译器，理解「图 → kernel」链路 |

### C. 生产引擎与内核（第 3 个月起读源码）

| 项目 | 规模 | 定位 |
|---|---|---|
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | ★ 95.4k | 事实标准。PagedAttention + 连续批处理，生态最全 |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | ★ 38.1k | RadixAttention 前缀复用 + 结构化输出，高并发场景常更快 |
| [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | ★ 14.9k | NVIDIA 官方：kernel 融合 + in-flight batching，GPU 上极限性能 |
| [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) | ★ 8.1k | 国产模型适配好，量化+服务一体化（TurboMind） |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | ★ 132k | CPU/Metal/边缘事实标准，GGUF 生态源头 |
| [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | ★ 19.6k | 异构（CPU+GPU）推理与 MoE 优化，单机跑大模型 |
| [LMCache/LMCache](https://github.com/LMCache/LMCache) | ★ 12.2k | KV Cache 跨请求/跨实例复用，长上下文降本关键 |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | ★ 6.8k | 服务场景的 attention/kernel 库，读懂 decode 阶段优化 |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | ★ 25.2k | IO-aware attention 的源头实现 |
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | ★ 5.4k | 量化/蒸馏/剪枝/NAS 一站式工具箱 |

### D. 底层功力（CUDA 与编译，决定上限）

| 资源 | 规模 | 为什么看 |
|---|---|---|
| [srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles) | ★ 12.6k | 用解谜学 CUDA 思维（线程/共享内存/归约），一晚上入门 |
| [BBuf/how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) | ★ 3.3k | 中文 CUDA 算子优化合集（含 FlashAttention、reduce、gemm 拆解） |
| [triton-lang/triton](https://github.com/triton-lang/triton) | ★ 高 | 用 Python DSL 写 kernel，工业界写自定义算子的主流方式 |

---

## 二、端侧与通用模型推理

> 端侧的约束不是「吞吐」而是「内存带宽 + 功耗 + 冷启动」。同一套技术（量化、算子融合、图优化）目标函数完全不同。

### 2.1 四套主栈对照（先选栈，再学）

| 栈 | 适合 | 学习入口 | 关键概念 |
|---|---|---|---|
| **TensorRT / TensorRT-LLM** | NVIDIA GPU（服务器 + Jetson） | [SDK 文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html) · [样例仓库](https://github.com/NVIDIA/TensorRT) | engine 构建、层融合、精度校准（FP16/INT8/FP8）、plugin |
| **ONNX Runtime** | 跨平台 CPU/GPU/NPU，Windows/Linux/移动 | [文档](https://onnxruntime.ai/docs/) · [GenAI 扩展](https://github.com/microsoft/onnxruntime-genai) | ONNX 图、Execution Provider（EP）、图优化、ORT Format |
| **TVM / MLC** | 任意后端（含国产 NPU、WebGPU） | [文档](https://tvm.apache.org/docs/) · [MLC 课程](https://mlc.ai/courses.html) | 计算图 → TE/TIR → auto-schedule → 目标代码 |
| **GGUF + llama.cpp** | 本地/CPU/Mac/边缘，分发友好 | [GGUF 规范](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) · [llama.cpp](https://github.com/ggml-org/llama.cpp) | 量化格式（Q4_K_M 等）、mmap、内存布局 |

### 2.2 端侧框架矩阵

| 框架 | 平台 | 规模 | 一句话 |
|---|---|---|---|
| [pytorch/executorch](https://github.com/pytorch/executorch) | 移动/嵌入式 | ★ 5.1k | PyTorch 官方端侧路线，`.pte` 格式 + 后端委托 |
| [google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT) | Android/嵌入式 | ★ 3.5k | TensorFlow Lite 后继者，Android NNAPI/GPU delegate |
| [apple/coremltools](https://github.com/apple/coremltools) | Apple 设备 | ★ 5.4k | 转 Core ML，吃 ANE；配 [Core ML 文档](https://developer.apple.com/documentation/coreml) |
| [alibaba/MNN](https://github.com/alibaba/MNN) | 移动/边缘 | ★ 16.3k | 阿里出品，端侧 LLM/多模态落地多 |
| [Tencent/ncnn](https://github.com/Tencent/ncnn) | 移动 | ★ 24.0k | 极致轻量，手机端 CV 经典选择 |
| [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino) | Intel CPU/GPU/NPU | ★ 11.1k | x86 与 Intel 加速卡部署首选 |
| [mlc-ai/web-llm](https://github.com/mlc-ai/web-llm) | 浏览器 | ★ 19.3k | WebGPU 跑 LLM，编译到 WASM/WebGPU |
| [microsoft/BitNet](https://github.com/microsoft/BitNet) | 边缘/CPU | ★ 41.2k | 1-bit LLM 专用推理框架 |
| [Tiiny-AI/PowerInfer](https://github.com/Tiiny-AI/PowerInfer) | 本地/单卡 | ★ 9.9k | 稀疏激活 + 热专家缓存，端侧提速思路经典 |

### 2.3 压缩与量化工具

| 工具 | 规模 | 用途 |
|---|---|---|
| [intel/neural-compressor](https://github.com/intel/neural-compressor) | ★ 2.7k | INT8/FP8/INT4/MXFP4/NVFP4 量化 + 稀疏，CPU 侧最全 |
| [openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf) | ★ 1.2k | OpenVINO 配套压缩框架，PTQ/QAT 流程清晰 |
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | ★ 5.4k | GPU 侧量化/蒸馏/剪枝统一入口 |

> 量化格式速记：**GGUF**（分发/CPU）· **AWQ / GPTQ**（GPU 权重仅量化）· **SmoothQuant**（W8A8）· **FP8 / MXFP4 / NVFP4**（新硬件）· **1-bit（BitNet）**（极端压缩）。详见 [docs/03-edge-and-general.md](docs/03-edge-and-general.md)。

### 2.4 端侧课程与书

- **[efficientml.ai](https://efficientml.ai/)** — MIT HAN Lab《TinyML and Efficient AI Computing》，剪枝/量化/NAS/蒸馏 + 端侧部署，**端侧方向最系统的课**
- **[mlsysbook.ai](https://mlsysbook.ai/)** — Harvard & MIT Press《Machine Learning Systems》，含模型压缩与部署章节，网页免费
- **[tinyml.org](https://www.tinyml.org/)** — TinyML 社区与入门资料（MCU 级部署）

### 2.5 本仓脚本：一条命令跑完 HF → GGUF → 对比

```bash
# 量化 Q4_K_M / Q5_K_M / Q8_0，并用 llama-bench 出速度
./scripts/quantize-gguf.sh --model Qwen/Qwen2.5-1.5B-Instruct --quants Q4_K_M,Q5_K_M,Q8_0
# 只看看会执行什么（不下载、不编译）
./scripts/quantize-gguf.sh --model Qwen/Qwen2.5-1.5B-Instruct --dry-run
```

脚本负责：clone/编译 llama.cpp → 下载或读取模型 → 转 f16 → 逐档量化 → `llama-bench` 压测 → 体积汇总表。
对应 [docs/01-roadmap.md](docs/01-roadmap.md) B 轨 W2–W3；判读方法见 [docs/04-benchmark.md](docs/04-benchmark.md)。

---

## 三、书

### 中文

| 书 | 说明 |
|---|---|
| **《vLLM 与 SGLang：大模型高效推理双引擎实战》** 李明飞，电子工业出版社 2026-08，468 页 · [出版社页](https://www.phei.com.cn/module/goods/wssd_content.jsp?bookid=70871) | **中文里目前最贴题的专题书**：双引擎为主线，从原理到生产部署 |
| **《大语言模型》** · [llmbook-zh.github.io](https://llmbook-zh.github.io/) | **免费 PDF + 全套课件**；第 9 章「解码与部署」正是推理加速 |
| **《大规模语言模型：从理论到实践》** · [intro-llm.github.io](https://intro-llm.github.io/) | 复旦体系教材，解码与部署章节扎实 |
| **《大模型轻量化：模型压缩与训练加速》** | 以 DeepSeek-V3 为案例讲压缩与推理加速 |

### 英文

| 书 | 说明 |
|---|---|
| **AI Systems Performance Engineering**（Chris Fregly, O'Reilly）· [配套代码](https://github.com/cfregly/ai-performance-engineering) | 千页级，GPU/CUDA → 训练与推理调优，逐章 benchmark，目前最完整 |
| **[Machine Learning Systems](https://mlsysbook.ai/)**（Harvard & MIT Press） | 网页免费，系统视角 + 部署与压缩，先读这本再读专著 |
| **[ML Engineering](https://github.com/stas00/ml-engineering)**（Stas Bekman） | 免费在线书，工程细节密度极高：显存、并行、调试 |
| **AI Engineering**（Chip Huyen）· [作者书目](https://huyenchip.com/books/) | 推理优化（量化/批处理/缓存）单独成章，偏产品与决策视角 |
| **Hands-On LLM Serving and Optimization** · [配套代码](https://github.com/orca3/llm-model-inference) | 书名即内容，服务化实操 |
| **The Physics of LLM Inference** · [配套代码](https://github.com/Infatoshi/physics-llm-inference) | 小众但第一性原理推导扎实（Roofline/带宽/容量） |
| **LLM Inference Handbook** · [GitHub](https://github.com/modular/llm-inference-handbook) | 手册型，指标定义与优化手段一页一个，查概念最快 |

---

## 四、课程

| 课程 | 说明 |
|---|---|
| [CMU 11-868 Large Language Model Systems](https://llmsystem.github.io/llmsystem2026fall/docs/Syllabus) | GPU 编程 → MiniTorch → 解码与投机解码，**slides 与作业全公开**，硬核 |
| [MLC 系列课程](https://mlc.ai/courses.html) | 4 门，含 *Towards Efficient Generative LLM Serving*、*Efficient Systems for LLM Serving* |
| [Efficiently Serving LLMs](https://learn.deeplearning.ai/courses/efficiently-serving-llms/information)（DeepLearning.AI × Predibase） | 短课，batching/量化/LoRA 服务化，一晚上建立直觉 |
| [efficientml.ai](https://efficientml.ai/)（MIT） | 端侧与高效 AI 计算，见 §2.4 |

---

## 五、论文主线（14 篇）

| 主题 | 论文 |
|---|---|
| 综述总纲 | [A Survey on Efficient Inference for LLMs](https://arxiv.org/abs/2404.14294) |
| KV 管理与调度 | [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)、[Sarathi-Serve](https://arxiv.org/abs/2403.02310)、[DistServe](https://arxiv.org/abs/2401.09670) |
| Attention 内核 | [FlashAttention](https://arxiv.org/abs/2205.14135) |
| 量化 | [SmoothQuant](https://arxiv.org/abs/2211.10438)、[AWQ](https://arxiv.org/abs/2306.00978)、[GPTQ](https://arxiv.org/abs/2210.17323) |
| 投机解码 | [Speculative Decoding](https://arxiv.org/abs/2211.17192)、[Medusa](https://arxiv.org/abs/2401.10774)、[EAGLE](https://arxiv.org/abs/2401.15077) |
| 前缀复用 | [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104) |
| 架构与集群 | [DeepSeek-V2 (MLA)](https://arxiv.org/abs/2405.04434)、[Mooncake](https://arxiv.org/abs/2407.00079) |

阅读顺序与每篇「一句话贡献 + 该看哪节」见 [docs/05-papers.md](docs/05-papers.md)。

---

## 六、落地计划

两条轨，各 8 周，详细周计划（任务/产出/验收）见 **[docs/01-roadmap.md](docs/01-roadmap.md)**。

**A 轨 · 服务端**

| 周 | 做什么 | 产出 |
|---|---|---|
| 1 | Roofline + 手算 KV Cache 容量 | 一页纸公式推导 |
| 2–3 | 同模型部署 vLLM / SGLang，压测扫描并发 | 双引擎对比表 + 系统拐点 |
| 4–5 | 精读 nano-vllm，手写 paged KV + 连续批处理 | 可用小引擎 + 性能数字 |
| 6 | AWQ/GPTQ/SmoothQuant 精度-延迟对照 | 曲线图 |
| 7 | EAGLE/Medusa 接入，测接受率与加速比 | 加速报告 |
| 8 | 选型总结 | 可直接当面试材料 |

**B 轨 · 端侧**

| 周 | 做什么 | 产出 |
|---|---|---|
| 1 | 算内存墙：模型大小/带宽/理论 token/s | 端侧容量模型 |
| 2–3 | GGUF 量化对比（Q4_K_M / Q5 / Q8） | 质量-速度-体积三维表 |
| 4–5 | ONNX Runtime 或 TensorRT 端到端转换 + 图优化 | 可复现转换脚本 |
| 6 | 算子/内存剖析（perf、Nsight、EP 选择） | 瓶颈定位报告 |
| 7 | 一条边端链路落地（手机/Mac/浏览器） | demo |
| 8 | 移动端延迟与功耗实测 | 端侧选型文档 |

---

## 七、指标与压测

| 指标 | 含义 | 谁在意 |
|---|---|---|
| **TTFT** | 首 token 延迟（≈ prefill + 排队） | 交互体验 |
| **ITL / TPOT** | 相邻 token / 每 token 输出时间（≈ decode） | 阅读流畅度 |
| **E2E latency** | 端到端时延 | SLA |
| **Throughput** | tokens/s（总） | 成本 |
| **Goodput** | 满足 SLA 的有效吞吐 | **真正该优化的目标** |
| **KV 命中率 / 显存水位** | 前缀缓存效率与 OOM 风险 | 稳定性 |

- 压测工具：[vLLM benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)、[llmperf](https://github.com/ray-project/llmperf)、[GuideLLM](https://github.com/neuralmagic/guidellm)、[NVIDIA NIM benchmarking](https://docs.nvidia.com/nim/benchmarking/llm/latest/index.html)
- 本仓脚本：[scripts/bench-serving.sh](scripts/bench-serving.sh)（无依赖，直接打 OpenAI 兼容端点，输出 TTFT/ITL/吞吐）
- 方法、坑与判读：见 [docs/04-benchmark.md](docs/04-benchmark.md)

---

## 技术地图（docs/）

| 文档 | 内容 |
|---|---|
| [docs/01-roadmap.md](docs/01-roadmap.md) | 12 周双轨计划 + 验收标准 |
| [docs/02-serving-llm.md](docs/02-serving-llm.md) | 服务端技术地图：概念 → 实现 → 源码入口 |
| [docs/03-edge-and-general.md](docs/03-edge-and-general.md) | 端侧与通用推理：四栈对照、量化格式、硬件 |
| [docs/04-benchmark.md](docs/04-benchmark.md) | 指标定义、压测方法、常见坑 |
| [docs/05-papers.md](docs/05-papers.md) | 论文清单、阅读顺序、一句话贡献 |

## 收录标准

1. **能动**：有代码或可复现实验，纯概念文章不收；
2. **活着**：近 18 个月有实质更新，或属稳定经典；
3. **可定位**：能说清「解决哪个环节的哪个问题」；
4. 不收破解、盗版书、拉群引流类链接。

欢迎 PR，格式与流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 数据时效

- 星标与链接状态：**2026-10-25** 抓取，之后会漂移，属正常；
- 书目信息以出版社/作者页面为准；框架版本请以官方发布为准。

---

<p align="center"><sub>SLOW IS A DESIGN CHOICE · MEASURE FIRST</sub></p>