# 端侧与通用模型推理 · 技术地图

> 端侧优化的目标函数和服务器完全不同：**不是吞吐，是「能不能跑、跑多久、烫不烫、延迟稳不稳」**。
> 同一批技术（量化、算子融合、图优化）在两端用法相反：服务器追求 batch 打满，端侧追求 batch=1 也要快。

---

## 1. 端侧第一性原理：三堵墙

| 墙 | 公式 / 表现 | 直接后果 |
|---|---|---|
| **内存墙** | 权重字节数 = 参数量 × 位宽 / 8 | 4-bit 下 7B ≈ 3.5–4 GB；超过设备可用内存就 OOM 或频繁换页 |
| **带宽墙** | decode 每 token 几乎要读一遍权重 → `token/s ≲ 带宽 / 权重字节` | 手机 LPDDR5 带宽 ~50 GB/s，4-bit 7B 理论上限 ~12 tok/s |
| **功耗/散热墙** | 持续高占用 → 降频 → 延迟抖动 | 冷启动前几十 token 快，之后掉速 |

**推论**：端侧优化 80% 的收益来自「少读内存」——量化、稀疏、KV 压缩、分层缓存，而不是加算力。

---

## 2. 四套主栈对照

| | TensorRT / TRT-LLM | ONNX Runtime | TVM / MLC | GGUF + llama.cpp |
|---|---|---|---|---|
| 目标硬件 | NVIDIA GPU、Jetson | CPU/GPU/NPU（跨平台，靠 EP） | 任意后端（含国产 NPU、WebGPU） | CPU、Metal、Vulkan |
| 输入 | ONNX / PyTorch → engine | ONNX（或 ORT Format） | Relay/TE 图 → 目标代码 | GGUF（转换 + 量化一体） |
| 核心机制 | 层融合、精度校准、plugin、kernel 自动调优 | Execution Provider 委托、图优化器 | auto-schedule、算子 lowering | 量化 kernel、mmap、内存布局 |
| 学习曲线 | 中（构建慢、版本敏感） | 低-中 | 高（编译器思维） | 低（开箱即用） |
| 典型问题 | engine 与驱动/TensorRT 版本绑定 | 算子不支持 → 回退 CPU，性能断崖 | 调度不理想时不如手写 | 极限性能不如前两者 |
| 文档 | [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html) | [ORT](https://onnxruntime.ai/docs/) | [TVM](https://tvm.apache.org/docs/)、[MLC 课程](https://mlc.ai/courses.html) | [GGUF 规范](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) |

### 转换链路（记住这张图，选栈就不会乱）

```
PyTorch 模型
   ├─ torch.export ──────────────▶ ExecuTorch (.pte) ──▶ 手机/嵌入式
   ├─ torch.onnx.export ─▶ ONNX ─┬▶ ONNX Runtime（选 EP）
   │                             └▶ TensorRT（.engine）
   ├─ convert_hf_to_gguf.py ────▶ GGUF（量化）──▶ llama.cpp / Ollama
   ├─ TVM/MLC 前端 ─────────────▶ 任意后端（含 WebGPU）
   └─ coremltools.convert ──────▶ Core ML (.mlpackage) ──▶ Apple ANE
```

---

## 3. 量化格式对照

| 格式 | 位宽 / 粒度 | 谁在用 | 注意 |
|---|---|---|---|
| **GGUF `Q4_K_M`** | 4-bit，K-quant 分块 + mixed precision | llama.cpp / Ollama | 分发首选；`_K_M` 表示中等混合精度档 |
| **GGUF `Q8_0` / `Q6_K`** | 8/6-bit | 同上 | 质量接近 FP16，体积降一半 |
| **AWQ** | 权重 4-bit，激活保留 | vLLM / SGLang / TensorRT-LLM | 激活感知，比朴素 RTN 好；需校准集 |
| **GPTQ** | 权重 3–4-bit，逐层误差补偿 | 同上 | 生态老牌，需校准集 |
| **SmoothQuant** | W8A8（激活也量化） | 服务端高吞吐 | 需把激活离群值「摊平」到权重 |
| **FP8 (E4M3/E5M2)** | 8-bit 浮点 | H100 及更新 GPU | 精度损失小，硬件原生支持 |
| **MXFP4 / NVFP4** | 4-bit 微缩放浮点 | 新 GPU / 新工具链 | 需要硬件与工具链同时支持 |
| **1-bit（BitNet）** | 三值/二值权重 | [BitNet](https://github.com/microsoft/BitNet) | 需专门训练，不是直接量化出来的 |

> 选择顺序：先看**硬件支不支持** → 再看**任务对 4-bit 的敏感度** → 最后才比速度。

---

## 4. 端侧 LLM 的专用技巧

| 技巧 | 思路 | 代表 |
|---|---|---|
| **KV Cache 量化/压缩** | KV 从 FP16 降到 INT8/INT4，长上下文时才见效 | llama.cpp KV cache 类型、KV 压缩库 |
| **分层/分块加载** | 权重不常驻，按需从闪存流式读 | llama.cpp mmap、Apple「LLM in a flash」思路 |
| **稀疏激活 + 专家缓存** | 只算被激活的专家，热专家驻留内存 | [PowerInfer](https://github.com/Tiiny-AI/PowerInfer)、[ktransformers](https://github.com/kvcache-ai/ktransformers) |
| **投机解码** | 小模型起草降低 decode 次数 | 端侧大模型 + 小模型组合 |
| **算子融合 + 静态图** | 减少 kernel 启动与内存往返 | TensorRT、Core ML、LiteRT |
| **浏览器端** | WebGPU + WASM 跑小模型 | [web-llm](https://github.com/mlc-ai/web-llm) |

---

## 5. 硬件平台对照（选型用）

| 平台 | 典型 NPU/加速器 | 首选栈 |
|---|---|---|
| NVIDIA Jetson | GPU + DLA | TensorRT |
| Apple iPhone / Mac | ANE + GPU | Core ML（ANE）/ llama.cpp（Metal） |
| Android 旗舰 | 高通 Hexagon / 联发科 APU | LiteRT delegate / MNN / NCNN |
| Intel PC / 边缘盒子 | CPU + iGPU + NPU | OpenVINO |
| 浏览器 | WebGPU | web-llm / TVM WASM |
| 国产 NPU（昇腾等） | 专用 | CANN / MindIE / vllm-ascend、TVM 后端 |

---

## 6. 端侧学习顺序（4 步）

1. **先建立容量感**：算内存/带宽/理论 token/s（见 §1），能预判「跑不跑得动」
2. **用现成栈跑通**：llama.cpp + GGUF 量化，理解 `Q4_K_M` 到底是什么
3. **换一次栈**：PyTorch → ONNX → ORT（或 TensorRT），亲手踩「算子回退」的坑
4. **再往下**：算子剖析（perf / Nsight / ORT profiling）与 kernel 级优化；此时回到 [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) 与 [Triton](https://github.com/triton-lang/triton)

配套课程：[efficientml.ai](https://efficientml.ai/)（MIT，剪枝/量化/NAS/蒸馏 + 部署）、[mlsysbook.ai](https://mlsysbook.ai/)（系统视角）。

---

## 7. 端侧常见坑

1. **算子不支持悄悄回退 CPU**：一定要看 profiling 里每个节点的 EP 落点，别只看总延迟。
2. **dynamic shape 与静态图**：固定 shape 通常更快，但要注意 pad 带来的浪费。
3. **layout 转换开销**：NCHW ↔ NHWC 的搬运可能吃掉融合收益。
4. **精度对齐**：量化后先跑一遍与 FP 参考的 logits 差异，再看任务指标。
5. **内存峰值而非均值**：端侧 OOM 往往发生在中间张量峰值。
6. **冷启动**：模型加载 + 图构建 + 首次推理，用户感知的是这一段。
7. **保温降频**：长时间跑会掉速，报告里要写「第 N 分钟后的 tok/s」。