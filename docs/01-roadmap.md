# 12 周落地计划

> 两条轨不必都走完。先按 [README 的 30 秒选路](../README.md#30-秒选路) 选一条，再回来看细节。
> 每周都有**可验收的产出物**；没有产出的周等于没学。

---

## A 轨 · 服务端 LLM 推理（8 周主线 + 4 周加练）

### W1 · 把「为什么慢」算清楚

- **任务**：Roofline 模型；给定 7B/70B 模型与 GPU 规格，算出算术强度、理论 token/s、KV Cache 占用量
- **资料**：[time-to-first-token](https://github.com/patchy631/time-to-first-token) W1–W2、[llm-internals](https://github.com/amitshekhariitbhu/llm-internals)
- **产出**：一页纸推导，含 `KV bytes = 2 × layers × kv_heads × head_dim × seq_len × batch × dtype_bytes`
- **验收**：能解释「为什么长上下文先撞显存、短上下文先撞带宽」

### W2–W3 · 把引擎跑起来并压出拐点

- **任务**：同一模型分别部署 vLLM 与 SGLang；开环扫描 QPS 与闭环扫描并发；记录 TTFT/ITL/吞吐/显存
- **资料**：[vLLM benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)、[GuideLLM](https://github.com/neuralmagic/guidellm)、本仓 [scripts/bench-serving.sh](../scripts/bench-serving.sh)
- **产出**：双引擎对比表 + 一张「并发 → 吞吐/延迟」曲线
- **验收**：能找到吞吐饱和点，并说清瓶颈是算力、带宽还是调度

### W4–W5 · 手写一个 mini 引擎

- **任务**：跟读 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，实现 block 管理 + 连续批处理；再读 [mini-sglang](https://github.com/sgl-project/mini-sglang) 的 RadixCache
- **产出**：能跑的小引擎 + 与 vLLM 的对照数字（哪怕慢 3 倍）
- **验收**：能画出一次请求从 tokenize → schedule → forward → detokenize 的完整路径

### W6 · 量化三件套

- **任务**：AWQ / GPTQ / SmoothQuant 各跑一次，测精度（任务指标）与延迟
- **资料**：[Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)、对应论文见 [papers](05-papers.md)
- **产出**：精度-延迟-显存三维表
- **验收**：能说明「为什么 W4A16 比 W8A8 在 batch 小时更划算」

### W7 · 投机解码

- **任务**：接入 EAGLE 或 Medusa（SGLang/vLLM 都有实现），测接受率 α、加速比与额外显存
- **产出**：加速比 vs α 曲线，附「什么负载不适合投机」的结论
- **验收**：能解释投机解码的收益边界（低 batch 高延迟场景）

### W8 · 选型与总结

- **产出**：3000 字选型文档：场景 → 引擎 → 量化策略 → 硬件 → 成本估算
- **验收**：能给一个真实业务提方案并说服人

### W9–W12 · 加练（可选）

1. 读 [LMCache](https://github.com/LMCache/LMCache)，做前缀缓存命中率实验
2. 读 [Sarathi-Serve](https://arxiv.org/abs/2403.02310)，在引擎里找 chunked prefill 实现
3. 读 [Mooncake](https://arxiv.org/abs/2407.00079)，理解 PD 分离与 KV 传输
4. 自己写一个 attention kernel（Triton），对比 FlashAttention

---

## B 轨 · 端侧与通用推理（8 周）

### W1 · 端侧第一性原理

- **任务**：算内存墙。模型 4-bit 量化后大小、目标设备带宽、理论 token/s 上限
- **资料**：[efficientml.ai](https://efficientml.ai/) 前两讲、[mlsysbook.ai](https://mlsysbook.ai/)
- **产出**：端侧容量模型表（手机 / 车机 / Jetson / Mac）
- **验收**：能预判「这个模型能不能在这个设备上跑起来」

### W2–W3 · GGUF 与本地推理

- **任务**：同一模型导出 Q4_K_M / Q5_K_M / Q8_0，测质量、速度、体积
- **资料**：[GGUF 规范](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)、[llama.cpp](https://github.com/ggml-org/llama.cpp)
- **产出**：三维对比表 + 一页「量化档位怎么选」
- **验收**：能解释 K-quant 的分块量化与 mixed precision

### W4–W5 · 换栈：ONNX Runtime 或 TensorRT

- **任务**：PyTorch → ONNX → ORT（选 EP）/ 或 TensorRT engine 构建；开图优化与精度校准
- **资料**：[ORT 文档](https://onnxruntime.ai/docs/)、[TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)
- **产出**：可复现转换脚本 + 前后性能对比
- **验收**：能定位「是算子没融合，还是 EP 没吃上」

### W6 · 剖析瓶颈

- **任务**：perf / Nsight / ORT profiling；找热点算子与内存拷贝
- **产出**：瓶颈定位报告（CPU 抢占？H2D 拷贝？layout？）
- **验收**：能给出一个具体优化并量化收益

### W7 · 落地一条边端链路

- 选一个：手机（ExecuTorch / MNN）、Mac（Core ML / Metal）、浏览器（WebLLM）
- **产出**：可运行的 demo + 冷启动/首 token/功耗数据

### W8 · 端侧选型文档

- **产出**：设备 × 框架 × 量化 × 时延的决策矩阵

---

## 节奏建议

| 每天 | 内容 |
|---|---|
| 30 min | 读资料（跟 time-to-first-token 的日节奏） |
| 60 min | 动手（跑实验/改代码） |
| 15 min | 写一句结论进笔记：**今天验证了什么假设** |

> 判断自己是否真的学会了：能不能在白板上画出数据流，并指出瓶颈在哪一层。