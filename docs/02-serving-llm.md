# 服务端 LLM 推理 · 技术地图

> 目的：把「概念」和「去哪个仓库看实现」对上。路径随版本变动，这里给的是**模块级定位**，以各仓 main 分支为准。

---

## 1. 一次请求的生命周期

```
prompt ──tokenize──▶ [排队] ──▶ prefill（并行算全序列）──▶ decode（逐 token，自回归）
                                   │                          │
                                   └── 写 KV Cache ───────────┴── 读 KV Cache
                                                                 │
                                                    detokenize ◀──┘ ──▶ 流式返回
```

| 阶段 | 特征 | 主要瓶颈 |
|---|---|---|
| Prefill | 大矩阵乘，算力密集 | GPU 算力（compute-bound） |
| Decode | 每步只算 1 token，但要把整个 KV 读一遍 | 显存带宽（memory-bound） |
| 排队 | 调度器决定谁先跑 | 调度策略与公平性 |

**一句话记住**：prefill 拼算力，decode 拼带宽，系统优化大多在「怎么让这两件事不互相干扰」。

---

## 2. 核心概念 → 实现入口

| 概念 | 解决什么 | 去哪看 |
|---|---|---|
| **KV Cache** | 避免 decode 重复计算历史 token 的 K/V | 任意引擎的 attention backend；nano-vllm 最易读 |
| **PagedAttention** | KV 按 block 分页，消除碎片、支持共享 | [vLLM 论文](https://arxiv.org/abs/2309.06180)；vLLM `block manager` 类；nano-vllm 对应实现 |
| **Continuous batching** | 请求级动态进出，而不是等整批结束 | 引擎 `scheduler`；Orca 论文思想 |
| **Chunked prefill** | 长 prompt 切片，避免阻塞 decode | [Sarathi-Serve](https://arxiv.org/abs/2403.02310)；vLLM/SGLang scheduler |
| **Prefix caching / RadixAttention** | 相同前缀只算一次 | [SGLang 论文](https://arxiv.org/abs/2312.07104)；mini-sglang 的 RadixCache |
| **PD 分离** | prefill 与 decode 分机部署，各自扩缩容 | [DistServe](https://arxiv.org/abs/2401.09670)、[Mooncake](https://arxiv.org/abs/2407.00079) |
| **投机解码** | 小模型起草、大模型校验，多出 token | [Speculative Decoding](https://arxiv.org/abs/2211.17192)、[EAGLE](https://arxiv.org/abs/2401.15077)；SGLang 有完整实现 |
| **量化** | 降权重/KV 位宽，省带宽和显存 | [AWQ](https://arxiv.org/abs/2306.00978)/[GPTQ](https://arxiv.org/abs/2210.17323)/[SmoothQuant](https://arxiv.org/abs/2211.10438)；Model-Optimizer |
| **并行（TP/PP/EP）** | 单卡放不下时切分 | vLLM/SGLang 分布式执行器；Megatron 文档（训练侧同源） |
| **KV 复用/卸载** | 跨请求、跨实例复用；显存不够落 CPU/SSD | [LMCache](https://github.com/LMCache/LMCache) |
| **结构化输出约束** | 语法约束下的采样加速 | SGLang 的 grammar backend |

---

## 3. 引擎选型矩阵

| 场景 | 首选 | 理由 |
|---|---|---|
| 通用在线 API，模型多、生态全 | **vLLM** | 模型覆盖最广，社区最大，出问题搜得到 |
| 高并发 + 大量共享前缀（RAG/Agent） | **SGLang** | RadixAttention + 调度激进，前缀命中场景优势明显 |
| NVIDIA GPU 极限性能、可接受构建成本 | **TensorRT-LLM** | kernel 融合与 in-flight batching 更彻底 |
| 国产模型 / 国产卡适配 | **LMDeploy**、vllm-ascend 等 | 适配跟进快 |
| 单机 CPU / Mac / 边缘 | **llama.cpp** | GGUF 生态，零依赖分发 |
| 超长上下文 + 多轮，想省重复计算 | vLLM + **LMCache** | KV 落盘/跨实例复用 |

**选型时真正该问的三个问题**：① 延迟 SLA 是 TTFT 还是 ITL 主导？② 前缀复用率高不高？③ 显存/卡型受不受限？

---

## 4. 常见误区

1. **只优化吞吐**：吞吐上去了但 p99 TTFT 崩了，用户体验更差。用 **goodput**（满足 SLA 的吞吐）做目标。
2. **忽略排队**：压测时并发一高，瓶颈常常在调度与显存回收，不在 kernel。
3. **拿平均值下结论**：LLM 延迟是长尾分布，看 p50/p95/p99。
4. **量化不测质量**：4-bit 在长上下文和代码任务上退化最明显，必须在目标任务上回归。
5. **迷信单点优化**：KV 量化 + 连续批处理 + 前缀缓存叠加时收益不线性，要整体压测。
6. **忘记 warmup**：CUDA graph 捕获、kernel 自动调优都要预热，冷启动数据会误导决策。

---

## 5. 观测清单（上生产必备）

| 类别 | 指标 |
|---|---|
| 延迟 | TTFT p50/p95/p99、ITL、E2E |
| 吞吐 | tokens/s、requests/s、goodput |
| 资源 | GPU 利用率、显存占用、KV Cache 水位、swap 次数 |
| 缓存 | prefix cache 命中率、KV 命中率 |
| 队列 | 等待请求数、排队时长、抢占/重算次数 |

vLLM 暴露 Prometheus 指标（`/metrics`），可直接接 Grafana；把「TTFT / ITL / 队列深度 / KV 水位」四个面板先搭起来。