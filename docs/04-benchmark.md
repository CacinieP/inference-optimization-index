# 指标与压测方法

> 结论先行：**压测的目的是找拐点，不是刷峰值数字。** 单点最高吞吐没有意义，能说清「在什么并发下 p99 开始崩」才有意义。

---

## 1. 指标定义

| 指标 | 定义 | 关键点 |
|---|---|---|
| **TTFT**（Time To First Token） | 请求发出 → 收到第一个 token | ≈ 排队时间 + prefill 时间；交互体验主要看它 |
| **ITL**（Inter-Token Latency） | 相邻两个输出 token 的时间间隔 | ≈ decode 单步时间；决定「读起来卡不卡」 |
| **TPOT**（Time Per Output Token） | 总生成时间 / 输出 token 数 | 与 ITL 常互换使用，注意口径 |
| **E2E Latency** | 请求发出 → 最后一个 token | SLA 常用 |
| **Throughput** | 单位时间总输出 tokens / 请求数 | 成本指标 |
| **Goodput** | **满足 SLA 的有效吞吐** | SLO 约束下的真实能力，优化目标应选它 |
| **Prefix/KV 命中率** | 命中缓存的 token 占比 | 长系统提示、RAG、多轮对话场景决定成败 |
| **KV Cache 水位** | 已用 / 可用 block 数 | 接近满会触发抢占与重算，延迟骤升 |

**关系速记**：`E2E ≈ TTFT + (输出 token 数 − 1) × ITL`。

---

## 2. 压测三条原则

1. **开环 vs 闭环要分清**
   - 开环（固定到达速率，如 QPS）→ 测「系统能不能扛住这个负载」
   - 闭环（固定并发数）→ 测「在这么多并发下的延迟」
   - 两者混用会得出互相矛盾的结论。
2. **必须 warmup**：CUDA graph 捕获、kernel autotune、前缀缓存预热都会让前几轮数据偏慢。丢弃前 10–20% 请求。
3. **看分位数不看均值**：报告 p50 / p95 / p99；均值会把长尾藏起来。

---

## 3. 用本仓脚本

```bash
# 依赖：bash + curl + python3（无第三方包）
./scripts/bench-serving.sh \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --requests 20 \
  --concurrency 4 \
  --max-tokens 128
```

输出：每个请求的 TTFT、ITL、输出 tokens、总时长，以及汇总的 p50/p95/p99。

**注意**：这是轻量自测脚本，用于快速对照与回归；正式报告建议用 [vLLM benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks)、[GuideLLM](https://github.com/neuralmagic/guidellm) 或 [llmperf](https://github.com/ray-project/llmperf)，它们能更好地控制到达过程与统计口径。

---

## 4. 压测流程（可直接照着做）

1. **定 SLO**：例如「TTFT p95 < 800 ms，ITL p95 < 80 ms」
2. **固定变量**：模型、量化、max_tokens、prompt 长度分布、硬件
3. **扫并发**：1 / 2 / 4 / 8 / 16 / 32 / 64 …，每档跑够样本（≥ 100 请求或 ≥ 60 s）
4. **记录拐点**：吞吐不再线性增长、或 p99 突增的那一档
5. **换变量重跑**：只改一个（引擎 / 量化 / chunked prefill / 前缀缓存）
6. **写结论**：一张表 + 一句「在 X 并发内满足 SLO，超过则 Y 成为瓶颈」

### prompt 长度分布要贴近真实

| 场景 | 输入长度 | 输出长度 |
|---|---|---|
| 闲聊 | 短 | 短 |
| RAG / 知识库 | 长（2k–32k） | 中 |
| Agent 工具调用 | 中长且反复追加 | 短 |
| 代码生成 | 中 | 长 |

压测只用一种长度分布，结论一定失真。

---

## 5. 数据判读表

| 现象 | 可能原因 | 先查什么 |
|---|---|---|
| TTFT 高但 ITL 正常 | 排队久 / prefill 重 / 长 prompt | 队列深度、chunked prefill 是否开启 |
| ITL 抖动大 | 批大小波动、抢占重算、显存换页 | KV 水位、抢占次数、batch 统计 |
| 并发上去吞吐不涨 | 已到算力或带宽上限；或调度退化 | GPU 利用率、显存带宽利用率 |
| 吞吐高但 p99 崩 | 调度不公平 / 长请求阻塞短请求 | 调度策略、是否开启优先级/分块 |
| 命中率上不去 | 前缀不一致（时间戳、随机 ID 在系统提示里） | 实际 prompt 前缀对比 |
| 量化后速度没提升 | 未在目标 batch 下走量化 kernel；或回退 FP | profiling 看 kernel 名与位宽 |
| 冷启动很慢 | 图构建 / engine 反序列化 / 首次加载 | 分阶段计时 |

---

## 6. 报告模板（抄这个就行）

```
硬件：<GPU/CPU> × N ｜ 引擎：<name+version> ｜ 模型：<name+量化>
SLO：TTFT p95 < ?ms，ITL p95 < ?ms
负载：并发 1/8/32，输入 ~2k tokens，输出 256 tokens，样本 200/档

| 并发 | TTFT p50/p95 | ITL p50/p95 | 吞吐 tok/s | goodput | KV 水位 |
|---|---|---|---|---|---|
| 1  | | | | | |
| 8  | | | | | |
| 32 | | | | | |

结论：在 ___ 并发内满足 SLO；瓶颈是 ___；下一步优化 ___。
```