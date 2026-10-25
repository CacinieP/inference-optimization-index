# 论文主线（14 篇）

> 读法建议：**先综述建坐标系，再按「你手上的瓶颈」挑 1–2 篇精读，其余只读摘要与系统设计章节。**
> 每篇都标了「一句话贡献」和「该看哪」，避免从头读到尾的时间黑洞。

---

## 阅读顺序

| 阶段 | 读什么 | 目的 |
|---|---|---|
| ① 建坐标系（1 天） | 综述 2404.14294 | 知道优化手段分几类：数据级 / 模型级 / 系统级 |
| ② 系统主线（1 周） | PagedAttention → Sarathi-Serve → DistServe → Mooncake | 理解调度、显存、部署拓扑这条主线 |
| ③ 压缩主线（1 周） | SmoothQuant → AWQ → GPTQ | 知道每种量化的适用边界 |
| ④ 加速主线（3 天） | Speculative Decoding → Medusa → EAGLE | 理解「用算力换延迟」的家族 |
| ⑤ 内核与架构（按需） | FlashAttention、SGLang、DeepSeek-V2 | 内核优化、前缀复用、MLA 架构 |

---

## 1. 综述

| 论文 | 一句话贡献 | 重点看 |
|---|---|---|
| [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294)（清华 + 无问芯穹） | 把 LLM 推理优化整理成「数据级 / 模型级 / 系统级」三层分类，附各手段的收益与代价对比 | 分类框架图与各章小结；先建立 taxonomy |

---

## 2. 系统与调度

| 论文 | 一句话贡献 | 重点看 |
|---|---|---|
| [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（vLLM） | 把 KV Cache 按 block 分页管理，显存碎片几乎归零，吞吐提升数倍 | §4 内存管理设计、§5 实现；对照 nano-vllm 代码 |
| [Taming Throughput-Latency Tradeoff with Sarathi-Serve](https://arxiv.org/abs/2403.02310) | chunked prefill + stall-free batching：把长 prefill 切片，避免 decode 被阻塞 | §3 调度策略；对应 vLLM 的 `max_num_batched_tokens` 类参数 |
| [DistServe: Disaggregating Prefill and Decoding](https://arxiv.org/abs/2401.09670) | prefill 与 decode 分机部署，各自按瓶颈扩缩容，goodput 显著提升 | §4 资源分配与放置策略 |
| [Mooncake: A KVCache-centric Disaggregated Architecture](https://arxiv.org/abs/2407.00079)（月之暗面） | 以 KV Cache 为中心重构集群：KV 池化、传输与调度协同 | §3 架构、§5 调度器（Conductor） |
| [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) | RadixAttention 用前缀树复用 KV，结构化输出用有限状态机加速 | §4 RadixAttention；对照 mini-sglang |

---

## 3. 量化与压缩

| 论文 | 一句话贡献 | 重点看 |
|---|---|---|
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | 把激活的离群值按通道「迁移」到权重，使 W8A8 可行 | §3 数学推导；理解为什么激活比权重难量化 |
| [AWQ](https://arxiv.org/abs/2306.00978) | 只保护 1% 显著权重通道即可大幅降低量化误差，无需反向传播 | §3 激活感知缩放；实践中精度通常优于 GPTQ |
| [GPTQ](https://arxiv.org/abs/2210.17323) | 逐层二阶误差补偿的一次性权重量化，3–4 bit 可用 | §3 算法；注意校准集与 group size 的影响 |

---

## 4. 解码加速

| 论文 | 一句话贡献 | 重点看 |
|---|---|---|
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | 小模型起草 + 大模型并行校验，输出分布严格不变 | §2 算法与接受率；收益公式与 α 的关系 |
| [Medusa](https://arxiv.org/abs/2401.10774) | 在模型上加多个解码头并行预测后续 token，用树形注意力校验 | §3 训练与树形验证 |
| [EAGLE](https://arxiv.org/abs/2401.15077) | 在特征层（而非 token 层）做投机，接受率更高 | §3 特征级起草；工程落地最常用的方案之一 |

---

## 5. 内核与架构

| 论文 | 一句话贡献 | 重点看 |
|---|---|---|
| [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware 分块计算，把 attention 从「内存受限」变「算力受限」，结果精确不近似 | §3 分块与重算策略；配合 GPU-Puzzles 建立直觉 |
| [DeepSeek-V2（MLA）](https://arxiv.org/abs/2405.04434) | Multi-head Latent Attention 把 KV 压成低秩潜向量，长上下文显存大幅下降 | §2.1 架构；对照 DeepSeek-V3 报告看工程化 |

---

## 6. 延伸（按需，不入主线）

- 稀疏/激活：PowerInfer、Mixtral 系 MoE 推理
- KV 压缩：KIVI、KV 量化类工作（配合 [LMCache](https://github.com/LMCache/LMCache) 实践）
- 长上下文：MInference 类稀疏注意力
- 卸载与异构：[ktransformers](https://github.com/kvcache-ai/ktransformers) 相关实践

---

## 7. 怎么读才不浪费时间

1. **先看摘要最后两句**：多数论文会把「提升了多少」写在结论句。
2. **看图 1 与系统架构图**：这是作者对问题的最浓缩表达。
3. **只精读你当前瓶颈对应的那一节**：调度问题读 §3，显存问题读 §4。
4. **读完立刻找一个实现**：论文里的「block 分页」在 nano-vllm 里长什么样，比读十遍公式有用。
5. **记一条反例**：每篇都问「什么情况下这个优化会失效」，这是面试和实战的分水岭。