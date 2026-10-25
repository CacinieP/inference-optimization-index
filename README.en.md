# Inference Optimization Index

A curated, link-verified index for learning **LLM inference optimization** — on both the serving side and the device/edge side. Repositories, books, courses and papers, each annotated with *why it is worth your time*.

- **Serving**: vLLM / SGLang / TensorRT-LLM / LMDeploy, KV cache, scheduling, quantization, speculative decoding
- **Edge & general**: TensorRT, ONNX Runtime, TVM/MLC, GGUF + llama.cpp, ExecuTorch, LiteRT, Core ML, MNN, OpenVINO, WebLLM
- **Books & courses**: MIT `efficientml.ai`, `mlsysbook.ai`, CMU 11-868, MLC courses, and Chinese titles on vLLM/SGLang
- **Papers**: 14 verified arXiv entries in a recommended reading order
- **Plans**: two 8-week tracks (serving / edge) with weekly deliverables and acceptance criteria
- **Benchmarking**: metric definitions, methodology, pitfalls, and a dependency-free load-test script

![Links checked](https://img.shields.io/badge/LINKS-CHECKED%202026--10--25-69c8bd?style=flat-square)
![License](https://img.shields.io/badge/LICENSE-MIT-14262b?style=flat-square)

> The main index is written in Chinese: **[README.md](./README.md)**.
> Quick maps: [serving](./docs/02-serving-llm.md) · [edge & general](./docs/03-edge-and-general.md) · [benchmarking](./docs/04-benchmark.md) · [papers](./docs/05-papers.md) · [roadmap](./docs/01-roadmap.md)

## How to use it

1. Pick your track from the "30-second routing" table in the Chinese README.
2. Follow the weekly plan — every week must end with an artifact (a table, a script, a number), not just reading.
3. Before reading engine source code, deploy one engine and measure it. Measured intuition beats memorized theory.

## Contributing

PRs welcome. Entry format: `name | scale/platform | one-sentence reason to read it`. Every link must pass `python3 scripts/check-links.py`. See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT. Links point to third-party work; their content is owned by their authors.