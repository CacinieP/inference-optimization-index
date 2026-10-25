#!/usr/bin/env bash
# 对 OpenAI 兼容的 chat/completions 端点做轻量压测：测量 TTFT / ITL / 吞吐。
# 依赖：bash + curl 不需要；只用 python3 标准库（urllib）。
#
# 用法:
#   ./scripts/bench-serving.sh --url http://127.0.0.1:8000/v1/chat/completions \
#       --model Qwen2.5-7B-Instruct --requests 20 --concurrency 4 --max-tokens 128
#
# 说明:
#   - 这是自测/回归用的轻量脚本，正式压测请用 vLLM benchmarks / GuideLLM / llmperf。
#   - 结果会打印每请求明细与 p50/p95/p99 汇总；如需 CSV 用 --csv results/out.csv。
set -euo pipefail

exec python3 - "$@" <<'PY'
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

p = argparse.ArgumentParser(description="轻量 LLM 服务压测（流式）")
p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
p.add_argument("--model", required=False, default="default")
p.add_argument("--requests", type=int, default=20, help="总请求数")
p.add_argument("--concurrency", type=int, default=1, help="并发数")
p.add_argument("--max-tokens", type=int, default=128)
p.add_argument("--prompt", default="用三句话解释什么是 KV Cache。")
p.add_argument("--timeout", type=float, default=120.0)
p.add_argument("--api-key", default="", help="可选，Bearer token")
p.add_argument("--csv", default="", help="把每请求结果写入 CSV")
args = p.parse_args(sys.argv[1:])

lock = threading.Lock()
rows = []


def pct(values, q):
    if not values:
        return float("nan")
    data = sorted(values)
    idx = min(len(data) - 1, max(0, int(round((q / 100) * (len(data) - 1)))))
    return data[idx]


def one_request(i):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "stream": True,
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    req = urllib.request.Request(
        args.url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    start = time.perf_counter()
    ttft = None
    stamps = []
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - start
                    stamps.append(now)
                    tokens += 1
        end = time.perf_counter()
    except Exception as exc:
        with lock:
            rows.append({"i": i, "ok": False, "err": f"{type(exc).__name__}: {exc}"})
            print(f"  req#{i:<3} FAILED {type(exc).__name__}: {str(exc)[:80]}")
        return

    itl = [b - a for a, b in zip(stamps, stamps[1:])]
    row = {
        "i": i,
        "ok": True,
        "ttft_ms": (ttft or 0) * 1000,
        "itl_ms": statistics.mean(itl) * 1000 if itl else 0.0,
        "tokens": tokens,
        "total_s": end - start,
        "tok_s": tokens / (end - start) if end > start else 0.0,
    }
    with lock:
        rows.append(row)
        print(
            f"  req#{i:<3} TTFT={row['ttft_ms']:7.1f}ms  "
            f"ITL={row['itl_ms']:6.1f}ms  tok={tokens:<4}  "
            f"total={row['total_s']:5.2f}s  {row['tok_s']:6.1f} tok/s"
        )


print(f"目标: {args.url}")
print(f"模型: {args.model} | 请求 {args.requests} | 并发 {args.concurrency} | max_tokens {args.max_tokens}")
print("说明: 前几个请求可能受 warmup 影响，建议丢弃前 10%~20%\n")

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
    list(pool.map(one_request, range(args.requests)))
wall = time.perf_counter() - t0

ok = [r for r in rows if r.get("ok")]
if not ok:
    print("\n全部请求失败，检查端点与模型名。")
    sys.exit(1)

ttfts = [r["ttft_ms"] for r in ok]
itls = [r["itl_ms"] for r in ok]
toks = sum(r["tokens"] for r in ok)

print("\n--- 汇总 ---")
print(f"成功 {len(ok)}/{len(rows)}")
print(f"TTFT  p50={pct(ttfts,50):7.1f}ms  p95={pct(ttfts,95):7.1f}ms  p99={pct(ttfts,99):7.1f}ms")
print(f"ITL   p50={pct(itls,50):7.1f}ms  p95={pct(itls,95):7.1f}ms  p99={pct(itls,99):7.1f}ms")
print(f"吞吐  {toks / wall:8.1f} tok/s（墙钟 {wall:.2f}s，输出 {toks} tokens）")
print(f"请求  {len(ok) / wall:8.2f} req/s")

if args.csv:
    import csv

    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["i", "ttft_ms", "itl_ms", "tokens", "total_s", "tok_s"]
        )
        writer.writeheader()
        for row in sorted(ok, key=lambda r: r["i"]):
            writer.writerow({k: row[k] for k in writer.fieldnames})
    print(f"\n已写入 {args.csv}")
PY