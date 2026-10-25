#!/usr/bin/env bash
# HF 模型 → GGUF 量化 → 本地速度/体积对比（端侧学习 B 轨 W2–W3 的动手脚本）
#
# 用法:
#   ./scripts/quantize-gguf.sh --model Qwen/Qwen2.5-1.5B-Instruct
#   ./scripts/quantize-gguf.sh --model ./local-model --quants Q4_K_M,Q5_K_M,Q8_0
#   ./scripts/quantize-gguf.sh --model Qwen/Qwen2.5-1.5B-Instruct --dry-run   # 只打印将执行的命令
#
# 依赖: python3（含 torch/transformers/sentencepiece/huggingface_hub）、git、cmake
# 说明: 首次运行会 clone llama.cpp 并编译 llama-quantize / llama-bench，耗时较长。
set -euo pipefail

MODEL=""
OUT="./gguf-out"
QUANTS="Q4_K_M,Q8_0"
LLAMA_DIR="${LLAMA_DIR:-$HOME/Developer/llama.cpp}"
THREADS=""
PROMPT_TOKENS=512
GEN_TOKENS=128
REPEATS=3
DRY_RUN=0
SKIP_BENCH=0
INSTALL_DEPS=0

usage() {
  cat <<'EOF'
HF 模型 → GGUF 量化 → 本地速度/体积对比

参数:
  --model <id|dir>      HuggingFace 仓库 id（如 Qwen/Qwen2.5-1.5B-Instruct）或本地模型目录
  --out <dir>           输出目录（默认 ./gguf-out）
  --quants <a,b,c>      量化档位（默认 Q4_K_M,Q8_0）
  --llama-dir <dir>     llama.cpp 位置（默认 $LLAMA_DIR 或 ~/Developer/llama.cpp）
  --threads <n>         压测线程数（默认自动）
  --prompt-tokens <n>   llama-bench 的 prompt 长度（默认 512）
  --gen-tokens <n>      llama-bench 的生成长度（默认 128）
  --repeats <n>         每个档位重复次数（默认 3）
  --skip-bench          只量化，不跑 llama-bench
  --install-deps        缺 Python 依赖时自动 pip install（默认只提示）
  --dry-run             只打印命令，不执行
  -h, --help            显示本帮助

示例:
  ./scripts/quantize-gguf.sh --model Qwen/Qwen2.5-1.5B-Instruct --quants Q4_K_M,Q5_K_M,Q8_0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --quants) QUANTS="${2:-}"; shift 2 ;;
    --llama-dir) LLAMA_DIR="${2:-}"; shift 2 ;;
    --threads) THREADS="${2:-}"; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS="${2:-}"; shift 2 ;;
    --gen-tokens) GEN_TOKENS="${2:-}"; shift 2 ;;
    --repeats) REPEATS="${2:-}"; shift 2 ;;
    --skip-bench) SKIP_BENCH=1; shift ;;
    --install-deps) INSTALL_DEPS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1（用 --help 查看用法）" >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "错误: 必须提供 --model" >&2; usage; exit 2; }

CPU_JOBS="$( (sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4) )"
[[ -n "$THREADS" ]] || THREADS="$CPU_JOBS"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    printf '+ %s\n' "$*"
    "$@"
  fi
}

# ---------- 0. 依赖检查 ----------
say "检查依赖"
for bin in python3 git cmake; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    if [[ $DRY_RUN -eq 1 ]]; then echo "警告: 缺少 $bin（dry-run 继续）" >&2
    else echo "缺少 $bin，请先安装" >&2; exit 1; fi
  fi
done

if python3 -c "import torch, transformers, sentencepiece, huggingface_hub" >/dev/null 2>&1; then
  echo "Python 依赖就绪"
elif [[ $DRY_RUN -eq 1 ]]; then
  echo "警告: 未检测到 Python 依赖（dry-run 继续）"
elif [[ $INSTALL_DEPS -eq 1 ]]; then
  echo "安装 Python 依赖（torch / transformers / sentencepiece / huggingface_hub）..."
  run python3 -m pip install --upgrade torch transformers sentencepiece huggingface_hub
else
  cat >&2 <<'EOF'
缺少 Python 依赖。二选一：
  1) 加 --install-deps 让脚本安装
  2) 手动: python3 -m pip install torch transformers sentencepiece huggingface_hub
EOF
  exit 1
fi

if [[ $DRY_RUN -eq 0 ]]; then
  mkdir -p "$OUT"
  OUT="$(cd "$OUT" && pwd)"
fi

# ---------- 1. llama.cpp ----------
say "准备 llama.cpp（$LLAMA_DIR）"
if [[ -f "$LLAMA_DIR/convert_hf_to_gguf.py" ]]; then
  echo "已存在，跳过 clone"
else
  run git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi

# ---------- 2. 模型 ----------
say "解析模型"
if [[ -d "$MODEL" ]]; then
  MODEL_DIR="$MODEL"
  echo "本地目录: $MODEL_DIR"
else
  MODEL_DIR="$OUT/model"
  echo "从 HuggingFace 下载: $MODEL"
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '[dry-run] snapshot_download("%s" -> %s)\n' "$MODEL" "$MODEL_DIR"
  else
    python3 - "$MODEL" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, target = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id=repo,
    local_dir=target,
    allow_patterns=[
        "*.json", "*.safetensors", "*.gguf", "*.model",
        "*.txt", "*.py", "tokenizer*", "*.tiktoken", "*.md",
    ],
)
print(f"download ok: {path}")
PY
  fi
fi

# ---------- 3. 转 f16 ----------
BASE="$(basename "$MODEL")"
F16="$OUT/${BASE}-f16.gguf"
say "转换到 f16（中间产物，量化都从它出发）"
run python3 "$LLAMA_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" --outfile "$F16" --outtype f16

# ---------- 4. 编译工具 ----------
say "编译 llama-quantize / llama-bench"
QUANT_BIN="$LLAMA_DIR/build/bin/llama-quantize"
BENCH_BIN="$LLAMA_DIR/build/bin/llama-bench"
if [[ -x "$QUANT_BIN" && -x "$BENCH_BIN" ]]; then
  echo "已编译，跳过"
else
  run cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release
  run cmake --build "$LLAMA_DIR/build" -j "$CPU_JOBS" --target llama-quantize llama-bench
fi
# 兼容旧目录布局
if [[ ! -x "$QUANT_BIN" && $DRY_RUN -eq 0 ]]; then
  QUANT_BIN="$(find "$LLAMA_DIR/build" -type f -name 'llama-quantize' -perm -u+x 2>/dev/null | head -1 || true)"
fi
if [[ ! -x "$BENCH_BIN" && $DRY_RUN -eq 0 ]]; then
  BENCH_BIN="$(find "$LLAMA_DIR/build" -type f -name 'llama-bench' -perm -u+x 2>/dev/null | head -1 || true)"
fi
[[ $DRY_RUN -eq 1 || -x "$QUANT_BIN" ]] || { echo "找不到 llama-quantize（编译失败？）" >&2; exit 1; }

# ---------- 5. 逐档量化 ----------
IFS=',' read -r -a QUANT_ARR <<<"$QUANTS"
for q in "${QUANT_ARR[@]}"; do
  q="$(echo "$q" | tr -d '[:space:]')"
  [[ -n "$q" ]] || continue
  dst="$OUT/${BASE}-${q}.gguf"
  say "量化: $q → $(basename "$dst")"
  run "$QUANT_BIN" "$F16" "$dst" "$q"
done

# ---------- 6. 压测 ----------
if [[ $SKIP_BENCH -eq 0 ]]; then
  [[ $DRY_RUN -eq 1 || -x "$BENCH_BIN" ]] || { echo "找不到 llama-bench" >&2; exit 1; }
  for q in "${QUANT_ARR[@]}"; do
    q="$(echo "$q" | tr -d '[:space:]')"
    src="$OUT/${BASE}-${q}.gguf"
    [[ -f "$src" || $DRY_RUN -eq 1 ]] || { echo "跳过（文件不存在）: $src"; continue; }
    say "压测: $q（pp$PROMPT_TOKENS / tg$GEN_TOKENS / -r $REPEATS / -t $THREADS）"
    run "$BENCH_BIN" -m "$src" -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -t "$THREADS" -r "$REPEATS"
  done
fi

# ---------- 7. 汇总 ----------
if [[ $DRY_RUN -eq 0 ]]; then
  say "体积对照"
  printf '%-14s %10s\n' "QUANT" "SIZE"
  for f in "$F16" "$OUT/${BASE}"-*.gguf; do
    [[ -f "$f" ]] || continue
    printf '%-14s %10s MB\n' "$(basename "$f" .gguf)" \
      "$(awk -v b="$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")" 'BEGIN{printf "%.0f", b/1048576}')"
  done
  printf '\n下一步: 用 llama-cli 做主观质量对比，把「体积 / 速度 / 质量」三维填进 docs/03 的量化表。\n'
fi