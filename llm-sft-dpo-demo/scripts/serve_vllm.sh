#!/usr/bin/env bash
# 用 vLLM 起 OpenAI 兼容服务（可选：先合并 LoRA 或做 AWQ 4bit 量化）
#
#   bash scripts/serve_vllm.sh                 # 直接加载基座
#   MODEL=out/merged bash scripts/serve_vllm.sh  # 加载合并后的模型
#   MODEL=out/qwen7b-awq bash scripts/serve_vllm.sh
set -euo pipefail

MODEL="${MODEL:-models/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name cs-assistant \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --port "$PORT"

# 调用示例
#   curl http://127.0.0.1:8000/v1/chat/completions \
#     -H 'Content-Type: application/json' \
#     -d '{"model":"cs-assistant","messages":[{"role":"user","content":"我的订单三天没动了"}]}'
