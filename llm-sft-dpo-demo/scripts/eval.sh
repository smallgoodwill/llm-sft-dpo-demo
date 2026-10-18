#!/usr/bin/env bash
# 评测：基座 vs 微调，300 条测试集（意图全量 + 合规抽样）
# 结果同时写入 results/result.txt，便于跨 checkpoint 对比
set -euo pipefail

export BASE_MODEL="${BASE_MODEL:-models/Qwen2.5-7B-Instruct}"
mkdir -p logs results
python run.py eval 2>&1 | tee logs/eval.log
