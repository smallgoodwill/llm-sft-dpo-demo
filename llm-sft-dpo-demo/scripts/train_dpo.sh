#!/usr/bin/env bash
# DPO：依赖 out/sft，L20 48G 约 20 分钟；TRL 不兼容会打印原因并跳过
set -euo pipefail

export BASE_MODEL="${BASE_MODEL:-models/Qwen2.5-7B-Instruct}"
[ -d out/sft ] || { echo "未找到 out/sft，先跑 scripts/train_sft.sh"; exit 1; }

mkdir -p logs
nohup python run.py dpo > logs/dpo.log 2>&1 &
echo "DPO 已在后台启动，日志：tail -f logs/dpo.log"
