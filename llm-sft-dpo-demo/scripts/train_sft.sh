#!/usr/bin/env bash
# QLoRA SFT：L20 48G 约 15 分钟（2.2k 样本 × 3 epoch）
set -euo pipefail

export BASE_MODEL="${BASE_MODEL:-models/Qwen2.5-7B-Instruct}"
[ -d "$BASE_MODEL" ] || { echo "未找到基座：$BASE_MODEL，先跑 scripts/download_model.sh"; exit 1; }

mkdir -p out logs
python run.py data

# 断开 SSH 也不中断
nohup python run.py sft > logs/sft.log 2>&1 &
echo "SFT 已在后台启动，日志：tail -f logs/sft.log"
