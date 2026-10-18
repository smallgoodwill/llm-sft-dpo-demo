#!/usr/bin/env bash
# 从 ModelScope 拉取基座（国内可达；如需 HF 请自行改 --model 与源）
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DIR="${DIR:-models/$(basename "$MODEL")}"

mkdir -p "$(dirname "$DIR")"
modelscope download --model "$MODEL" --local_dir "$DIR"

echo "已下载到 $DIR"
echo "export BASE_MODEL=$(cd "$(dirname "$DIR")" && pwd)/$(basename "$DIR")"
