#!/usr/bin/env bash
# 服务器一次性环境准备：venv + 依赖 + 基座下载
#
#   bash scripts/setup_env.sh              # 建环境 + 装依赖
#   bash scripts/setup_env.sh --download   # 再顺手下载基座
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
# torch 单独装，按显卡 CUDA 版本调整 cu124 / cu121
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

if [ "${1:-}" = "--download" ]; then
  bash scripts/download_model.sh
fi

echo
echo "环境就绪。下次进入：source .venv/bin/activate"
echo "跑数据：python run.py data"
