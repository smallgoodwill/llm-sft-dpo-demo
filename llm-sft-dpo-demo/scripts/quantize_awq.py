#!/usr/bin/env python3
"""AWQ 4bit 量化（可选步骤，autoawq 新版 torch 下容易装不上，失败不影响 SFT/DPO 结论）。

    pip install autoawq
    python scripts/quantize_awq.py --model out/merged-sft --out out/qwen7b-awq
    MODEL=out/qwen7b-awq bash scripts/serve_vllm.sh

约需 20G 显存做校准，7B 模型量化后权重约 4.5G，单卡 48G 可跑且能留出 KV cache。
"""
from __future__ import annotations

import argparse

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",  # GEMM 适合 vLLM 的算子和 batch 推理
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/merged-sft")
    ap.add_argument("--out", default="out/qwen7b-awq")
    ap.add_argument("--samples", type=int, default=128, help="校准样本条数，默认取自带数据集")
    a = ap.parse_args()

    print(f"加载待量化模型：{a.model}")
    model = AutoAWQForCausalLM.from_pretrained(a.model, low_cpu_mem_usage=True, use_cache=False)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    print("开始量化（4bit / group_size=128 / GEMM）…")
    model.quantize(tok, quant_config=QUANT_CONFIG, export_compatible=True)
    model.save_quantized(a.out)
    tok.save_pretrained(a.out)
    print("量化完成 →", a.out)
    print("起服务：MODEL=%s bash scripts/serve_vllm.sh" % a.out)


if __name__ == "__main__":
    main()
