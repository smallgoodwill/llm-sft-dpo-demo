#!/usr/bin/env python3
"""把 LoRA adapter 合并回基座，输出可直接被 vLLM 加载的完整模型。

    python scripts/merge_lora.py --adapter out/sft --out out/merged-sft
    python scripts/merge_lora.py --adapter out/dpo --out out/merged-dpo

注意：合并需要完整 bf16 权重（7B 约 15G 显存/内存），48G 卡没问题；
      若显存不足，可加 --cpu 在内存里合并（慢但稳）。
"""
from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.getenv("BASE_MODEL", "models/Qwen2.5-7B-Instruct"))
    ap.add_argument("--adapter", default="out/sft")
    ap.add_argument("--out", default="out/merged")
    ap.add_argument("--cpu", action="store_true", help="在 CPU 内存中合并")
    a = ap.parse_args()

    print(f"基座：{a.base}\nadapter：{a.adapter}\n输出：{a.out}")
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)

    if a.cpu:
        model = AutoModelForCausalLM.from_pretrained(
            a.base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": "cpu"}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            a.base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
        )

    model = PeftModel.from_pretrained(model, a.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(a.out, safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(a.out)
    print("合并完成 →", a.out)
    print("起服务：MODEL=%s bash scripts/serve_vllm.sh" % a.out)


if __name__ == "__main__":
    main()
