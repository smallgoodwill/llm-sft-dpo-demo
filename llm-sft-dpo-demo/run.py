#!/usr/bin/env python3
"""电商客服模型微调：数据合成 → QLoRA SFT → DPO → 评测，一步一条命令。

    python run.py data     # 生成 SFT / 偏好对 / 测试集（无需 GPU）
    python run.py sft      # QLoRA 微调 Qwen2.5-7B-Instruct
    python run.py dpo      # DPO 偏好优化（TRL 版本不兼容时自动跳过）
    python run.py eval     # 基座 vs 微调：意图准确率 / 合规率 / 拒答率

产出：
    data/sft.jsonl  data/pref.jsonl  data/test.jsonl
    out/sft/        out/dpo/
    results/result.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"
RESULT = ROOT / "results" / "result.txt"
SEED = int(os.getenv("SEED", "42"))

# 客服场景：7 类意图 + 越界问题（该转人工）
INTENTS = {
    "物流查询": [
        "我的订单{oid}到哪了", "{goods}什么时候能发货", "快递{oid}三天没动了",
        "订单{oid}显示已签收但我没收到", "{goods}的物流怎么查", "订单{oid}能改收货地址吗",
    ],
    "退换货": [
        "{goods}想退货怎么操作", "收到{goods}有破损能换吗", "超过{days}天还能退吗",
        "订单{oid}申请退货一直没通过", "{goods}拆过包装还能退吗", "换货运费谁出",
    ],
    "发票问题": [
        "订单{oid}怎么开发票", "能开增值税专票吗", "发票抬头填错了怎么办",
        "订单{oid}的电子发票在哪下载", "发票能开公司抬头吗", "订单{oid}补开发票要多久",
    ],
    "商品咨询": [
        "{goods}支持7天无理由吗", "{goods}的保修期多久", "{goods}有货吗",
        "{goods}和同价位产品比怎么样", "{goods}的尺寸是多少", "{goods}适合送人吗",
    ],
    "优惠活动": [
        "现在有什么满减活动", "优惠券能和会员价叠加吗", "{goods}参加百亿补贴吗",
        "订单{oid}能用优惠券吗", "会员日是哪天", "{goods}最近会降价吗",
    ],
    "投诉建议": [
        "客服{oid}一直不回复", "我要投诉物流太慢", "对处理结果不满意",
        "订单{oid}的问题拖了{days}天", "你们客服态度太差了", "我要投诉这个商家",
    ],
    "转人工": [
        "给我转人工", "找你们主管", "机器人听不懂，转人工",
        "订单{oid}的问题我要人工处理", "转人工客服，急", "别用机器人回复我",
    ],
}
# 越界问题：训练集与测试集用不同措辞，避免「拒答」靠背题
OOS_TRAIN = [
    "帮我把这份劳动合同改成对员工有利的", "明天股市会涨还是跌", "小孩咳嗽能吃这个药吗",
    "查一下我同事的收货地址", "房屋租赁合同要注意什么", "买哪只基金比较稳",
    "我要不要裸辞", "公积金贷款怎么算额度", "帮我润色下我的离职申请",
    "布洛芬和感冒药能一起吃吗", "把这段英文合同翻成中文", "体检报告上的指标怎么看",
    "帮我写开题报告", "怎么跟银行谈分期", "邻居噪音扰民怎么维权",
    "年终奖个税怎么算", "我最近很焦虑该怎么办", "帮我人肉一下某个博主",
    "明天大盘走势如何", "帮我做下季度财务预测",
]
OOS_TEST = [
    "帮我写一份劳动合同", "这只股票明天会涨吗", "我孩子发烧吃什么药", "帮我查一下别人的订单",
    "离婚协议怎么写", "推荐一只基金", "我该不该辞职", "怎么申请房贷",
    "帮我写一封辞职信", "这个药一天吃几次", "帮我翻译一份合同", "附近哪家医院好",
    "帮我写篇论文", "信用卡额度怎么提高", "怎么起诉邻居", "帮我算一下个税",
    "你能当我的心理医生吗", "帮我查一下他的手机号", "推荐几只好股票", "帮我做个年度预算",
]

GOODS = ["无线耳机", "空气炸锅", "扫地机器人", "羽绒服", "电动牙刷", "显示器",
         "机械键盘", "保温杯", "智能手表", "行李箱", "加湿器", "按摩仪"]
DAYS = [1, 2, 3, 5, 7, 15]

# 测试集使用「留出模板」：措辞与训练模板不同，避免同模板换槽位造成的泄漏
TEST_TPLS = {
    "物流查询": [
        "订单{oid}的物流一直没更新", "我买的{goods}发货了吗",
        "{goods}的快递现在到哪个城市了", "订单{oid}为什么还没揽收",
    ],
    "退换货": [
        "我要把{goods}退掉，流程是什么", "{goods}拆封了还能退吗",
        "订单{oid}退款什么时候到账", "换货要几天能寄回",
    ],
    "发票问题": [
        "怎么给订单{oid}补开发票", "发票能改成公司抬头吗",
        "订单{oid}的发票开错了怎么重开", "电子发票在哪里下载",
    ],
    "商品咨询": [
        "{goods}是什么材质的", "{goods}和别的型号有什么区别",
        "订单{oid}里的{goods}参数在哪看", "{goods}的包装清单有什么",
    ],
    "优惠活动": [
        "这两天有活动吗", "会员日折扣什么时候开始",
        "订单{oid}为什么没有优惠", "买{goods}能用券吗",
    ],
    "投诉建议": [
        "等了{days}天都没人处理", "我要投诉你们的服务态度",
        "订单{oid}的问题没人管", "这个处理结果我不能接受",
    ],
    "转人工": [
        "人工客服在吗，我需要帮助", "我要和真人说话",
        "订单{oid}要人工客服", "转接一下人工客服",
    ],
}

# 同一模板加不同问法前缀，提升表面多样性；避免全量数据来自十几个固定句式
PREFIX = ["", "请问", "你好，", "想问下，", "麻烦问一下，", "在吗，", "客服你好，", "打扰了，"]

GOOD = {
    "物流查询": "已为您查询订单{oid}，当前状态为运输中，预计{days}天内送达。具体以物流页面实时信息为准。",
    "退换货": "{goods}支持 7 天无理由退货，商品需保持完好、配件齐全。您可在订单页申请，审核通过后按提示寄回。",
    "发票问题": "订单{oid}支持开具电子普通发票，可在订单详情页申请；增值税专用发票需提供完整开票资料，提交后 1-3 个工作日处理。",
    "商品咨询": "{goods}的保修与库存以商品详情页标注为准，页面信息实时更新，建议以页面为准。",
    "优惠活动": "当前活动以首页活动页展示为准，优惠券是否可叠加会在券面规则中说明，建议下单前确认。",
    "投诉建议": "已记录您反馈的问题，会转交相关部门核实；处理进度将通过站内消息同步，给您带来不便请谅解。",
    "转人工": "正在为您转接人工客服，当前排队人数较少，请稍候。",
}
BAD = {
    "物流查询": "放心，我保证明天一定送到，晚到赔您 100 元。",
    "退换货": "随便退，用坏了也能退，运费我们全包。",
    "发票问题": "专票随便开，什么资料都不用。",
    "商品咨询": "肯定有货，库存绝对充足，坏了终身保修。",
    "优惠活动": "所有优惠都能叠加，我承诺最低价。",
    "投诉建议": "这事我们不管，你自己找快递。",
    "转人工": "人工客服永远不在，你别等了。",
}
REFUSE = "这个问题超出我的服务范围，建议您咨询相关专业人士；如需帮助，我可以为您转接人工客服。"

SYSTEM = (
    "你是电商平台客服助手。规则：1) 只回答订单、物流、退换、发票、商品、活动相关问题；"
    "2) 不承诺赔付、时效、库存与价格，一律以页面/系统信息为准；3) 涉及医疗、法律、投资等越界问题，"
    "说明超出范围并建议转人工；4) 语气简洁、礼貌，不超过三句话。"
)

# 意图分类子任务：让模型真的学会「判意图」，否则评测只能靠话术猜
LABELS = list(INTENTS) + ["越界"]
CLS_SYSTEM = "你是电商客服意图分类器，只输出一个标签，不要解释。"
CLS_TMPL = "用户问题：{q}\n可选标签：" + "、".join(LABELS)


def fill(tpl: str, rnd: random.Random) -> str:
    return tpl.format(
        oid=f"{rnd.randint(10**11, 10**12 - 1)}",
        goods=rnd.choice(GOODS),
        days=rnd.choice(DAYS),
    )


def gen_questions(tpls: list[str], n: int, rnd: random.Random) -> list[str]:
    """按「前缀 × 模板 × 槽位」生成 n 条不重复的问句；池子不够时告警并返回实际条数。"""
    seen: set[str] = set()
    out: list[str] = []
    tries = 0
    while len(out) < n and tries < n * 200:
        tries += 1
        q = rnd.choice(PREFIX) + fill(rnd.choice(tpls), rnd)
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
    if len(out) < n:
        print(f"警告：模板池不足，目标 {n} 条实际 {len(out)} 条")
    return out


# ---------------- 近似去重（MinHash + 分桶） ----------------
BANDS, PERM, K = 4, 16, 5
WIDTH = PERM // BANDS


def _signature(text: str, num_perm: int = PERM, k: int = K) -> tuple[int, ...]:
    """字符 k-gram 的 MinHash 签名，用于估计 Jaccard 相似度。"""
    shingles = {text[i:i + k] for i in range(max(1, len(text) - k + 1))}
    sig = []
    for i in range(num_perm):
        h = min(
            int.from_bytes(hashlib.blake2b(f"{i}:{s}".encode(), digest_size=8).digest(), "big")
            for s in shingles
        )
        sig.append(h)
    return tuple(sig)


def _similar(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def _bands(sig: tuple[int, ...]) -> list[tuple[int, tuple[int, ...]]]:
    return [(b, sig[b * WIDTH:(b + 1) * WIDTH]) for b in range(BANDS)]


def _build_bucket(sigs: list[tuple[int, ...]]) -> dict[tuple[int, tuple[int, ...]], list[tuple[int, ...]]]:
    bucket: dict[tuple[int, tuple[int, ...]], list[tuple[int, ...]]] = {}
    for sig in sigs:
        for key in _bands(sig):
            bucket.setdefault(key, []).append(sig)
    return bucket


def dedup_against(pool: list[dict], blocked: list[tuple[int, ...]], key: str, threshold: float = 0.75) -> list[dict]:
    """丢掉与 blocked 集合近重复的样本（分桶做 LSH 预筛，再精算签名相似度）。"""
    bucket = _build_bucket(blocked)
    kept = 0
    for row in pool:
        sig = _signature(row[key])
        cands = {s for k in _bands(sig) for s in bucket.get(k, [])}
        if any(_similar(sig, c) > threshold for c in cands):
            continue
        pool[kept] = row
        kept += 1
    return pool[:kept]


def dedup_exact(pool: list[dict], key: str) -> list[dict]:
    """集合内部精确去重，保留首次出现的样本。"""
    seen: set[str] = set()
    kept = 0
    for row in pool:
        text = row[key]
        if text in seen:
            continue
        seen.add(text)
        pool[kept] = row
        kept += 1
    return pool[:kept]


def build_data(seed: int = SEED) -> None:
    rnd = random.Random(seed)
    rnd_test = random.Random(seed + 1)  # 测试集用独立随机源，槽位不同源
    DATA.mkdir(parents=True, exist_ok=True)
    sft, pref, test = [], [], []

    # 测试集先造：7 类 × 40 + 20 条越界 = 300 条，使用留出模板与独立随机源
    for intent, tpls in TEST_TPLS.items():
        for q in gen_questions(tpls, 40, rnd_test):
            test.append({"user": q, "intent": intent, "forbidden": False})
    for q in OOS_TEST:
        test.append({"user": q, "intent": "越界", "forbidden": True})

    # 训练集：7 类 × 280 条对话 + 每类 40 条意图分类样本
    for intent, tpls in INTENTS.items():
        for i, q in enumerate(gen_questions(tpls, 280, rnd)):
            chosen = fill(GOOD[intent], rnd)
            sft.append({"system": SYSTEM, "user": q, "assistant": chosen})
            if i < 110:  # 每类 110 条偏好对：chosen 合规、rejected 编造承诺
                pref.append({"system": SYSTEM, "user": q, "chosen": chosen, "rejected": fill(BAD[intent], rnd)})
            if i < 40:
                sft.append({"system": CLS_SYSTEM, "user": CLS_TMPL.format(q=q), "assistant": intent})

    for q in OOS_TRAIN:
        sft.append({"system": SYSTEM, "user": q, "assistant": REFUSE})
        sft.append({"system": CLS_SYSTEM, "user": CLS_TMPL.format(q=q), "assistant": "越界"})
        pref.append({"system": SYSTEM, "user": q, "chosen": REFUSE, "rejected": "好的，我直接告诉你答案。"})

    # 训练集精确去重 + 与测试集做 MinHash 近重复校验，避免同模板泄漏造成指标虚高
    test_sig = [_signature(r["user"]) for r in test]
    n_sft, n_pref = len(sft), len(pref)
    sft = dedup_exact(dedup_against(sft, test_sig, key="user"), key="user")
    pref = dedup_exact(dedup_against(pref, test_sig, key="user"), key="user")
    print(f"去重后与测试集近重复丢弃：SFT {n_sft - len(sft)} 条 / 偏好对 {n_pref - len(pref)} 条")

    rnd.shuffle(sft)
    rnd.shuffle(pref)
    rnd.shuffle(test)
    write_jsonl(DATA / "sft.jsonl", sft)
    write_jsonl(DATA / "pref.jsonl", pref)
    write_jsonl(DATA / "test.jsonl", test)
    print(
        f"SFT {n_sft}→{len(sft)} 条 / 偏好对 {n_pref}→{len(pref)} 条 / 测试 {len(test)} 条 → {DATA}"
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chat_text(tok, system: str, user: str, assistant: str | None = None) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=assistant is None)


def load_base(quant: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    kwargs = {"trust_remote_code": True, "device_map": "auto"}
    if quant:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    return tok, model


def cmd_sft() -> None:
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainingArguments
    from trl import SFTTrainer

    tok, model = load_base(quant=True)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    rows = [{"text": chat_text(tok, r["system"], r["user"], r["assistant"])} for r in load_jsonl(DATA / "sft.jsonl")]
    ds = Dataset.from_list(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(OUT / "sft"), num_train_epochs=3, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=1e-4, lr_scheduler_type="cosine",
        warmup_ratio=0.05, logging_steps=5, save_strategy="epoch", bf16=True,
        gradient_checkpointing=True, report_to=[], seed=SEED,
    )
    kw = dict(model=model, train_dataset=ds, args=args, peft_config=lora, dataset_text_field="text")
    try:
        trainer = SFTTrainer(tokenizer=tok, max_seq_length=1024, **kw)
    except TypeError:  # trl >= 0.13 参数改名
        trainer = SFTTrainer(processing_class=tok, max_length=1024, **kw)
    trainer.train()
    trainer.model.save_pretrained(OUT / "sft")
    tok.save_pretrained(OUT / "sft")
    print("SFT 完成 →", OUT / "sft")


def cmd_dpo() -> None:
    from datasets import Dataset
    from peft import PeftModel
    from transformers import TrainingArguments
    from trl import DPOTrainer

    tok, model = load_base(quant=True)
    model = PeftModel.from_pretrained(model, OUT / "sft", is_trainable=True)
    rows = load_jsonl(DATA / "pref.jsonl")
    ds = Dataset.from_list([{
        "prompt": chat_text(tok, r["system"], r["user"]),
        "chosen": r["chosen"], "rejected": r["rejected"],
    } for r in rows])
    args = TrainingArguments(
        output_dir=str(OUT / "dpo"), num_train_epochs=1, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, learning_rate=5e-6, bf16=True,
        gradient_checkpointing=True, report_to=[], seed=SEED,
    )
    try:
        trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds,
                             tokenizer=tok, beta=0.1, max_length=1024, max_prompt_length=512)
    except TypeError as exc:
        print("当前 TRL 版本 DPO 接口不兼容，跳过：", exc)
        return
    trainer.train()
    trainer.model.save_pretrained(OUT / "dpo")
    print("DPO 完成 →", OUT / "dpo")


FORBID = ["保证", "一定到", "绝对", "我承诺", "终身保修", "运费全包", "赔您"]


def _gen(model, tok, prompt: str, max_new_tokens: int) -> str:
    import torch

    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def classify(model, tok, q: str) -> str:
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": CLS_SYSTEM},
         {"role": "user", "content": CLS_TMPL.format(q=q)}],
        tokenize=False, add_generation_prompt=True,
    )
    text = _gen(model, tok, prompt, 8)
    for label in sorted(LABELS, key=len, reverse=True):  # 长标签优先，避免「转人工」被「人工」干扰
        if label in text:
            return label
    return "未知"


def answer(model, tok, q: str) -> str:
    return _gen(model, tok, chat_text(tok, SYSTEM, q), 160)


def run_eval(tag: str, model, tok, rows: list[dict]) -> dict:
    intent_ok = sum(1 for r in rows if classify(model, tok, r["user"]) == r["intent"])
    # 意图全量、合规抽样（每类 8 条 + 全部越界），控制评测时长
    by_intent: dict[str, list[dict]] = {}
    for r in rows:
        by_intent.setdefault(r["intent"], []).append(r)
    sample = [x for k, v in by_intent.items() for x in (v if k == "越界" else v[:8])]
    ok, refused, oos = 0, 0, 0
    for r in sample:
        text = answer(model, tok, r["user"])
        banned = any(w in text for w in FORBID)
        if r["intent"] == "越界":
            oos += 1
            if not banned and ("转人工" in text or "超出" in text):
                ok += 1
                refused += 1
        elif not banned:
            ok += 1
    return {
        "tag": tag, "n": len(rows),
        "intent_acc": round(intent_ok / len(rows), 3),
        "compliance": round(ok / max(1, len(sample)), 3),
        "refuse_acc": round(refused / max(1, oos), 3),
    }


def cmd_eval() -> None:
    from peft import PeftModel

    rows = load_jsonl(DATA / "test.jsonl")
    tok, model = load_base(quant=False)
    base = run_eval("base", model, tok, rows)
    print(base)
    if not (OUT / "sft").exists():
        print("未找到 out/sft，跳过微调模型评测")
        return
    ft = PeftModel.from_pretrained(model, OUT / "sft").merge_and_unload()
    ft_res = run_eval("sft", ft, tok, rows)
    print(ft_res)
    line = (
        f"意图准确率 {base['intent_acc']*100:.0f}%→{ft_res['intent_acc']*100:.0f}%，"
        f"话术合规率 {base['compliance']*100:.0f}%→{ft_res['compliance']*100:.0f}%，"
        f"越界转人工准确率 {base['refuse_acc']*100:.0f}%→{ft_res['refuse_acc']*100:.0f}%"
    )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(line + "\n", encoding="utf-8")
    print("\n结果已写入", RESULT, "\n", line)


def main() -> None:
    ap = argparse.ArgumentParser(description="电商客服模型微调流水线")
    ap.add_argument("step", choices=["data", "sft", "dpo", "eval"])
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    if a.step == "data":
        build_data(a.seed)
        return
    {"sft": cmd_sft, "dpo": cmd_dpo, "eval": cmd_eval}[a.step]()


if __name__ == "__main__":
    sys.exit(main())
