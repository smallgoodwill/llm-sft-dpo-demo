# 电商客服大模型微调：SFT → DPO → 量化部署

以 Qwen2.5-7B-Instruct 为基座，做「意图识别 + 标准话术 + 该拒答就拒答」的垂直场景对齐，并给出可复现的数据构造、训练、评测与部署链路。

> 单卡 48G（L20）可跑；QLoRA 4bit 训练显存约为全参 SFT 的 1/5。

## 为什么做这个

通用模型在客服场景有三类典型问题：

| 问题 | 表现 | 本项目做法 |
|---|---|---|
| 越界回答 | 用户问医疗/法律/投资也照答 | SFT 中加入拒答样本，偏好对中把「编造承诺」标为 rejected |
| 乱承诺 | 「保证明天到」「终身保修」 | 系统提示约束 + DPO 惩罚违规话术 + 关键词规则做合规判定 |
| 意图漂移 | 问退货答成物流 | 增加意图分类子任务，评测时单独统计意图准确率 |

## 流水线

```
data（合成+去重+分层留出） → sft（QLoRA r=16 α=32） → dpo（β=0.1） → eval（基座 vs 微调）
                                                          ↓
                                              AWQ 4bit 量化 → vLLM 部署
```

## 快速开始

```bash
# 0. 环境（venv + torch + 依赖 + 基座）
bash scripts/setup_env.sh --download
source .venv/bin/activate
export BASE_MODEL=$PWD/models/Qwen2.5-7B-Instruct

# 1. 造数据（无需 GPU，秒级）
python run.py data

# 2. 训练与评测
bash scripts/train_sft.sh     # 约 15 min
bash scripts/train_dpo.sh     # 约 20 min
bash scripts/eval.sh          # 基座 vs 微调对比

# 3. 合并 LoRA → 可选 AWQ 4bit 量化
python scripts/merge_lora.py --adapter out/sft --out out/merged-sft
python scripts/quantize_awq.py --model out/merged-sft --out out/qwen7b-awq   # 可选

# 4. 部署（OpenAI 兼容）
MODEL=out/qwen7b-awq bash scripts/serve_vllm.sh
```

## 数据构造

- **SFT 2280 条**：7 类客服意图 × 280 条（6 条模板 × 8 种问法前缀 × 槽位随机），每条含标准话术；另混入 280 条意图分类样本（输入问题，输出单个标签）+ 20 条越界拒答样本。
- **偏好对 790 条**：chosen 为合规话术，rejected 为「保证送达 / 随便退 / 终身保修」等违规承诺。
- **测试集 300 条**：7 类 × 40（**留出模板**，措辞与训练集不同）+ 20 条越界；训练集用 MinHash 与测试集做近重复校验，当前丢弃 0 条。

细节见 [`docs/dataset.md`](docs/dataset.md)。数据为模板 + 槽位随机合成，未使用真实工单，评测结论也只在该分布内成立。

## 评测口径

| 指标 | 判定方式 |
|---|---|
| 意图准确率 | 模型输出标签与金标一致（全量 300 条） |
| 话术合规率 | 命中「保证/一定到/绝对/我承诺」等违规词即判不合规（每类抽样 8 条） |
| 越界转人工准确率 | 越界问题中既不含违禁词、又给出「超出范围/转人工」的比例 |

结果见 [`results/README.md`](results/README.md)（跑完回填）。评测脚本同时输出一行汇总，方便对比不同 checkpoint。

## 目录结构

```
.
├── run.py                 # 四个阶段一条命令：data / sft / dpo / eval
├── configs/               # 训练超参（与 run.py 默认值一致）
├── scripts/
│   ├── setup_env.sh       # 一次性：venv + torch + 依赖 (+ 下载基座)
│   ├── download_model.sh  # ModelScope 拉基座
│   ├── train_sft.sh       # nohup 后台 SFT
│   ├── train_dpo.sh       # nohup 后台 DPO
│   ├── eval.sh            # 基座 vs 微调
│   ├── merge_lora.py      # LoRA 合并成完整权重
│   ├── quantize_awq.py    # AWQ 4bit 量化（可选）
│   └── serve_vllm.sh      # OpenAI 兼容服务
├── data/                  # samples 入库，全量数据本地生成不入库
├── docs/                  # 数据构造与评测口径说明
└── results/               # 评测结果占位
```

## 后续计划

- [ ] 把 `run.py` 按 `data / train / eval / serve` 拆成 `src/` 包，补单元测试
- [ ] 接入真实脱敏工单，替换模板合成数据，做一次领域迁移对比
- [ ] 补 AWQ 量化前后显存与吞吐实测数据
- [ ] 增加 LLM-as-judge 评分，与关键词规则交叉验证

## 参考

- Qwen2.5-7B-Instruct、TRL（SFTTrainer / DPOTrainer）、PEFT（LoRA / QLoRA）、vLLM、AutoAWQ

## License

MIT
