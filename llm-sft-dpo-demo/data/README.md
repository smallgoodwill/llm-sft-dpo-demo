# data 目录

| 路径 | 是否入库 | 说明 |
|---|---|---|
| `samples/sft_sample.jsonl` | 是 | 从真实生成结果里截的 3 条，方便看格式 |
| `samples/pref_sample.jsonl` | 是 | 2 条偏好对样例 |
| `sft.jsonl` / `pref.jsonl` / `test.jsonl` | 否 | `python run.py data` 生成，可完全复现，不入库 |

## 字段

`SFT`（`sft.jsonl`）
```json
{"system": "服务规则…", "user": "用户问题", "assistant": "标准话术"}
```

`偏好对`（`pref.jsonl`）
```json
{"system": "…", "user": "…", "chosen": "合规话术", "rejected": "编造承诺的话术"}
```

`评测集`（`test.jsonl`）
```json
{"user": "…", "intent": "退换货", "forbidden": false}
```

## 重新生成

```bash
python run.py data            # 默认 seed=42
python run.py data --seed 7   # 换随机种子
```

生成后终端会打印「与测试集近重复丢弃」的条数，正常应为 0；不为 0 说明模板留出被破坏。
