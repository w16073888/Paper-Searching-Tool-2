---
tags: [索引, 数据]
---

# scope_v1

Laya multilingual scope 判别器的冻结评测数据。本目录只保存独立测试样本和评测产物，
不保存训练集。

## 当前口径

- `test.jsonl`：16 条 v2 验收题，12 条 `in_scope`、4 条 `out_of_scope`。
- 该测试集只用于最终评估，不参与训练、校准或阈值选择。
- `smoke.jsonl`：只验证 Laya 训练输入格式与 dry-run，不进入正式训练。
- `results/`：脚本生成，可重建；包含逐题结果、汇总指标和算力探针。
- `dataset/`：800 条 scope silver 训练/校准/开发测试数据，详见其 README。

## 运行

```powershell
python scripts/eval_scope_classifier.py --backend all --train-dry-run
python scripts/build_scope_dataset.py
```

本机 CPU、无网络缓存时可分开运行：

```powershell
python scripts/eval_scope_classifier.py --backend laya --laya-repeat 3
python scripts/eval_scope_classifier.py --backend llm --llm-limit 16
```

## 相关

- `../_index.md`
- `../../scripts/eval_scope_classifier.py`
- `../../规划/2026-10-23-Jev调参-Laya多语言scope微调.md`
