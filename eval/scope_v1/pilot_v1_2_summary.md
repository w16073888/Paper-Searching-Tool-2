# scope-dataset-v1.2 pilot · 2026-10-24

## 结论

参数化训练入口和显式 train/calibration 已可用。v1.2 head-only pilot 仍不达标；
按约定升级到最后 2 层 encoder 可改善范围内召回和概率质量，但显著恶化 OOD 误放行。
本轮不继续解冻更多层，也不切换线上 backend。

## 固定口径

- 数据：`scope-dataset-v1.2`，train 697 / calibration 178。
- `soft-ce`、`max_len=256`、seed 0、4 epochs、CPU。
- calibration 只用于温度与阈值；internal_test 在阈值冻结后评估。
- 外部 16 题未重复使用。

## Calibration

| 模型 | Accuracy | NLL | Brier | 范围内误拒 | OOD 误放行 |
|---|---:|---:|---:|---:|---:|
| base | 0.6404 | 0.7457 | 0.4876 | 0.3837 | 0.3370 |
| v1.2 head-only | 0.6067 | 0.6743 | 0.4771 | 0.3488 | 0.4348 |
| v1.2 last-2-layers | 0.6685 | 0.6116 | 0.4172 | 0.1744 | 0.4783 |

最后 2 层相对 head-only 明显改善了 accuracy、NLL/Brier 和范围内召回，
但把 OOD 误放行推向更高。

## 门控与 internal_test

| 模型 | 阈值/覆盖 | 总 accuracy | 范围内误拒 | OOD 误放行 |
|---|---|---:|---:|---:|
| v1.2 head-only | 10.1% cal / 11.2% test | 0.9355 | 0.0735 | 0.0536 |
| v1.2 last-2-layers | 19.1% cal / 19.2% test | 0.9435 | 0.0588 | 0.0536 |

两个模型均未达到：范围内误拒 ≤2%、OOD 误放行 ≤5%、本地覆盖 ≥60%。
last-2-layer 的热推理 P95 为 0.4054 秒，延迟满足要求。

## 判定

- head-only：表示/任务边界能力不足。
- last-2-layers：容量增加能学习更多 ML 词面与结构，但把专业领域技术词进一步当作 in_scope，
  OOD 风险反而上升。
- 继续解冻更多层不是当前最合理的下一步；应先修正类别偏置目标或加入专门负例学习，
  并保持 calibration-only 阈值选择。

## 证据

- `.models/jev/ft/scope-v12-head-only-seed0/pilot_report.json`
- `.models/jev/ft/scope-v12-last2-seed0/pilot_report.json`
- `gate_report_v1_2_head_only.json`
- `gate_report_v1_2_last2.json`
- `results_v12_last2/summary.json`

## OOD 非对称损失单次实验

为验证 OOD 误放行是否主要来自对称 CE，固定其他条件，仅把 OOD 样本
soft-CE 权重设为 2.0、in-scope 权重保持 1.0，在 last-2-layers 上训练一次。

| 指标 | last-2 baseline | OOD weight 2.0 |
|---|---:|---:|
| Calibration accuracy | 66.85% | 71.35% |
| Calibration NLL | 0.6116 | 0.5991 |
| Calibration Brier | 0.4172 | 0.3987 |
| 范围内误拒（argmax） | 17.44% | 25.58% |
| OOD 误放行（argmax） | 47.83% | 31.52% |
| 门控覆盖率（test） | 19.2% | 21.6% |
| 门控后准确率 | 94.35% | 94.35% |
| 门控后范围内误拒 | 5.88% | 4.41% |
| 门控后 OOD 误放行 | 5.36% | 7.14% |
| 热推理 P95 | 0.4054s | 0.4089s |

非对称损失改善了概率指标、整体 accuracy 和 OOD argmax，但固定门控仍只覆盖
21.6%，且误拒和误放行不能同时达标。按实验前约定不继续搜权重、不尝试双头，
不切换线上 backend。

## 非对称实验证据

- `.models/jev/ft/scope-v12-last2-oodw2-seed0/pilot_report.json`
- `gate_report_v1_2_last2_oodw2.json`
- `results_v12_oodw2/summary.json`
