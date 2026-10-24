# Laya scope head-only pilot · 2026-10-24

## 结论

冻结 encoder、只训练 decision head 的 pilot **未达到生产接管门槛**。
模型概率质量有小幅改善，但分类边界和选择性覆盖仍不足，不进入多 seed 或
encoder 解冻阶段。线上 scope backend 继续使用当前 LLM。

## 配置

- 数据：`scope-dataset-v1.1`，train 555 / calibration 120 / internal_test 125。
- 基座：`convaiinnovations/laya-multilingual`，revision `1720e3e...`。
- 训练：冻结 encoder、`soft-ce`、`max_len=256`、seed 0、4 epochs、CPU。
- 训练耗时：1321.22 秒。
- 最终温度：`[0.7186, 1.0, 1.0]`；无塌缩告警。

## Calibration

| 指标 | 基座 | head-only |
|---|---:|---:|
| Accuracy | 0.6833 | 0.6750 |
| NLL | 0.6083 | 0.5899 |
| Brier | 0.4146 | 0.4011 |
| top-1 Brier | 0.2073 | 0.2005 |
| ECE | 0.0359 | 0.0621 |
| 范围内误拒 | 0.3182 | 0.2879 |
| OOD 误放行 | 0.3148 | 0.3704 |

NLL/Brier 改善说明 head 学到了信号，但 accuracy、ECE 和 OOD 误放行没有同步改善。

## 开发期内部测试

| 模型/策略 | Accuracy | 范围内误拒 | OOD 误放行 | Laya 覆盖率 | 热推理 P95 |
|---|---:|---:|---:|---:|---:|
| 当前 LLM | 0.9516* | 0.0882 | 0.0000 | 0% | 6.37s |
| Laya zero-shot | 0.6800 | 0.3188 | 0.3214 | 100% | 0.346s |
| head-only argmax | 0.6880 | 0.2899 | 0.3393 | 100% | 0.441s |
| head-only + LLM fallback | 0.9355* | 0.0882 | 0.0357 | 9.6% | 0.441s |

\* LLM 有 1 条 `scope_timeout`，整体准确率按成功完成的 124 条计算。

三路门控阈值只在 calibration 上选择：

- `tau_in_scope=0.810`
- `tau_out_of_scope=0.835`
- calibration 覆盖率 12.5%，accepted accuracy 86.67%。
- calibration 高置信范围内误拒 1.52%，高置信 OOD 误放行 1.85%。

内部测试上 Laya 只接管 9.6%，远低于 60% 目标；整体范围内误拒仍为 8.82%。
因此门控不能带来有效的延迟收益，且相对纯 LLM 降低准确率。

## 证据

- `dataset_v1_1/`
- `results_internal_v1_1/summary.json`
- `results_trained_v1_1/summary.json`
- `gate_report.json`
- `.models/jev/ft/scope-v1-head-only-seed0/pilot_report.json`

## 下一步门槛

在以下任一问题解决前，不启动多 seed 或解冻最后 2–4 层：

- 补足边界和跨域应用中的人工金标，区分数据质量问题与表示能力问题；
- 降低英文/混合语言的范围内误拒；
- 证明在安全风险约束下本地覆盖率能达到 60% 以上。
