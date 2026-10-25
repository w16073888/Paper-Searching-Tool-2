---
tags: [评测, holdout, 比较题]
---

# v2 比较题 holdout v1 · 正式判定标准

## 冻结规则

- 本题集只允许运行一次。首次运行已完成，之后不得再运行。
- 不得使用本题集的结果调整提示词、检索排序、模型参数或阈值。
- 后续修改必须新建版本化 holdout，禁止复用 v1。
- `run_v2_eval.py` 会读取同目录 manifest；`do_not_run_again=true` 时拒绝再次执行。

## 自动判定

以下五项必须同时通过：

| Gate | 要求 |
|---|---|
| 回答率 | `answered / n = 100%` |
| 可引用 claim 率 | `supported / claims ≥ 85%` |
| 全量 grounding 题比例 | 所有 claim 均 supported 的题 ≥ 70% |
| 延迟 | P95 ≤ 60 秒 |
| 错误 | 请求错误数 = 0 |

单题 operational pass 同时要求：

```text
route == local
refused == false
error == null
supported > 0
citations > 0
```

## 首次结果

```text
answered=10/10
errors=0
claims=38
supported=34
filtered=4
supported_claim_rate=0.8947
fully_grounded_questions=7/10
full_grounding_rate=0.7
avg=23.414s
p95=30.840s
automatic_gate_passed=true
```

## 口径限制

自动 gate 验证路由、引用支持的 claim 和延迟，不替代人工语义正确性评审。
由于题目已经冻结且本题集只运行一次，不能为了补人工评分而重新运行 v1；
若需要人工正确性结论，应使用答案导出结果或新建下一版 holdout。

## 复现评分

评分器只读取已有结果，不调用模型：

```bash
python scripts/evaluate_comparison_holdout.py \
  --result workspace/ledger/v2_comparison_holdout_v1_result.json \
  --out workspace/ledger/v2_comparison_holdout_v1_summary.json
```
