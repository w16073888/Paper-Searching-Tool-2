---
tags: [索引, 数据]
---

# scope-dataset-v1.1

移动云 MaaS 生成的 scope silver 数据集，用于 Laya 微调、校准和开发期评估。
这些标签不是人工金标，不能直接作为最终论文指标。

## 数据口径

- 共 800 条：范围内 448、域外 352。
- 构造类别：语料接地范围内 400、明确 OOD 320、跨域边界 80。
- 父组切分：train 555、calibration 120、internal_test 125。
- 语言：中文 408、中英混合 130、英文 262。
- 父组总数 248；同一论文或同一领域组不会跨 split。
- 16 条 `../test.jsonl` 是独立外部测试集，不参与本数据集训练、校准或阈值选择。

## 文件

- `train.jsonl` / `calibration.jsonl` / `internal_test.jsonl`：完整元数据。
- `laya_train.jsonl` / `laya_calibration.jsonl` / `laya_internal_test.jsonl`：
  可直接交给 Laya 训练/评测格式。
- `all.jsonl`：800 条合并数据，只用于审计。
- `review.csv`：80 条全部边界题 + 66 条分层抽样；146 条已完成复核，
  含 Laya 预检预测、`review_decision` 和 `review_notes`。
- `quality_report.json`：结构、切分泄漏、近重复与外部测试重叠检查。
- `manifest.json`：数据哈希、模型、prompt 版本和数量口径。
- `generation_cache.jsonl`：批级断点缓存；脚本读取，不人工浏览。

## 复现

```powershell
python scripts/build_scope_dataset.py
```

密钥只从 `config.yaml` 的 `llm.api_key_env` 指向的环境变量读取，不落盘。
成功批次会写入 `generation_cache.jsonl`，重跑只请求缺失批次。

## 版本信息

- 父版本：`scope-dataset-v1`
- 复核：146/146 `accept`，未改标签
- manifest SHA256：`0bc723ef31f298ae3f1786679a6e85a21a94fd5f6a1ffacf8ae524c395fc2213`
- 原 v1 目录保持独立，不随 v1.1 修改

## 已知限制

- `label_source=llm_generated_silver`；80 条边界题和 66 条分层样本已完成复核，
  其余训练/测试标签仍为 silver。
- `laya.train_cli` 直接训练时会在输入文件内再随机划出少量校准项。正式训练应使用
  `laya_calibration.jsonl` 作为独立校准集的自定义入口，避免校准口径漂移。
- `internal_test.jsonl` 用于开发期选模；最终验收仍优先使用冻结的外部测试集。

## 相关

- `../../_index.md`
- `../../../scripts/build_scope_dataset.py`
- `../../../规划/2026-10-23-Jev调参-Laya多语言scope微调.md`
