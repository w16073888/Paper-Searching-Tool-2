---
tags: [索引, 数据]
---

# scope-dataset-v1.2

在 v1.1 基础上完成 29 条错误复核和定向补强的 scope silver 数据集。

## 数据口径

- 共 1000 条：范围内 548、
  域外 452。
- 类别：in_scope 460、
  OOD 380、
  boundary 160。
- 切分：train 697、
  calibration 178、
  internal_test 125。
- 200 条新增样本只进入 train/calibration；internal_test 的 125 条内容保持冻结。
- 16 条 `../test.jsonl` 仍是独立外部测试集。

## v1.2 补强

- 复核 head-only 在 internal_test 中产生的 29 条未审错误，全部 `accept`。
- 新增 60 条英文 ML 正例。
- 新增 60 条中文 OOD 技术负例，覆盖石油工程、药理学和航天。
- 新增 80 条跨域最小对比题（40 正、40 负）。
- 新增样本按父组切分，无跨 split 泄漏、精确重复、近重复或外部测试重叠。

## 文件

- `train.jsonl` / `calibration.jsonl` / `internal_test.jsonl`
- `laya_*.jsonl`：Laya 训练/评测格式
- `all.jsonl`：全量审计数据
- `review.csv`：v1.1 的 146 条复核
- `review_errors_29.csv`：本轮 29 条错误复核
- `quality_report.json`：结构与泄漏审计
- `manifest.json` / `manifest.sha256`
- `augmentation_cache.jsonl`：本轮模型生成缓存

## 版本关系

- 父版本：`scope-dataset-v1.1`
- 父 manifest SHA256：`0bc723ef31f298ae3f1786679a6e85a21a94fd5f6a1ffacf8ae524c395fc2213`
- 本版本 manifest SHA256：`df157f0bda147710199f96bf9985dceb686145bd329bb6d79a512084ab147be3`
- v1 和 v1.1 均保持独立，不被覆盖。

## 复现

```powershell
python scripts/augment_scope_dataset.py
```

密钥只从 `config.yaml` 的 `llm.api_key_env` 指向的环境变量读取，不落盘。

## 相关

- `../../_index.md`
- `../../../scripts/augment_scope_dataset.py`
- `../../../scripts/train_scope_laya.py`
- `../../../规划/2026-10-23-Jev调参-Laya多语言scope微调.md`
