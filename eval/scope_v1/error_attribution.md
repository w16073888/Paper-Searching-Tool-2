# Laya scope pilot 误差归因

## 结论

head-only 改善了 NLL/Brier，但没有改善 argmax 决策边界；主要瓶颈是边界/跨域定义和英文/混合语言表示。全量 146 条复核样本中错误样本标签均为 accept，因此当前证据不支持把失败主要归因于已复核标签错误。安全门控需要较高阈值才能在 calibration 控制风险，导致覆盖率不足 60%。

## 核心证据

| 指标 | calibration base | calibration trained | internal base | internal trained |
|---|---:|---:|---:|---:|
| accuracy | 0.6833 | 0.6750 | 0.6800 | 0.6880 |
| in_scope_false_refusal_rate | 0.3182 | 0.2879 | 0.3188 | 0.2899 |
| ood_false_pass_rate | 0.3148 | 0.3704 | 0.3214 | 0.3393 |

## 训练翻转

- Calibration: `{"correct_to_correct": 79, "wrong_to_wrong": 36, "wrong_to_correct": 2, "correct_to_wrong": 3}`
- Internal test: `{"wrong_to_wrong": 37, "correct_to_correct": 83, "wrong_to_correct": 3, "correct_to_wrong": 2}`

## 主要错误方向

- Calibration trained: false refusal 19，false pass 20；reviewed error items 6/39。
- Internal trained: false refusal 20，false pass 19；reviewed error items 10/39。

## 语言与题型

完整分组指标见同名 JSON。最明显的短板是 boundary 和英文/混合语言，而内部测试里的 OOD 误放行主要来自技术词触发。

## 门控失败原因

- Calibration 选阈值时覆盖 12.5%，到 internal_test 后实际覆盖 9.6%。
- 本地接管样例中仍有 2 条错误。
- Fallback 错误 7 条；其中包含当前 LLM 的一条 timeout。

## 决策

先修数据与任务定义，再考虑 encoder 解冻；多 seed 只会重复当前瓶颈。

## 归因排序

- **primary · head-only 只改善了概率排序，没有移动主要决策边界**：internal_test 仅 3 条错转对、2 条对转错，37 条继续错；NLL/Brier 改善但 accuracy 仅从 0.680 到 0.688。 下一步：先用已标错误构造更难/更均衡的 head-only 诊断集，不先加 seed。
- **primary · 英文 ML 问题被误拒，中文专业领域问题被误放行**：20 条范围内误拒中 13 条为英文；19 条 OOD 误放行中 11 条为中文。OOD 误放行集中在 petroleum_engineering 6、pharmacology 4、aerospace 3。 下一步：补英文 ML 术语正例和中文 OOD 技术词负例，按领域分层复核。
- **primary · 跨域边界定义仍是主要混淆源**：boundary accuracy 0.60；该组 6 条错误中 ml_for_education 4、physics_informed_ml 1、abbreviation_and_polysemy 1。 下一步：扩充边界金标，重点覆盖“用 ML 优化领域”与“领域传统方法”的最小对比对。
- **secondary · 正确/错误样本的置信度可分离性不足**：internal_test 正确均值 0.6760、错误均值 0.6393，仅差 0.0367；安全门控因此只覆盖 0.096。 下一步：若数据修正后仍有此问题，再评估更大容量 head 或 encoder 解冻。
- **not_primary_but_incomplete · 标签/切分错误不是已证实主因，但仍有银标覆盖缺口**：切分泄漏和近重复均为 0；internal_test 的 39 条错误中 10 条经过复核且全部为 accept，其余 29 条未逐条审阅。 下一步：优先人工复核 29 条未审错误，再判断是否需要调整 silver 标签。
