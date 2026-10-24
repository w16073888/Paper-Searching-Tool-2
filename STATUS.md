# 当前开发状态

> 后续会话先读本文件；这里只保留当前目标、进度、下一命令。

更新：2026-10-24

## 当前目标

算法 v2 paper-first、离线 topic snapshot、契约 v2 和完整 HTTP 演示已跑通。
当前不继续改检索算法，重点转向稳定性、超时归因和前端联调。

## 基线

- Git 快照：`b517988`（旧实现可完整恢复）。
- 当前分支：`codex/agent-v2-reset`。
- `main` 保留删除前可运行版本。

## 已保留

- 论文数据、metadata、1024 维 embedding 与只读语料层。
- `hotspot.api` 契约、HTTP/SSE、前端适配。
- `reasoning/evidence.py`、`answer.py`、`verify.py`。
- `structure/embed.py`、`venue_matrix.py`。
- 问题集与 evidence/verify fixtures。
- Trace 汇总与发布/服务入口。

## 已删除

- 旧 `subcluster_adaptive`、direction、BM25/dense/RRF/MMR/router/rewrite/vector。
- 旧 global coverage、nearest-centroid coarse-to-fine。
- rlRAG/RF/MPS/共引/依赖抽取/旧聚类卡片。
- custom harness、旧 G0–G3 执行器与检索 benchmark。

## 当前状态

- LanceDB 全量索引已建立：22,747 篇、1024 维、vector + FTS。
- `Engine.ask` 已接 PydanticAI agent；范围内问题可返回带引用答案。
- Agent 可选 `search_papers` / `fetch_papers` / `get_topic_overview`；
  evidence 输出必须有 fetched ID。
- 输出后强制 `verify_and_render`，未通过引用校验的 claim 不进入答案。
- 32 条精确标题探测 32/32；v2 工具集 Recall@10=0.291、Recall@50=0.558。
- 5 条 OOD smoke 5/5 拒答；全局趋势问题会调用离线 topic snapshot。
- Trace v2 已写入 `workspace/ledger/retrieval_trace.jsonl`。
- Silver qrels：7 查询、367 候选、0 缺失；grade 0/1/2 = 83/20/264；
  其中 hard negatives + random negatives 已覆盖。
- Silver 工具基线：Recall@10=0.2115、Recall@50=0.8316。
- 契约 v2：旧参数已删除，保留 5 个 Agent/回答参数 + 3 个独立超时参数。
- `meta` 已记录 PydanticAI、LanceDB、topic snapshot、trace 能力和 arXiv 索引缺失提示。
- usage 已映射模型请求数、输入/输出 token 与耗时；成本仍为 0（待价格表）。
- 16 题验收：9 条正常带引用回答、3 条 OOD 拒答、1 条范围内拒答、3 条超时。
- 完整 HTTP 演示：DPO 返回 3 supported claims / 3 citations；
  158.036s、4 次模型请求、11032 输入 token、1497 输出 token。
- 历史 3 条超时均无 scope decision、无工具事件，归因为 scope 阶段超时。
- scope/agent/verify 已拆为 30/150/30 秒独立预算；scope 不重试，agent 最多一次。
- agent 异常已分类为 timeout / connection / output-invalid；verify 硬超时后回退本地抽取式证据。
- 接口文档已同步 v2：route、SSE 三阶段、空 topics/graph、超时码、verify 状态、
  arXiv 不可用和 usage 口径全部更新。
- scope v1 冻结集已完成双基线：当前 LLM 16/16、P50 6.34s / P95 10.09s；
  Laya zero-shot 11/16，范围内 7/12、OOD 4/4、热推理 P95 0.345s；
  5 条范围内误拒集中在 Dreamer、visuomotor、TD3/SAC、逆 RL、离线 RL。
  Laya 冷加载 25.92s；训练 dry-run 4/4 条有效、无跳过。
- scope-dataset-v1 已用移动云模型构造 800 条 silver 数据：范围内 448、域外 352；
  train/calibration/internal_test = 555/120/125，248 个父组无跨 split 泄漏。
  100% 通过结构、切分、外部测试重叠和近重复审计；边界领域均含正负例。
- `scope-dataset-v1.1` 已完成 146 条复核（全部 accept，无改标），保留 v1；
  manifest SHA256 `0bc723ef...fc2213`。Laya 在复核样本中有 55 条分歧，
  其中 35 条是边界题，主要将跨域应用中的 ML 方法误判为 OOD。
- internal_test 开发期基线：当前 LLM 完成 124/125、accuracy 95.16%、
  范围内误拒 8.82%、OOD 误放行 0%；Laya zero-shot 68.0%，
  范围内误拒 31.88%、OOD 误放行 32.14%。
- Laya head-only pilot 已完成 4 epochs（1321s）：calibration NLL 0.6083→0.5899、
  Brier 0.4146→0.4011，但 accuracy 0.6833→0.6750，OOD 误放行
  0.3148→0.3704。热推理 P95 0.441s。
- 三路门控阈值仅由 calibration 选出 `tau_in=0.81 / tau_out=0.835`，
  internal_test 本地覆盖仅 9.6%，整体 accuracy 93.55%、范围内误拒 8.82%；
  覆盖率与误拒目标未达标。pilot 判决：不扩展模型，线上继续用 LLM scope。
- 误差归因已完成：head-only 在 internal_test 只把 3 条错转对、2 条对转错，
  其余 37 条继续错；20 条范围内误拒中 13 条为英文，19 条 OOD 误放行
  中 11 条为中文，集中在石油工程、药理学和航天。边界题 accuracy 0.60，
  教育应用 4 条错误。正确/错误平均置信度仅差 0.0367，安全门控因此覆盖率低。
  39 条错误中 10 条经过人工复核且全部 accept；其余 29 条已完成追加复核。
- 29 条追加错误复核：16 条 in_scope、13 条 OOD，全部 `accept`；
  复核表见 `eval/scope_v1/dataset_v1_2/review_errors_29.csv`。
- `scope-dataset-v1.2` 已完成：新增 60 条英文 ML 正例、60 条中文 OOD
  技术负例（石油/药学/航天）和 80 条跨域最小对比题；总 1000 条，
  train/calibration/internal_test = 697/178/125。新增 200 条只进
  train/calibration，internal_test 内容变化 0、ID 冲突 0。
  质量审计无切分泄漏、精确重复、0.90 近重复或外部测试重叠。
- 训练入口已参数化：支持显式 train/calibration、manifest 校验、loss、冻结策略、
  解冻层数、batch/accumulation、LR、seed、长度、device 与 dry-run。
- v1.2 head-only pilot 未达标：calibration accuracy 0.6067、范围内误拒 0.3488、
  OOD 误放行 0.4348；internal_test 门控覆盖 11.2%。
- 按约定升级到最后 2 层 encoder：calibration accuracy 0.6685、NLL 0.6116、
  Brier 0.4172，但 OOD 误放行升至 0.4783；internal_test 门控覆盖 19.2%、
  范围内误拒 0.0588、OOD 误放行 0.0536。三项硬目标仍未达标，
  热推理 P95 0.4054s。停止继续解冻，考虑类别偏置目标和专门负例学习。
- OOD 非对称损失单次实验已完成：last-2-layers + OOD weight 2.0。
  calibration accuracy 0.7135、NLL 0.5991、Brier 0.3987；
  门控覆盖率 21.6%、范围内误拒 0.0441、OOD 误放行 0.0714。
  热推理 P95 0.4089s。三项硬目标仍未同时达标，按约定止损，
  不继续搜权重、不尝试双头，线上继续使用 LLM scope。

## 下一步

1. 用新超时预算重跑历史 3 条超时问题，验证错误分层。
2. 排查 RLOO 范围内拒答；不改检索召回策略。
3. 前端按 contract v2 联调，验证来源禁用与 usage 展示。
4. 旧 Jev 权重和外部源码已删除，只保留 `scope_probe*.json` 测试记录。
   `.models/jev` 已替换为 `convaiinnovations/laya-multilingual`（mmBERT-base、
   322M，revision `1720e3e...`）；权重 SHA256 已与官方 LFS 元数据一致。
   scope v1、v1.1、v1.2 数据和 head-only pilot 均已完成。当前不切换 Laya
   backend，不跑多 seed、不解冻 encoder。误差归因见
   `eval/scope_v1/error_attribution.md`；29 条错误复核与定向补强已完成。
   v1.2 head-only、最后 2 层解冻和 OOD 2:1 非对称损失均未通过。
   本轮已按预算止损，默认继续使用 LLM scope；后续 scope 工作需重新立项，
   不再在当前方向继续调权重或扩大层数。

## 固定验收

```powershell
python -m py_compile hotspot/api/engine.py hotspot/api/contract.py hotspot/api/http.py hotspot/reasoning/evidence.py hotspot/reasoning/answer.py hotspot/reasoning/verify.py
python scripts/check_api.py --quiet
python scripts/check_tree.py --vault .
```

旧 Trace 仍可用 `scripts/summarize_trace.py` 离线汇总。
