# 当前开发状态

> 后续会话先读本文件；这里只保留当前目标、进度、下一命令。

更新：2026-10-26

## 当前目标

项目已迁移到云服务器并使用本地 Qwen3-8B-AWQ + vLLM。
当前重点是保持云/Git 同步；全量主题快照和趋势题端到端修复已完成验收。

## 2026-10-25 云迁移收尾

- Qwen：`127.0.0.1:8000`，Qwen3-8B-AWQ，vLLM 0.32.0。
- 项目服务：`127.0.0.1:8080`，配置 `config.server.yaml`。
- 16 题验收连续两轮 `16/16`，逐题 route/refusal/claims/citations 完全一致。
- 比较题已支持对象拆词、分路检索和 RRF 融合。
- 比较题 holdout v1 已运行一次：`10/10` 正常回答，自动 gate 通过。
- holdout v1 状态为 `retired`，`do_not_run_again=true`，不得再次运行或用于调优。
- 契约自检 `35/35`；README 索引树完整。

## 2026-10-25 arXiv 周报云端日更

- 原周报抓取器已 vendored 到 `hotspot/vendor/arxiv_weekly/`，源目录保持只读。
- 周成员只按 `first_seen` 的 ISO 周绑定；`replace/cross` 只进公告旁路文件。
- 历史四周已生成 harness：W38 1,069 / W39 2,758 / W40 542 / W41 804 篇，
  均因缺失工作日保持 `partial`。
- arXiv 缺失向量已补齐 804 篇，LanceDB 为 5,173 篇；`scope.week_id` 已逐条验证。
- 云端 cron：上海工作日 13:30 日更，周日 21:00 封卷；封卷要求五个有效 `.done`。
- 2026-10-25 是周末，隔离冒烟只得到预期空 feed；下一工作日观察首轮正式日更。

## 2026-10-26 8B / 14B 模型选型

- Qwen3-14B-AWQ 已在 12 GB RTX 3060 上完成加载、性能和 20 分钟稳定性测试。
- 公平配置下 8B 约 59.0 tok/s，14B 约 36.7 tok/s；14B 为 8B 的约 62%。
- 14B 最大上下文为 8,128；项目 16 题中的比较题实际达到 8,129 tokens 并报 400。
- 24 题冻结选型集自动分：8B 11/24、14B 12/24，未达到预设的至少提升 2 题。
- 项目 16 题：8B 15/16（`a_c03` 误拒）、14B 15/16（`a_c01` 上下文超限）、
  移动云 API 16/16。
- 最终决定：**生产默认继续使用 Qwen3-8B-AWQ**，14B 保留作离线候选，
  移动云 API 仅作为可选外部对照。
- 测试代码、冻结题集、原始结果和恢复验证已落盘；生产 8B 已恢复。

## 2026-10-26 全量主题聚类

- 新增 `hotspot/structure/topic_cluster.py`：PCA-128 + 分桶局部 kNN +
  Leiden，巨簇拆分、小簇合并、全量强制归属。
- 新增 `scripts/build_topic_snapshot.py` 与 `scripts/check_topic_snapshot.py`。
- 全量 22,747 篇构建完成：220 簇、覆盖率 100%、成员 22/103/222
  （min/median/max）、320,421 条图边、平均度 28.17、构建耗时 20.9 秒。
- 生产快照 `hotspot/data/topic_snapshot_v1.json.gz` 为 1,065,456 bytes，
  随代码部署；`workspace/metrics/` 旧快照只作兼容回退。
- `TopicStore` 改为词法 + PCA 质心混合选簇；弱相关的
  `process supervision` 查询返回空，由 Agent 降级到 `search_papers`。
- `meta.capabilities.topic_snapshot` 改为读取真实快照可用性。
- `g01–g20` 工具层探针 19/20 命中；仅 `g11 process supervision` 无可靠簇，
  按设计返回空并走论文检索 fallback。
- 快照加载 0.096s；BGE 冷启动后新查询约 0.24–0.25s（词法 + 质心 + 选簇）。

## 2026-10-27 趋势题失败修复

- `g01–g20` 首轮端到端为 18/20；`g02` 是 malformed final claims，
  `g12` 是重复 search/fetch 后耗尽请求预算。
- Agent 新增非对象 claim 修复、未 fetch evidence 的安全降级、重复工具调用缓存，
  并把 search/fetch 工具重试上限设为 3。
- 三条故障题复测全部通过；完整 `g01–g20` 复测为 **20/20**。
- 100 claims / 94 supported / 110 citations；P50 15.156s、P95 27.259s。
- 重复调用拦截实际触发于 `g06` search 和 `g19` fetch，模型随后正常完成回答。
- `check_api` 46/46；生产 8080 已重启到修复版本并通过 HTTP 冒烟。

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

1. 比较题 holdout v1 已退役，不再运行；后续算法改动另建 v2 holdout。
2. 云端当前不能直接访问 GitHub，代码同步以本地 Git bundle 为准，直到配置代理或迁移远端。
3. 前端按 contract v2 联调，并补鉴权、反向代理或 SSH 访问方案；引擎不加账号状态。
4. 下一工作日核对 cron 产出的 `.done`、当周 harness、向量增量与新 arXiv 索引。
5. 如需长期在线，配置用户级开机自启；`.models` 仅在需要 Laya 实验时迁移。
6. 若未来转向 16B 以上模型，需要更小量化/更强 GPU 或重新设计上下文预算。

## 固定验收

```powershell
python -m py_compile hotspot/api/engine.py hotspot/api/contract.py hotspot/api/http.py hotspot/reasoning/evidence.py hotspot/reasoning/answer.py hotspot/reasoning/verify.py
python scripts/check_api.py --quiet
python scripts/check_tree.py --vault .
python scripts/check_weekly_pipeline.py
python scripts/daily_arxiv_sync.py --preflight
python scripts/evaluate_comparison_holdout.py --result eval/v2_comparison_holdout_v1.result.json
```

holdout v1 只允许对已归档结果做离线复算，禁止用 `run_v2_eval.py` 再次运行。
旧 Trace 仍可用 `scripts/summarize_trace.py` 离线汇总。
