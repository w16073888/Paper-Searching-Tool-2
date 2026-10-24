---
tags: [实验, 日志, Jev, Laya, scope]
---

# 2026-10-24 从 Jev 社区试错到 Laya scope 微调

> 承接 [[2026-10-20-P0算法性能改造]]：前序工作已经把论文检索、回答和引用校验收口到 v2 Agent；
> 这两天要解决的是它前面最昂贵、也最容易超时的一步——判断问题是否属于本地 AI/ML 论文库。

**环境**：Windows 10 · Python 3.13.9 · 8 CPU 线程 · NVIDIA MX330 2 GB · `laya==0.4.13` ·
会议论文 22,747 篇 · scope v1.2 共 1,000 条 silver 数据。

## 1. 研究问题：把 scope 判断从慢速云端 LLM 下沉到本地判别器

当前请求链路的第一道门是 scope classification。此前它由移动云 LLM 完成，输入很短、
输出只有 `in_scope/out_of_scope`，但一次请求实测要数秒，而且一旦网络或模型服务抖动，
就会直接阻塞后面的 Agent 工具调用。

这条科研故事的问题定义不是“能不能让本地模型回答论文问题”，而是：

> 能否用一个本地、低延迟、可校准的 System-1 判别器，接管高置信范围内的正负样本；
> 对不确定样本继续回退现有 LLM，从而降低平均延迟和 scope 超时，同时不恶化误拒与误放行？

对应的长期方案在 [[2026-10-23-Jev调参-Laya多语言scope微调]]。本日志只记录实际动作、数据和证据。

## 2. 第一轮路线：社区 Jev 复现没有达到项目门槛

最初尝试的是社区 openJev 风格的本地决策模型，并把权重放在 `.models/jev` 下。
首个 151M 级候选在 17 条 scope 探针上的实测结果很差：

```text
model_dir           : .models/jev/oj141m
load_s              : 25.374
rss_mb              : 684
sample_count        : 17
batch_accuracy      : 0.2353
in_scope_recall     : 0
out_of_scope_recall : 1
abstain_rate        : 0
```

它对域外问题全部判对，但 13 条范围内问题一条也没识别出来。也就是说，模型退化成了
“见谁都判 OOD”的保守护栏，短标签、英文改写和关闭校准都没有改善。

这个失败排除了一个看似诱人的故事：

> “找一个开源 Jev 小模型，接到现有 router 上，就能低成本替代 LLM。”

实际结论是，社区复现可以用作负基线，但不能因为名字像 Jev 就把它当生产模型。
旧权重和外部源码后来被清理，只保留 `scope_probe*.json` 作为失败证据。

## 3. 第二轮路线：改用 Laya multilingual 作为本地决策基座

第二轮把基座换成 `convaiinnovations/laya-multilingual`：

- 架构：mmBERT-base encoder + 2 层 decision head，总计 322M；
- revision：`1720e3e3357cfe1e281542e223f8273b0890ca34`；
- 本地位置：`.models/jev`；权重没有进入 Git；
- `model.safetensors`：643,835,514 字节；
- SHA256：`9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`；
- 运行时：`laya==0.4.13`。

这里保留 `.models/jev` 这个路径，是历史命名；当前实际模型已经是 Laya，不再是旧 openJev。

本机实测：

```text
load_s              : 25.9217
warmup_s            : 0.6329
timed_inference_calls: 48
mean latency_s      : 0.3168
p50 latency_s       : 0.3175
p95 latency_s       : 0.3445
```

Laya 的冷加载约 26 秒，因此服务不能在首个用户请求里懒加载，必须启动时预加载和 warm-up。

## 4. 工程接入：先建立可比较的评测入口，不直接切换生产路由

检查当前代码后确认：

- `hotspot.agent.runtime::_classify_scope()` 仍直接调用云端 LLM；
- `ScopeDecision` 只有类别、领域和原因，没有置信度；
- 还没有生产级的 Laya adapter、门控或灰度开关。

因此这一步没有把 Laya 强行塞进生产链路，而是先增加了评测和实验基础设施：

- `scripts/eval_scope_classifier.py`：统一比较 LLM、Laya zero-shot、加载和热推理延迟；
- `eval/scope_v1/test.jsonl`：冻结 16 条外部测试题；
- `eval/scope_v1/manifest.json`：记录测试集哈希，并禁止它参与训练和阈值选择；
- `scripts/check_tools_v2.py` 和其他现有离线检查继续验证工程边界。

这一步的工程价值是建立了“同一测试集、同一 prompt、同一指标”的对照条件。
生产 `_classify_scope()` 暂时不动，避免在模型尚未证明有效时引入线上回归。

## 5. 冻结测试与开发测试：Laya 的强项和短板同时暴露

### 5.1 外部 16 题 sanity check

| 后端 | 正确 | 范围内召回 | OOD 召回 | P50 | P95 |
|---|---:|---:|---:|---:|---:|
| 当前 LLM | 16/16 | 12/12 | 4/4 | 6.3388s | 10.0901s |
| Laya zero-shot | 11/16 | 7/12 | 4/4 | 0.3263s | 0.3554s |

Laya 的域外判断非常保守：4 条明确 OOD 全部拒绝；代价是 12 条范围内问题误拒 5 条。
误拒集中在 Dreamer、visuomotor policy、TD3/SAC、逆强化学习和离线强化学习。

这组 16 题后来只作为 sanity check 保存，不再反复拿来做阈值选择。

### 5.2 内部 125 题开发期对照

构造数据集后，在 `internal_test` 的 125 条样本上比较当前 LLM 与 Laya zero-shot：

| 后端 | 完成 | 准确率 | 范围内召回 | OOD 召回 | 范围内误拒 | OOD 误放行 | P50 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前 LLM | 124/125 | 0.9516 | 0.9118 | 1.0000 | 0.0882 | 0.0000 | 3.4803s | 6.3735s |
| Laya zero-shot | 125/125 | 0.6800 | 0.6812 | 0.6786 | 0.3188 | 0.3214 | 0.3047s | 0.3460s |

LLM 有 1 条 `scope_timeout`；按已完成样本计算分类指标，同时在日志中单独保留超时，
不把超时伪装成普通的 OOD 判断。

Laya 的延迟优势非常稳定，但零样本能力不够：

- 边界题准确率只有 0.5333；
- 中文范围内召回尚可，但 OOD 误放行较高；
- 英文和混合语言的整体 accuracy 分别为 0.6512、0.5714；
- 它会把“图神经网络做气象预报”“PINN 训练”“RL 做交易策略”这类问题当成应用领域本身，
  这正是项目要专门修掉的“技术名词 -> OOD”偏差。

## 6. 数据集构造：从“凭感觉找几条题”变成可回放的双层数据

数据集使用 `Mobile_trash_API_KEY` 指向的移动云 MaaS 模型生成，密钥只从环境变量读取，
没有写入任何数据、manifest 或日志。

### 6.1 数据设计

首版 v1 构造了 800 条 silver 样本：

- 400 条论文接地范围内问题：从 22,747 篇语料中选 200 篇，每篇 2 种问法；
- 320 条明确 OOD：32 个领域，每领域 10 条；
- 80 条跨域边界：16 个应用领域，每领域 5 条，并要求同时包含“领域本身”和“领域中的 ML 方法”；
- 语言分布：中文 408、中英混合 130、英文 262；
- 父组总数 248，按论文或领域整组切分，避免同论文/同领域的改写题跨 split；
- split：train 555、calibration 120、internal_test 125。

相关脚本：

- `scripts/build_scope_dataset.py`：数据生成、缓存、格式校验、切分和质量审计；
- `scripts/finalize_scope_review.py`：人工复核结果、Laya disagreement 回填和新版本生成。

生成过程不是一次成功：

- 移动云结构化响应耗时在约 44–390 秒之间波动；
- `ood-02` 两次返回数量不完整/超额，最后改成两个领域小批，再二分到单领域生成；
- 35 个成功批次直接复用缓存，避免重复请求；
- 最终所有 36 个批次的输入和输出都落在 `generation_cache.jsonl`。

### 6.2 质量门槛

质量报告显示：

- 800/800 条字段完整；
- 248 个父组中跨 split 组数为 0；
- 200 个论文题组全部恰好 2 条；
- 16 个边界领域全部同时有正负例；
- 与外部 16 题测试集精确重叠为 0；
- 0.90 字符 trigram 阈值下，同 split 近重复 0 对，跨 split 近重复 0 对；
- 问题长度：18–273 字符，均值 92.02；
- silver confidence：0.63–0.99，均值 0.913。

Laya 训练格式 dry-run 也通过：

```text
train: 555 rows read, 555 valid items, skipped: {}
calibration: 120 rows read, 120 valid items, skipped: {}
```

## 7. v1.1 人工复核：重点不是“模型都说了什么”，而是边界口径是否一致

复核表 `review.csv` 包含 146 条：80 条全部边界题，加 66 条分层抽样。

逐条复核结论：

- 146/146 条 `review_decision=accept`；
- 没有发现需要改标签的样本；
- 每条 `review_notes` 都写明了“领域机制”还是“领域中的 ML 方法”；
- 增加了 Laya zero-shot 预测和选中概率，作为第二意见和未来讲数据质量故事的证据。

Laya 与人工判断不一致 55 条，其中：

- 边界 35 条；
- 普通范围内 10 条；
- OOD 10 条。

这些不一致主要不是数据错了，而是模型偏见：

Laya 看到“气候/医学/金融/材料”就倾向判 OOD，即使问题明确在问 GNN、PINN、
Transformer、RL、LSTM 或损失函数。例如，它会将明确使用 LSTM 的金融预测题
误拒为 OOD。这给后续训练提供了非常直接的错误切片。

新版本 `scope-dataset-v1.1`：

- 保留原 v1 目录不动；
- 新目录：`eval/scope_v1/dataset_v1_1/`；
- 新 manifest SHA256：`0bc723ef31f298ae3f1786679a6e85a21a94fd5f6a1ffacf8ae524c395fc2213`；
- 记录父版本 v1 的 manifest hash；
- 原训练文本不变，但增加 review 元数据，因此文件哈希全部重算。

## 8. 显式 train/calibration：绕开官方 CLI 的隐式校准切分

Laya 自带 `laya.train_cli` 很方便，但它会在输入文件里再随机切走约 10% 做校准。
这与项目已经冻结的独立 calibration 集不一致。dry-run 输出直接暴露了这个问题：

```text
dry-run: 555 rows read, 555 valid items, skipped: {}
dry-run: optimizer budget: 500 train item(s), effective batch up to 64
(micro_batch 8 x grad_accum 8), 32 optimizer updates over 4 epoch(s)
```

555 条输入最后只留下 500 条训练项，说明不能直接把 `laya_train.jsonl` 交给官方 CLI
并声称“train=555、calibration=120”。

因此新增 `scripts/train_scope_laya.py`：

- 训练只读 `laya_train.jsonl`，硬校验 555 条；
- 温度拟合只读 `laya_calibration.jsonl`，硬校验 120 条；
- 不读取 `laya_internal_test.jsonl`；
- 冻结 encoder，只训练 decision head；
- `soft-ce`；
- `max_len=256`、`head_max_len=256`；
- seed 0；
- 4 epochs；
- micro-batch 4、gradient accumulation 4；
- 每 epoch 保存 `checkpoint_latest`；
- 训练后用 calibration 拟合温度并输出 Brier/NLL、混淆矩阵和 abstention threshold。

## 9. v1.1 head-only：概率指标改善，但决策边界没有移动

第一次训练前，自定义脚本把 `expected={"scope": ...}` 当字符串比较，导致：

```text
base calibration: acc=0.0 in_recall=0.0 ood_recall=0.0
```

修正后恢复为 `acc=0.6833`。这说明显式训练入口必须对训练前指标做独立 sanity check。

完整 head-only pilot 结果：

| Calibration | Base | Head-only |
|---|---:|---:|
| Accuracy | 0.6833 | 0.6750 |
| NLL | 0.6083 | 0.5899 |
| Brier | 0.4146 | 0.4011 |
| 范围内误拒 | 0.3182 | 0.2879 |
| OOD 误放行 | 0.3148 | 0.3704 |

三路门控阈值只在 calibration 选出 `tau_in=0.81 / tau_out=0.835`；到 internal_test
只覆盖 9.6%，总 accuracy 0.9355、范围内误拒 0.0882、OOD 误放行 0.0357。
覆盖率和范围内误拒均不达标。热推理 P95 为 0.441s。

## 10. 误差归因与 v1.2 补强：英文 ML 正例和中文技术词负例不够

对 base→head-only 的逐题翻转和错误切片分析：

- internal_test 仅 3 条错转对、2 条对转错，37 条仍然错误；
- 20 条范围内误拒中 13 条为英文；
- 19 条 OOD 误放行中 11 条为中文；
- OOD 误放行集中在石油工程 6、药理学 4、航天 3；
- 正确与错误样本平均置信度只差 0.0367，导致安全门控无法覆盖；
- 39 条错误中 10 条已审且全部 `accept`，剩余 29 条也经追加复核全部 `accept`。

据此构造 `scope-dataset-v1.2`：

- 新增 60 条英文 ML 正例；
- 新增 60 条中文石油/药学/航天 OOD 技术负例；
- 新增 80 条跨域最小对比题（40 正、40 负）；
- 总 1,000 条，train/calibration/internal_test = 697/178/125；
- 新增数据只进 train/calibration，internal_test 题干和标签变化为 0；
- 无父组跨 split、精确重复、0.90 近重复或外部测试重叠。

manifest SHA256：

```text
df157f0bda147710199f96bf9985dceb686145bd329bb6d79a512084ab147be3
```

## 11. v1.2 容量升级：last-2-layers 改善 ML 判断，却放大 OOD 误放行

先跑同口径 head-only（697/178，`soft-ce`、`max_len=256`、seed 0、4 epochs）：

| Calibration | Base | v1.2 head-only |
|---|---:|---:|
| Accuracy | 0.6404 | 0.6067 |
| NLL | 0.7457 | 0.6743 |
| Brier | 0.4876 | 0.4771 |
| 范围内误拒 | 0.3837 | 0.3488 |
| OOD 误放行 | 0.3370 | 0.4348 |

仍不达标后，按预设升级到最后 2 层 encoder（约 10.03M encoder 参数可训练，
encoder LR `1e-5`）：

| Calibration | Base | last-2 |
|---|---:|---:|
| Accuracy | 0.6404 | 0.6685 |
| NLL | 0.7457 | 0.6116 |
| Brier | 0.4876 | 0.4172 |
| 范围内误拒 | 0.3837 | 0.1744 |
| OOD 误放行 | 0.3370 | 0.4783 |

门控后在 internal_test 上 coverage 0.192、范围内误拒 0.0588、OOD 误放行
0.0536；三项目标仍未同时满足。热推理 P95 0.4054s。

## 12. OOD 非对称损失单次实验：明显改善，但仍未达到可接管门槛

预注册唯一一次实验：保持 last-2-layers、数据、seed 和超参不变，只把
OOD soft-CE 权重设为 2.0、in-scope 保持 1.0。

| Calibration | last-2 baseline | OOD weight 2.0 |
|---|---:|---:|
| Accuracy | 0.6685 | 0.7135 |
| NLL | 0.6116 | 0.5991 |
| Brier | 0.4172 | 0.3987 |
| 范围内误拒（argmax） | 0.1744 | 0.2558 |
| OOD 误放行（argmax） | 0.4783 | 0.3152 |
| internal_test 门控 coverage | 0.192 | 0.216 |
| 门控后 accuracy | 0.9435 | 0.9435 |
| 门控后范围内误拒 | 0.0588 | 0.0441 |
| 门控后 OOD 误放行 | 0.0536 | 0.0714 |
| 热推理 P95 | 0.4054s | 0.4089s |

非对称损失显著改善了 OOD argmax 和概率指标，但固定门控只能覆盖 21.6%，
且范围内误拒与 OOD 误放行不能同时达标。按实验前约定：

- 不继续搜 OOD 权重；
- 不尝试双头；
- 不继续扩大解冻层数；
- 线上默认继续使用 LLM scope，Laya 仅保留为实验模型。

## 13. 这个故事当前能怎么讲

可以把研究叙事组织成四段，而不是“我们换了一个模型”：

1. **问题识别**：scope 判断短、频繁、高延迟，是整条 RAG 链路的成本瓶颈；
2. **负结果**：社区 Jev 风格 151M 模型在 17 条探针中范围内召回为 0，
   说明“本地判别器”不能靠同名开源模型直接落地；
3. **方法修正**：改用 Laya 322M System-1 模型，建立独立测试、双后端对照、
   分组数据构造和人工边界复核；
4. **证据驱动迭代**：Laya 的速度达标但零样本质量不达标，因此冻结 encoder 做 head-only 微调，
   后续只让高置信样本本地接管，中间区域回退 LLM。

最有价值的科研故事不是“准确率涨了多少”，而是：

> 我们先证明了社区 Jev 在本任务上会退化成单类拒答，再把 scope 从自然语言生成任务
> 重构为可校准的二分类决策问题，最后用带边界正负例的 silver 数据和独立 calibration，
> 区分“模型不会判断”与“模型置信度不可信”两种错误。

## 发现的问题

### 1. 【最严重】社区 Jev 在项目范围内几乎完全失效

- 现象：17 条探针中准确率 0.2353，范围内 13 条召回为 0。
- 根因：该模型/标签格式不适合本项目的中文 AI/ML 论文问题，退化为把所有问题判成 OOD。
- 证据：`batch_accuracy=0.2353`、`in_scope_recall=0`、`out_of_scope_recall=1`。

### 2. Laya zero-shot 速度快，但边界和混合语言质量不足

- 现象：125 条内部测试 accuracy 0.68，边界 accuracy 0.5333，范围内误拒与 OOD 误放行均约 32%。
- 根因：领域名词先验过强，模型把应用领域本身与“应用领域中的 ML 方法”混淆。
- 证据：内部测试 `in_scope_false_refusal_rate=0.3188`、
  `ood_false_pass_rate=0.3214`。

### 3. 官方训练 CLI 的隐式校准切分与项目口径不一致

- 现象：输入 555 条，dry-run 显示只使用 500 条训练项。
- 根因：`laya.train_cli` 会在输入文件内部再随机切出约 10% 做校准。
- 证据：`dry-run: 555 rows read ... optimizer budget: 500 train item(s)`。

### 4. 自定义训练脚本的 expected dict 解析错误曾在训练前制造假信号

- 现象：第一次训练启动时，未训练模型的 calibration accuracy 被打印为 0。
- 根因：把 laya 数据行的 `expected={"scope": ...}` 当普通字符串，正负例集合为空。
- 证据：`base calibration: acc=0.0 in_recall=0.0 ood_recall=0.0`；
  修正后为 `acc=0.6833 in_recall=0.6818 ood_recall=0.6852`。

### 5. 【最严重】Laya 训练能把范围内判断做强，但没有学会“领域本身不是 ML”

- 现象：v1.2 head-only accuracy 0.6067；last-2 提升到 0.6685，但 OOD 误放行从
  0.3370 升到 0.4783；OOD 2:1 加权后仍为 0.3152（argmax），门控后 0.0714。
- 根因：模型将“技术性、优化、反馈、决策”等词面当作 in_scope 信号，语言与领域上下文
  仍没有形成稳定可分边界。
- 证据：last-2 的 internal_test OOD recall 0.6、boundary accuracy 0.5333；
  三组模型均未达到 coverage ≥60%、误拒 ≤2%、OOD 误放行 ≤5%。

### 6. Windows 相对路径在报告保存阶段两次导致训练已完成但 artifact 未落盘

- 现象：train 和 calibrate 报告在 `Path.relative_to(ROOT)` 处抛 `ValueError`。
- 根因：传入相对路径，而 `ROOT` 是绝对路径。
- 证据：`ValueError: 'eval\\scope_v1\\...' is not in the subpath of 'D:\\...'`。
- 修复：统一用 resolve/display path；训练权重未丢，使用 `finalize_scope_training.py`
  从 epoch 4 checkpoint 恢复。

## 待完成

- [x] head-only 4 epochs、温度校准、三路门控和 internal_test 验收均完成。
- [x] 判断 pilot 不通过，未继续多 seed 或盲目扩大解冻层数。
- [x] 完成 29 条错误复核和 v1.2 定向补强。
- [x] 完成最后 2 层容量升级和唯一一次 OOD 非对称损失实验。
- [x] 实验结果仍不达标，默认线上继续使用 LLM scope。
- [ ] 若重新开启 scope 方向，需要新方案与新一轮预算；不要在本轮继续调权重或双头。
- [ ] 生产 `_classify_scope()` 尚未切换 Laya adapter；影子运行和 canary 未开始，也不应在本轮启动。

## 相关

- [[2026-10-20-P0算法性能改造]]
- [[2026-10-23-Jev调参-Laya多语言scope微调]]
- `eval/scope_v1/dataset_v1_1/README.md`
- `eval/scope_v1/results_internal_v1_1/summary.json`
- `scripts/build_scope_dataset.py`
- `scripts/finalize_scope_review.py`
- `scripts/eval_scope_classifier.py`
- `scripts/train_scope_laya.py`
- `.models/jev/scope_probe.json`
- `eval/scope_v1/dataset_v1_2/README.md`
- `eval/scope_v1/pilot_v1_2_summary.md`
- `eval/scope_v1/gate_report_v1_2_last2_oodw2.json`
- `scripts/augment_scope_dataset.py`
- `scripts/finalize_scope_training.py`
