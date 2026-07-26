# base-dense-v3-500m 最终 validation 报告

本报告针对不可变的 `base-dense-v3-500m` final checkpoint (step 1,912, 累计
500,009,962 个训练 token), 在独立、已认证的 validation 语料上完成
candidate/shared/teacher 三角色的全量 NLL 评测。评测全程为 `torch.inference_mode()` 前向:
没有 optimizer state、没有 backward, 也没有任何参数更新。

> **方法学勘误（2026-07-26）**：训练后独立代码审查确认，本轮虽然严格加载了
> Qwen3.5 checkpoint 的 15 张原生 `mtp.*` 参数，并按
> `h_t + embed(x_(t+1)) -> x_(t+2)` 构造辅助目标，但 MTP decoder 的 RoPE
> `position_ids` 错用了 `t`，正确位置应为 `t+1`。因此，本报告记录的 NTP-only
> candidate/shared/teacher validation NLL、PPL、吞吐和训练日志仍是原始实测事实；
> 但 v3 的 MTP 路径不能再称为“完全原生对齐”，也不能据此作 MTP 增益的因果结论。
> 该问题已在 v4 启动前由提交 `c9a08cf` 修复并加入独立位置对齐回归测试。

## 核心结论

| 角色 | 预测 token | 平均 NLL | 困惑度 | wall tok/s |
| --- | --- | --- | --- | --- |
| candidate | 20,009,445 | 2.376669 | 10.7690 | 15,425 |
| shared | 20,009,445 | 2.597585 | 13.4313 | 35,443 |
| teacher | 20,009,445 | 2.066529 | 7.8974 | 7,453.7 |

- teacher gap closed fraction 为 **0.41599**; 项目中 dense gate 的 0.10
  阈值结论为 **通过**。
- 三个角色都覆盖 120 / 120
  个 shard、4,947 条序列; 预测 token 数完全一致,
  因而比较使用的是同一组 label。
- `candidate` 为训练后的 transfer 分支; `shared` 在同一 0.8B backbone 上关闭 transfer
  分支; `teacher` 为冻结的 Qwen3.5-9B 参照模型。

![最终 validation NLL](charts/validation_nll.svg)

![最终 validation 困惑度](charts/validation_perplexity.svg)

![各角色 validation 吞吐](charts/validation_role_throughput.svg)


## 已认证的 baseline 对照

Baseline 为 `base-dense-v2-500m`, 当前结果为 `base-dense-v3-500m`。对照只在以下条件全部
严格相等后才生成: validation prepared-manifest SHA256、dataset fingerprint、序列/输入
token/shard 数, 以及 candidate/shared/teacher 的预测 token 数。因此这里的绝对和相对变化
来自同一 held-out 任务与同一组 label, 不是跨数据集比较。

| 角色 | baseline NLL | 当前 NLL | NLL 绝对变化 | NLL 相对变化 | baseline PPL | 当前 PPL | PPL 绝对变化 | PPL 相对变化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate | 2.377578 | 2.376669 | -0.000910 | -0.038% | 10.7788 | 10.7690 | -0.0098 | -0.091% |
| shared | 2.597585 | 2.597585 | +0.000000 | +0.000% | 13.4313 | 13.4313 | +0.0000 | +0.000% |
| teacher | 2.066529 | 2.066529 | +0.000000 | +0.000% | 7.8974 | 7.8974 | +0.0000 | +0.000% |

| 指标 | baseline | 当前 | 绝对变化 | 相对变化 |
| --- | --- | --- | --- | --- |
| teacher gap closed | 0.414281 | 0.415994 | +0.001713 | +0.413% |

### 同一 held-out 集的跨版本历史

下表与两张图仅使用已认证 lineage 中记录的版本; 每一行都对应完全相同的 prepared
validation manifest 和三角色预测 token 口径。

| run_id | candidate mean NLL | teacher gap closed |
| --- | --- | --- |
| base-dense-v1 | 2.542525 | 10.368% |
| base-dense-v2-500m | 2.377578 | 41.428% |
| base-dense-v3-500m | 2.376669 | 41.599% |

![跨版本 candidate NLL](charts/validation_candidate_nll_history.svg)

![跨版本 teacher gap closed](charts/validation_teacher_gap_closed_history.svg)

NLL/困惑度的负变化表示改善; teacher-gap-closed 的正变化表示改善。Baseline 的
`summary.json` 由相邻 `MANIFEST.json` 清单认证, 该 manifest 再由 `COMPLETE` 认证:

- Baseline summary SHA256: `3bc9a9246dbe4fd46e431f9b0e7f3b27e9e97d72d6557e450b9fa2542dc140b8`
- Baseline report manifest SHA256: `4798a5dc4ef0b456c2ac979976e44f0058b8c707e2b4a68f94d8bf59cba49dc4`
- Baseline checkpoint manifest SHA256: `7318c071507dec896521a669d4b682de0359e43a70dcd992d11a8d1e3ad870d9`
- Baseline evaluation manifest SHA256: `c178d2b92c5ca362cc09ec740988a5555b2382f56d7d528e6a87e58ad39f359e`


## base-dense-v3-500m 训练过程

- 汇总 compute 吞吐为 **6,820.2 tok/s**;
  optimizer-step wall 吞吐为 **6,715.8 tok/s**;
  从 train_start 到 final checkpoint 的端到端吞吐为
  **5,379.0 tok/s**。
- 首次 `train_start` 到 final 的 elapsed 跨度为
  25.821 小时, 恢复感知口径在下方单列。
  共写入 58 个 checkpoint; 写盘总耗时
  17.05 分钟, 平均
  17.63 秒, 最大
  43.13 秒。
- data wait 合计 152.079 秒, 只占 compute-step 时间的
  0.2074%。训练峰值显存为
  27.320 GiB allocated /
  27.559 GiB reserved。
- 裁剪前 grad norm 在 499 /
  1912 步超过阈值 1.000, 比例
  26.10%。hidden-alignment batch 为
  96 步, ordinary batch 为 1816 步。


### 恢复与墙钟口径

- 从首次 `train_start` 到 `train_complete` 的 elapsed wall 为
  **25.821 h**, 对应 **5,379.0
  tok/s**。这个分母有意包含跨 session 停顿、重新初始化及回滚后的重放工作。
- 最终 canonical 日志中已提交 optimizer step 的 active wall 合计
  **20.681 h**, 对应
  **6,715.8 tok/s**。
- 两者相差 **5.140 h**; 该差值还包含 checkpoint 写盘、模型构建/preflight
  等开销, 不能全部解释为 GPU 空闲。
- 生命周期记录 3 个 session、2 次 resume、
  1 次 graceful stop; 1 个 session 没有终止事件。

| 事件 | session | step | token | UTC | 详情 |
| --- | --- | --- | --- | --- | --- |
| session_start | ca1b8eb040194aa19d6882845ba04b2a | n/a | n/a | 2026-07-25T03:39:37.029011+00:00 | n/a |
| initialized | ca1b8eb040194aa19d6882845ba04b2a | 0 | 0 | 2026-07-25T03:45:15.797150+00:00 | fork=step-000000000383-milestone-complete |
| train_start | ca1b8eb040194aa19d6882845ba04b2a | 0 | 0 | 2026-07-25T03:45:15.825070+00:00 | n/a |
| session_start | e14d4c7b82744b77a0acec91b95f0824 | n/a | n/a | 2026-07-25T14:26:04.471664+00:00 | n/a |
| resume | e14d4c7b82744b77a0acec91b95f0824 | 847 | 221,504,027 | 2026-07-25T14:31:55.366697+00:00 | checkpoint=step-000000000847-periodic |
| train_start | e14d4c7b82744b77a0acec91b95f0824 | 847 | 221,504,027 | 2026-07-25T14:31:55.399009+00:00 | n/a |
| graceful_stop | e14d4c7b82744b77a0acec91b95f0824 | 1064 | 278,247,681 | 2026-07-25T16:54:11.593191+00:00 | n/a |
| session_start | aee06a8f9be5456d9d7f9e26ec95b34d | n/a | n/a | 2026-07-25T20:22:51.101559+00:00 | n/a |
| resume | aee06a8f9be5456d9d7f9e26ec95b34d | 1064 | 278,247,681 | 2026-07-25T20:27:47.715453+00:00 | checkpoint=step-000000001064-interrupt-request-000001 |
| train_start | aee06a8f9be5456d9d7f9e26ec95b34d | 1064 | 278,247,681 | 2026-07-25T20:27:47.748750+00:00 | n/a |
| train_complete | aee06a8f9be5456d9d7f9e26ec95b34d | 1912 | 500,009,962 | 2026-07-26T05:34:30.929211+00:00 | checkpoint=step-000000001912-milestone-complete |


- 前 50 步到后 50 步的 token 加权指标: NTP 2.71694 到
  1.89440, teacher KD 0.33295 到
  0.33543, anchor KL 0.06074 到
  0.20352。这些训练 batch 的混合目标有明显采样噪声, 不能替代 held-out NLL。

![训练 loss 分量](charts/training_loss.svg)

![训练 loss 分量 (50-step 滑动平均)](charts/training_loss_smoothed.svg)


![裁剪前 grad norm](charts/gradient_norm.svg)

![学习率曲线](charts/learning_rate.svg)

![训练吞吐](charts/throughput.svg)

![训练显存](charts/gpu_memory.svg)

![Checkpoint 写盘耗时](charts/checkpoint_duration.svg)


## Validation 期间的 GPU 遥测

- 只读采集 4709 个 1 秒样本, 全程 P-state 为
  `{'P0': 1, 'P1': 4489, 'P3': 15, 'P5': 96, 'P8': 108}`; 没有挂载 profiler 或 CUPTI subscriber。
- 功耗平均 **448.31 W**, 范围 31.58 至
  579.75 W, 功耗上限恒为 600 W。
- GPU 利用率平均 77.79% (0 至
  100%), 显存利用率平均 19.29%; 温度平均
  63.31 °C。
- Validation 平均只达到功耗上限的 74.7%, 因此该只读评测没有触及 600 W 功耗墙; 这不代表训练阶段也没有触及。
- 利用率会在 microbatch、词表 head/CE 与 shard 原子提交边界之间波动; 不能仅凭功耗值
  断言算子是否饱和。
- 原始 CSV SHA256: `a26ce774dfd799acec90945b00959faa4fd627340a2b6d72fefd1cd56452e2aa`。

![Validation GPU 功耗](charts/validation_gpu_power.svg)

![Validation GPU 利用率](charts/validation_gpu_utilization.svg)



## 按数据源校正的训练分析

使用不可变 manifest 和确定性 cursor 逐步重放后, 所有 data phase 均与日志一致。
warmup 后的 primary 阶段中,
1,572 /
1,702 个 optimizer batch
(92.36%)
是单一数据源。这个固定窗口同时包含 primary 的 stable 与 cosine-decay 段, 排除 warmup
和 quality cooldown。
下表对该阶段每项指标拟合 `source mix 固定效应 + committed tokens`;
置信区间为 lag 50 的 Newey-West HAC 95% CI。

| 指标 | 每 100M slope | HAC 95% CI | R² | raw SD | 残差 SD |
| --- | --- | --- | --- | --- | --- |
| loss | -0.04206 | [-0.05005, -0.03408] | 0.9837 | 0.9664 | 0.1235 |
| ntp | -0.03122 | [-0.03904, -0.02340] | 0.9847 | 0.9490 | 0.1173 |
| teacher_kd | -0.01157 | [-0.01351, -0.00963] | 0.9263 | 0.1238 | 0.0336 |
| mtp | -0.01454 | [-0.02138, -0.00770] | 0.9819 | 1.1144 | 0.1500 |
| anchor_kl | 0.02179 | [0.01637, 0.02722] | 0.5890 | 0.1232 | 0.0790 |

| 数据源 | 等效 step | total | NTP | KD | MTP |
| --- | --- | --- | --- | --- | --- |
| chinese_fineweb2_cmn_hani | 418.7 | 4.4276 | 3.7498 | 0.2006 | 4.4849 |
| code_github_clean_allowlisted | 260.8 | 1.6645 | 1.0809 | 0.4427 | 1.3195 |
| english_fineweb_edu_dedup | 605.0 | 3.2004 | 2.5979 | 0.2974 | 2.9917 |
| math_finemath_4plus | 250.5 | 2.2627 | 1.6527 | 0.4033 | 1.9336 |
| science_cosmopedia_openstax | 85.8 | 2.5467 | 1.7760 | 0.5225 | 2.3202 |
| science_cosmopedia_stanford | 81.2 | 2.6141 | 1.7404 | 0.6269 | 2.3161 |

total loss 的来源组成解释了 98.37% raw 方差;
控制来源后, 趋势为每 100M token
-0.04206。因此原始曲线接近平台并不等于
"完全没有学习": 不同语料难度和多目标混合遮住了显著为负的 within-source 趋势。
这仍是训练集证据, 不能替代 held-out NLL。

![Raw 与 source-adjusted loss](charts/training_source_adjusted_loss.svg)

![各来源 post-warmup primary loss](charts/training_loss_by_source.svg)


## 按数据源拆分的 validation

| 数据源 | shard | 输入 token | candidate NLL | shared NLL | teacher NLL | gap closed |
| --- | --- | --- | --- | --- | --- | --- |
| chinese_fineweb2_cmn_hani | 30 | 5,001,101 | 3.65619 | 4.37832 | 3.63614 | 0.9730 |
| code_github_clean_allowlisted | 18 | 3,000,084 | 1.02816 | 1.04375 | 0.73577 | 0.0506 |
| english_fineweb_edu_dedup | 42 | 7,000,153 | 2.60331 | 2.62559 | 2.10321 | 0.0426 |
| math_finemath_4plus | 18 | 3,012,338 | 1.57736 | 1.64901 | 1.25676 | 0.1827 |
| science_cosmopedia_openstax | 6 | 1,000,155 | 1.64885 | 1.87425 | 1.31699 | 0.4045 |
| science_cosmopedia_stanford | 6 | 1,000,561 | 1.57293 | 1.73894 | 1.14185 | 0.2780 |

表中 source NLL 按预测 token 加权, 并非对 shard 均值做简单平均。`summary.json` 同时保留每个
shard 的精确 NLL、token 数、source ID 和 candidate 相对 shared 的改善量。

![按 source 的 NLL](charts/validation_source_nll.svg)

![Validation 数据组成](charts/validation_source_tokens.svg)

![逐 shard NLL](charts/validation_shard_nll.svg)

## 欠拟合、MTP 与退火口径

- 是否欠拟合应首先看本次 held-out candidate/shared/teacher 差距, 不能只看训练 loss。
  如果 teacher gap closed 仍低, 并且后续同口径 validation checkpoint sweep 在更多 token 后持续改善,
  才能更有把握地支持“增加数据和 token budget”。本轮没有 validation 时间序列, 因此本次
  单个 final 点不能确定最佳停止位置, 也不能被称为 best checkpoint。
- 本轮 loss 日志中明确启用了 MTP `0.1`；15 张 checkpoint 原生参数保持
  frozen、不进入 optimizer，MTP loss 会经 student hidden state 回传到可训练适配矩阵。
  但其 RoPE 位置存在上述一 token 错位，因此这里只陈述实际执行路径，不再把它表述为
  完全对齐的 Qwen3.5 原生 MTP forward。
- 不可变 LR 日程为: 5,000,000 token 线性 warmup, 随后稳定到 250,000,000 token, 再用最后 250,000,000 token 余弦退火到峰值的 0.100 倍。数据流在 450,000,000 token 同时切换到已认证的 quality-cooldown 语料; 该边界的 loss 跳变与数据分布变化完全混杂, 不能只归因于 LR。

## 数据治理限定

- 当前 lineage 为 `authenticated_extracted_corpus`, 角色为 `validation`;
  **research_only=true**,
  **ready_for_training=false**。
- 尚未完成的审计: `cross_source_near_dedup, full_contextual_pii_scan, project_benchmark_13gram_scan`。在 cross-source near-dedup、完整上下文 PII 扫描和项目
  benchmark 13-gram contamination 扫描完成前, 本结果只能作为研究证据, 不能宣称数据无污染或
  production-ready。

## 完整性、source-tree 与复现限定

- Final checkpoint: `/media/data1/Project/AI/Twen1/runs/base-dense-v3-500m/step-000000001912-milestone-complete`
- Checkpoint manifest SHA256: `ef43670d7c1cbc8ed3908b258659c7426b4cfe10e14c9b7db54968e2481b0e9a`; 清单中的
  4 个文件已逐一重新计算 SHA256。
- Evaluation manifest SHA256: `97b9f59d968fef1aa3a9a0234cac542391151645650969283cd449ac0056dd3f`
- Evaluation PLAN SHA256: `e2f903a91d01f5d550582b4bb7733cad333829693e8462d6a0e789ef6e49f2e7`
- Validation prepared manifest SHA256: `4d3877bfaf4e01551d41913e11d37db08c6097b513ac94ae4a63b8a640b68e7f`
- 运行环境: torch `2.11.0+cu130`、CUDA `13.0`、
  `NVIDIA GeForce RTX 5090`、compute capability `12.0`、
  dtype `bfloat16`、`FLA_TILELANG=0`、
  `CUDA_HOME=/usr/local/cuda-13.2`。`FLA_TILELANG=0` 明确使用 production Triton 路径。
- Checkpoint 加载模式为 `forward_only_lineage_compatible_not_exact_resume`。saved/current exact-training fingerprint 是否
  一致: `True`; saved/current source tree 是否一致:
  `True`。训练时 source tree 为
  `a7aff762d504557a32d0cfbeb7bfde6a812f9303333480255fcbdfec0ca0cdbd`, 评测时为
  `a7aff762d504557a32d0cfbeb7bfde6a812f9303333480255fcbdfec0ca0cdbd`。

这里的结论是: source model、calibration、训练/KD data manifest、loss weights、top_k、run
geometry 和 DCP trainable key/shape 均通过认证。saved/current exact-training fingerprint 与 source tree 均匹配。此次操作仍然只是 forward-only inference validation; 是否允许训练恢复继续由 checkpoint loader 的完整合同判定。

复现时使用绝对 checkpoint 路径最清晰:

```bash
FLA_TILELANG=0 bash scripts/with_cuda_toolchain.sh .venv/bin/python -m twen evaluate nll \
  --config /media/data1/Project/AI/Twen1/runs/base-dense-v3-500m/resolved_config.yaml \
  --checkpoint /media/data1/Project/AI/Twen1/runs/base-dense-v3-500m/step-000000001912-milestone-complete \
  --prepared-manifest /media/data1/Project/AI/Twen1/artifacts/data/base-validation/manifest.json \
  --output /media/data1/Project/AI/Twen1/artifacts/evaluations/base-dense-v3-500m-final-validation \
  --role candidate --role shared --role teacher --batch-size 1 --device cuda:0
```

`MANIFEST.json` 认证中英文报告、`summary.json`、GPU 遥测 CSV 和全部 SVG; `COMPLETE`
再认证该 manifest。
