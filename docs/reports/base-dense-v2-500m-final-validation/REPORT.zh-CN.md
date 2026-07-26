# base-dense-v2-500m 最终 validation 报告

本报告针对不可变的 `base-dense-v2-500m` final checkpoint (step 1,912, 累计
500,009,962 个训练 token), 在独立、已认证的 validation 语料上完成
candidate/shared/teacher 三角色的全量 NLL 评测。评测全程为 `torch.inference_mode()` 前向:
没有 optimizer state、没有 backward, 也没有任何参数更新。

## 核心结论

| 角色 | 预测 token | 平均 NLL | 困惑度 | wall tok/s |
| --- | --- | --- | --- | --- |
| candidate | 20,009,445 | 2.377578 | 10.7788 | 15,050 |
| shared | 20,009,445 | 2.597585 | 13.4313 | 35,090 |
| teacher | 20,009,445 | 2.066529 | 7.8974 | 9,957.8 |

- teacher gap closed fraction 为 **0.41428**; 项目中 dense gate 的 0.10
  阈值结论为 **通过**。
- 三个角色都覆盖 120 / 120
  个 shard、4,947 条序列; 预测 token 数完全一致,
  因而比较使用的是同一组 label。
- `candidate` 为训练后的 transfer 分支; `shared` 在同一 0.8B backbone 上关闭 transfer
  分支; `teacher` 为冻结的 Qwen3.5-9B 参照模型。

![最终 validation NLL](charts/validation_nll.svg)

![最终 validation 困惑度](charts/validation_perplexity.svg)

![各角色 validation 吞吐](charts/validation_role_throughput.svg)


## 已认证的 v1 baseline 对照

Baseline 为 `base-dense-v1`, 当前结果为 `base-dense-v2-500m`。对照只在以下条件全部
严格相等后才生成: validation prepared-manifest SHA256、dataset fingerprint、序列/输入
token/shard 数, 以及 candidate/shared/teacher 的预测 token 数。因此这里的绝对和相对变化
来自同一 held-out 任务与同一组 label, 不是跨数据集比较。

| 角色 | baseline NLL | 当前 NLL | NLL 绝对变化 | NLL 相对变化 | baseline PPL | 当前 PPL | PPL 绝对变化 | PPL 相对变化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate | 2.542525 | 2.377578 | -0.164946 | -6.488% | 12.7117 | 10.7788 | -1.9330 | -15.206% |
| shared | 2.597585 | 2.597585 | +0.000000 | +0.000% | 13.4313 | 13.4313 | +0.0000 | +0.000% |
| teacher | 2.066529 | 2.066529 | +0.000000 | +0.000% | 7.8974 | 7.8974 | +0.0000 | +0.000% |

| 指标 | baseline | 当前 | 绝对变化 | 相对变化 |
| --- | --- | --- | --- | --- |
| teacher gap closed | 0.103680 | 0.414281 | +0.310601 | +299.575% |

NLL/困惑度的负变化表示改善; teacher-gap-closed 的正变化表示改善。Baseline 的
`summary.json` 由相邻 `MANIFEST.json` 清单认证, 该 manifest 再由 `COMPLETE` 认证:

- Baseline summary SHA256: `ecf4f7b9a8bef4d4a34fba9bb2e4e9b330f243a1b6a22cb9d9f30d5c4a6949a7`
- Baseline report manifest SHA256: `1e79536e8252031083bd1af20175ace5e3f33848462563a34202f7b584e5b521`
- Baseline checkpoint manifest SHA256: `8f1da11708cc2b4c05cadc11106b9f46484a77c472511e8953d43cb82018ae29`
- Baseline evaluation manifest SHA256: `00bdd9a7211cc278f9eaa4afaceb715241d2af6005a6df8ea525fb683b680d9b`


## base-dense-v2-500m 训练过程

- 汇总 compute 吞吐为 **6,765.4 tok/s**;
  optimizer-step wall 吞吐为 **6,649.3 tok/s**;
  从 train_start 到 final checkpoint 的端到端吞吐为
  **5,769.6 tok/s**。
- 总墙钟 24.073 小时, 共写入
  58 个 checkpoint; 写盘总耗时
  19.04 分钟, 平均
  19.69 秒, 最大
  53.94 秒。
- data wait 合计 175.667 秒, 只占 compute-step 时间的
  0.2377%。训练峰值显存为
  27.320 GiB allocated /
  27.539 GiB reserved。
- 裁剪前 grad norm 在 678 /
  1912 步超过阈值 1.000, 比例
  35.46%。hidden-alignment batch 为
  96 步, ordinary batch 为 1816 步。
- 前 50 步到后 50 步的 token 加权指标: NTP 2.71679 到
  1.89735, teacher KD 0.33309 到
  0.34138, anchor KL 0.06087 到
  0.20995。这些训练 batch 的混合目标有明显采样噪声, 不能替代 held-out NLL。

![训练 loss 分量](charts/training_loss.svg)

![训练 loss 分量 (50-step 滑动平均)](charts/training_loss_smoothed.svg)


![裁剪前 grad norm](charts/gradient_norm.svg)

![学习率曲线](charts/learning_rate.svg)

![训练吞吐](charts/throughput.svg)

![训练显存](charts/gpu_memory.svg)

![Checkpoint 写盘耗时](charts/checkpoint_duration.svg)


## Validation 期间的 GPU 遥测

- 只读采集 4158 个 1 秒样本, 全程 P-state 为
  `{'P1': 3905, 'P3': 10, 'P5': 3, 'P8': 240}`; 没有挂载 profiler 或 CUPTI subscriber。
- 功耗平均 **441.97 W**, 范围 26.61 至
  576.82 W, 功耗上限恒为 600 W。
- GPU 利用率平均 73.69% (0 至
  100%), 显存利用率平均 19.26%; 温度平均
  62.33 °C。
- Validation 平均只达到功耗上限的 73.7%, 因此该只读评测没有触及 600 W 功耗墙; 这不代表训练阶段也没有触及。
- 利用率会在 microbatch、词表 head/CE 与 shard 原子提交边界之间波动; 不能仅凭功耗值
  断言算子是否饱和。
- 原始 CSV SHA256: `588d14ddf1103a8623d61a49fe2a5809356158e3faa7454c1bce0924fccd1ccd`。

![Validation GPU 功耗](charts/validation_gpu_power.svg)

![Validation GPU 利用率](charts/validation_gpu_utilization.svg)



## 按数据源校正的训练分析

使用不可变 manifest 和确定性 cursor 逐步重放后, 所有 data phase 均与日志一致。
warmup 后的 primary 阶段中,
1,572 /
1,702 个 optimizer batch
(92.36%) 是单一数据源。
下表对该阶段每项指标拟合 `source mix 固定效应 + committed tokens`;
置信区间为 lag 50 的 Newey-West HAC 95% CI。

| 指标 | 每 100M slope | HAC 95% CI | R² | raw SD | 残差 SD |
| --- | --- | --- | --- | --- | --- |
| loss | -0.04089 | [-0.04892, -0.03285] | 0.9835 | 0.9630 | 0.1236 |
| ntp | -0.03099 | [-0.03886, -0.02313] | 0.9846 | 0.9464 | 0.1174 |
| teacher_kd | -0.01064 | [-0.01272, -0.00856] | 0.9255 | 0.1244 | 0.0339 |
| mtp | -0.01502 | [-0.02189, -0.00815] | 0.9819 | 1.1125 | 0.1496 |
| anchor_kl | 0.02242 | [0.01680, 0.02804] | 0.5833 | 0.1287 | 0.0831 |

| 数据源 | 等效 step | total | NTP | KD | MTP |
| --- | --- | --- | --- | --- | --- |
| chinese_fineweb2_cmn_hani | 418.7 | 4.4223 | 3.7434 | 0.2015 | 4.4805 |
| code_github_clean_allowlisted | 260.8 | 1.6705 | 1.0822 | 0.4472 | 1.3207 |
| english_fineweb_edu_dedup | 605.0 | 3.2002 | 2.5975 | 0.2979 | 2.9908 |
| math_finemath_4plus | 250.5 | 2.2629 | 1.6523 | 0.4035 | 1.9335 |
| science_cosmopedia_openstax | 85.8 | 2.5490 | 1.7751 | 0.5252 | 2.3202 |
| science_cosmopedia_stanford | 81.2 | 2.6150 | 1.7405 | 0.6275 | 2.3161 |

total loss 的来源组成解释了 98.35% raw 方差;
控制来源后, 趋势为每 100M token
-0.04089。因此原始曲线接近平台并不等于
"完全没有学习": 不同语料难度和多目标混合遮住了显著为负的 within-source 趋势。
这仍是训练集证据, 不能替代 held-out NLL。

![Raw 与 source-adjusted loss](charts/training_source_adjusted_loss.svg)

![各来源 stable-primary loss](charts/training_loss_by_source.svg)


## 按数据源拆分的 validation

| 数据源 | shard | 输入 token | candidate NLL | shared NLL | teacher NLL | gap closed |
| --- | --- | --- | --- | --- | --- | --- |
| chinese_fineweb2_cmn_hani | 30 | 5,001,101 | 3.66014 | 4.37832 | 3.63614 | 0.9677 |
| code_github_clean_allowlisted | 18 | 3,000,084 | 1.02839 | 1.04375 | 0.73577 | 0.0499 |
| english_fineweb_edu_dedup | 42 | 7,000,153 | 2.60281 | 2.62559 | 2.10321 | 0.0436 |
| math_finemath_4plus | 18 | 3,012,338 | 1.57178 | 1.64901 | 1.25676 | 0.1969 |
| science_cosmopedia_openstax | 6 | 1,000,155 | 1.65269 | 1.87425 | 1.31699 | 0.3976 |
| science_cosmopedia_stanford | 6 | 1,000,561 | 1.58724 | 1.73894 | 1.14185 | 0.2541 |

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
- 本轮明确启用了 **Qwen3.5 原生 MTP**, loss 权重为 **0.1**。MTP 源头的 15 张参数保持 frozen、不进入 optimizer, 但 MTP loss 会经 student hidden state 回传到可训练适配矩阵。
- 不可变 LR 日程为: 5,000,000 token 线性 warmup, 随后稳定到 450,000,000 token, 再用最后 50,000,000 token 余弦退火到峰值的 0.100 倍。数据流在 450,000,000 token 同时切换到已认证的 quality-cooldown 语料; 该边界的 loss 跳变与数据分布变化完全混杂, 不能只归因于 LR。

## 数据治理限定

- 当前 lineage 为 `authenticated_extracted_corpus`, 角色为 `validation`;
  **research_only=true**,
  **ready_for_training=false**。
- 尚未完成的审计: `cross_source_near_dedup, full_contextual_pii_scan, project_benchmark_13gram_scan`。在 cross-source near-dedup、完整上下文 PII 扫描和项目
  benchmark 13-gram contamination 扫描完成前, 本结果只能作为研究证据, 不能宣称数据无污染或
  production-ready。

## 完整性、source-tree 与复现限定

- Final checkpoint: `/media/data1/Project/AI/Twen1/runs/base-dense-v2-500m/step-000000001912-milestone-complete`
- Checkpoint manifest SHA256: `7318c071507dec896521a669d4b682de0359e43a70dcd992d11a8d1e3ad870d9`; 清单中的
  4 个文件已逐一重新计算 SHA256。
- Evaluation manifest SHA256: `c178d2b92c5ca362cc09ec740988a5555b2382f56d7d528e6a87e58ad39f359e`
- Evaluation PLAN SHA256: `bfc4055716d4b598a42bd5ba858771e34005f12fe9ad59f78c17987c6b5a6c22`
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
  --config /media/data1/Project/AI/Twen1/runs/base-dense-v2-500m/resolved_config.yaml \
  --checkpoint /media/data1/Project/AI/Twen1/runs/base-dense-v2-500m/step-000000001912-milestone-complete \
  --prepared-manifest /media/data1/Project/AI/Twen1/artifacts/data/base-validation/manifest.json \
  --output /media/data1/Project/AI/Twen1/artifacts/evaluations/base-dense-v2-500m-final-validation \
  --role candidate --role shared --role teacher --batch-size 1 --device cuda:0
```

`MANIFEST.json` 认证中英文报告、`summary.json`、GPU 遥测 CSV 和全部 SVG; `COMPLETE`
再认证该 manifest。
