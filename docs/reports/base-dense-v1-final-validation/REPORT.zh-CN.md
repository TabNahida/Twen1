# Base Dense v1 最终 validation 报告

本报告针对不可变的 v1 final checkpoint (step 383, 累计
100,151,046 个训练 token), 在独立、已认证的 validation 语料上完成
candidate/shared/teacher 三角色的全量 NLL 评测。评测全程为 `torch.inference_mode()` 前向:
没有 optimizer state、没有 backward, 也没有任何参数更新。

## 核心结论

| 角色 | 预测 token | 平均 NLL | 困惑度 | wall tok/s |
| --- | --- | --- | --- | --- |
| candidate | 20,009,445 | 2.542525 | 12.7117 | 14,663 |
| shared | 20,009,445 | 2.597585 | 13.4313 | 33,413 |
| teacher | 20,009,445 | 2.066529 | 7.8974 | 9,394.2 |

- teacher gap closed fraction 为 **0.10368**; 项目中 dense gate 的 0.10
  阈值结论为 **通过**。
- 三个角色都覆盖 120 / 120
  个 shard、4,947 条序列; 预测 token 数完全一致,
  因而比较使用的是同一组 label。
- `candidate` 为训练后的 transfer 分支; `shared` 在同一 0.8B backbone 上关闭 transfer
  分支; `teacher` 为冻结的 Qwen3.5-9B 参照模型。

![最终 validation NLL](charts/validation_nll.svg)

![最终 validation 困惑度](charts/validation_perplexity.svg)

![各角色 validation 吞吐](charts/validation_role_throughput.svg)

## v1 训练过程

- 汇总 compute 吞吐为 **4,438.6 tok/s**;
  optimizer-step wall 吞吐为 **4,290.1 tok/s**;
  从 train_start 到 final checkpoint 的端到端吞吐为
  **4,286.0 tok/s**。
- 总墙钟 6.491 小时, 共写入
  38 个 checkpoint; 写盘总耗时
  12.85 分钟, 平均
  20.28 秒, 最大
  24.20 秒。
- data wait 合计 4.455 秒, 只占 compute-step 时间的
  0.0197%。训练峰值显存为
  22.388 GiB allocated /
  22.654 GiB reserved。
- 裁剪前 grad norm 在 262 /
  383 步超过阈值 1.000, 比例
  68.41%。hidden-alignment batch 为
  20 步, ordinary batch 为 363 步。
- 前 50 步到后 50 步的 token 加权指标: NTP 2.46595 到
  2.45797, teacher KD 0.47200 到
  0.38491, anchor KL 0.02312 到
  0.06540。这些训练 batch 的混合目标有明显采样噪声, 不能替代 held-out NLL。

![训练 loss 分量](charts/training_loss.svg)

![裁剪前 grad norm](charts/gradient_norm.svg)

![学习率曲线](charts/learning_rate.svg)

![训练吞吐](charts/throughput.svg)

![训练显存](charts/gpu_memory.svg)

![Checkpoint 写盘耗时](charts/checkpoint_duration.svg)


## Validation 期间的 GPU 遥测

- 只读采集 25 个 1 秒样本, 全程 P-state 为
  `{'P1': 25}`; 没有挂载 profiler 或 CUPTI subscriber。
- 功耗平均 **441.51 W**, 范围 320.77 至
  464.83 W, 功耗上限恒为 600 W。
- GPU 利用率平均 78.08% (18 至
  99%), 显存利用率平均 25.08%; 温度平均
  69.44 °C。
- 因此“约 400 W”这一观察属实, 但当前 **没有触及 600 W 功耗墙**。利用率在
  microbatch、词表 head/CE 与 shard 原子提交边界之间波动, 说明 batch2 值得单独做固定
  shard A/B; 不能仅凭功耗值断言算子已经饱和。
- 原始 CSV SHA256: `01420465a766a8ca46d5ca54e7bbd1065c178a983708daa918ccf1b320932ffb`。

![Validation GPU 功耗](charts/validation_gpu_power.svg)

![Validation GPU 利用率](charts/validation_gpu_utilization.svg)


## 按数据源拆分的 validation

| 数据源 | shard | 输入 token | candidate NLL | shared NLL | teacher NLL | gap closed |
| --- | --- | --- | --- | --- | --- | --- |
| chinese_fineweb2_cmn_hani | 30 | 5,001,101 | 4.20722 | 4.37832 | 3.63614 | 0.2305 |
| code_github_clean_allowlisted | 18 | 3,000,084 | 1.04101 | 1.04375 | 0.73577 | 0.0089 |
| english_fineweb_edu_dedup | 42 | 7,000,153 | 2.61885 | 2.62559 | 2.10321 | 0.0129 |
| math_finemath_4plus | 18 | 3,012,338 | 1.62777 | 1.64901 | 1.25676 | 0.0541 |
| science_cosmopedia_openstax | 6 | 1,000,155 | 1.80531 | 1.87425 | 1.31699 | 0.1237 |
| science_cosmopedia_stanford | 6 | 1,000,561 | 1.68099 | 1.73894 | 1.14185 | 0.0971 |

表中 source NLL 按预测 token 加权, 并非对 shard 均值做简单平均。`summary.json` 同时保留每个
shard 的精确 NLL、token 数、source ID 和 candidate 相对 shared 的改善量。

![按 source 的 NLL](charts/validation_source_nll.svg)

![Validation 数据组成](charts/validation_source_tokens.svg)

![逐 shard NLL](charts/validation_shard_nll.svg)

## 欠拟合、MTP 与退火口径

- 是否欠拟合应首先看本次 held-out candidate/shared/teacher 差距, 不能只看训练 loss。
  如果 teacher gap closed 仍低, 并且 v2 的 validation checkpoint sweep 在更多 token 后持续改善,
  才能更有把握地支持“增加数据和 token budget”。v1 训练期间没有 validation 时间序列, 因此本次
  单个 final 点不能确定最佳停止位置, 也不能被称为 best checkpoint。
- v1 的不可变 resolved config 中 MTP loss 权重为 **0.0**, 而且
  `mtp` 字段在原始 loss 配置中不存在。
  所以本次结果必须表述为 **MTP=0、没有训练 Qwen3.5 原生 MTP 头的 v1**。
- v1 实际使用 10,000,000 warmup token 和旧版 cosine decay,
  总预算 100,000,000 token; 它没有单独的 WSD stable phase 或预先定义的
  cooldown/annealing phase。v2 可以新增这些实验设计, 但不能把 v1 结果追溯描述为使用了新策略。

## 数据治理限定

- 当前 lineage 为 `authenticated_extracted_corpus`, 角色为 `validation`;
  **research_only=true**,
  **ready_for_training=false**。
- 尚未完成的审计: `cross_source_near_dedup, full_contextual_pii_scan, project_benchmark_13gram_scan`。在 cross-source near-dedup、完整上下文 PII 扫描和项目
  benchmark 13-gram contamination 扫描完成前, 本结果只能作为研究证据, 不能宣称数据无污染或
  production-ready。

## 完整性、source-tree 与复现限定

- Final checkpoint: `/media/data1/Project/AI/Twen1/runs/base-dense-v1/step-000000000383-milestone-complete`
- Checkpoint manifest SHA256: `8f1da11708cc2b4c05cadc11106b9f46484a77c472511e8953d43cb82018ae29`; 清单中的
  4 个文件已逐一重新计算 SHA256。
- Evaluation manifest SHA256: `00bdd9a7211cc278f9eaa4afaceb715241d2af6005a6df8ea525fb683b680d9b`
- Evaluation PLAN SHA256: `cab35495461902531e9c1440b04630bac6cbf65fb5b6cc591aac1801ace5f9b9`
- Validation prepared manifest SHA256: `4d3877bfaf4e01551d41913e11d37db08c6097b513ac94ae4a63b8a640b68e7f`
- 运行环境: torch `2.11.0+cu130`、CUDA `13.0`、
  `NVIDIA GeForce RTX 5090`、compute capability `12.0`、
  dtype `bfloat16`、`FLA_TILELANG=0`、
  `CUDA_HOME=/usr/local/cuda-13.2`。`FLA_TILELANG=0` 明确使用 production Triton 路径。
- Checkpoint 加载模式为 `forward_only_lineage_compatible_not_exact_resume`。saved/current exact-training fingerprint 是否
  一致: `False`; saved/current source tree 是否一致:
  `False`。训练时 source tree 为
  `4775fc259454c410e610c9349bab816a3622dc97b9d4877cff280cf05028947c`, 评测时为
  `8197ce12d1d553ec94ed28716388b69b609bd0e2b5626c9c2c275bf823912f17`。

这里的结论是: source model、calibration、训练/KD data manifest、loss weights、top_k、run
geometry 和 DCP trainable key/shape 均通过认证, 因此允许 **forward-only inference validation**;
但 source tree 已变化, **绝对不能把它表述为 exact-resume-compatible, 也不能据此继续旧 run 训练**。
默认 export/recovery/fold 与任何 optimizer resume 仍保持严格 fingerprint gate。

复现时使用绝对 checkpoint 路径最清晰:

```bash
FLA_TILELANG=0 bash scripts/with_cuda_toolchain.sh .venv/bin/python -m twen evaluate nll \
  --config runs/base-dense-v1/resolved_config.yaml \
  --checkpoint /media/data1/Project/AI/Twen1/runs/base-dense-v1/step-000000000383-milestone-complete \
  --prepared-manifest artifacts/data/base-validation/manifest.json \
  --output artifacts/evaluations/base-dense-v1-final-validation \
  --role candidate --role shared --role teacher --batch-size 1 --device cuda:0
```

`MANIFEST.json` 认证中英文报告、`summary.json`、GPU 遥测 CSV 和全部 SVG; `COMPLETE`
再认证该 manifest。
