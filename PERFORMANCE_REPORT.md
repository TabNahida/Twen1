# Twen1 RTX 5090 性能、功耗与显存报告

更新时间：2026-07-19（Asia/Shanghai）

本文的唯一 canonical 性能来源是：

- `artifacts/benchmarks/rtx5090-base-dense-utilization-report.json`，SHA256
  `cf40eac976767681a704676bee960d8c220d700a847b73d62e1f2358ea15ab38`；
- 同名前缀的 Markdown、三张 SVG、`MANIFEST.json` 和 `COMPLETE`；
- benchmark 源码 SHA256
  `93c9610cbced74111f554f0306d1ef4ebecc537f767e5a194b53d4bca821abaa`。

下文提到的旧 AC4/AC24、`6405.436 tok/s`、B2 folded 等数据只保留为历史诊断，均不再是
production 配置。

## 当前结论

单张 RTX 5090、4096 context、top-64 KD、原生 Qwen3.5 MTP、teacher split offload、1.5 GiB
Adam-moment 等价占位下，正式选择为：

| 图 | physical batch | expanded outer / inner | production 吞吐 | 最小估算余量 |
|---|---:|---:|---:|---:|
| ordinary | 1 | 0 / 0 | 约 6,859 tok/s | 3.948 GiB |
| hidden alignment | 1 | 8 / 16 | 图内约 3,902 tok/s；含 staging 约 2,365 tok/s | 3.513 GiB |

按 95% ordinary / 5% alignment 做 harmonic mixture，canonical 结果是
**6,263.998 tok/s**。训练配置固定：

```yaml
data:
  micro_batch_size: 1
  global_batch_tokens: 262144
runtime:
  dense_transfer_execution: expanded
  activation_checkpointing: true
  activation_checkpointing_on_alignment_only: true
  activation_checkpoint_layer_count: 0
  hidden_alignment_activation_checkpoint_layer_count: 8
  dense_transfer_token_checkpoint: true
  dense_transfer_checkpoint_layer_count: 0
  hidden_alignment_dense_transfer_checkpoint_layer_count: 16
  loss_chunk_tokens: 512
```

单卡 accumulation 因而是 `262144 / (1 × 4096) = 64`。outer 与 inner 层集合严格互斥；
alignment 的 8 个 outer 层和其余 16 个 inner 层由 resolver 确定性选择，并由 preflight 与
runtime actual-state 回读共同认证。

所有正式 case 都满足：72/72 trainable gradient finite、全部启用的 loss finite、原生 MTP
causal SDPA、teacher CPU offload、`no_optimizer_created=true`、`no_optimizer_steps=true`。
性能探针没有启动训练。

## 为什么不选 B2

expanded B2 ordinary 的安全点是 outer 0 / inner 24，约 **6,239 tok/s**，比 canonical B1
慢 **9.964%**。B2 outer 4 / inner 4 曾 OOM；outer 8 / inner 8 虽能运行，但物理余量只有约
1.68 GiB，低于 3 GiB 门。B2 可作为功耗/调度对照，不能作为生产推荐。

B2 的高利用率采样更接近板卡 600 W 上限：约 471 W、峰值 584.20 W；B1 ordinary 约
444 W、峰值 476.83 W。更高功耗并没有换来更高吞吐，所以不能用“瓦数更高”替代
端到端 tok/s、正确性和显存余量门。

## 400 W 读数的解释

5090 不存在 400 W 功耗墙。独立 alignment 稳态复核测得约 591–595 W，峰值约 604 W；GPU
work union busy 为 93.765%，且没有大于等于 1 ms 的 GPU 空洞。BF16 Tensor GEMM 占主要
kernel 时间。

早期 Web 页面约 2 秒一次的轮询会漏掉约 1 秒 step 的峰值，因此看到 400 W 左右不代表 GPU
没吃满。当前 Dashboard 使用单个 `nvidia-smi --loop-ms=100` 采样进程，UI 每 1 秒刷新；正式
benchmark 也保存 100 ms telemetry。报告功耗时应同时给出采样周期、active/high-util window、
均值、p95 和峰值。

Nsight Compute 仍被宿主 `ERR_NVGPUCTRPERM` 阻挡，本文不伪造 Tensor pipe、occupancy、DRAM
SOL 或 cache 命中率。

## 显存账本：optimizer 很小不等于整图显存很小

用户关于 optimizer 的判断是正确的：冻结 0.8B/9B/MTP 权重不会产生权重梯度或 Adam moments。
但是它们仍需作为 forward/backward 输入驻留或分阶段 stage，且为了求 A/B 梯度，反向仍要穿过
冻结 donor FFN。

A/B 不是 rank-16 LoRA，而是每层完整的 `4096×1024` 跨宽度映射：

| 训练状态 | 约占用 |
|---|---:|
| 48 张 A/B + 24 scale，FP32 参数 | 0.75 GiB |
| 对应 FP32 gradient | 0.75 GiB |
| Adam 一、二阶 moments | 1.50 GiB |
| 参数 + gradient + moments | 3.00 GiB |

ordinary 静态账本还包括约 1.401 GiB 的 0.8B text body、6.750 GiB 的 mapped donor FFN，以及
CUDA workspace/allocator。teacher-exclusive 约 8.033 GiB 保留在 CPU，只有 alignment batch
才 stage 到 GPU；mapped donor 与 teacher MLP 使用 exact alias，不复制第二份 6.75 GiB。

真正随 checkpoint 策略变化的是 4K 激活、hidden tuple、MTP body、词表 loss 临时平面及其
backward saved tensor，而不是 optimizer moments。expanded selective checkpoint 的真实 KD
全图验收把 allocated 从 24.453 GiB 降到 17.336 GiB，同时保持 output/input/A/B/scale、全图
loss 和 24 层重算门 bitwise 通过。这正是 alignment 采用 outer 8 / inner 16 的原因。

训练 DCP 只保存 72 个 trainable tensor；冻结 backbone、donor、tied head、channel map 和
15 个 MTP source tensor从锁定 source checkpoint 重载。

## 原生 Qwen3.5 MTP

当前实现严格加载 checkpoint 顶层的 15 个 `mtp.*` tensor，使用原生单层 decoder、共享
embedding/LM head、`L-2` target 与 causal SDPA。MTP 参数保持 frozen/eval，不进入 optimizer，
但 loss 会沿 main hidden state 回传 A/B。

旧 v1 训练的 `losses.mtp=0`，所以旧 checkpoint 没有训练 MTP；v1 validation 不受后来 causal
修复影响。用户已为 v2 明确选择 `losses.mtp=0.1`。该值是项目实验选择，不是 Qwen3.5 官方默认，
并且是 resume-critical 配置。

## Profiler 结论

清洗后的 canonical Nsight 针对 B1 ordinary expanded 0/0：dense GEMM 是第一热点，compiled
Triton reduction、FLA recurrent attention 与 MTP SDPA flash 明显更小。数据等待不是瓶颈；
旧 run 的 `data_wait_fraction` 约 0.02%。

分块词表 loss 曾是碎片化热点，compiled reduction 已消除主要重复归约。把 chunk 直接增到
1024 会制造约 6 GiB 额外临时平面并拖慢 backward；当前 v2 使用已经通过完整图容量门的
`loss_chunk_tokens=512`。

raw `.nsys-rep`/SQLite 可能包含进程环境，只按本地机密处理。正式 bundle 只包含清洗摘要；
修复前的 eager/non-causal MTP trace 均标为 `INVALID_SUPERSEDED`。

## 数值准入：folded 拒绝，expanded 通过

`differentiable_folded` 在真实 v1 checkpoint、4 个真实 top-64 KD microbatch、完整累积门下失败：
scale 14/24 层未过，最坏相对误差 **203.17%**。它只能保留为 experimental/rejected 路径，
不得生成 production admission，也不得出现在最终 v2 配置。

`expanded` selective checkpoint 则正式通过：isolated CUDA 的 output/input/A/B/scale bitwise
一致，真实 KD 全图 loss bitwise 一致，24 层 token core 都按约定重算，72 个参数未变化。

证据：

- `artifacts/audits/differentiable-fold-numerical-admission/FULL_GRAPH_V1_REAL_KD_ACCUMULATION_REPORT.md`
- `artifacts/audits/differentiable-fold-numerical-admission/EXPANDED_SELECTIVE_CHECKPOINT_FULL_GRAPH_REPORT.md`

历史 B2 folded、ordinary AC3、alignment AC24 与 `6405.436 tok/s` 记录仍可用于解释优化过程，
但结论均为 **rejected/superseded**。

## 当前训练前状态

- v1 final validation 已完成：candidate/shared/teacher NLL 为
  `2.542525 / 2.597585 / 2.066529`，teacher gap closed `10.368%`；它验证 final checkpoint，
  但没有多 checkpoint 时间序列，因此不能称 validation-selected best。
- 500M prepared 已完成：641 shards、126,457 sequences、516,719,389 tokens，manifest SHA256
  `9290665ac1e09fbd5b9aea1966bed7a51095bab66f460a0124af4532b1805fd9`。
- 500M top-64 KD 的 attempt 4 正在 optimizer-free 后台运行；当前没有最终 KD manifest、
  orchestration `COMPLETE`，不能写成 KD 已完成。
- WSD 与 50M 六来源 quality-cooldown policy/materializer 代码已就绪；真实 policy 必须在完整 KD
  后按真实 parent SHA 生成，当前尚未发布真实 policy/cooldown bundle。
- 用户已选择 MTP `0.1` 与 peak LR：adapter/LoRA `2e-4`，router/scale `1e-3`。
- 正式训练仍未启动，并且只能由用户按自己的时间启动。
