# Base v2 训练前准备审计

更新日期：2026-07-24。本文记录 Base v2 从数据门到正式启动的闭合状态。

## 状态总览

| Gate | 状态 |
|---|---|
| v1 final validation | 完成；teacher gap closed 10.368% |
| Qwen3.5 原生 MTP | causal SDPA、15 tensor、L-2 target 已接入 |
| folded numerical admission | **FAIL / rejected** |
| expanded selective checkpoint | **PASS / production** |
| canonical RTX 5090 性能 bundle | 完成；expanded B1，6,263.998 tok/s mixture |
| 500M audit/prepared | 完成；516,719,389 tokens |
| 500M top-64 KD | 完成；641 shards / 516,719,389 tokens |
| WSD | 代码与最终参数已锁 |
| 50M quality cooldown | 完成；68 shards / 53,221,680 tokens / 136 hardlinks |
| MTP/LR 决策 | 用户已确认 |
| 最终 v2 YAML/Web profile | 已发布；config SHA `ce8b171b...ea9b` |
| optimizer 训练 | 已由用户授权启动；Web profile running |

## v1 validation 边界

v1 final validation 的 candidate/shared/teacher NLL 为
`2.542525 / 2.597585 / 2.066529`，teacher gap closed `10.368%`。这验证了 final checkpoint，
但 v1 没有 checkpoint 时间序列，因此不能称 validation-selected best。v1 训练时
`losses.mtp=0`，不受后来 MTP causal 修复影响。

## 唯一 production execution contract

canonical report 是
`artifacts/benchmarks/rtx5090-base-dense-utilization-report.json`，SHA256
`cf40eac976767681a704676bee960d8c220d700a847b73d62e1f2358ea15ab38`。

最终运行几何：

```yaml
data:
  micro_batch_size: 1
  global_batch_tokens: 262144
runtime:
  activation_checkpointing: true
  activation_checkpoint_layer_count: 0
  hidden_alignment_activation_checkpoint_layer_count: 8
  dense_transfer_execution: expanded
  dense_transfer_token_checkpoint: true
  dense_transfer_checkpoint_layer_count: 0
  hidden_alignment_dense_transfer_checkpoint_layer_count: 16
  teacher_cpu_offload: true
  activation_checkpointing_on_alignment_only: true
  loss_chunk_tokens: 512
```

即 ordinary outer/inner 0/0、alignment outer/inner 8/16、单卡 accumulation 64。ordinary
约 6,859 tok/s，alignment 图内约 3,902 tok/s、含 staging 约 2,365 tok/s；95%/5%
harmonic mixture 为 **6,263.998 tok/s**。

expanded B2 的安全 ordinary 点约 6,239 tok/s，比 B1 慢 9.964%。它虽然更接近 600 W，
但没有吞吐收益，因此 rejected。旧 B2 folded、ordinary AC3、alignment AC24、B1
`6405.436 tok/s` 都是历史证据，不得再进入 production YAML。

## 数值准入

`differentiable_folded` 在真实 v1 checkpoint、真实 top-64 KD microbatch 和完整累积门中失败：
scale 14/24 层未过，最坏相对误差 203.17%。不得放宽门、伪造 admission 或把它描述为
“待补一个正式 report”；结论已经是 rejected。

`expanded` selective checkpoint 已通过 isolated CUDA bitwise 门、真实 KD 全图 loss bitwise
门、24 层重算门及 72 参数不变门，并把全图 allocated 从 24.453 GiB 降到 17.336 GiB。

## 500M 数据

六来源配额如下：

| 来源 | train quota |
|---|---:|
| FineWeb-Edu-Dedup 英文 | 175M |
| FineWeb2 `cmn_Hani` 中文 | 125M |
| GitHub clean allowlisted code | 75M |
| FineMath-4+ | 75M |
| Cosmopedia Stanford | 25M |
| Cosmopedia OpenStax | 25M |

最终 audit attestation SHA256 为
`27c57994fe8b7e7ac77a69d6c591ca001e6aac0ea3ec0b68981260f3bb7a8ed0`。cross-source exact/near、
contextual PII、project benchmark 13-gram、train-vs-validation exact 五项 gate 全部通过，
findings/rejections 为 0；六来源 train/validation quota 全部通过。

prepared 已完成并通过 full validator：

- 641 shards；
- 126,457 sequences；
- 516,719,389 tokens；
- dataset fingerprint
  `9b4f169f9834d2351ce09bafc34057db115edd1a02b18753e7f3d957cff5c5bb`；
- manifest SHA256
  `9290665ac1e09fbd5b9aea1966bed7a51095bab66f460a0124af4532b1805fd9`；
- `ready_for_training=true`、`research_only=false`、`pending_audits=[]`。

历史 partial 以原目录整体保留，不参与当前 lineage；当前 641 shard 全部绑定 current generator
与 pipeline fingerprint。

## 500M KD：complete

KD 生产点保持 batch 2、4096 context、logits chunk 64、BF16、top-64、temperature 2。
最终 641/641 shards 已生成并独立 index；KD manifest SHA256 为
`242ed2d0fb899cb333939bbd581f8a5632e97228f0c1fda2fee14bea7291efe9`，orchestration
MANIFEST/COMPLETE SHA 链与 generation/index exit 0 均已认证。

动态进度只引用
`artifacts/data/base-v2-500m-kd-orchestration/status.json`。在以下文件全部存在并通过 SHA
认证前，KD gate 保持未完成：

- `artifacts/data/base-v2-500m-kd/manifest.json`；
- `artifacts/data/base-v2-500m-kd-orchestration/MANIFEST.json`；
- `artifacts/data/base-v2-500m-kd-orchestration/COMPLETE`。

失败、STOP 或 partial shard 都不会写总 `COMPLETE`。不要删除 `.incomplete` 或已完成 shard。

## WSD 与六来源 quality cooldown

500M scheduler 已锁为：

- 0–5M：linear warmup；
- 5M–450M：保持 peak LR；
- 450M–500M：cosine decay；
- 500M：达到 peak 的 0.1 倍。

末段 50M 使用独立 high-quality whole-shard view。六来源 policy 目标为
15M / 15M / 7.5M / 5M / 5M / 2.5M；生成器按固定 seed/hash 和 source quota 选择 whole shard，
primary/cooldown 使用双 cursor，resume 与 world-size 变化保持确定性。prepared 与 KD tensor
通过 hardlink 复用，不重跑 teacher。

真实 policy 已绑定最终 primary prepared/KD manifest SHA，selection-plan SHA 为
`e245ab483724c4c1c3dbf54990e8ac87117757709bd7de6d20fc0c0903ee9b16`；cooldown
prepared/KD manifest 与 bundle `COMPLETE` 已生成并通过闭合树、SHA、inode 审计。

## 已确认的训练超参数

用户已经确认：

```text
MTP loss weight = 0.1
adapter peak LR = 2e-4
router peak LR  = 1e-3
LoRA peak LR    = 2e-4
scale peak LR   = 1e-3
```

这些值与 WSD、execution contract、数据/KD/cooldown SHA 一起属于 resume-critical fingerprint。

## 最终发布与 Web

fail-closed finalizer 已认证完整 KD、真实 cooldown、canonical expanded report、expanded PASS、
folded FAIL、source 与 v1 fork checkpoint，并发布独立 `base-dense-v2-500m` config/evidence/profile。
首次运行使用 `resume=none` fork v1 final，checkpoint 策略为 100 step / 30 分钟。

Dashboard 已监听 `0.0.0.0:8765`，100 ms GPU 采样、约 1 秒 UI 刷新；v1 profile monitor-only，
v2 profile 已启用并显示 running。若以后改为 systemd user service，退出所有会话后自动拉起仍需：

```bash
sudo loginctl enable-linger "$USER"
```

## 当前运行

上述七步已全部闭合。正式 run 的动态状态只读
`runs/base-dense-v2-500m/{rank0-session.json,events.jsonl,metrics.jsonl,telemetry.jsonl}`；停止后
Web 会显示“可恢复”，再次启动时完整认证 checkpoint 后自动切换为 `--resume auto`，不再重复 fork。
