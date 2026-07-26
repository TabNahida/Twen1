# Twen1 Base 训练就绪记录

更新时间：2026-07-24（Asia/Shanghai）

## 当前结论

旧 `base-dense-v1` optimizer 训练及其 final validation 已完成。Base v2 的 500M prepared/KD、
真实 quality-cooldown、性能/功耗/数值门、finalizer、preflight/dry-run/graph-smoke 和 Web 管理面
也已全部通过。用户于 2026-07-24 明确授权后，`base-dense-v2-500m` 已从 v1 final fork 并进入
正式 optimizer 训练；动态真值以 `runs/base-dense-v2-500m/` 中的 session/events/metrics 为准。

## v1 final 与 validation

- run：`runs/base-dense-v1`；
- final：`step-000000000383-milestone-complete`；
- 383 optimizer steps，100,151,046 committed input tokens；
- checkpoint `COMPLETE` 与逐文件 SHA256 已通过；
- v1 resolved config、checkpoint 和日志均没有 MTP loss/指标；`losses.mtp=0`；
- ordinary step 平均约 4,523 tok/s，完整墙钟约 4.29k tok/s；
- 262/383 step 的裁剪前 grad norm 大于 1。

final validation 已在 120/120 个认证 shard、20,009,445 个共同预测 token 上完成：

| 模型 | NLL |
|---|---:|
| candidate | 2.542525 |
| shared-only | 2.597585 |
| teacher | 2.066529 |

teacher gap closed 为 **10.368%**，通过项目 10% dense gate。报告位于
`artifacts/reports/base-dense-v1-final-validation/REPORT.zh-CN.md`。由于 v1 没有多个
checkpoint 的 validation 时间序列，它仍只能称 **final**，不能追溯称为 validation-selected best。

当前源码已增加 resume-critical 语义，不能用它对旧 run 做 `--resume auto`，也不能以
`--resume none` 覆盖旧目录。v2 必须建立独立 run，并显式 fork v1 final。

## 模型与原生 MTP

- 0.8B source commit：`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`；
- 9B source commit：`68c46c4b3498877f3ef123c856ecfde50c39f404`；
- loader 严格要求 0.8B checkpoint 顶层 15 个 `mtp.*` tensor；
- 原生语义为 `h_t + embed(x_(t+1)) -> predict x_(t+2)`，使用 `L-2` target 与 causal SDPA；
- MTP 参数保持 frozen/eval，不进入 Adam，但 loss 可沿 main hidden state回传 A/B；
- training DCP 只保存 72 个 trainable tensor，MTP 与其他冻结 source state 从锁定 checkpoint
  重载；
- 导出时 dense MTP 转换为合法 native MoE MTP 布局。

用户已明确选择 v2 的 `losses.mtp=0.1`。这不是 Qwen3.5 官方默认，而是本项目的
resume-critical 实验选择。

## 500M 数据与 KD

最终 clean audit 五项 gate、六来源 train/validation quota 全部通过。prepared 当前为：

- 641 shards；
- 126,457 sequences；
- 516,719,389 train tokens；
- manifest SHA256
  `9290665ac1e09fbd5b9aea1966bed7a51095bab66f460a0124af4532b1805fd9`；
- `ready_for_training=true`、`research_only=false`、`pending_audits=[]`；
- pipeline `COMPLETE` SHA256
  `a3a0856f64161e44d3d78b7fc84617a22f99c6e1dd0c4fc94ab26a001646da20`。

top-64 KD 使用 batch 2、sequence 4096、logits chunk 64、BF16、top-64、temperature 2，现已
完整结束：641 shards / 126,457 sequences / 516,719,389 tokens，manifest SHA256 为
`242ed2d0fb899cb333939bbd581f8a5632e97228f0c1fda2fee14bea7291efe9`。历史编排记录位于：

- `artifacts/data/base-v2-500m-kd-orchestration/status.json`；
- `artifacts/data/base-v2-500m-kd-orchestration/console.log`；
- `artifacts/data/base-v2-500m-kd-orchestration/logs/`。

最终 `generate-kd` 与独立 `index-kd` 均 exit 0；orchestration `MANIFEST.json`/`COMPLETE` 与上述
manifest SHA 的三重认证已通过。

## Canonical 性能与数值门

canonical report SHA256 是
`cf40eac976767681a704676bee960d8c220d700a847b73d62e1f2358ea15ab38`。生产候选固定为：

- `dense_transfer_execution=expanded`；
- physical microbatch 1、单卡 accumulation 64；
- ordinary outer/inner = 0/0；
- alignment outer/inner = 8/16；
- 95%/5% harmonic mixture = **6,263.998 tok/s**；
- teacher split offload、loss chunk 512、1.5 GiB optimizer-state reserve。

expanded selective checkpoint 已通过真实 KD 全图 bitwise 准入。`differentiable_folded` 在真实
checkpoint/KD 累积门中 scale 14/24 层失败，最坏相对误差 203.17%，只能保留为
experimental/rejected。任何 B2 folded、ordinary AC3、alignment AC24 或 `6405.436 tok/s`
配置都不是当前 production 候选。

expanded B2 安全 ordinary 点约 6,239 tok/s，比 B1 慢 9.964%；虽然 B2 功耗更接近 600 W，
但瓦数更高没有转化为吞吐收益，所以生产仍选 B1。

## 显存边界

optimizer 只管理适配参数，冻结权重没有 Adam moments；但冻结权重与它们的激活仍占显存。
A/B 也不是 rank-16 LoRA，而是 48 张完整 `4096×1024` FP32 映射：

- A/B/scale 参数约 0.75 GiB；
- gradient 约 0.75 GiB；
- 两份 Adam moments 约 1.50 GiB；
- trainable 参数、gradient、moments 合计约 3.00 GiB。

此外还包括 0.8B body、6.75 GiB mapped donor、4K activation、hidden tuple、MTP body、词表
loss 临时平面和 CUDA workspace。teacher-exclusive 约 8.033 GiB 平时留在 CPU，只在
alignment optimizer batch stage 一次。expanded selective checkpoint 的意义是减少 saved
activation，而不是给冻结参数“省 optimizer”。

正式基准实际触碰了 1.5 GiB Adam-moment 等价 reserve，但没有创建 optimizer；因此它证明
容量可放入 5090，不等于 optimizer-step 吞吐/恢复已经验收。

## v2 学习率、WSD 与 quality cooldown

用户已选择 peak LR：

| 参数组 | peak LR |
|---|---:|
| adapter | `2e-4` |
| router | `1e-3` |
| LoRA | `2e-4` |
| scale | `1e-3` |

500M token scheduler 固定为 token-based WSD：5M linear warmup，5M–450M 保持 peak，最后
50M cosine decay 到 0.1 倍。scheduler state、committed token 和配置字段都属于 exact-resume
语义。

最后 50M 使用独立六来源 high-quality whole-shard cooldown view。真实 policy 已绑定最终 parent
prepared/KD SHA；bundle 为 68 shards / 53,221,680 tokens / 136 hardlinks，根 `COMPLETE`
绑定 manifest SHA `ea4e0280520cdad8f674372d6dcb6aeffcd2f007372c88b1fdd94ee0e74849e3`。

## Web 与训练控制

Dashboard 当前监听 `0.0.0.0:8765`，HTTP Basic 凭据为 mode 0600。GPU 以 100 ms 采样，UI
约 1 秒刷新，磁盘只写 10 秒聚合桶并做有界轮转。页面展示 loss、LR、grad、compute/wall
tok/s、ETA、显存、功耗、利用率和 Data/KD pipeline 状态。

Web 只能控制固定 allowlist profile 的训练 start/save/stop；v1 profile 是 monitor-only，v2 profile
已 launch-enabled 且当前 running。Dashboard 以独立 session 在 Linux 后台监听 `0.0.0.0:8765`；
若以后迁移为 systemd user service 并要求退出所有 Linux 会话后自动拉起，仍需用户执行：

```bash
sudo loginctl enable-linger "$USER"
```

## 剩余 gate

- [x] v1 final validation；
- [x] native causal MTP；
- [x] folded FAIL / expanded PASS 数值结论；
- [x] canonical expanded B1 性能/功耗/显存 bundle；
- [x] 500M audit 与 prepared；
- [x] MTP coefficient、四组 peak LR、WSD 与 cooldown policy 设计；
- [x] 完成并 index 500M top-64 KD，取得真实 KD manifest/COMPLETE；
- [x] 基于真实 KD SHA 生成并审批六来源 50M policy，物化 cooldown bundle；
- [x] 运行配置发布入口，生成最终 v2 YAML 与 Web profile；
- [x] 对最终 lineage执行 validate、preflight、dry-run 与无 optimizer graph-smoke；
- [x] 用户授权后从 v1 final fork 启动正式 Base v2 训练。

因此当前准确状态是：**正式 Base v2 正在运行，Web、完整日志、GPU telemetry 与安全 checkpoint
策略均已启用。**
