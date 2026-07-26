# Base Dense v4 Muon / LR 与 checkpoint 漂移审计

## 结论

`12.8x` 是 PyTorch 2.11 `Muon(adjust_lr_fn="match_rms_adamw")`
对 `4096 x 1024`（及其转置）矩阵使用的原生形状系数：

`0.2 * sqrt(max(4096, 1024)) = 12.8`

Twen 没有再次放大 Muon LR。`1e-4` 是 nominal peak，日志中的
`1.2788e-3` 是正交化矩阵更新所用的 shape-adjusted coefficient；
二者必须同时记录，但 adjusted coefficient 不能直接当作 AdamW LR 比较。

真正异常的是 branch scale。相对 v3 final，v4 smoke step 40 时：

| checkpoint | Adapter 相对 L2 漂移 | Scale 相对 L2 漂移 | scale 均值 |
|---|---:|---:|---:|
| v3 final | 0 | 0 | 0.0204080 |
| step 40 | 1.1657% | 24.2033% | 0.0155377 |
| step 50 | 1.3035% | 24.1501% | 0.0156480 |
| step 60 | 1.3476% | 24.3013% | 0.0156127 |
| step 62 | 1.3522% | 24.3305% | 0.0156049 |

step 62 有 23/24 个 scale 下降。scale 漂移在 step 40 已基本完成，后续
cosine 降 LR 也没有恢复。与此同时，frozen held-out candidate NLL 从 v3 final
的 `2.3766688031972105` 恶化到 `2.384702292258207`，即
`+0.008033489061 / +0.338%`；62 步的 grad norm 全部低于 `1.0`，因此不是
gradient clipping 掩盖了高 LR。

## 参数与 optimizer 账本

- 48 张完整 FP32 A/B：`201,326,592` 个参数，仅由 Muon 更新；
- 24 个 FP32 branch scale：24 个参数，仅由 AdamW 更新；
- dense v4 没有 router 或 LoRA 参数组；
- 冻结的 backbone、donor 和 MTP 参数不进入 optimizer；
- A/B 本身约 `0.75 GiB`，Muon momentum 约 `0.75 GiB`；optimizer state
  很小于 frozen weights、4K activation、词表 loss 平面和 CUDA workspace。

checkpoint DCP 中的 scheduler 状态确认 base LR 为 `[1e-4, 3e-4]`，
step 62 的 live LR 为 `[1e-5, 3e-5]`，consumed tokens 为 `16,197,992`。
Muon/AdamW 参数组、所有 Muon 数值选项、base LR、warmup、schedule、
min ratio 与 max tokens 都属于 critical fingerprint；bundle 的 model、
optimizer、scheduler 恢复已有逐字节等价测试。

## `scale_lr=0` 审查

当前配置校验明确要求 `scale_lr > 0`，所以不能把 `0` 偷渡为“冻结”。
若绕过校验，PyTorch 的 AdamW scale group 仍会存在，scale 参数不会变化，
但 gradient 和 Adam moments 仍会计算、`step` 仍会递增；这不是语义完整的冻结。

真正冻结需要新增 resume-critical 开关、将 scale 设为
`requires_grad=false`、允许 Muon-only optimizer schema，并重新验证 DCP/recovery。
在正式 pilot 前临时改变这些架构语义风险高于收益，因此首选保留 scale group，
把 peak LR 降到 `1e-5`。

`1e-5` 的依据：

- 相对 smoke 的 `3e-4` 是 30 倍下调；
- 同形 13M 校准中，离散 cosine 的累计 scale LR 约为 smoke 实际累计值的
  `2.68%`；
- 250M、5M warmup、全程 cosine 时累计 scale LR 约 `0.0052355`，
  是 smoke 实际 `0.0098254` 的 `53.3%`；若使用 `3e-5` 则会升到
  smoke 的 `159.9%`。

## 低 LR 校准候选（不自动启动）

配置：`configs/base/dense-v4-13m-low-lr-calibration.yaml`

- 从 v3 final model-only checkpoint fork，重新初始化 Muon/AdamW；
- `max_tokens=13,000,000`；
- `adapter_lr=5e-5`，shape-adjusted peak `6.4e-4`；
- `scale_lr=1e-5`；
- `warmup_tokens=5,000,000`，全程 cosine，`min_lr_ratio=0.1`；
- NTP `1.0` + native MTP `0.1`；
- global batch `262,144`，physical micro-batch 1；
- `allow_corpus_reuse=false`；
- Web profile 固定 fork checkpoint；完成新数据认证与 no-reuse dry-run 后，
  `launch_enabled=true`，只允许用户手动安排这轮校准。

新版内容质量门从原 governed train/validation 中进一步拒绝 2,605 篇文档；
二次审计全部 gate 归零后重新 prepared 得到 13,733,818 个 unique token。
校准预算因此从 15M 收紧到 13M，并按新容量重新归一化 source weights。
完整 no-reuse dry-run 已通过：physical micro-batch 为 1、gradient accumulation
为 64、global batch 为 262,144 token，13M 预算含完整尾批仍不触发 epoch 回绕。
为消除审计后扫描器源码变更造成的身份漂移，又使用当前
`src/twen/data/audits.py`（SHA256
`b5f7b2f28545dc47fd797d418a77d2ed2cf2c0d3129180179b1eaaa58d1a5637`）
对同一 filtered candidate/frozen corpus 完整重跑 quality-v3 审计。六项 gate
再次全部通过且 0 findings；新 attestation SHA256 为
`73f973c34fff3d8035c72c0898b9ee3d27c2a98e74bdb068c817491157bc4986`，
prepared manifest SHA256 为
`fff506fd87c69b75d6bd1f86a96a96ee5b9c8e66ae0ba76f5b670154031f4393`，
authenticated source-map SHA256 为
`63f09c7a01a7ca21b1bee6edcd54f314c60f3530464f99570d3fe7ce115031fc`。
Web 只开放该校准 profile；250M formal profile 在本节后续质量 gate 和正式数据
认证全部通过前继续禁用。

校准完成后只在以下 gate 全部满足时进入 250M：

1. checkpoint 40、50、final 使用同一 frozen v3 validation 口径；
2. 最佳且 final aggregate NLL 均不得高于 `2.3766688031972105`；
3. 中文来源 NLL 不得高于 v3 的 `3.656194313354557`；
4. final scale 相对 v3 的 L2 漂移不得超过 `5%`；
5. `reused sequences/tokens = 0`，所有 reference 的 `epoch=0`；
6. loss、NTP、MTP、grad norm、nominal/adjusted LR 全部 finite，clip fraction 为 0；
7. 仅选择带完整 manifest/COMPLETE 身份的 checkpoint。

## 复现

以下命令只读取 DCP trainable delta，不构建模型、不创建 optimizer、不初始化 CUDA：

```bash
.venv/bin/python scripts/audit_dense_checkpoint_drift.py \
  --baseline runs/base-dense-v3-500m/step-000000001912-milestone-complete \
  --candidate runs/base-dense-v4-16m-smoke/step-000000000040-periodic \
  --candidate runs/base-dense-v4-16m-smoke/step-000000000050-periodic \
  --candidate runs/base-dense-v4-16m-smoke/step-000000000060-periodic \
  --candidate runs/base-dense-v4-16m-smoke/step-000000000062-milestone-complete \
  --output docs/reports/base-dense-v4-optimizer-drift-audit/analysis.json
```

完整机器可读值见 `analysis.json`；`MANIFEST.json` 记录 payload 身份，
`COMPLETE` 认证 manifest。
