# Base v2 高质量数据 cooldown 契约

状态：真实 policy 与 hardlink bundle 已于 2026-07-24 发布并通过全量认证。policy 选择
68 shards / 13,025 sequences / 53,221,680 tokens，selection-plan SHA256 为
`e245ab483724c4c1c3dbf54990e8ac87117757709bd7de6d20fc0c0903ee9b16`；bundle 含
136 个同 inode tensor hardlinks，dataset fingerprint 为
`52232e4512917e973a9acbff1909d7f1799d6b1755ea173fa237e4eee09bba67`。Base v2 已启用该
cooldown，并在 450M committed token 后切换。LR WSD 与数据 cooldown 是两个独立机制；前者
改变学习率，后者在 token 坐标上切换认证语料。

## 配置契约

v2 目标是在 450M committed token 后使用独立的 prepared + top-64 KD manifest：

```yaml
data:
  quality_cooldown_manifest_path: artifacts/data/base-v2-500m-quality-bundle/prepared/manifest.json
  quality_cooldown_manifest_sha256: <真实 SHA256>
  quality_cooldown_teacher_kd_manifest_path: artifacts/data/base-v2-500m-quality-bundle/kd/manifest.json
  quality_cooldown_teacher_kd_manifest_sha256: <真实 SHA256>
  quality_cooldown_start_tokens: 450000000
```

五个字段必须全有或全无。默认全无时，resolved YAML、canonical dict、critical fingerprint、单语料
cursor 与旧 v1 checkpoint 语义保持不变。启用后，四个 manifest identity 和 start token 都是
resume-critical；start 必须小于训练总 token budget。

切换发生在 optimizer batch 边界：若一个 batch 从 450M 前开始并在 commit 后跨过 450M，该完整
batch 仍来自主语料；下一整个 batch 才进入 cooldown。这避免按 world size/microbatch 几何拆分同一
optimizer batch。准确边界、最后一个 primary batch 的起点和 phase 都写入 event/metric/checkpoint。

## 无需重新生成 teacher KD 的 subset view

推荐从最终 500M 主 prepared/KD 中选择经过人工规则认证的整 shard，而不是重新读取文本或猜测
同一 manifest 内的动态 source 权重：

1. `source_id` 从父 prepared 绑定的 extracted corpus manifest 中按输出路径反查，不依赖文件名猜测；
   selection policy 输出有序 parent `shard_id`、逐 shard source ID 和逐来源 token 数。
2. cooldown prepared view 为每个选中 parent shard 保留相同 `source_path/source SHA/prepared tensor
   SHA/sequence count/token count`。prepared tensor 文件可 hardlink。
3. cooldown KD view 必须选择相同顺序的 `source_shard_id`，并保留相同 source prepared tensor SHA、
   KD tensor SHA、sequence/token count。大体积 KD tensor 文件可 hardlink，因此不需要再运行 9B
   teacher。
4. 不能直接复制主 corpus manifest。新 view 必须有独立 dataset fingerprint、有序 shard inventory，
   并从 0 重排连续 global sample/token range。KD shard manifest 绑定 dataset fingerprint 和 global
   range，因此需要重写这份小型元数据及其 COMPLETE/SHA；`kd_tensors.safetensors` 本身不变。
5. cooldown 必须是主语料的严格 whole-shard subset，至少覆盖 50M token，且 prepared lineage 要
   `ready_for_training=true`、`research_only=false`、无 pending audit。

preflight 会完整验证主/cooldown 两套 prepared 与 KD 文件 SHA、sequence length、tokenizer、teacher
model/revision/manifest、temperature、top-64 normalization 和 exact coverage，再逐 shard join 回主
prepared/KD tensor identity。重复 parent shard、伪造 tensor、source ID 无法由父 extracted manifest
认证、source mix 统计不等于选中 token、复用主 dataset fingerprint 或不足 50M 均 fail closed。

## 50M 选择策略生成器

Base v2 使用锁定的六来源最低配额，总目标正好 50M token：

| source_id | 最低 token |
|---|---:|
| `english_fineweb_edu_dedup` | 15,000,000 |
| `math_finemath_4plus` | 15,000,000 |
| `code_github_clean_allowlisted` | 7,500,000 |
| `chinese_fineweb2_cmn_hani` | 5,000,000 |
| `science_cosmopedia_openstax` | 5,000,000 |
| `science_cosmopedia_stanford` | 2,500,000 |

`twen data generate-cooldown-policy` 先完整认证父 prepared、top-64 KD exact coverage 和 extracted
lineage，再从 extracted manifest 的输出路径映射真实 `source_id`，不按文件名猜来源。每个来源独立用
固定 seed 对包含 parent dataset fingerprint、shard ID、source ID、source/prepared tensor SHA 和
sequence/token count 的 shard identity 做 SHA256 排序，取达到该来源最低配额的最短 whole-shard
前缀；六组并集再用独立 global scope SHA256 排序。末 shard 允许小幅 overshoot，不切 shard、也不
动态重权。

默认命令是严格只读 dry plan：认证与选择完成后只向 stdout 返回 draft，其中
`approved_for_quality_cooldown=false`；不会创建 output、staging 或 lock，也不会运行 teacher KD 或
训练。以下 dry-plan 命令是已经执行过的可复现审计入口；真实 policy 已发布，不要对同一 output
重复执行：

```bash
.venv/bin/python -m twen data generate-cooldown-policy \
  --prepared-manifest artifacts/data/base-v2-500m/manifest.json \
  --kd-manifest artifacts/data/base-v2-500m-kd/manifest.json \
  --output artifacts/data/base-v2-500m-quality-policy
```

父 prepared/KD 全部完成并审阅 dry plan 后，必须显式增加 `--approve` 才会原子发布：

```bash
.venv/bin/python -m twen data generate-cooldown-policy \
  --prepared-manifest artifacts/data/base-v2-500m/manifest.json \
  --kd-manifest artifacts/data/base-v2-500m-kd/manifest.json \
  --output artifacts/data/base-v2-500m-quality-policy \
  --approve
```

发布目录是封闭的五文件 bundle：

- `quality-cooldown-policy.json`：与物化器现有 locked schema 直接兼容；
- `AUDIT.json`：目标/实际/overshoot/占比、候选与选中 shard 数、固定 seed、逐 shard source/global
  hash、输入 SHA 和全部 gate；
- `REPORT.md`：同一审计的人类可读报告；
- `MANIFEST.json`：认证上述三个 payload；
- `COMPLETE`：绑定 manifest SHA。

所有产物都记录 `training_started=false` 与 `teacher_kd_started=false`。来源不足、父 manifest 在认证或
发布前发生 SHA 漂移、重复来源目标/parent shard、无法形成严格 parent subset、已有 output、staging
或 symlink 均 fail closed；生成器不会覆盖或“修复”已有目录。

## Policy schema 与物化 CLI

策略文件使用封闭字段集合；所有 SHA 和 token 数都必须来自已经完成认证的父 manifest，不能使用下面的
占位值直接运行：

```json
{
  "schema_version": 1,
  "kind": "twen_quality_cooldown_selection_policy",
  "policy_id": "reviewed-policy-v1",
  "approved_for_quality_cooldown": true,
  "selection_basis": "explicit reviewed whole-shard rule",
  "parent_prepared_manifest_sha256": "<64位真实SHA256>",
  "parent_kd_manifest_sha256": "<64位真实SHA256>",
  "required_cooldown_tokens": 50000000,
  "ordered_shards": [
    {"shard_id": "shard-000123", "source_id": "math_finemath_4plus"}
  ],
  "declared_source_mix_token_counts": {
    "math_finemath_4plus": 50000000
  }
}
```

先 dry-run；它只认证输入并输出计划，不创建 output、staging 或 lock：

```bash
.venv/bin/python -m twen data materialize-cooldown \
  --prepared-manifest artifacts/data/base-v2-500m/manifest.json \
  --kd-manifest artifacts/data/base-v2-500m-kd/manifest.json \
  --selection-policy artifacts/data/base-v2-500m-quality-policy/quality-cooldown-policy.json \
  --output artifacts/data/base-v2-500m-quality-bundle \
  --required-cooldown-tokens 50000000 \
  --dry-run
```

人工核对 dry-run 的 shard 顺序、source mix、token 总数与 fingerprint 后，移除 `--dry-run` 才会
物化。bundle 内实际训练入口分别为 `prepared/manifest.json` 和 `kd/manifest.json`。

物化器只允许同文件系统 hardlink，不会在 hardlink 失败时退化为复制，也不会重新运行 tokenizer、
teacher KD 或训练。所有小 manifest、每 shard COMPLETE、顶层 bundle/COMPLETE 都重写并重新认证；
global sample/token range 从 0 连续重排。发布使用独占 lock、带 identity 的 staging 和单次原子 rename。
同一输入和 policy 的再次调用只复验已发布 bundle 并返回 `skipped_existing=true`；缺文件、symlink、额外
文件、SHA/inode 改变、损坏 staging、不同 policy 或 source/output 路径重叠都会拒绝，验证过程不会
补写或修复已有输出。

## deterministic cursor 与 checkpoint

`DeterministicCooldownCursor` 内含两个独立的 `DeterministicGlobalCursor`：

- primary 与 cooldown 都使用固定 seed 的 `shard-local-affine-v1`，不保存大 shuffle table；
- phase 只由全局 committed token 与锁定的 `quality_cooldown_start_tokens` 决定；
- checkpoint 同时保存 primary cursor、cooldown cursor、全局 sample/token、active phase、两套 dataset
  fingerprint 和切换点；
- resume 会逐项认证两个 cursor 的 dataset identity、局部计数与全局计数之和；
- world size 改变时，只重新把同一个全局 optimizer batch 按 rank stride 分片，不改变 phase 或样本
  顺序。

当前没有合格的 quality subset manifest，因此功能保持关闭。不得手工填占位 SHA、把完整主 manifest
同时填入 cooldown 字段，或仅按文件名/source 猜“高质量”。v2 finalizer 还会要求 policy 五文件
bundle 与 cooldown 顶层 `MANIFEST.json`/`COMPLETE` 完整，且六来源分别达到上述最低配额；只留下
`prepared/manifest.json` 与 `kd/manifest.json` 的半成品目录不能发布训练配置。
