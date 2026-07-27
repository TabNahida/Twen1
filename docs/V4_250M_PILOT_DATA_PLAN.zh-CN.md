# v4 250M Pilot 两阶段数据计划（launch-disabled）

## 结论

本计划已经固定为 `225M primary + 25M cooldown`，并把训练契约固定为：

- `max_tokens=250,000,000`
- `quality_cooldown_start_tokens=225,000,000`
- `global_batch_tokens=262,144`
- `sequence_length=4096`
- `data.allow_corpus_reuse=false`
- 从 `runs/base-dense-v3-500m/step-000000001912-milestone-complete`
  使用 `--resume none --fork-from ...` 分叉

当前只允许继续做中文来源替换、metadata resolve、语料物化、治理审计和 prepare，
**不允许启动训练**。
`configs/base/dense-v4-250m-pilot.blocked.yaml` 中的 prepared manifest 与
source-map 使用显式 `PENDING_*` 值，因此即使绕过 Web 直接调用 CLI，也会在配置校验阶段
失败。`locks/base-dense-v4-250m-pilot.readiness.json` 与容量 attestation 同时保持
`launch_enabled=false`。旧 primary/cooldown 的 refill、prepared、两阶段/validation
不相交、formal v3 baseline 与 governed controller 证据曾经完成，但 2026-07-27 的
中文 semantic-quality 全量复核推翻了这套数据的训练准入；这些身份必须在中文
remediation 后重新生成。13M low-LR calibration 也尚未运行，且旧 calibration prepared
复用了失败来源，不能启动。

## 正式超参、fork 与暂停评测门

13M low-LR calibration 仍保持 Adapter `5e-5`、scale `1e-5`、5M warmup；
它是正式启动前必须由用户安排并通过的质量门，但不是 250M 的 warm start。250M
始终从 v3 final 重新 fork，并使用更保守的独立合同：

- Adapter/Lora nominal LR `3e-5`，scale LR `3e-6`；
- 10M warmup，随后全程 cosine 到 peak 的 `0.1`；
- physical micro-batch 1、GA 64、global batch `262,144` token；
- 每 50 step 保存 checkpoint，`keep_last=3`；
- 必须从 v3 final model-only checkpoint fork，并重置 optimizer、scheduler 和 cursor；
  禁止从 16M smoke 或 13M calibration checkpoint 继续训练；v3 final 的 `COMPLETE`
  SHA256 固定为
  `3a21a50e35de74ecd0ff5b8f00aa29ed6c83f746fc2cf97d4da6b0536262b6c7`。

13M calibration 尚未执行。完成后必须同时认证训练报告、checkpoint 40/50/final 的
同口径 frozen-v3 validation、checkpoint drift 报告及各 checkpoint 的
manifest/`COMPLETE`，并满足：

- 最佳和 final aggregate NLL 均不高于 `2.3766688031972105`；
- 中文 NLL 不高于 `3.656194313354557`；
- final branch-scale 相对 v3 L2 漂移不超过 `5%`；
- reused sequence/token 均为 0，所有 reference `epoch=0`；
- loss、NTP、MTP、grad norm、nominal/adjusted LR 全部 finite，clip fraction 为 0。

这些要求已作为 pending `calibration_gate` 写入 readiness；在完整认证证据回填并重新
计算 gate 前，它明确 `passed=false`、`authorizes_training=false`。

正式任务在约 13M、26M、52M、105M、157M、210M、223M（cooldown 前）、
236M（cooldown 中）和 250M token 暂停评测。任一 non-finite、数据
`epoch>0`/reuse、阶段身份或不相交证明失败、checkpoint lineage 失败、scale 相对
v3 L2 漂移超过 5%，都立即停止；rolling 50-step clip fraction 超过 1% 时暂停审查。
validation 还采用以下 fail-closed 门：

- aggregate NLL 单次高于 v3 `0.010`，或连续两次高于 `0.005`；
- 中文 NLL 单次高于 v3 `0.05`，或连续两次高于 `0.03`；
- 相对最佳 checkpoint 连续两次回退超过 `0.001`；
- 连续两个大评测点改善不足 `1e-4` 时停止，`1e-4` tie 内选择更早 checkpoint。

这里的“暂停”是外部 governed controller 的合同，语义是提交第一个令累计 token
达到或越过阈值的完整 optimizer batch 后暂停。**当前 controller 尚未实现**；下面列出的
`twen.cli train` 命令只会启动训练，不会自动暂停、运行 NLL validation 或执行这些
post-launch hard stop。因此 controller 本身也是 launch blocker，不能把 readiness 中的
阈值误当成已经接入引擎的功能。

现有 frozen validation 只覆盖中文、英文、数学、GitHub code、OpenStax 和 Stanford。
正式启动前必须为 ArXiv、StackV2、Common Corpus、LibreTexts 与 Gutenberg 增加冻结
validation 并重算 v3 baseline；还必须证明 primary+cooldown train union 与两份
validation union 在 stable ID、normalized exact 和 MinHash near-duplicate `0.8` 三层
不相交。formal validation summarizer 只生成报告，不授权训练；其 bundle 与不相交
attestation 必须由最终 readiness 另行认证。否则新增来源没有可比质量门，250M 保持
blocked。

## 两阶段 mix 与 raw quota

`raw quota = ceil(clean quota / smoke 治理保留率 × 1.10, 1000 tokens)`。
它是物化扫描预算，不是已经获得的容量。只有最终 prepared manifest 中的 unique
token/sample 容量才能解除容量门。

Primary recipe 请求 `225.270M` clean tokens，使其高于 `225M + 262,144` 的完整尾批门。

| Primary 来源 | Mix | Clean quota | Raw quota | 保留率 |
|---|---:|---:|---:|---:|
| FineWeb2 中文 | 24% | 54,064,800 | 68,343,000 | 87.02% |
| FineWeb-Edu | 25% | 56,317,500 | 65,217,000 | 94.99% |
| FineMath 4+ | 14% | 31,537,800 | 47,046,000 | 73.74% |
| GitHub Code Clean | 9% | 20,274,300 | 29,403,000 | 75.85% |
| Cosmopedia OpenStax | 4% | 9,010,800 | 11,557,000 | 85.77% |
| Cosmopedia Stanford | 3% | 6,758,100 | 9,974,000 | 74.54% |
| ArXiv permissive | 12% | 27,032,400 | 65,584,000 | 45.34% |
| Stack v2 Edu | 4% | 9,010,800 | 10,701,000 | 92.63% |
| Common Corpus | 5% | 11,263,500 | 15,407,000 | 80.42% |
| **合计** | **100%** | **225,270,000** | **323,232,000** | — |

Cooldown recipe 请求 `25.270M` clean tokens，使其高于 `25M + 262,144` 的完整尾批门。

| Cooldown 来源 | Mix | Clean quota | Raw quota | 保留率 |
|---|---:|---:|---:|---:|
| FineWeb2 中文 | 25% | 6,317,500 | 7,986,000 | 87.02% |
| FineMath 4+ | 22% | 5,559,400 | 8,294,000 | 73.74% |
| ArXiv permissive | 22% | 5,559,400 | 13,488,000 | 45.34% |
| Cosmopedia OpenStax | 10% | 2,527,000 | 3,241,000 | 85.77% |
| Cosmopedia Stanford | 8% | 2,021,600 | 2,984,000 | 74.54% |
| LibreTexts | 4% | 1,010,800 | 1,213,000 | 91.67% |
| Project Gutenberg | 4% | 1,010,800 | 3,229,000 | 34.44% |
| GitHub Code Clean | 5% | 1,263,500 | 1,833,000 | 75.85% |
| **合计** | **100%** | **25,270,000** | **42,268,000** | — |

不能把全局约 80% 的保留率套到所有来源。尤其 Gutenberg、ArXiv 的实测保留率较低，
raw 扫描预算必须按逐来源值放大。

## 固定文件与许可证

两份 recipe 已经通过真实 Hub metadata 对比，resolved lock 的
`remote_identity_verification=verified_against_hub_metadata`。Primary 固定文件
总量 `6,193,268,750 B`，cooldown 固定文件总量 `3,778,042,685 B`，合计
`9,971,311,435 B`（约 `9.286 GiB`）。这是远端 locked 范围，不代表全部内容都会下载：
Parquet 使用 range read，gzip JSONL 必须完整下载。

共同来源使用不同上游文件：

| 来源 | Primary | Cooldown |
|---|---|---|
| FineWeb2 中文 | `004_00073.parquet` | `004_00072.parquet` |
| FineMath | `train-00060-of-00064.parquet` | `train-00061-of-00064.parquet` |
| GitHub Code | `train-00570-of-00880.parquet` | `train-00571-of-00880.parquet` |
| OpenStax | `train-00001-of-00002.parquet` | `train-00000-of-00002.parquet` |
| Stanford | `train-00002-of-00013.parquet` | `train-00003-of-00013.parquet` |
| ArXiv | `arxiv-papers-0005.json.gz` | `arxiv-papers-0006.json.gz` |

peS2o 小样中的 `metadata.oa_license` 主要是无版本 `CCBY`。项目不会把它擅自解释为
`CC-BY-4.0`；正式 recipe 已将 peS2o 替换为同类科学论文来源
`common-pile/arxiv_papers_filtered`，并只接收能够规范化为明确版本 CC BY/CC0 的行。
许可证 allowlist 仍然在逐文档物化时 fail-closed，实际许可保留量尚未通过。

OpenCSG FineWeb-Edu-Chinese V2.2 的 Hub 标记和 README 使用条款存在冲突，不在正式
recipe 中。USGPO 实测保留率仅 1.67%，也不进入本计划。

## 容量与不相交门

`locks/base-data-sources-v4-250m.capacity-attestation.json` 当前是 blocked
attestation。它已经绑定：

- config、两份 recipe 和两份 remote-resolved lock 的 SHA256；
- `max_tokens`、cooldown 边界、global batch 和 no-reuse 契约；
- 每个来源 required clean/raw quota；
- Primary 最低 `225,262,144 tokens / 54,996 samples`；
- Cooldown 最低 `25,262,144 tokens / 6,168 samples`。

以下字段必须在真实物化后回填，当前为 `null/false`：

- extracted/prepared manifest SHA256；
- prepared dataset fingerprint；
- authenticated source-map SHA256；
- overall/per-source available、margin、passed；
- 审计 attestation 与 attribution manifest SHA256；
- 两阶段 stable-ID exact、normalized-text exact 和 near-duplicate 不相交结果。

仅用 smoke 提前停止位置外推大文件容量是无效的。最终必须分别证明 locked raw
available、治理保留率和 prepared unique capacity。

中文确定性规则目前覆盖赌博/SEO 拼接、繁简字符高频混写、重复段落、乱码、异常短句界
和采集站 boilerplate，并记录独立 reason code。它不宣称能够高精度识别“大年夜、
软件体系、乾坤”等同一脚本的机械转换语义污染；该项在正式物化后仍需小样统计/人工
复核，不能因为确定性规则通过就自动解除质量门。

2026-07-27 的正式全量统计复核已经证明该担忧真实存在，而不是待确认的理论风险：
primary 79,209 篇与 cooldown 9,970 篇中文文档的全部输入 shard 均通过 size/SHA
认证后，扫描到 170 篇高置信机械转换文档和 5,755 篇异常标点拼接指标文档；确定性
风险/对照样本还显示更广泛的跨主题串文、SEO 与同义词改写污染。因此当前 FineWeb2
中文输入及复用它的 13M calibration prepared 均为 **FAIL**，Web 启动门已经关闭。
不能只删除少量正则命中后沿用本文件后续的旧 capacity/evidence：中文来源替换或重新
筛选后，必须从 extracted audit、prepared capacity、phase/formal disjointness、formal
validation baseline、closure 和 calibration 身份开始重做。

## 可执行命令

### 1. 重新认证 Hub metadata（不下载语料）

```bash
.venv/bin/python -m twen.cli data resolve-sources \
  --recipe locks/base-data-sources-v4-primary.json \
  --output locks/base-data-sources-v4-primary.resolved.json \
  --network-policy fallback \
  --proxy http://172.23.240.1:8080

.venv/bin/python -m twen.cli data resolve-sources \
  --recipe locks/base-data-sources-v4-cooldown.json \
  --output locks/base-data-sources-v4-cooldown.resolved.json \
  --network-policy fallback \
  --proxy http://172.23.240.1:8080
```

### 2. 可选缓存复用

只复用 source ID、repo、revision、path、size 和 SHA256 全部相同的已认证 gzip cache。
推荐 copy-on-write/reflink，避免修改旧 smoke 证据：

```bash
mkdir -p data/base-v4-250m-primary-r1/.source-cache/{downloads,derived}
cp -a --reflink=auto \
  data/base-v4-smoke-r3/.source-cache/downloads/code_stackv2_edu_permissive \
  data/base-v4-250m-primary-r1/.source-cache/downloads/
cp -a --reflink=auto \
  data/base-v4-smoke-r3/.source-cache/derived/code_stackv2_edu_permissive \
  data/base-v4-250m-primary-r1/.source-cache/derived/

mkdir -p data/base-v4-250m-cooldown-r1/.source-cache/{downloads,derived}
for source in education_libretexts_permissive public_domain_project_gutenberg; do
  cp -a --reflink=auto \
    "data/base-v4-smoke-r3/.source-cache/downloads/$source" \
    data/base-v4-250m-cooldown-r1/.source-cache/downloads/
  cp -a --reflink=auto \
    "data/base-v4-smoke-r3/.source-cache/derived/$source" \
    data/base-v4-250m-cooldown-r1/.source-cache/derived/
done
```

复制后仍由下载器验证 size/SHA/manifest；不要复用同 repo 的不同 shard，也不要把旧的
extracted JSONL 混入新输出。

### 3. Raw 真实物化（已完成，命令仅供身份复现）

当前 primary/cooldown raw corpus 都已有认证 `COMPLETE`，manifest SHA256 分别为
`c09cc7c81beaf53487d623012fc005dc7a06f13d238df040d751df527294ec54` 与
`ac4a1ffbfce4cae153662699313df6565cebc6362aab6e942566506245edd4b8`。
两者仍为 `ready_for_training=false`；不要把 raw 完成状态解释成 final governed
prepared corpus，也不要无理由重复执行下面的下载/物化命令。

```bash
.venv/bin/python -m twen.cli data build-base \
  --recipe locks/base-data-sources-v4-primary.json \
  --resolved-lock locks/base-data-sources-v4-primary.resolved.json \
  --output data/base-v4-250m-primary-r1 \
  --tokenizer artifacts/models/qwen3.5-0.8b-base \
  --tokenizer-manifest-sha256 5e847be98c2d114d2fb85a67ac77f4eef416e11871c1a5580d1bc52663e2e643 \
  --profile materialization \
  --network-policy fallback \
  --proxy http://172.23.240.1:8080 \
  --stop-file data/base-v4-250m-primary-r1/STOP \
  --progress always

.venv/bin/python -m twen.cli data build-base \
  --recipe locks/base-data-sources-v4-cooldown.json \
  --resolved-lock locks/base-data-sources-v4-cooldown.resolved.json \
  --output data/base-v4-250m-cooldown-r1 \
  --tokenizer artifacts/models/qwen3.5-0.8b-base \
  --tokenizer-manifest-sha256 5e847be98c2d114d2fb85a67ac77f4eef416e11871c1a5580d1bc52663e2e643 \
  --profile materialization \
  --network-policy fallback \
  --proxy http://172.23.240.1:8080 \
  --stop-file data/base-v4-250m-cooldown-r1/STOP \
  --progress always
```

后续必须依次完成 refill → 用同一 frozen v3 validation 做最终 audit → prepare，并额外
完成 primary/cooldown train-to-train 与 train-union/validation-union 的
stable-ID/exact/near 不相交审计。即使容量 attestation 的所有 `passed` 都为 true，也只能
回填数据身份；还必须等待 calibration、formal validation 和外部 pause controller 三个
readiness gate 通过，才能生成独立的最终 config。当前 blocked draft 的 `PENDING_*` 不应
被手工替换。
