# v4 Base 文本数据源、schema v2 与治理运行手册

状态：**r3 已真实物化并完成治理；16.014M unique-token smoke 已 ready for training**

对应 recipe：[`locks/base-data-sources-v4.json`](../locks/base-data-sources-v4.json)

当前已认证 resolved lock：
[`locks/base-data-sources-v4.resolved.json`](../locks/base-data-sources-v4.resolved.json)

本方案面向纯文本 causal NTP + Qwen3.5 原生 MTP 的 v4 Base Dense Adapter
训练。它不生成、下载或消费 9B teacher logits，也不启用 teacher KD、anchor KL
或 teacher hidden alignment。

recipe 使用稳定的 `schema_version: 2` 和
`kind: twen_base_data_source_recipe_v2`。运行时现在已经实现并 fail-closed
验证以下能力：

- schema-v2 recipe/activation 校验；
- 锁定 LFS 身份的 Parquet HTTP Range 读取；
- gzip JSONL 完整断点下载、压缩文件 SHA256 验证、流式解压和派生文件认证；
- dotted-field 访问、许可证规范化与逐文档 allowlist；
- source map、按有效 token 配比和确定性恢复所需的 lineage；
- `token-deficit-corrected-source-mix-bp-v2` 训练 batch mixer。

截至 2026-07-26，本地离线 `data inspect`、`data resolve --dry-run` 和使用已认证
resolved lock 的 `data build --dry-run` 均已通过。r1 真实 build 随后在 29.6M /
40M token 处按合同失败：LibreTexts 满足 1.4M train quota，但 book-level split 后
validation 只有 174,726 / 1.4M token。r2 因此把新来源 validation smoke 缩到 2M，
训练配比和 20M train quota 保持不变；随后又因 OER Commons 全文件只有
183,015 train / 22,975 validation token，低于 400,000 / 40,000 quota 而
fail-closed。r3 用固定 revision 的 USGPO 和 Project Gutenberg 替换
OER Commons/Pressbooks。r3 最终物化出 20,081,154 train +
2,069,128 validation token；治理 pass-001 随后按 rejection ledger 去除
4,067,482 个 train token，pass-002 在过滤后的 train 与冻结 v3 validation 上得到
0 findings。最终 prepared train 为 16,013,672 token、3,923 条 4096-token sequence，
`ready_for_training=true`、`research_only=false`。

## 1. 三档规模与 recipe 配比

v4 在三个规模上使用完全相同的来源比例：

| 来源 | 新旧 | 比例 | 20M smoke | 250M pilot | 500M formal |
|---|---:|---:|---:|---:|---:|
| FineWeb-Edu dedup | 旧 | 24% | 4.8M | 60M | 120M |
| FineWeb2 `cmn_Hani` | 旧 | 18% | 3.6M | 45M | 90M |
| FineMath 4+ | 旧 | 14% | 2.8M | 35M | 70M |
| GitHub Code Clean permissive | 旧 | 8% | 1.6M | 20M | 40M |
| Cosmopedia OpenStax | 旧 | 3% | 0.6M | 7.5M | 15M |
| Cosmopedia Stanford | 旧 | 3% | 0.6M | 7.5M | 15M |
| LibreTexts permissive | 新 | 7% | 1.4M | 17.5M | 35M |
| USGPO public domain | 新 | 2% | 0.4M | 5M | 10M |
| Project Gutenberg public domain | 新 | 4% | 0.8M | 10M | 20M |
| Open-license ArXiv | 新 | 5% | 1M | 12.5M | 25M |
| Stack v2 Edu permissive | 新 | 4% | 0.8M | 10M | 20M |
| Common Corpus permissive sample | 新 | 8% | 1.6M | 20M | 40M |
| **合计** | **旧 70% / 新 30%** | **100%** | **20M** | **250M** | **500M** |

下表是 raw materialization 的 recipe 配比，不是过滤后 smoke 的有效采样比例。新来源
smoke validation 保留 2M token，并采用同一来源比例。因此 r3 的真实
物化总目标是 **20M train + 2M validation = 22M token**。LibreTexts validation quota
从 1.4M 缩到 140,000 token，低于 r1 实测的 174,726 token；训练配比没有改变。
任何来源在许可、过滤、去重后无法满足 quota，都必须失败并修订 recipe；不得静默
重分配到其他来源。最终跨版本质量比较继续使用已冻结的 20M project validation，
不能与这个 2M 新来源 smoke validation 混为同一统计口径。

2M validation 中最小来源只有 40K–60K token，约 10–15 条 4096-token 序列；
它们足以发现数据管线和 loss 的明显异常，不足以支撑可靠的逐来源优劣排序。

治理后没有用重复采样伪装 20M unique quota：当前运行明确改名为 **16M governed
smoke**，并把有效 source weights 绑定到过滤后的 unique-token 容量。它只验证下载锁、
解压、字段、许可证、去重、切分、确定性恢复、纯文本目标、Muon 和 batch mixing，
不用于得出模型质量结论。250M pilot 用于确认 v4 的纯文本目标、
优化器和学习率设置。只有 pilot 的数据审计和 held-out 结果均通过后，才允许启动
500M formal。

## 2. 为什么选择这些新增来源

### 2.1 开放教材、公共领域书籍与政府出版物

`common-pile/libretexts_filtered` 提供教材、课程章节、讲义和问题集。
`common-pile/project_gutenberg_filtered` 增加公共领域英文书籍，
`common-pile/usgpo_filtered` 增加美国联邦政府公共领域出版物。三者都有自包含的
`text`，并在 `metadata.license`、`metadata.url` 和 `metadata.provenance`
中保留逐文档许可与来源。

这些数据比继续扩大 Common Crawl web 比例更能增加长篇解释、叙事和正式文书。
LibreTexts 可能包含 OpenStax 内容，Gutenberg/USGPO 也可能与 Common Corpus 的
Open Culture/Open Government collection 重合，因此必须执行跨源 exact +
near-duplicate 检查。对于 Cosmopedia 的合成改写，语义重叠无法只靠文本哈希完全
消除，应在报告中单独统计。

LibreTexts 的固定 revision 在 dataset card 中声称 3.6GB UTF-8，但
datasets-server 对同一 revision 给出约 386.8MB decoded bytes。锁文件保留了这项
差异；实际 admission 只能依据本地物化后的行数和 tokenizer token 数，不能依据 card
估算。

r1 全文件实测还证明 LibreTexts permissive 子集只提供 1,401,983 train token；
这只够 1.4M smoke quota，分别只覆盖 17.5M pilot quota 的约 8% 和 35M formal quota
的约 4%。进入 pilot 前必须增加新的不可变文件/来源，或通过新 recipe 显式降低其
700 bp 权重并重新分配；禁止循环重复这 1.4M token 来伪装容量。

r2 全文件实测证明 OER Commons permissive 子集同样不足，且不仅是 validation
比例问题，因此 r3 不再降低 quota 掩盖容量，而是替换来源。Gutenberg 和 USGPO
锁定文件虽然分别为 570,083,371 B 和 760,402,814 B，但压缩大小不是有效 token
证明；131,072-token 文档上限、逐行许可和稳定 group split 后仍必须用真实 tokenizer
完成 quota scan。

### 2.2 数学与科学论文

`common-pile/arxiv_papers_filtered` 只收录上传元数据声明为 CC BY、CC BY-SA 或
CC0 的论文，并保留 `metadata.license` 和原始 URL。v4 默认进一步排除 CC BY-SA，
因此只使用 CC0 和 CC BY。

ArXiv 和 FineMath 的侧重点不同：前者是论文，后者是经过教育质量分类的数学网页。
两者仍可能在网页镜像、作者主页和引用材料上重合，也可能包含 benchmark 题目或解答，
所以必须经过跨源 near-dedup 和 benchmark decontamination。

recipe 只锁定最小的 `arxiv-papers-0007.json.gz`。如果该尾分片存在时间或学科偏置，
20M smoke 的分布审计必须拒绝它；后续只能通过带新 SHA256 的 recipe 修订加入更多
分片，不能在下载器中偷偷扩大 glob。

### 2.3 教育质量代码

`common-pile/stackv2_edu_filtered` 是 Stack v2 的教育质量子集，JSON 行中直接包含
`text`，不需要再向 Software Heritage 在线取 blob。它带有 repo、path、revision、
detected licenses 和 provenance。

上游接受 Blue Oak 列表中的多种许可证；v4 不继承整个列表，只沿用项目已有的
Apache-2.0、BSD-2-Clause、BSD-3-Clause、CC0-1.0、ISC、MIT allowlist。必须继续
执行 secret scan、generated/vendor filtering、repo 级 validation split，以及和
现有 `codeparrot/github-code-clean` 的 repo/path/content 去重。

### 2.4 Common Corpus permissive fallback

原计划中的中文 Wikipedia 被替换为
`PleIAs/common_corpus@307910e4c5d040d6f318e6edf2a2b97849155771` 的固定
Parquet sample。这样可以默认排除 CC-BY-SA/GFDL。

该默认 config 实际只声明
`common_corpus_1/subset_100_1.parquet`，包含 69,907 行，而不是完整的
2.27T-token Common Corpus。v4 只允许：

- `language` 为 `Chinese` 或 `English`；
- `language_type` 为 `Written`；
- `open_type` 为 `Open Government`、`Open Science` 或 `Open Culture`；
- 许可证规范化后落入 permissive allowlist；
- 排除 `Wikipedia`、`Github Open Source` 和 `Wikidata` collection。

这仍是异构 sample，不能假定 40M formal quota 一定充足或分布均衡。smoke 必须输出
语言、collection、open_type、许可证、文档长度和 token 数分布；如果不满足 quota
或中文覆盖不足，应失败并重新选择固定数据，而不是解除过滤条件。

## 3. 固定文件范围

recipe 中每个来源只列一个已经通过 Hub API 核验 path、size、LFS SHA256 的最小文件，
`file_patterns` 是精确路径而不是 wildcard。完整 SHA256 见锁文件。

| 来源 | 固定最小文件 | 压缩/Parquet 大小 |
|---|---|---:|
| FineWeb-Edu | `fineweb-edu-dedup/train-00068-of-00234.parquet` | 2,251,835,383 B |
| FineWeb2 中文 | `data/cmn_Hani/train/004_00073.parquet` | 1,137,716,850 B |
| FineMath 4+ | `finemath-4plus/train-00060-of-00064.parquet` | 284,239,808 B |
| GitHub Code Clean | `data/train-00570-of-00880.parquet` | 345,453,831 B |
| Cosmopedia OpenStax | `data/openstax/train-00001-of-00002.parquet` | 173,491,421 B |
| Cosmopedia Stanford | `data/stanford/train-00002-of-00013.parquet` | 253,591,785 B |
| LibreTexts | `libretexts-0000.json.gz` | 115,037,937 B |
| USGPO | `usgpo-0000.json.gz` | 760,402,814 B |
| Project Gutenberg | `project_gutenberg-dolma-0000.json.gz` | 570,083,371 B |
| ArXiv | `arxiv-papers-0007.json.gz` | 210,893,770 B |
| Stack v2 Edu | `stack-edu-0094.json.gz` | 474,450,587 B |
| Common Corpus | `common_corpus_1/subset_100_1.parquet` | 429,962,586 B |

新来源最小文件共 2,560,831,065 B（约 2.385GiB）；全部十二个最小文件共
7,007,160,143 B（约 6.526GiB）。这只是锁定范围，不代表文件已经下载，也不证明
过滤后的 token 容量足够。

## 4. 许可证门

逐文档许可证先规范化成 canonical identifier，再执行 allowlist。默认允许：

- Public Domain；
- CC0-1.0；
- CC BY 1.0/2.0/2.5/3.0/4.0；
- Apache-2.0、BSD-2-Clause、BSD-3-Clause、ISC、MIT。

默认排除：

- CC-BY-SA；
- GFDL；
- GPL、AGPL、LGPL、MPL 等 copyleft；
- unknown、no-license 和无法可靠规范化的值。

数据集级 ODC-By 和 Apache-2.0 来源继续沿用现有 v1 治理范围。所有来源都必须生成
attribution manifest；许可证允许使用并不意味着可以丢弃 provenance。

嵌套字段如 `metadata.license` 是 schema v2 的真实字段路径；已实现的统一
dotted-field accessor 会读取它，缺失任何 required field 都会拒绝该行并统计原因。
旧 schema v1 的平面字段解析契约没有被悄悄放宽。

## 5. 去重、污染与安全门

正式 admission 顺序固定如下：

1. 以 repo、commit、path、compressed size、compressed SHA256 验证来源身份。
2. 解压 JSON Lines 或读取 Parquet，验证 required fields 和 stable ID。
3. 规范化并过滤逐行许可证，同时生成 attribution。
4. 运行现有 PII、低质量、异常长度和代码 secret scan。
5. 以规范化文本 SHA256 做 exact dedup。
6. 在所有 v4 来源和既有 v1/v2/v3 语料之间做 MinHash/near-duplicate 检查。
7. 对锁定的 validation、GSM8K、MATH、MMLU、ARC、HumanEval、MBPP 等评测材料执行
   13-gram contamination 检查。
8. 代码以 repo 分组，教材以 book/document stable ID 分组后再做 deterministic
   train/validation split，禁止同一 repo 或同一本书跨 split。
9. 物化后按来源 token quota 验收，并输出被拒原因、underfill、overshoot 和最终
   source mix。
10. 训练 loader 必须做 source-stratified optimizer-batch mixing，避免连续
    source-pure batches 重新引入周期性 loss 波动。

near-dedup、完整 contextual PII 和 benchmark 13-gram 已实现为独立的认证审计，
不是 extractor 暗中修改输入的步骤。`data audit-base` 当前默认 near-duplicate threshold
为 `0.8`，可通过 CLI 显式调整；最终值、扫描器源码摘要、输入 manifest、findings 和
rejection ledger 都必须进入 attestation。阈值变化要求重新审计，不能沿用旧 attestation。

## 6. 可运行 CLI、网络策略与治理状态

### 6.1 离线检查与 resolve

先做完全离线的 recipe 检查：

```bash
.venv/bin/python -m twen.cli data inspect-recipe \
  --recipe locks/base-data-sources-v4.json
```

当前结果为 schema v2、12 个来源、10,000 basis points、7,007,160,143 locked
bytes，activation 的 `missing_implementation=[]`。随后可离线预演 exact lock plan：

```bash
.venv/bin/python -m twen.cli data resolve-sources \
  --recipe locks/base-data-sources-v4.json \
  --output /tmp/base-data-sources-v4.resolved.candidate.json \
  --dry-run
```

`--dry-run` 只核验 recipe 内嵌计划，`remote_identity_verification=deferred`，不能代替
真实 Hub metadata 认证。需要重建 resolved lock 时才执行联网 resolve，并先写候选文件，
不要直接覆盖已经审阅的 lock：

```bash
.venv/bin/python -m twen.cli data resolve-sources \
  --recipe locks/base-data-sources-v4.json \
  --output /tmp/base-data-sources-v4.resolved.candidate.json \
  --network-policy fallback \
  --proxy <PROXY_URL>
```

2026-07-26 的宿主网络中 HF 直连曾超时，因此实现保持：

1. Hugging Face 先直连；
2. 只在网络错误时由同一进程切到配置代理；
3. GitHub 始终通过 GitHub proxy wrapper；
4. 不把全局代理污染到其他 host。

已认证 r3 resolved lock 当前 SHA256 为
`d552f5df400b6f6e4bd8516fa23dbe1a03f77dfecba08618ee7e2082b45817d3`，
其中 `materialization_audit.complete=true`；该文件来自一次真实 Hub metadata
比对，不是上面的离线 dry-run 计划。如果候选文件不同，必须审查差异并修订
recipe/lock，不能静默接受 mutable `main`。

### 6.2 build dry-run 与真实 22M smoke

在不联网、不写输出的情况下，先认证 recipe、resolved lock、tokenizer 和 22M
per-source quota：

```bash
.venv/bin/python -m twen.cli data build-base \
  --recipe locks/base-data-sources-v4.json \
  --resolved-lock locks/base-data-sources-v4.resolved.json \
  --output /tmp/base-v4-smoke-dry-run \
  --tokenizer artifacts/models/qwen3.5-0.8b-base \
  --tokenizer-manifest-sha256 5e847be98c2d114d2fb85a67ac77f4eef416e11871c1a5580d1bc52663e2e643 \
  --profile smoke \
  --dry-run
```

r3 resolved lock 生成后，dry-run 必须明确给出 `target_train_tokens=20000000`、
`target_validation_tokens=2000000`。真实物化使用新输出目录：

```bash
.venv/bin/python -m twen.cli data build-base \
  --recipe locks/base-data-sources-v4.json \
  --resolved-lock locks/base-data-sources-v4.resolved.json \
  --output data/base-v4-smoke-r3 \
  --tokenizer artifacts/models/qwen3.5-0.8b-base \
  --tokenizer-manifest-sha256 5e847be98c2d114d2fb85a67ac77f4eef416e11871c1a5580d1bc52663e2e643 \
  --profile smoke \
  --network-policy fallback \
  --proxy <PROXY_URL> \
  --stop-file data/base-v4-smoke-r3/STOP \
  --progress always
```

这一步只提取和认证数据，不启动训练。只有 corpus manifest 的十二个来源都达到各自
train/validation quota，才算 22M smoke 物化通过。r1 的
`data/base-v4-smoke` 和 `data/base-v4-smoke-r2` 保留为 underfill 证据，不能和
r3 产物混拼。

### 6.3 audit、`research_only` 与 prepare

`build` 完成的 extracted corpus 会把格式、许可、稳定 split、exact dedup、基础 PII、
代码 secret 和 provenance 标为 complete，但仍把以下三项标成 pending：

- `cross_source_near_dedup`；
- `full_contextual_pii_scan`；
- `project_benchmark_13gram_scan`。

因此 extracted manifest 是 `ready_for_data_prepare=true`、
`ready_for_training=false`。默认 `data prepare` 会拒绝它。正式路径应先执行
`data audit-base`，必要时按 rejection ledger 执行 `data materialize-audit` 并对过滤
结果重新审计；只有 attestation 的所有 gate 通过后，prepare 才能在不使用 override 的
情况下生成 `ready_for_training=true` 的产物。

本轮没有使用 pending-audit override。真实治理链为：

```bash
.venv/bin/python -m twen.cli data audit-base \
  --extracted-manifest data/base-v4-smoke-r3/corpus-manifest.json \
  --frozen-validation-manifest data/base-v3/corpus-manifest.json \
  --benchmark-registry locks/base-benchmark-registry.json \
  --benchmark-root data/benchmarks/base-v2 \
  --output data/base-v4-smoke-r3-audit-pass-001 \
  --near-duplicate-threshold 0.8 \
  --max-findings 10000

.venv/bin/python -m twen.cli data materialize-audit \
  --audit-attestation data/base-v4-smoke-r3-audit-pass-001/attestation.json \
  --output data/base-v4-smoke-r3-filtered-pass-001

# 对 filtered candidate train + filtered frozen-v3 validation 再审计；
# pass-002 的所有 gate 均为 complete，findings/rejections 均为 0。
.venv/bin/python -m twen.cli data audit-base \
  --extracted-manifest data/base-v4-smoke-r3-filtered-pass-001/corpus-manifest.json \
  --frozen-validation-manifest \
    data/base-v4-smoke-r3-filtered-pass-001/corpus-manifest.json \
  --benchmark-registry locks/base-benchmark-registry.json \
  --benchmark-root data/benchmarks/base-v2 \
  --output data/base-v4-smoke-r3-audit-pass-002 \
  --near-duplicate-threshold 0.8 \
  --max-findings 10000

.venv/bin/python -m twen.cli data prepare \
  --extracted-manifest \
    data/base-v4-smoke-r3-filtered-pass-001/corpus-manifest.json \
  --role train \
  --audit-attestation \
    data/base-v4-smoke-r3-audit-pass-002/attestation.json \
  --output artifacts/data/base-v4-smoke-r3-filtered-train \
  --tokenizer artifacts/models/qwen3.5-0.8b-base \
  --tokenizer-manifest-sha256 5e847be98c2d114d2fb85a67ac77f4eef416e11871c1a5580d1bc52663e2e643 \
  --sequence-length 4096 \
  --progress always
```

已认证身份如下：

- raw manifest SHA256:
  `9c62f5d80561a96bf7307b83a5a5a9ab8c7c3e46dad9e5125caa3091fc2acce8`；
- filtered manifest SHA256:
  `de9710487cd55c00791ae42e8762c49a0f8cddd5b792fcfd3af63310f9c8c837`；
- prepared manifest SHA256:
  `61afef2502d4703edb8ac44c0c7f74c4e75ee9335b9596fb2d3101ac282641b5`；
- prepared dataset fingerprint:
  `3071f7e6ef36e016da873d1a7d7983a0d525a99b8777c623c8b002de2168a8ec`；
- prepared source-map SHA256:
  `40d24b13d3f3e4a8a5f9108f9a004db46a7ec1a8778cef63d2952a3630318cf1`。

可随时重新认证并打印配置使用的 source-map SHA：

```bash
.venv/bin/python -m twen.cli data inspect-prepared \
  --manifest artifacts/data/base-v4-smoke-r3-filtered-train/manifest.json
```

pass-001 共扫描 36,902 篇 train/frozen-validation 文档，记录 2,596 个 gate event：
1,745 个 contextual-PII、839 个 benchmark overlap、12 个 train-vs-validation
near-duplicate。去重后过滤 1,456 篇 train 文档/4,067,482 token 与 1,099 篇 frozen
validation 文档/1,492,759 token。pass-002 对 15,337 篇 train 与 19,010 篇过滤后的
frozen validation 文档复扫，exact/near/PII/benchmark 全部为 0。

filtered manifest 的 validation role 有意来自 frozen v3 validation；原始 v4 的 2M
source-smoke validation 不进入最终 train prepared tensors。后续跨版本模型质量比较仍
单独使用原始、固定的 v3 project validation 口径。v4 是纯文本目标，prepare 后
**不执行** `generate-kd`。

## 7. 磁盘预算、停止与确定性恢复

十二个锁定输入合计约 6.526GiB，但这不是物化峰值。七个 Parquet 由 HTTP Range
流式读取；五个 gzip JSONL 必须完整持久化，压缩文件合计 2,130,868,479 B
（约 1.985GiB），还会保留经 SHA256 认证的未压缩 JSONL、row-offset index。
此外还有 22M-token train/validation JSONL、attribution/provenance ledger 和 prepared
safetensors。未压缩比、过滤保留率、行级 metadata 和 padding 尚未经真实 smoke 测量，
所以不能给出精确峰值；首次执行建议至少预留 **30GiB** 可用空间，并持续检查
`du -sh data/base-v4-smoke-r3 artifacts/data/base-v4-smoke-r3-*`。30GiB 是运行安全余量，
不是数据容量证明。

恢复边界如下：

- gzip 下载保留 partial bytes 并续传；验证过的压缩对象可复用；
- 解压失败保留 identity-bound `.incomplete`，重跑会从已验证压缩对象确定性重放；
- extracted 数据约每 1M token 事务提交 chunk、输出 SHA 和 source cursor；
- `STOP` 只在 durable chunk 边界停止；移走 `data/base-v4-smoke-r3/STOP` 后用**完全相同**
  的 build 命令恢复；
- prepare 逐 extracted shard 事务提交，重跑只处理未完成 shard；
- recipe、resolved lock、tokenizer、profile 或 pipeline fingerprint 改变时必须使用新
  输出目录，不能删除 marker 后把不兼容产物拼接在一起。

## 8. v4 16M governed smoke 配置

已发布候选配置为
[`configs/base/dense-v4-16m-smoke.yaml`](../configs/base/dense-v4-16m-smoke.yaml)。
它从 v3 final model-only checkpoint fork，但显式重新初始化 Muon/AdamW state、
scheduler 和逐来源 cursor。优化目标不读取 9B teacher logits：

- `data.mode=prepared-text`；
- causal NTP `1.0` + Qwen3.5 原生 frozen MTP head `0.1`；
- teacher KD、anchor KL、hidden alignment 全部为 `0.0`；
- 48 个二维 FFN Adapter 使用 Muon，24 个 branch scale 使用 AdamW；
- Adapter nominal peak LR `1e-4`、scale peak LR `3e-4`；
- 5M-token warmup，随后全程 cosine 到 peak 的 `0.1`；
- 目标为 16M committed token，global batch 初值为 262,144 token。

Muon 的 `match_rms_adamw` 会按矩阵形状调整实际 update coefficient；4096×1024
Adapter 的 factor 是 12.8。因此日志同时记录 nominal LR、adjusted LR 与 factor，
不能把 Muon 的 adjusted coefficient 直接当成 AdamW peak LR 比较。

审计后的 source 容量与 recipe 差距很大，尤其 USGPO 只保留
6,873 / 411,025 token。若按原 2% 权重训练 16M，它会被重复约 47 次；authenticated
refill 的保守估算又需要额外约 67M raw token，其中约 60M 都用于 USGPO。当前 smoke
因此显式启用 `source_mix_allow_weight_override=true`，以 largest-remainder 将
过滤后的 unique-token 容量归一化到 10,000 bp：

| source | clean token | lineage bp | effective bp |
|---|---:|---:|---:|
| FineWeb-Edu | 4,563,046 | 2,400 | 2,849 |
| FineWeb2 Chinese | 3,134,944 | 1,800 | 1,958 |
| FineMath | 2,070,788 | 1,400 | 1,293 |
| GitHub code | 1,213,922 | 800 | 758 |
| OpenStax | 514,821 | 300 | 322 |
| Stanford | 447,622 | 300 | 280 |
| LibreTexts | 1,284,747 | 700 | 802 |
| USGPO | 6,873 | 200 | 4 |
| Gutenberg | 288,518 | 400 | 180 |
| ArXiv | 460,080 | 500 | 287 |
| StackV2 | 741,447 | 400 | 463 |
| Common Corpus | 1,286,864 | 800 | 804 |
| **total** | **16,013,672** | **10,000** | **10,000** |

preflight 会同时认证 lineage 与 effective weights，并把
`source_mix_weight_override=true`、两组权重及新的 source-mix dataset fingerprint
写入日志和 checkpoint。默认仍 fail-closed：没有显式 flag 时，任何权重差异都会拒绝。

`sources.donor` 仍作为冻结 FFN/Adapter 架构血缘存在；`sources.teacher` 因 schema
兼容保留但不会构造在线 teacher。真正保证“不读取 9B logits/KD”的是 prepared-text
路径和三个 teacher-side loss 为零。

真实 optimizer-step 门禁已经完成：

| 候选 | 结论 | aggregate wall tok/s | peak reserved | 物理显存/功耗证据 |
|---|---|---:|---:|---|
| B1 | **采用** | 8,028 | 25.75 GiB | 最小 headroom 6.09 GiB；util p95 98%；power p95 604.19 W |
| B2 + AC20 | 可运行但拒绝 | 7,131 | 28.20 GiB | 比 B1 慢约 12.6%，headroom 仅 3.64 GiB |
| B2 无 checkpoint | 拒绝 | — | 外部峰值约 32,067 MiB | 首次 forward 触发 WSL/CUDA driver 容量边界 |
| B4 | 拒绝 | — | — | 即使 AC24 graph-smoke 仍失败 |

连续运行与 STOP → resume → SIGUSR1 分支在 1,048,576 token 后逐字节等价；最终
`artifacts/configuration/v4-optimizer-ab/summary.json` 同时要求 power、完整 profiler
trace 与 recovery，结论为 `accepted=true`。因此 Web allowlist 现在只对 v4 设
`launch_enabled=true`；v1/v2/v3 继续 monitor-only。

## 9. 明确排除的候选

| 候选 | 本轮不纳入原因 |
|---|---|
| `open-web-math/open-web-math` | 现有 FineMath card 明确说明加入 OpenWebMath URL，并使用相同抽取流程，重叠过高。 |
| SmolLM `python-edu` / HuggingFaceTB `stack-edu` | 行中只有 `blob_id` 和元数据，没有正文；需要在线访问 Software Heritage。 |
| `EleutherAI/proof-pile-2` | 依赖 dataset script，viewer 返回 script unsupported；脚本中的数据 URL 硬编码 `main`，且底层许可不统一。 |
| OpenCSG Chinese FineWeb Edu v2.x | Hub 的 Apache-2.0 tag 与 README 中 OpenCSG Community License、商业使用需邮件许可的要求冲突。 |
| BAAI CCI3-HQ / IndustryCorpus2 | 主仓库 gated；ungated 行业子仓库缺少可核验的 README/license 声明。 |
| `wikimedia/wikipedia` 中文 | 数据质量高，但本 recipe 默认排除 CC-BY-SA/GFDL，因此用 Common Corpus permissive sample 替代。 |
| `PleIAs/common_corpus` 完整仓库 | README 描述 2.27T token，但 default config 只绑定一个 sample；不得把完整规模宣传当作当前锁定输入规模。 |

## 10. 进入 500M formal 前的硬门

500M formal 只有在以下证据齐全后才可启动：

- schema v2、`jsonl_gzip`、prepared-text source mixer 和 Muon 恢复测试通过；
- 每个 locked file 的 compressed SHA256 和 derived artifact SHA256 完整；
- 当前 16M governed smoke 的 Muon、恢复、source-mix override 和 Web 运行门全部通过；
- 新 recipe/refill 以 unique token 恢复至少 20M clean train，并替换或显著降低
  USGPO 这类单个 13-gram 命中会淘汰整篇长文档的病态来源；不得用重复采样冒充容量；
- 250M pilot 没有 silent underfill、source-pure batch 或数据读取瓶颈；
- Common Corpus 的实际语言/collection 分布被报告并满足用途；
- 单个最小尾分片不存在不可接受的时间、领域或语言偏置；
- 新增或替换任何分片都通过 recipe revision 和新锁完成，而不是修改现有 manifest。
