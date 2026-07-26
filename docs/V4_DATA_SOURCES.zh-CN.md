# v4 Base 文本数据源与治理草案

状态：**设计草案，不可直接运行**

对应锁文件：[`locks/base-data-sources-v4.json`](../locks/base-data-sources-v4.json)

本方案面向纯文本 causal NTP + Qwen3.5 原生 MTP 的 v4 Base Dense Adapter
训练。它不生成、下载或消费 9B teacher logits，也不启用 teacher hidden
alignment。

锁文件明确使用 `schema_version: 2` 和
`kind: twen_base_data_source_recipe_v2_draft`。当前 v1 数据解析器只接受原生
Parquet，不能解析其中五个 `jsonl_gzip` 来源，也不能执行 dotted metadata field
和许可证规范化。因此，在 schema v2、gzip 输入和 source-stratified batch mixer
完成并通过测试前，不得把该文件交给现有 `data resolve-sources` 或
`data build-base` 当作正式配置。

## 1. 三档规模与固定配比

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
| OER Commons permissive | 新 | 2% | 0.4M | 5M | 10M |
| Pressbooks permissive | 新 | 4% | 0.8M | 10M | 20M |
| Open-license ArXiv | 新 | 5% | 1M | 12.5M | 25M |
| Stack v2 Edu permissive | 新 | 4% | 0.8M | 10M | 20M |
| Common Corpus permissive sample | 新 | 8% | 1.6M | 20M | 40M |
| **合计** | **旧 70% / 新 30%** | **100%** | **20M** | **250M** | **500M** |

严格 held-out validation 另保留 20M token，并采用同一来源比例。任何来源在许可、
过滤、去重后无法满足 quota，都必须失败并修订 recipe；不得静默重分配到其他来源，
否则不同版本的训练不再可比。

20M smoke 只验证下载锁、解压、字段、许可证、去重、切分、token quota、确定性恢复
和 batch mixing，不用于得出模型质量结论。250M pilot 用于确认 v4 的纯文本目标、
优化器和学习率设置。只有 pilot 的数据审计和 held-out 结果均通过后，才允许启动
500M formal。

## 2. 为什么选择这些新增来源

### 2.1 开放教材

`common-pile/libretexts_filtered`、`common-pile/oercommons_filtered` 和
`common-pile/pressbooks_filtered` 提供教材、课程章节、讲义、问题集和教学材料。
三者都有自包含的 `text`，并在 `metadata.license`、`metadata.url` 和
`metadata.provenance` 中保留逐文档许可与来源。

它们比继续扩大 Common Crawl web 比例更能增加教材式解释。三个站点可能转载同一本
开放教材，LibreTexts 和 Pressbooks 也可能包含 OpenStax 内容，因此必须在三个来源
之间，并与已有 Cosmopedia OpenStax 做 exact + near-duplicate 检查。对于
Cosmopedia 的合成改写，语义重叠无法只靠文本哈希完全消除，应在报告中单独统计。

LibreTexts 的固定 revision 在 dataset card 中声称 3.6GB UTF-8，但
datasets-server 对同一 revision 给出约 386.8MB decoded bytes。锁文件保留了这项
差异；实际 admission 只能依据本地物化后的行数和 tokenizer token 数，不能依据 card
估算。

### 2.2 数学与科学论文

`common-pile/arxiv_papers_filtered` 只收录上传元数据声明为 CC BY、CC BY-SA 或
CC0 的论文，并保留 `metadata.license` 和原始 URL。v4 默认进一步排除 CC BY-SA，
因此只使用 CC0 和 CC BY。

ArXiv 和 FineMath 的侧重点不同：前者是论文，后者是经过教育质量分类的数学网页。
两者仍可能在网页镜像、作者主页和引用材料上重合，也可能包含 benchmark 题目或解答，
所以必须经过跨源 near-dedup 和 benchmark decontamination。

草案只锁定最小的 `arxiv-papers-0007.json.gz`。如果该尾分片存在时间或学科偏置，
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

草案中每个来源只列一个已经通过 Hub API 核验 path、size、LFS SHA256 的最小文件，
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
| OER Commons | `oercommons-0000.json.gz` | 17,028,282 B |
| Pressbooks | `pressbooks-0000.json.gz` | 191,725,499 B |
| ArXiv | `arxiv-papers-0007.json.gz` | 210,893,770 B |
| Stack v2 Edu | `stack-edu-0094.json.gz` | 474,450,587 B |
| Common Corpus | `common_corpus_1/subset_100_1.parquet` | 429,962,586 B |

新来源最小文件共 1,439,098,661 B（约 1.340GiB）；全部十二个最小文件共
5,885,427,739 B（约 5.481GiB）。这只是锁定范围，不代表文件已经下载，也不证明
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

嵌套字段如 `metadata.license` 是 schema v2 的真实字段路径，不是当前 v1 parser
支持的平面字段。实现时应提供统一的 dotted-field accessor，缺失任何 required
field 都要拒绝该行并统计原因。

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

near-dedup 参数和阈值必须在实现 admission 前进入锁或 manifest。本文不臆造一个
未经 benchmark 的阈值。

## 6. 网络与下载策略

2026-07-26 的核验中，HF 直连在宿主网络连接 10 秒后超时，随后才切换到项目代理。
正式实现仍应保持：

1. Hugging Face 先直连；
2. 只在网络错误时由同一进程切到配置代理；
3. GitHub 始终通过 GitHub proxy wrapper；
4. 不把全局代理污染到其他 host。

schema v2 实现完成后的预期命令为：

```bash
.venv/bin/python -m twen.cli data resolve-sources \
  --recipe locks/base-data-sources-v4.json \
  --output locks/base-data-sources-v4.resolved.json \
  --network-policy fallback \
  --proxy http://172.23.240.1:8080
```

当前版本不得执行这条命令，因为 v1 resolver 会拒绝 schema v2/jsonl_gzip。已明确
确认直连不可用时才可把 `fallback` 改成 `proxy`。HF 不应套用
`scripts/with_github_proxy.sh`。

## 7. 明确排除的候选

| 候选 | 本轮不纳入原因 |
|---|---|
| `open-web-math/open-web-math` | 现有 FineMath card 明确说明加入 OpenWebMath URL，并使用相同抽取流程，重叠过高。 |
| SmolLM `python-edu` / HuggingFaceTB `stack-edu` | 行中只有 `blob_id` 和元数据，没有正文；需要在线访问 Software Heritage。 |
| `EleutherAI/proof-pile-2` | 依赖 dataset script，viewer 返回 script unsupported；脚本中的数据 URL 硬编码 `main`，且底层许可不统一。 |
| OpenCSG Chinese FineWeb Edu v2.x | Hub 的 Apache-2.0 tag 与 README 中 OpenCSG Community License、商业使用需邮件许可的要求冲突。 |
| BAAI CCI3-HQ / IndustryCorpus2 | 主仓库 gated；ungated 行业子仓库缺少可核验的 README/license 声明。 |
| `wikimedia/wikipedia` 中文 | 数据质量高，但本草案默认排除 CC-BY-SA/GFDL，因此用 Common Corpus permissive sample 替代。 |
| `PleIAs/common_corpus` 完整仓库 | README 描述 2.27T token，但 default config 只绑定一个 sample；不得把完整规模宣传当作当前锁定输入规模。 |

## 8. 进入 500M formal 前的硬门

500M formal 只有在以下证据齐全后才可启动：

- schema v2 和 `jsonl_gzip` 实现、单元测试、断点恢复测试通过；
- 每个 locked file 的 compressed SHA256 和 derived artifact SHA256 完整；
- 20M smoke 的每源 quota、许可证、attribution、去重和污染审计全部通过；
- 250M pilot 没有 silent underfill、source-pure batch 或数据读取瓶颈；
- Common Corpus 的实际语言/collection 分布被报告并满足用途；
- 单个最小尾分片不存在不可接受的时间、领域或语言偏置；
- 新增或替换任何分片都通过 recipe revision 和新锁完成，而不是修改现有 manifest。
