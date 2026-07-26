# Base 数据源锁定与采样建议

更新日期：2026-07-17。

本文锁定数据源、许可信息、字段、固定 revision、采样比例和审计规则。所有上游规模均来自
对应固定 revision 的官方 dataset card；实际配额由本项目锁定的 Qwen3.5-0.8B-Base
tokenizer 重新计算。本轮 ungated Base v3 已提取完成；本文仍不代表接受任何未使用的 gated
数据集条款。

## 结论

建议 Base 路线使用下列五类数据，目标比例为 **35% 英文通用、25% 中文、15% 代码、
15% 数学、10% 科学教育**。前四类可以直接锁定；代码类只有经过逐行许可筛选、来源留档和
secret/PII 扫描后才可以进入训练集。

| 角色 | 数据集 / config / split | 固定 commit | 页面许可和访问 | 官方规模、字段与结论 |
|---|---|---|---|---|
| 英文通用 | `HuggingFaceTB/smollm-corpus` / `fineweb-edu-dedup` / `train` | `3ba9d605774198c5868892d7a8deda78031a781f` | ODC-By；ungated | 220B tokens、190,168,005 documents；`text`, `id`, `metadata`。使用已经去重的版本，不使用未去重的原始 FineWeb-Edu。原生文件为 `fineweb-edu-dedup/train-*`。[固定 card](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/3ba9d605774198c5868892d7a8deda78031a781f/README.md) |
| 中文 | `HuggingFaceFW/fineweb-2` / `cmn_Hani` / `train` | `af9c13333eb981300149d5ca60a8e9d659b276b9` | ODC-By；ungated；另受 Common Crawl ToU 约束 | Mandarin Chinese 配置名是 `cmn_Hani`，不是 `zho_Hans`；636,058,984 documents、543,543,038,750 words、1.48TB on disk。`text`, `id`, `dump`, `url`, `date`, `file_path`, `language`, `language_score`, `language_script`, `minhash_cluster_size`, `top_langs`, `wordlist_ratio`。原生文件为 `data/cmn_Hani/{train,test}/*.parquet`。[固定 card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/blob/af9c13333eb981300149d5ca60a8e9d659b276b9/README.md) |
| 数学 | `HuggingFaceTB/finemath` / `finemath-4plus` / `train` | `e92b25a616738fe95dc186b64dfb19f9c8525594` | ODC-By；ungated；另受 Common Crawl ToU 约束 | 9.6B tokens、6,699,493 documents；`text`, `url`, `token_count`, `score`, `language` 等。已经 MinHash 去重，并以 13-gram 对 GSM8K、MATH、MMLU、ARC 去污染。原生文件为 `finemath-4plus/train-*`。[固定 card](https://huggingface.co/datasets/HuggingFaceTB/finemath/blob/e92b25a616738fe95dc186b64dfb19f9c8525594/README.md) |
| 科学教育 | `HuggingFaceTB/cosmopedia` / `stanford` + `openstax` / `train` | `0ae6ec63f91742bd2d1eaef4f02232c55d719385` | Apache-2.0；ungated | Mixtral-8x7B-Instruct-v0.1 合成；`text`, `prompt`, `text_token_length`, `seed_data`, `format`, `audience`。全库 25B tokens；这两个配置分别有 1,020,024 和 126,332 documents。已经对公开评测集去污染，但合成幻觉和风格模式是固有风险，故限制为 10%。原生文件为 `data/{stanford,openstax}/train-*`。[固定 card](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/blob/0ae6ec63f91742bd2d1eaef4f02232c55d719385/README.md) |
| 代码（条件源） | `codeparrot/github-code-clean` / Parquet `train` | `c48d40f9e70f0196f8236901ee35807f7d6c44c0` | 顶层 card 标 Apache-2.0，但源代码实际是逐行多许可证；ungated | 115M files，字段 `code`, `repo_name`, `path`, `license`, `size`；880 个原生 Parquet：`data/train-00000-of-00880.parquet` 至 `data/train-00879-of-00880.parquet`。只有逐行白名单、保留 attribution ledger 并完成 secret/PII 扫描后方可使用。[固定 card](https://huggingface.co/datasets/codeparrot/github-code-clean/blob/c48d40f9e70f0196f8236901ee35807f7d6c44c0/README.md) / [固定 loader](https://huggingface.co/datasets/codeparrot/github-code-clean/blob/c48d40f9e70f0196f8236901ee35807f7d6c44c0/github-code-clean.py) |

ODC-By 是数据库许可证，不自动证明每个网页正文的底层版权均已清权。因此这个组合适合作为
有来源、可复现的研究训练语料，但当前三项 pending 审计要求 prepared 产物保持
`research_only=true`，且不能被描述为“所有正文均 rights-cleared”的商业语料。
若有商业发布计划，仍需法务复核网页内容、Common Crawl ToU、归属展示和删除请求流程。

## 固定 token 配方与本轮校准子集

| 阶段 | 英文通用 | 中文 | 代码 | 数学 | 科学 | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| 单卡首训 PoC（20 个 global batches） | 1,835,008 | 1,310,720 | 786,432 | 786,432 | 524,288 | 5,242,880 |
| 本轮 train-only 校准子集（24 shards） | 6,750,936 | 4,973,188 | 3,312,171 | 3,226,077 | 1,684,755 | 19,947,127 |
| dense 正式语料 | 35,000,000 | 25,000,000 | 15,000,000 | 15,000,000 | 10,000,000 | 100,000,000 |
| sparse 扩展语料 | 175,000,000 | 125,000,000 | 75,000,000 | 75,000,000 | 50,000,000 | 500,000,000 |
| 固定验证集 | 7,000,000 | 5,000,000 | 3,000,000 | 3,000,000 | 2,000,000 | 20,000,000 |

代码内部建议按 token 再分：Python 35%；C/C++ 20%；JavaScript/TypeScript 20%；Java
10%；Go/Rust 10%；SQL/Shell 5%。Cosmopedia 的科学配额在 `stanford` 与 `openstax`
之间各取一半。FineWeb2 card 提到 rehydration 会改善部分语言的下游表现，但这里为了保持
低重复率不按 `minhash_cluster_size` 重采样；该偏离必须写入最终数据 manifest。

## 可复现获取方式

仓库实现使用以下两个 machine-readable lock：

- `locks/base-data-sources.json`：许可、字段、固定 commit、过滤规则及 PoC/dense/sparse/
  validation token 配额；
- `locks/base-data-sources.resolved.json`：1,563 个原生 Parquet 的固定 URL、LFS SHA256
  与 size，文件 SHA256 为
  `c5098da9b49c2f8fe755a4cb73d107677fa47e5e0beaea492207c8bf5e009d35`。

实际 CLI 使用 PyArrow + fsspec Range 读取并事务化写 JSONL/ledger：

```bash
python -m twen.cli data build-base \
  --recipe locks/base-data-sources.json \
  --resolved-lock locks/base-data-sources.resolved.json \
  --output data/base-v3 \
  --tokenizer artifacts/models/qwen3.5-0.8b-base \
  --tokenizer-manifest-sha256 5e847be98c2d114d2fb85a67ac77f4eef416e11871c1a5580d1bc52663e2e643 \
  --profile dense \
  --network-policy fallback \
  --stop-file data/base-v3/STOP \
  --progress always
```

唯一有效的 extracted corpus 是 `data/base-v3`；`base-v1`、`base-v2` 均有
`INVALIDATED.json`。Base v3 的 manifest SHA256 为
`2fb769833bf1507c8ed476e34c2ef6d7c2bb2d94b02807d6b075d350e8e76690`，corpus
fingerprint 为 `15dac54e0159d20b566a9e8002014aee5b76b298810bde32ed76cc5aec40d325`，
实际包含 100,007,485 train token、20,014,392 validation token，以及 120/120/120 个
train/validation/attribution 文件。

当前实现完成 exact dedup、基础 PII 正则 reject、代码 secret 正则和逐行许可过滤；MinHash
near-dedup、项目最终 benchmark 13-gram 与 contextual PII 必须作为后续审计执行，提取
manifest 会将它们标为 `pending` 并保持 `ready_for_training=false`。

`data prepare` 默认因此 fail closed。研究路线必须显式写出风险接受，并分别认证两个 role：

```bash
python -m twen.cli data prepare \
  --extracted-manifest data/base-v3/corpus-manifest.json \
  --role train \
  --allow-pending-research-audits \
  --output artifacts/data/base \
  --tokenizer artifacts/models/qwen3.5-0.8b-base \
  --tokenizer-manifest-sha256 5e847be98c2d114d2fb85a67ac77f4eef416e11871c1a5580d1bc52663e2e643 \
  --sequence-length 4096 --progress always

python -m twen.cli data prepare \
  --extracted-manifest data/base-v3/corpus-manifest.json \
  --role validation \
  --allow-pending-research-audits \
  --output artifacts/data/base-validation \
  --tokenizer artifacts/models/qwen3.5-0.8b-base \
  --tokenizer-manifest-sha256 5e847be98c2d114d2fb85a67ac77f4eef416e11871c1a5580d1bc52663e2e643 \
  --sequence-length 4096 --progress always
```

prepared schema v2 会记录 authenticated extracted lineage、全部 pending audits 和
`research_only=true`；KD schema v2 再通过 prepared dataset fingerprint 绑定这段 lineage。
preflight 的技术校验通过不等于三个数据治理审计已经完成，旧 prepared/KD schema v1 产物
必须重建。

优先使用 Hugging Face 原生 Parquet 和固定 commit，不引用会随时间变化的 `main`，也不依赖
未单独锁定 SHA 的 `refs/convert/parquet`。标准流式入口如下：

```python
from datasets import load_dataset

rows = load_dataset(
    "HuggingFaceTB/finemath",
    "finemath-4plus",
    split="train",
    revision="e92b25a616738fe95dc186b64dfb19f9c8525594",
    streaming=True,
)
```

其他三个标准数据集只需替换 repo、config、split 和 revision。中文的官方独立 split 可用
`split="test"`，且绝不能并入训练。原始文件也可以由 Hub API 列举后逐个下载：

```text
GET https://huggingface.co/api/datasets/<repo>/revision/<commit>
GET https://huggingface.co/api/datasets/<repo>/tree/<commit>/<directory>
GET https://huggingface.co/datasets/<repo>/resolve/<commit>/<file>
```

代码集不要依赖旧 Python builder 的可执行性。先用固定 revision 的 Hub API 列出 880 个
Parquet，再把固定 `resolve/<commit>/...parquet` URL 列表交给：

```python
load_dataset("parquet", data_files={"train": urls}, split="train", streaming=True)
```

HF 直连失败时可以使用项目现有 `fallback` 网络策略回退代理；下载 lock 必须保存 repo、
commit、config、split、逐文件相对路径、size、etag/LFS oid 和下载后 SHA256。

## 分流、去重和污染规则

1. 在写任何训练 JSONL 之前锁定 validation。稳定键分别为 FineWeb-Edu `id`、FineWeb2
   `id`、FineMath `url`、Cosmopedia `sha256(prompt + text)`；代码必须用 `repo_name` 分流，
   避免同仓库文件跨 train/validation。
2. 使用带 recipe seed 的 SHA256 做确定性分桶。验证集先达到上表固定配额；所有被分到验证
   桶但超过配额的记录也永久丢弃，不能回流训练。
3. 当前 extracted-corpus schema v1 按 recipe 中的 source 顺序、原生文件顺序和行顺序执行全局
   deterministic first-wins exact dedup；因此最终 train/validation 不会同时保留同一份
   规范化正文，但**不宣称 validation-first 优先级**。跨源 MinHash near-dedup 以及相对
   validation 的 near-duplicate 拒绝仍是训练前待完成审计。若未来要求 validation-first，
   必须升级为先冻结完整 validation 索引、再扫描 train 的两阶段 extractor，并变更 schema。
4. 对最终使用的评测题库维护独立 exclusion registry。所有训练文本做 13-gram overlap
   扫描；FineMath 自带的四套去污染结果不能替代本项目针对全部评测集的扫描。
5. 所有网页再次扫描 email、IP、电话、身份证样式和 URL query secrets。代码额外执行
   API key/private key/password 扫描，并排除 vendor、minified、generated、二进制转储和
   超长行；只依赖上游过滤不够。
6. 每条输出 JSONL 至少保留旁路 provenance（可存独立 ledger）：source repo、commit、
   config、source stable id、原 URL/repo/path、源许可证、normalized SHA256、split bucket、
   filter decisions。供训练的正文 JSONL 仍只需 `{"text": "..."}`。

代码许可白名单暂定：`mit`, `apache-2.0`, `bsd-2-clause`, `bsd-3-clause`, `isc`,
`cc0-1.0`。明确拒绝 `gpl-*`, `agpl-*`, `lgpl-*`, `epl-*`, `mpl-*`, `artistic-*`,
缺失/未知许可证。即使是白名单，也要保留 `repo_name`, `path`, `license` 和正文 hash 的
归属 ledger；Apache/BSD 等可能要求 notices/attribution。

## 不应直接使用的数据

| 数据集 | 不进入默认配方的原因 |
|---|---|
| `BAAI/CCI3-HQ` (`d6f3aa30cebfef497e822ff968ed68a18bf90b8f`) | gated；HF card 没有 license 字段，另要求外部 CCI 使用协议；来源正文可能含期刊等版权内容。未逐条审阅外部协议前不能安全自动使用。[固定 card](https://huggingface.co/datasets/BAAI/CCI3-HQ/blob/d6f3aa30cebfef497e822ff968ed68a18bf90b8f/README.md) |
| `BAAI/IndustryCorpus2` (`1721eecd696e4110d33a255440f3c7ce981140ee`) | 虽标 Apache-2.0，但 gated，且明确混合 Pile、BigCode、OpenWebMath 和未充分枚举的网页来源；缺少逐条许可/provenance，顶层标签不足以覆盖内容。[固定 card](https://huggingface.co/datasets/BAAI/IndustryCorpus2/blob/1721eecd696e4110d33a255440f3c7ce981140ee/README.md) |
| `Skywork/SkyPile-150B` (`d6395caa3005bcbf21dd80585c15f60004f77ccb`) | 使用 Skywork Community License + Apache 条款而非单一标准数据许可；card 明示仍可能含 email、电话、IP 和偏见。除非先审阅并接受自定义协议，否则不自动使用。[固定 card](https://huggingface.co/datasets/Skywork/SkyPile-150B/blob/d6395caa3005bcbf21dd80585c15f60004f77ccb/README.md) |
| `HuggingFaceTB/stack-edu` / `bigcode/the-stack-v2` | Stack-Edu 只有 SWH IDs，没有正文；正文批量获取涉及 Software Heritage/INRIA 协议。The Stack v2 gated、逐文件许可证、可能含 secrets/PII，并要求跟进持续 opt-out 更新，和永久固定旧 revision 存在治理冲突。[Stack-Edu card](https://huggingface.co/datasets/HuggingFaceTB/stack-edu/blob/eeec5caac5cc3758a18f1d3ba4416837a9ba814c/README.md) / [The Stack v2 条款](https://huggingface.co/datasets/bigcode/the-stack-v2/blob/7408bfbcfd48e5833d62fd3dba48afd20d109473/README.md) |
| `nampdn-ai/tiny-codes` (`9aebe5ee8b406356d5f5f2d603bc0a1684ee8ce7`) | card 标 MIT，但 gated，且没有充分披露具体生成模型、生成条款和逐样本 provenance；不作为审计友好的默认代码源。[固定 card](https://huggingface.co/datasets/nampdn-ai/tiny-codes/blob/9aebe5ee8b406356d5f5f2d603bc0a1684ee8ce7/README.md) |
| `open-web-math/open-web-math` | ODC-By 信息只在 dataset info 中，较早且不含 FineMath 的完整去污染流程。FineMath-4+ 已是更清楚的替代，避免两者重复混入。 |
