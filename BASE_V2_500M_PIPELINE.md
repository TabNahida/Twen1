# Base v2 500M 数据流水线

`scripts/prepare_base_v2_500m.py` 是一个只处理数据的可恢复状态机。它不会加载 teacher、不会
初始化 CUDA、不会创建 optimizer，也不会启动训练。

## 硬门与固定身份

- recipe SHA256：`aa9b774971480b634561557d731e1be9616044ead8a4354904708600e3254916`
- resolved lock SHA256：`c5098da9b49c2f8fe755a4cb73d107677fa47e5e0beaea492207c8bf5e009d35`
- benchmark registry SHA256：`defe66fa003eb4d5d00fa92e975ec7e923e1ec7399bddf0816bbc65cedfbb5e8`
- tokenizer manifest SHA256：`5e847be98c2d114d2fb85a67ac77f4eef416e11871c1a5580d1bc52663e2e643`
- 配额：500,000,000 train token + 20,000,000 validation token。
- `run` 还要求 batch-size/GPU 利用率报告
  `artifacts/benchmarks/rtx5090-base-dense-utilization-report.json` 已存在且
  `accepted=true`。最终推荐不硬编码 batch 数：门会验证 ordinary/alignment 使用同一 physical
  batch、每个 logical batch 都能整除 262,144 global token、所选两行的保守 physical headroom
  均不低于 3 GiB，并检查 finite loss/梯度/吞吐、72 个 trainable tensor 梯度、MTP/offload、
  optimizer reserve、no-optimizer 以及所选 benchmark artifact SHA。这样修复后的 SDPA sweep 可以
  安全选择 b1、b2 或 b4，同时边界显存档位不能被误当成 production 配置。
- 这个 performance gate 只授权数据状态机继续做 CPU audit/prepare，不授权训练配置。当前绑定的
  B1 approval 可以继续保护已在运行的数据流水线，但 v2 finalizer 不会把它复用为新的 B2 folded
  production evidence。
- 单独的 accepted report 仍不能放行。主任务在确认 eager 4K² MTP attention 问题已修复且新
  sweep 有效后，必须显式生成默认不存在的 companion approval。approval 绑定 report、
  `scripts/benchmark_full_dense_graph.py` 与 `src/twen/modeling/mtp.py` 的 SHA；任一文件变化都会
  重新关闭数据门。
- 旧的 `rtx5090-base-dense-batch2-utilization-report.json` 即使存在也只会显示为
  `legacy_report/accepted_for_pipeline=false`；新流水线不接受带误导性 batch2 名称的报告或
  approval。
- report 还必须带同前缀 `MANIFEST.json` 与 `COMPLETE`：MANIFEST 恰好认证 report JSON、
  Markdown 和三张 SVG 的 path/size/SHA，COMPLETE 再绑定 MANIFEST/report SHA、最终推荐与
  canonical source-provenance SHA；raw profiler 不属于 bundle。缺文件、额外文件或任一 SHA
  漂移都会 fail closed。

HF 采用 `fallback`：先直连，失败后才使用环境或 `--proxy` 指定的代理。resolved lock 虽包含
1,563 个原生 Parquet、远端全集约 2.51 TB，extractor 使用 HTTP Range 流式读取，只取达到配额
所需的数据。

## 目录隔离

- 初始提取：`data/base-v2-500m`
- 第 N 轮审计：`artifacts/data/base-v2-500m-audit-pass-NNN`
- 第 N 轮过滤：`data/base-v2-500m-filtered-pass-NNN`
- 第 N 代补量计划：`artifacts/data/base-v2-500m-refill-plan-NNN`
- 第 N 代不可覆盖 raw lineage：`data/base-v2-500m-refill-raw-NNN`
- 最终 train prepared：`artifacts/data/base-v2-500m`
- 状态、日志和总 COMPLETE：`artifacts/data/base-v2-500m-pipeline`

已有 `data/base-v1`、明确 invalidated 的 `data/base-v2`、以及冻结 validation 来源
`data/base-v3` 都不会被改写。审计或过滤目录一旦存在但身份无效，状态机会 fail closed，不会删除
或覆盖它。

## 审计后逐源补量

A0 的真实物化结果为 422,329,390 train token 和 18,521,633 validation token，不能通过降低
500M/20M 门或把 rejected document 重新计入来解决。状态机在每次
`materialize-audit` 后从完整 rejection ledger 去重 document identity，再从 candidate/frozen
attribution ledger 反查 `token_count_with_eos`，对六个来源分别计算 raw、clean、rejected 和
observed survival。事件数不能直接当文档数：同一文档可能同时命中 PII、benchmark 或 near-dup
gate。

每个来源的 train 和 validation runtime raw target 使用同一保守公式：

```text
clean_target = ceil(original_quota * (1 + 0.02))
guarded_survival = observed_survival - 0.01
runtime_raw_target = max(observed_raw, ceil(clean_target / guarded_survival))
```

`0.02` 是 clean token 余量，`0.01` 是 survival 下调百分点，CLI 只能调高不能调低。计划
`plan.json` 和 `COMPLETE` 绑定 A0 attestation SHA、rejection ledger、原 raw/materialized
manifest SHA、recipe/lock/tokenizer、六源原 cursor、原 chunk/source fingerprint、逐源公式和最终
runtime target。

refill builder 不修改 recipe，也不修改原 manifest。它在新目录中 hard-link 每个原 committed
chunk，认证相同 device/inode/size/SHA，从原 manifest 的逐源 `(file_index,row_index)` 后继续
HF Range 读取。全量 original raw 文本先进入 seen-hash set，因此续读不能重新接纳旧文档；新增
chunk 继续使用原 `sparse` pipeline/source fingerprint。merged raw 的 manifest/COMPLETE 额外
绑定 hardlink inventory、每个新增 chunk COMPLETE/output SHA 和 refill plan。代理仍是
`fallback`（HF 直连优先），持久化命令中的 `--proxy` 会脱敏。

每一代 merged raw 都必须重新执行完整 audit 和 materialize。若任一来源的 train 或 validation
仍低于原配额，状态机从新 cursor 生成下一代不可覆盖计划并循环；只有六来源逐项以及 500M/20M
总量同时通过、最终过滤产物又通过独立 re-audit，才允许进入 `data prepare`。不存在降低配额、
覆盖原 lineage、从头读取或跳过 gate 的路径。

独立 CLI（通常由状态机调用）为：

```bash
python -m twen data plan-base-refill \
  --audit-attestation artifacts/data/base-v2-500m-audit-pass-000/attestation.json \
  --base-raw-manifest data/base-v2-500m/corpus-manifest.json \
  --materialized-manifest data/base-v2-500m-filtered-pass-001/corpus-manifest.json \
  --output artifacts/data/base-v2-500m-refill-plan-001

python -m twen data build-base-refill \
  --plan artifacts/data/base-v2-500m-refill-plan-001/plan.json \
  --resolved-lock locks/base-data-sources.resolved.json \
  --output data/base-v2-500m-refill-raw-001 \
  --tokenizer artifacts/models/qwen3.5-0.8b-base \
  --tokenizer-manifest-sha256 5e847be98c2d114d2fb85a67ac77f4eef416e11871c1a5580d1bc52663e2e643 \
  --network-policy fallback
```

## 使用

只读检查：

```bash
.venv/bin/python scripts/prepare_base_v2_500m.py --action preflight
.venv/bin/python scripts/prepare_base_v2_500m.py --action plan
.venv/bin/python scripts/prepare_base_v2_500m.py --action status
```

性能门通过并经主任务协调后，启动或恢复相同流水线：

```bash
.venv/bin/python scripts/prepare_base_v2_500m.py --action run
```

companion approval 只能由主任务在新 sweep 验收后显式执行；数据流水线代理不会自行执行：

```bash
.venv/bin/python scripts/prepare_base_v2_500m.py \
  --action approve-performance \
  --acknowledge-native-mtp-attention-fix
```

已有 approval 不会被覆盖；其 report 或源码 SHA 不再匹配时必须重新走主任务验收流程。

若 HF 直连失败且环境没有代理，可显式提供 fallback proxy；URL 会在持久化状态和命令日志中
脱敏：

```bash
.venv/bin/python scripts/prepare_base_v2_500m.py --action run \
  --proxy http://127.0.0.1:7890
```

安全停止 extractor：

```bash
touch data/base-v2-500m/STOP
```

当前约 1M token chunk 完整提交后命令以 75 退出；删除 `STOP` 并重新执行完全相同的 `run`
命令即可恢复。已提交 chunk、SHA 和 sidecar 不会重写。

## 状态、SHA 与 ETA

`--action status` 会报告当前阶段、已提交 token、仅基于本次尝试新增提交量计算的吞吐和 ETA、
所有已完成 audit gate/metrics、prepared 身份以及每个持久化文件的 SHA256。每个子命令的合并
stdout/stderr 位于 `artifacts/data/base-v2-500m-pipeline/logs/`。

每个阶段都有初始 ETA 及其估算依据；build 在本次尝试提交新 chunk 后改用实测 committed-token
速率。每个完成阶段写入 `status.json.history`，包含开始/结束时间、耗时、退出码、输出 SHA 和
`eta_seconds=0`，并在 `phases/` 写独立、带 SHA 身份的 `*.COMPLETE.json`。流水线只有在 audit
全 gate 通过、train prepared manifest 再验证成功后才写总 `COMPLETE`。完成后它会停在“等待
协调 GPU teacher-KD”状态；不会自行继续 KD 或训练。

prepared 完成后的 GPU KD 也不会由本流水线自动串联。先用下面的只读入口认证 pipeline/
prepared/audit/teacher/磁盘和已有无 optimizer 性能证据：

```bash
PYTHONPATH=src .venv/bin/python scripts/orchestrate_base_v2_kd.py --action preflight
PYTHONPATH=src .venv/bin/python scripts/orchestrate_base_v2_kd.py --action plan
```

只有用户显式执行带 `--acknowledge-gpu-kd` 的 `--action run` 才会加载 9B teacher。完整的
batch/chunk 选择、15.4h/332.5GB 估算边界、单实例锁、STOP/resume、日志、Web 状态和最终
SHA-bound COMPLETE 契约见 `BASE_V2_500M_KD.md`。

## KD 与 50M quality cooldown 完成后的 fail-closed 配置 finalizer

`scripts/finalize_base_v2_config.py` / `twen config finalize-base-v2` 不包含训练路径。它只在以下
证据同时通过后写 `configs/base/dense-v2-500m.yaml`、认证 bundle 和固定 Web profile：

- pipeline 总 `COMPLETE`、最终 accepted audit attestation、prepared lineage/gates 相互绑定；
- prepared 至少覆盖 500M token，完整 top-64 KD 与其 shard/sample/token range 一一对应；KD
  orchestration `MANIFEST.json`/`COMPLETE` 必须认证最后成功的 `generate-kd` 与独立 `index-kd`；
- cooldown prepared/KD 必须是显式指定、独立 fingerprint 的 whole-shard 严格子集，绑定主
  prepared/KD tensor identity，至少覆盖最后 50M token；approved policy 五文件 bundle 与 cooldown
  顶层 `MANIFEST.json`/`COMPLETE` 都必须有效，六来源达到锁定最低配额；切换点固定为 450M；
- CPU preflight 对主 prepared/KD、cooldown prepared/KD、模型 download manifest、模型文件和
  calibration 文件执行一次完整 SHA 扫描；
- canonical 性能 report SHA 固定为
  `cf40eac976767681a704676bee960d8c220d700a847b73d62e1f2358ea15ab38`，且对应 approval、
  MANIFEST、COMPLETE 必须全部有效；推荐行必须恰好是 `b1-ordinary-ac0` /
  `b1-alignment-ac8`、microbatch 1、chunk 512、`dense_transfer_execution=expanded`；
- ordinary phase 固定 outer 0 / inner 0；5% alignment phase 固定 outer 8 的精确层
  `[0,3,7,10,13,16,20,23]`，inner 16 为精确补集
  `[1,2,4,5,6,8,9,11,12,14,15,17,18,19,21,22]`；preflight 再次解析并逐项锁定这些
  resume-critical indices；
- expanded selective-checkpoint 全图证据必须 PASS：
  `EXPANDED_SELECTIVE_CHECKPOINT_FULL_GRAPH_COMPLETE` 绑定 manifest/oracle/report，且 loss、执行模式、
  call count、参数未更新和 no-optimizer 安全字段全部通过；
- folded 四个真实 KD microbatch 累积证据必须仍为 FAIL / `experimental_only`：
  `FULL_GRAPH_V1_REAL_KD_ACCUMULATION_COMPLETE` 绑定 manifest/oracle/report，不能因吞吐结果改写为
  production；finalizer 不再接受也不需要 folded `PRODUCTION_ADMISSION.json`；
- `base-dense-v1/step-000000000383-milestone-complete` 全 checkpoint SHA 及 source lineage 有效；
- 新 run 固定为 `base-dense-v2-500m` / `runs/base-dense-v2-500m`，目录必须尚不存在。

historical numerical manifest 内嵌的 engine/modules/config SHA 作为“历史数学证据”原样记录；当前
config/preflight/engine/modules/benchmark/MTP 的文件 SHA 与 preflight source-tree SHA 作为“当前生产
实现”独立记录。phase control、日志等后续工程改动允许使当前 engine SHA 不同于历史 SHA，不要求二者
相等；当前执行语义由 canonical report、当前源码 identity 和 CPU preflight 的 exact indices 共同锁定。

MTP 系数和四组 peak/stable LR 以及 Web launch 都没有隐式默认决策。下面命令明确写出本次已选值，
并显式指定真实 cooldown bundle 的两个训练入口：

```bash
.venv/bin/python scripts/finalize_base_v2_config.py \
  --quality-cooldown-prepared-manifest artifacts/data/base-v2-500m-quality-bundle/prepared/manifest.json \
  --quality-cooldown-kd-manifest artifacts/data/base-v2-500m-quality-bundle/kd/manifest.json \
  --mtp-loss-weight 0.1 \
  --adapter-lr 0.0002 \
  --router-lr 0.001 \
  --lora-lr 0.0002 \
  --scale-lr 0.001 \
  --enable-web-launch
```

在主 prepared/KD 或真实 cooldown bundle 尚未完成时，上面的命令会在写任何 runnable 文件前失败；
不得创建占位 manifest/SHA。folded FAIL 已经决定生产路径必须保持 expanded，不能通过放宽数值门槛
解除。

产物固定使用 500M-token WSD：5M linear warmup、保持 peak LR 到 450M、最后 50M cosine decay
到 0.1 倍；450M 同时切换到独立 high-quality subset。`global_batch_tokens=262144`，B1 单卡
accumulation=64，checkpoint 间隔为 100 step 或 30 分钟。Web profile 默认
`launch_enabled=false`；只有同一次认证 finalization 显式加入 `--enable-web-launch` 才会启用页面
启动按钮。两种情况都只写配置，不创建 optimizer、不初始化 CUDA、不启动训练。缺少任一真实产物
或 SHA 漂移时，命令在写文件前退出；不会生成占位 SHA、半成品 runnable YAML 或 profile。

认证 bundle 位于 `artifacts/configuration/base-dense-v2-500m/`。dashboard 配置变化后需要重启
Web 服务以重新加载固定 allowlist；重启 dashboard 本身不会启动训练：

```bash
systemctl --user restart twen-dashboard.service
```
