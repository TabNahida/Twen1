# Base v2 500M top-64 KD 编排与审计

本文只覆盖冻结 Qwen3.5-9B-Base teacher 的 top-64 推理缓存，不包含 student 训练、optimizer
创建或 `optimizer.step()`。

## 当前状态

500M prepared 已完成：641 shards、126,457 sequences、516,719,389 tokens，manifest SHA256
`9290665ac1e09fbd5b9aea1966bed7a51095bab66f460a0124af4532b1805fd9`。

KD 已于 2026-07-22 完整生成并独立 index：641/641 shards、126,457 sequences、
516,719,389 tokens。KD manifest SHA256 为
`242ed2d0fb899cb333939bbd581f8a5632e97228f0c1fda2fee14bea7291efe9`；orchestration
`MANIFEST.json` SHA256 为
`bf9b5c7aa08f8160840d84348e5139bd399b93a1acb4287bd46b523b4635e6d9`，`COMPLETE`
SHA256 为 `760d2f235687f94888a216db0761caea74489198808357c80b8abc8008aad5d6`。
最终 generation 与独立 `index-kd` 均 exit 0，`optimizer_created=false`、
`training_started=false`。历史恢复命令保留在下文，仅用于审计，不应再次启动 KD。

## 生产点与容量

- batch 2 microbenchmark：约 10,425 input tok/s、17.7869 GiB peak；
- batch 4：约 10,518 tok/s、18.8880 GiB peak，只快 0.884% 且多 1.101 GiB；
- 因此 production 继续使用 batch 2，训练侧 microbatch 1 与它完全独立；
- v1 100,007,485-token 完整 KD 实跑为 3:04:20、约 9,042 wall tok/s；线性排期约
  15.4–15.7 小时，最终以当前 attempt telemetry/ETA 为准；
- tensor payload 按真实 padded capacity `sequence_count × 4096 × 665` 估算，并额外保留至少
  64 GiB 文件系统余量。

KD 包含读取、host copy、落盘和 SHA/index 阶段，本来就不会持续顶到 600 W。功耗不是完成门；
应看完整 wall tok/s、committed shard 和 SHA-bound `COMPLETE`。

证据：

- `artifacts/benchmarks/rtx5090-qwen35-9b-base-teacher-kd-batch2.json`
- `artifacts/benchmarks/rtx5090-qwen35-9b-base-teacher-kd-batch4.json`
- `artifacts/benchmarks/rtx5090-base-teacher-kd-full-run.json`

## fail-closed 启动与恢复门

`scripts/orchestrate_base_v2_kd.py` 在加载 CUDA 前验证：

1. data-only pipeline `COMPLETE`，且 `training_started=false`、`gpu_kd_started=false`；
2. pipeline、status 与 prepared path/size/SHA 完全绑定；
3. prepared 全部 shard/tensor SHA、4096 context、至少 500M token、authenticated train lineage；
4. audit attestation、rejection ledger、benchmark registry 与所有 gate；
5. 9B immutable revision/download manifest 和每个模型文件；
6. 已有 KD final shard 的 checksum/identity，staging `.incomplete` 不计进度；
7. batch/chunk 的本机 finite benchmark、磁盘门与单实例锁。

SHA 占位、手改 attestation、research override、额外 final 目录或不同 generator identity都会
fail closed。

## 只读检查

```bash
PYTHONPATH=src .venv/bin/python scripts/orchestrate_base_v2_kd.py --action status
PYTHONPATH=src .venv/bin/python scripts/orchestrate_base_v2_kd.py --action preflight
PYTHONPATH=src .venv/bin/python scripts/orchestrate_base_v2_kd.py --action plan
```

这些命令不启动 KD 或训练。当前运行状态优先查看 `status`，不要根据文档中的某个瞬时 shard
数字判断进度。

## 启动、停止与恢复

普通后台进程使用下面的固定生产合同。命令会写 PID 和独立 launcher 日志，但最终状态仍只以
orchestration `status.json` 为准：

```bash
mkdir -p .twen/background
setsid --fork /usr/bin/bash -c '
  echo "$$" > .twen/background/base-v2-kd.pid
  exec env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME=.cache/xdg TORCHINDUCTOR_CACHE_DIR=.cache/torchinductor \
    TRITON_CACHE_DIR=.cache/triton TILELANG_CACHE_DIR=.cache/tilelang \
    /usr/bin/bash scripts/with_cuda_toolchain.sh \
    .venv/bin/python scripts/orchestrate_base_v2_kd.py \
      --action run --acknowledge-gpu-kd --batch-size 2 \
      --logits-chunk-tokens 64 --poll-seconds 2
' >>.twen/background/base-v2-kd.log 2>&1 </dev/null
```

安全停止使用事务边界 STOP；普通后台进程会以 75 退出且不会自动重启：

```bash
touch artifacts/data/base-v2-500m-kd/STOP
```

STOP 在当前 shard 事务边界返回 75 并保持在磁盘。只有显式恢复时才删除 STOP，再运行同一后台命令：

```bash
rm artifacts/data/base-v2-500m-kd/STOP
```

不要删除 `.incomplete` 或已完成 shard；事务 writer 会认证 staging，完整 shard 不重算。

## 状态、日志与最终证据

- KD 数据：`artifacts/data/base-v2-500m-kd/`
- 实时状态：`artifacts/data/base-v2-500m-kd-orchestration/status.json`
- 合并 console：`artifacts/data/base-v2-500m-kd-orchestration/console.log`
- phase 日志：`artifacts/data/base-v2-500m-kd-orchestration/logs/`
- 最终编排 manifest：`artifacts/data/base-v2-500m-kd-orchestration/MANIFEST.json`
- 最终编排证明：`artifacts/data/base-v2-500m-kd-orchestration/COMPLETE`
- 最终 KD lock：`artifacts/data/base-v2-500m-kd/manifest.json`

generation 成功后还会独立运行 `index-kd` 全量复核。只有最终 `COMPLETE` 绑定真实
prepared/audit/pipeline/config/teacher/benchmark/KD/status SHA 后，后续 quality policy 与 v2
config 发布才可继续。

Dashboard 监听 `0.0.0.0:8765`，以约 1 秒 UI 刷新展示 Data/KD pipeline 的 phase、committed
token、百分比、wall tok/s 和 ETA。浏览器不能启动或停止 KD；KD 由独立后台进程与事务边界
STOP 控制，且不注册为 service。
