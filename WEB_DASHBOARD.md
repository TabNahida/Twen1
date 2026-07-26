# Twen 本地训练管理页面

Web 服务读取 rank-zero 已经写出的 `metrics.jsonl`、`telemetry.jsonl`、
`events.jsonl`、`rank0-session.json` 和 `console.log`，不会导入 PyTorch 或占用 GPU。
它不会自行启动训练，但可以在用户点击并通过二次确认后控制 allowlist 中训练的启动、保存与
优雅停止。当前按用户要求监听 `0.0.0.0:8765`；所有页面/API 都要求 HTTP Basic
认证，凭据位于 mode-0600 的 `.twen/dashboard/http-auth.json`。它适合受信局域网或 WSL
端口转发，未配置 TLS，因此不应直接暴露到公网。

页面把 training、KD/data pipeline 和 evaluation 统一为可选择的任务，任务目录保留
**运行中**、**可恢复**与**已完成**三类；运行中任务置顶，并作为首次打开页面时的默认监控对象。
当前 `base-dense-v2-500m` 因此直接成为主任务。KD 详情读取自己的 `status.json` 与 attempt log，
显示 phase、committed token、shard/sequence、百分比、实测
tok/s 和 ETA；training 才显示 loss、LR、gradient、checkpoint/events 等训练字段；evaluation
显示它自己的 role NLL/acceptance。Dashboard 不再扫描或展示 `reports/`、`artifacts/reports/`
中的报告文件。

界面采用纯色、中性深色、细边框和小圆角的传统控制台布局，左侧只列运行中/已完成任务，右侧按
任务类型显示适用信息。独立的只读 `nvidia-smi` 采样展示 GPU utilization、memory-controller
utilization、power draw/limit、温度、SM clock 和物理显存，并保留最近约 60 秒的曲线与
current/mean/p95/peak。实时 GPU 会显式绑定当前 active task；选择已完成任务时不会把另一个正在
运行任务的实时 GPU 数据冒充为该历史任务。浏览器约每 1 秒刷新一次。浏览器不会启动或停止 KD；
KD 使用持久 STOP 文件和独立的 fail-closed CLI，详见 `BASE_V2_500M_KD.md`。

Dashboard 自身有独立采样线程，即使没有打开浏览器也会维持单个固定 argv、`shell=False` 的
`nvidia-smi --loop-ms=100` 子进程。100 ms 原始样本只保留在内存中的最近约 60 秒窗口，用于实时
曲线和窗口统计；这不会向训练热路径加入 hook、CUDA 同步或额外 PyTorch 工作。采样器若无输出
超时、stdout 断开或子进程退出，会先把状态降级为 unavailable，再按有上限的 backoff 自动回收并
重启子进程。

磁盘不会按 10 Hz 无限追加原始行。Dashboard 每 10 秒把样本聚合为 count/error 以及各字段的
min/mean/p95/max/last，写入 `.twen/dashboard/gpu-telemetry.jsonl`；文件达到 16 MiB 后轮转，只保留
一个同样有界的 `.1` 段，总预算约 32 MiB。每个聚合桶仍会 fsync，因此页面关闭后仍能保留用于
报告的功耗、利用率、温度、时钟和物理显存证据。升级前已经存在的逐样本旧行可以与 schema-v2
聚合行共存，轮转后会自然淘汰。

## 安全模型

- 只有 `configs/web/dashboard.json` 中列出的 profile 才可能执行 start/save/stop；API 不接受
  shell、命令行参数或任意文件路径。
- 历史运行只读。`launch_enabled` 默认为 `false`；当前 v1 profile 明确禁止启动。
- start 必须由页面按钮发起，并再次逐字输入 `START <profile-id>`。服务端同时验证 CSRF token、
  固定 profile、finalizer 固化的 `config_sha256` 和全局无重复训练；跨进程文件锁还会拒绝从第二个
  Dashboard 实例并发启动。
- Web 只能执行固定的 `python -m twen train ...` argv，不能从 HTTP 传 shell、额外参数或
  `--dry-run/--graph-smoke` 等模式。该训练入口在导入 PyTorch、创建 optimizer 或执行训练前，
  无条件运行 coordinated training preflight；Web 没有跳过 preflight 的开关。
- save/stop 分别发送 `SIGUSR1`/`SIGTERM`；发送前验证 `rank0-session.json` 的 hostname、PID、
  `/proc/<pid>/cmdline`、cwd 和固定 config 身份。训练引擎收到 stop 后先 checkpoint 再退出。
- 每次接受的控制动作会 fsync 到 `.twen/dashboard/actions.jsonl`；controller PID 状态原子写入
  `.twen/dashboard/controller-state.json`。

v2 条目不应手工拼接。500M prepared 已完成，但 top-64 KD 仍在运行，真实 50M
quality-cooldown policy/bundle 也尚未生成。配置发布入口只有在完整 top-64 KD、cooldown、
audit、性能 bundle/approval、模型 source SHA 和 v1 final checkpoint 全部认证后，才会同时写
resolved config 与固定 profile。该条目把 `resume` 固定为 `none`、`fork_from` 固定为 v1 final，
默认 `launch_enabled=false`；只有 finalizer 的显式 `--enable-web-launch` 才能把它设为 true。
finalizer 会把 resolved config 的 SHA256 一并固化到 profile；重启 Dashboard 时 SHA 不一致会
fail closed。finalizer 本身不会启动训练。不要把 v2 路径临时从浏览器传给服务，也不要用占位 SHA
手改 profile。

finalizer 修改磁盘上的 allowlist 后，当前常驻进程仍保留启动时的旧配置；按下方普通后台进程的
停止与启动步骤重新加载。重启 Dashboard 不会启动训练，也不会误杀独立训练进程。

## 创建认证与前台验收（不启动训练）

```bash
PYTHONPATH=src .venv/bin/python -m twen web init-auth \
  --output .twen/dashboard/http-auth.json --username twen

PYTHONPATH=src .venv/bin/python -m twen web serve \
  --dashboard-config configs/web/dashboard.json \
  --host 0.0.0.0 --port 8765 \
  --auth-file .twen/dashboard/http-auth.json
```

浏览器打开 `http://<Linux-IP>:8765`，用户名为 `twen`，密码从本机私有凭据文件读取：

```bash
.venv/bin/python -c 'import json; print(json.load(open(".twen/dashboard/http-auth.json"))["password"])'
```

当前 v1 是 monitor-only，启动按钮禁用。HTTP Basic 在明文 HTTP 上传输，局域网若不完全受信，
应在反向代理上启用 HTTPS 或改用 SSH/VPN；不要做公网端口转发。

## 普通后台进程（不注册 service）

Dashboard 作为独立 session 的普通后台进程运行，不注册 systemd service，也不会随 WSL 启动
自动恢复。它收到 `SIGTERM` 后会停止并回收自己的 `nvidia-smi` 子进程，同时 flush 当前 10 秒
遥测聚合桶；它不会把该信号转发给训练。启动 Dashboard 不会启动训练：

```bash
mkdir -p .twen/background
setsid --fork /usr/bin/bash -c '
  echo "$$" > .twen/background/dashboard.pid
  exec env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME=.cache/xdg TORCHINDUCTOR_CACHE_DIR=.cache/torchinductor \
    TRITON_CACHE_DIR=.cache/triton TILELANG_CACHE_DIR=.cache/tilelang \
    /usr/bin/bash scripts/with_cuda_toolchain.sh \
    .venv/bin/python -m twen web serve \
      --dashboard-config configs/web/dashboard.json \
      --host 0.0.0.0 --port 8765 \
      --auth-file .twen/dashboard/http-auth.json
' >>.twen/background/dashboard.log 2>&1 </dev/null
```

只读检查监听与认证（不会启动训练）：

```bash
ss -ltnp 'sport = :8765'
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8765/
```

预期监听地址为 `0.0.0.0:8765`，未带认证的 HTTP 状态码为 `401`。

查看后台日志与进程：

```bash
tail -f .twen/background/dashboard.log
ps -p "$(cat .twen/background/dashboard.pid)" -o pid,etime,stat,cmd
```

停止 Dashboard 本身时，先核对上面 PID 的命令行，再发送 `SIGTERM`：

```bash
kill -TERM "$(cat .twen/background/dashboard.pid)"
```

训练控制应在页面完成；不要用 `kill -9`。普通后台进程在 WSL 实例关闭时也会退出，之后需要用户
显式重新启动。
