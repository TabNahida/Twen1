# v4 13M calibration Adapter/scale 漂移审计

本报告只分析已完成 checkpoint; 执行设备为 CPU, 未构建模型、未创建优化器、未启动训练。

| step | checkpoint | Adapter relative-L2 | scale relative-L2 |
|---:|---|---:|---:|
| 40 | `/media/data1/Project/AI/Twen1/runs/base-dense-v4-13m-low-lr-calibration/step-000000000040-periodic` | 0.00536767 | 0.01078333 |
| 50 | `/media/data1/Project/AI/Twen1/runs/base-dense-v4-13m-low-lr-calibration/step-000000000050-milestone-complete` | 0.00565633 | 0.01154244 |

## 5% scale gate

- 末 checkpoint scale relative-L2: `0.01154244`
- 上限: `0.05000000`
- 结论: **PASS**

## 输入身份

- formal closure MANIFEST SHA256: `b496a8f90fc23e70ebc101c0d54a6e4cc10158f14da57b47dfa3b03a7dd01885`
- calibration config SHA256: `15ce9dbf68643b6abbcbc687a698f2994e2200587c442974903b07613a43109d`
- v3 baseline manifest SHA256: `ef43670d7c1cbc8ed3908b258659c7426b4cfe10e14c9b7db54968e2481b0e9a`
- final checkpoint manifest SHA256: `c35d362a0944b402f26970cb5d269794dedd92b4ba9a0dc4542e4c92d76c1098`
- drift auditor SHA256: `4769a6d5d8b9015111bb8621abf7bb9a3af5344bd9036370ed9a03a21de8f696`
- bundle producer SHA256: `3ff880cce136706a219048762d05da0e511e3960d587f6bf946cdf5679e57f9b`
