# M07 Linux／remote/1 交接门

2026-09-27。M07当前Windows开发态有条件可用；下列仍为 proposed / blocked，未开放节点或服务器回退。

## 现场依赖与阻断范围

读取 `remote-protocol-v1.md`、`p06-remote-report.md`、P11能力代码：独立remote/1候选尚无主服务正式科学接线，原operation allowlist仅M01/M03/M04。M07未加入节点allowlist，不自行扩大公共调度器或执行007。现有NInfer可以进行纯核心对照，但本轮没有M07专属systemd/cgroup委托、运行时receipt与Linux REAPER执行证据。Windows逐位基线与Linux浮点差异尚未接受为科学等价。

因此 `m07_task.capability`在非win32拒绝，正式submit给出 `m07_platform_unverified`。这既不新建M07调度系统，也不以能运行其他模块替代M07资格。当前直接拒绝有明确原因，不宣称已经实现远程排队/自动回退。

## 供公共接线审阅的具体映射

| 现有M07 | remote/1目标 |
|---|---|
| operation `phonation_synthesis`，schema `m07/1` | 公共operation allowlist及node科学runner新增登记，不能由客户端自报 |
| action analyze/apply/generate | 不可变config.analysis中的M07Request，剥离服务器project/owner路径；保持输入引用 |
| source/target；apply/generate另含analysis | InputAsset按服务器ID/sha/size传输，analysis只接受同账号/项目成功产物 |
| 算法 `m07-v2-lpc-residual/1` | runtime_hash需绑定核心wheel hash、numpy/scipy/parselmouth构建、OS/arch、F0 backend与REAPER binary hash |
| 当前子进程1,000,000,000字节/120秒 | P11资源profile硬限制，服务器最多1槽，必须实际测Linux规模，不能沿用Windows本表开放 |
| ≤10秒/480000帧/8MB每输入，2–50步，完整组输出≤64MB | 节点接收预算和运行时再次检查；不能降低步数/采样率适配预算 |
| 正式job/generation、每组原子complete | attempt_id/generation/租约贯穿读写和complete；过期旧尝试拒绝，只接受一个有效逻辑结果 |
| batch_id/group_count/group_index | 已完成组保留，未完成组按公共重试策略恢复，已成功组不重启 |

新增准入必须按顺序完成：M07运行时receipt与等价判断 → Linux实际受限进程和REAPER双后端 → 正式PG/P07/remote桥 → HTTPS断连/超时/旧节点迟到/重复complete/取消/超额 → 实验室Linux及实际服务器。只在这些证据通过后登记相应规模和后端，服务器fallback回调才返回受验runtime hash。网络失败可重试执行，不承诺只执行一次；发布必须唯一。

本轮不需要新数据库列。现有公共桥若需007由统筹另行授权，M07不能迁移旧库。M07源码和Windows正式任务不依赖该门，已独立完成。
