# M11 功能、来源与验证映射

2026-09-27。完整模块 `in_progress`，已实现 Windows 本地正式任务与可选组件。平台边界见 [验收报告](../../testing/m11-report.md)。原 V2 说明书第 8.1–8.3 节与下列四份实际源码共同作为迁移依据。

| 功能 | V2 入口／实际行为 | V3 正式入口 | 正常与异常证据 |
| --- | --- | --- | --- |
| M11-F01 模型、词典、音频、输出 | `gui/dialogs/mfa_auto_alignment_dialog.py` 资源选择；`services/mfa_alignment_service.py` 外部 auto_alignment；管线递归复制语料 | 工作台 MFA 页面 → 组件检查／离线导入 → 模型及词典；目录或配对文件导入；独立结果子目录 | Qt 原生选择中文、空格、子目录的两份音频，真实任务及三个文件回读／保存；缺文本、重复文本、错误模型／词典有显式失败 |
| M11-F02 Beam 联动 | 默认 10/40，界面改变 Beam 时将 Retry 至少提高至四倍；service 仅在 Retry≤Beam 时归一化 | `mfa/state.ts` 与 `M11Config` 分别保留两层规则 | 独立前端测试、Python 参数测试；Qt 改 Beam=20 后 Retry=80 |
| M11-F03 进度／日志／帮助／错误 | 旧线程及外部管线输出日志，帮助入口 | 既有持久任务阶段／历史／结果；本机脱敏原生日志；帮助与 SRC-MFA、REF-MFA | 实际 LocalService、Qt 成功和恢复，原生日志回读；OOV、TextGrid 层规则、损坏模型、超时／崩溃失败 |
| M11-F04 新增真实取消 | V2 对话框 `setCancelButton(None)` 隐藏取消 | 既有 cancel API → worker 协作取消 → Windows Job Object 整组终止 | 正式运行任务取消；外部进程故障注入的 cancelled／timeout／crash 均 `group_cleaned=true` |
| 新增独立组件 | 旧 service `_ensure_runtime_app` 会复制旧 GUI，本轮不调用 | 用户级可选版本目录，可信清单，原地自检成功才切换；不再启动旧 GUI | 真实 809,661,654 B 候选包短路径导入成功；长路径 DLL 失败保留候选且未激活；单元测试失败保留旧指针 |
| 新增网页等待 | 原 V2 无账号或远程任务 | 已登录项目 → 相同 M11 页面／上传／原任务队列 | 真实新建 PG + 认证 ASGI 双账号隔离、当前配额、等待重开、普通 worker 不误领；真实节点计算仍阻断 |

## 原样迁移与显式修正

- `phonetic_toolbox/core/transcription/mfa_name_codec.py` 原样复制到 `packages/phonetic_core/src/phonetic_core/transcription/mfa_name_codec.py`，UTF-8 十六进制 `ptbx_` 编码及复合扩展名规则保留。说明书所述自动转拼音与当前源码不同，以实际编码为准，未增加拼音转换。
- 原 `mfa_alignment_pipeline.py` 单独加载，在公开合成输入复现旧 `NUMBA_DISABLE_JIT=1` 的 MFCC 错误，未得到可用于数值相等比较的旧 TextGrid。不能声称 V3 与旧失败运行输出等价。
- [ADR-M11-001](../../decisions/ADR-M11-001.md) 单列运行适配修正：任务级 JIT 缓存、3.3.8 的实际全局选项、独占 SQLite／根目录。MFA 版本、模型、Beam 默认值及 align/export 算法调用保留。
- 新输入安全／科研保护：缺失或重复转写提前拒绝；词典音素被 MFA 排除时显式失败；OOV 显式失败；MFA 的保留 TextGrid 层名说明清楚。原行为可能忽略词典条目或使用 OOV 代替，这些拒绝规则是明确的保护性修正，单列验证，不冒充原样迁移。
- 输出名使用序号加原始文件名，避免不同说话人子目录同名文件碰撞；溯源同时保留输入相对路径及原始 hash。原始音频、转写和输出目录中的既有文件不覆盖。

## 科学边界

公开合成 /a/ 样例只验证实际计算、时间轴与输出管线。两次并发任务的完整 TextGrid 时间／标签逐项相等，不能替代人工标注边界准确率。119.8 秒无分段合成转写在默认 Beam 下得到 `m11_no_alignments`，不自动调参；100 份短文件成功不能推广为 100 份两分钟自然录音已通过。
