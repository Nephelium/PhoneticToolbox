# M03-E3 批次错误反馈

2026-09-13。**verified，限定 Windows 开发态 Chrome 的参数拒绝、提交反馈和小窗滚动。完整 M03/E3 仍 in_progress。** 计划见[批次反馈收口](../plans/2026-09-13-m03-batch-feedback.md)。按井井最新要求，EXE 工作暂停。

## 问题与修复

原批次只有字体预检。数字输入的 min/max 未通过表单校验执行，高通填成低通的值后仍逐文件提交；异常只保留文件名，关闭弹窗并清空上一批保存入口。Chrome 首轮回归在期待弹窗内高通错误处失败，证据 `output/validation/m03-ui/chrome-e6168083c23045a09a2c097c0e93abf3/report.json` 和 `failed.png` 保留。

1. 前端按既有 M03/1 契约校验滤波、静音阈值、显著度、谱窗与 dB 范围，空值和非有限值拒绝。先校验再字体预检和任务提交；单文件共用参数检查，仍独立执行选区、采样率和计算预算检查。未改变默认值、科学算法或后端契约。
2. 首个新任务被实际接收后才替换本次批量 ID 和参数快照。全部提交被拒绝时保留弹窗、选中项和上一批保存入口。逐文件提示拒绝原因，部分成功时保留成功任务且不自动重复提交。
3. 提交阶段冻结参数与文件选择，公共 ModalDialog 新增默认关闭的 closeDisabled 选项，EGG 批次提交时同步禁用返回、关闭和 Escape。任务提交结束后恢复关闭；后续任务执行与取消沿用处理记录。共享弹窗其余用法保持默认可关闭。
4. 使用 V3 公共字段、弹窗、提示和主题样式。800×440 时普通滚轮可滚动正文，底部按钮固定可达。

## 实际验证

| 命令 | 结果 |
| --- | --- |
| `npm --prefix frontend run test` | 64 passed；含空值、非有限、契约端点及文件采样率独立检查 |
| `npm --prefix frontend run typecheck` / `run build` | 通过 |
| `node tests/e2e/m03-batch-feedback.cjs` | 6 组通过，0 页面错误；非法参数零任务/字体请求、修正后真实计算、全部拒绝保留旧入口、延迟时禁改、浅深/窄窗滚动、部分失败逐文件反馈 |
| `node tests/e2e/m03-fonts.cjs` | 3 组通过，0 页面错误；缺字体拒绝/保留、纯CSV、恢复字体后的真实CSV和三PNG，实际字体元数据回读 |
| `.venv/m09-ui/Scripts/python.exe scripts/validate_docs.py` / `scripts/check_architecture.py` | 571 文件、330 来源、32任务，errors=[]；历史快照失效链接单列保留，架构通过 |
| `npm --prefix frontend run contracts:check` / `run ui-data:check`；`git -c core.safecrlf=false diff --check` | 无生成漂移，差异检查通过 |

最终证据分别为 `output/validation/m03-ui/chrome-511e1ccb603a4cc58ae339b938cd2bfe/report.json` 与 `chrome-58c632471e1544709cd04a2c5e80aa2f/report.json`。浅色 800×600 和深色 800×440 截图已查看，滚轮效果另用实际 scrollTop 增长确认。中间功能通过轮 `chrome-553de69607b141449716bc11681aedd3` 保留，最终轮补公共关闭按钮状态与小窗滚轮。

正常任务使用既有真实本机服务和 `.venv/m03-compatible` 科学子进程。提交拒绝/延迟仅在测试浏览器 RPC 边界受控注入 `input_unavailable`，用于验证恢复流程，不冒充真实文件到期、生产网络故障或托管账号验收。数据库复用既有测试 schema 的隔离副本，`schema_applied=[]`。本轮未重跑 Qt、科学数值全集、多屏设备或真实语料。没有 DDL、依赖安装、v2 修改、EXE 构建、push 或发布。

来源依据为既有 `backend/src/ptb_api/egg_models.py` 的 M03/1 参数约束，无新增外部代码、方法或依赖，第三方登记无新增项。下一项为开发态结果窗口读取失败/到期反馈与剩余交互审阅；来源未闭合项继续单列，M04 不推进。
