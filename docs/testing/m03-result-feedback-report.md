# M03-E3 结果窗口反馈

2026-09-13。**verified，限定 Windows 独立 Chrome 开发态的结果读取、重读和反馈归属。完整 M03/E3 仍 in_progress。** 计划见[结果反馈方案](../plans/2026-09-13-m03-result-feedback.md)。EXE及相关探针按井井要求暂停。

## 修复行为

此前 PNG/WAV 读取失败后，错误写入主页面，已打开的结果弹窗只剩部分内容。元数据读取失败时没有结果窗口。保存和下载的异步反馈未绑定窗口，可能在关闭旧窗口后覆盖新结果的提示。

现在先打开公共结果弹窗并显示加载状态，元数据、图片、WAV与历史分析读取失败均在当前窗口提示，指出正在读取的内容。已读到的部分结果保留；重新读取结果只读该任务产物，不提交新计算。元数据未取得或读取尚在进行时，保存/下载按钮禁用，返回分析始终可用。读取错误与保存错误分别保留。

每次打开/关闭使用既有 viewEpoch 及任务 ID 共同确认窗口归属。旧读取成功/失败、旧保存或下载反馈均不能更新新窗口；在目录选择返回后也复核归属，避免旧窗口随后启动保存。已经启动的保存或下载可按原授权完成，关闭窗口不冒充取消文件操作。历史分析读取成功仍恢复原四图和参数，沿用原时间、数据与试听角色。

## 实际验证

| 命令 | 结果 |
| --- | --- |
| `npm --prefix frontend run test` | 64 passed |
| `npm --prefix frontend run typecheck` / `run build` | 通过 |
| `node tests/e2e/m03-result-feedback.cjs` | 10 组通过，0 页面错误：部分PNG失败、重读不新建任务、元数据失败、IF部分音频失败、迟到读成功/失败、迟到保存失败、正常完整保存、迟到下载失败及真实CSV校验、历史预览恢复、800×440浅深主题 |
| `node tests/e2e/m03-playback.cjs` | 4 组通过，0 页面错误；实际 Web Audio 样本、暂停/续播、声道交换、IF两角色切换及关闭、文件/标签切换与关闭焦点 |
| `.venv/m09-ui/Scripts/python.exe scripts/validate_docs.py` / `scripts/check_architecture.py` | 573文件、330来源、32任务，errors=[]；历史快照失效链接单列保留，架构通过 |
| `npm --prefix frontend run contracts:check` / `run ui-data:check`；`git -c core.safecrlf=false diff --check` | 无生成漂移，差异检查通过 |

最终证据为 `output/validation/m03-ui/chrome-3b52a2c93ef945b3a13e9215c75cdc75/report.json` 与 `chrome-df80430da0374a3e9af91c58ce547e2e/report.json`。CSV 浏览器下载另存 `downloaded.csv` 并与真实任务清单 SHA-256 精确比对；完整保存由既有本机授权目录适配完成。深色结果到期截图已查看，按钮边界在实际800×440视窗内；浅色截图同目录保留。

失败证据保留：`chrome-95cc8317fe57417c9ae50d9fb8e59f7a` 是实施前读错误未出现在弹窗的实际复现。`chrome-dc911caa385a4ad2b34b045d3f46e0b0` 为第一轮功能通过。扩展用例 `chrome-870a9583265a43f6a5237b5f26646d47` 暴露测试错误地假设三图清单顺序固定，实际到达故障图前可能已读两图；修正为依据实际清单选择最后一张作为故障目标，不改变产品图像顺序或放宽错误提示要求。

正常任务、文件与保存走真实本机服务和 `.venv/m03-compatible` 科学子进程，测试数据库为既有 schema 的独立副本，`schema_applied=[]`。仅在浏览器 RPC 边界注入 `asset_expired` 和延迟，验证错误恢复和迟到反馈，不代表真实七天到期或生产网络故障。`m03-live.html?downloads=1` 为可选测试下载适配，读取真实托管产物字节，不冒充生产网页账号/认证路径。本轮未重跑 Qt、生产账号、物理声卡、多屏设备或科学数值全集。

无新增外部依赖、算法或引用，来源登记不变。未改 v2、数据库 schema、科学环境、全局配置、EXE或现存研究数据，未push或发布。下一项为按V2功能清单集中复核开发态剩余缺口，方法与代码许可未闭合项继续单列；不推进M04。
