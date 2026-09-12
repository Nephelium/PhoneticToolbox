# M03-D 统一 EGG 页面与 Windows 开发态操作

2026-09-12。**M03-D verified，限定本轮 Windows 开发态四图、参数/播放、逐文件批次、持久记录与导出操作。完整 M03 仍 in_progress，下一项 E 联合收口。** 不将此报告扩展为完整 30 项、网页账号/TTL、自然语料页面或冻结 EXE 验收。

## 实现与原因

页面使用共同 AppShell、主题 tokens、公共字体、AudioTransport、WaveformViewport、TaskPanel、ModalDialog 和方法来源入口。新增 ScientificPlot 是主题/字号自适应 SVG 显示组件，没有事件或 F0 算法。EGG 页面按 V2 的左 CQ/SQ/语谱、右音频/EGG 微观、下方参数与总览布局实现。井井指出下方按钮散乱后，第一行改为选区与更新、试听、导出与逆滤波三组，第二行分为事件检测、语谱图与 F0，高低通移到 EGG 图上方。按组换行，大字号减少时间刻度数量而不缩小字号。

新增 preview 模式返回有界数值 JSON、归一化分析音频 WAV、无坐标 PSD 栅格图。核心新增的微观显示和 IF 对照数值路径来自原 V2 GUI，独立双轮冻结后逐样本比较。±100 ms 微观显示、±50 ms 事件、processed ±100 ms CQ 不混用。IF 图的实际中心窗为总长 50 ms，原误写标题已明确修正。普通 single/batch/inverse 产物集合不变；preview 字段不进入旧模式的幂等哈希。

实际 Qt 首轮发现共同 task 桥仅允许单请求，平行读回 WAV/PNG 与轮询发生冲突。公共桌面能力新增有界顺序队列，失败请求不会阻塞下一请求，科研子进程与任务并行策略不变。页面按输入/参数签名和代次拒绝过期回读，原文件总览与归一化试听分开。

## 验证命令和证据

下列 Python 检查使用进程内 `PYTHONPATH=backend/src;desktop/src` 的绝对路径，不修改全局环境。

| 检查 | 命令/结果 |
| --- | --- |
| 原版显示基准 | `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/capture_m03_display.py`：在原 v2 环境、offscreen、禁写字节码，两个独立进程输出数组完全相同；公开冻结文件 `tests/fixtures/m03/ui-display.npz` 和 `ui-source.json` |
| 安装核心、导出、字体 | `scripts/Invoke-M03-Python.ps1 -PythonArguments @('-X','utf8','-m','pytest','-c','tests/pytest.ini','backend/tests/test_m03_exports.py','backend/tests/test_m03_preview.py','backend/tests/test_fonts.py','tests/parity/test_egg_analysis.py','packages/phonetic_core/tests/test_egg_config.py','tests/parity/test_m03_capture_contract.py','-q')`：122 passed；保留重复时间案例的既有两条数值警告 |
| M03 契约 | `.venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini backend/tests/test_m03_contract.py -q`：14 passed，含旧模式幂等配置兼容 |
| 前端 | `npm --prefix frontend test`：47 passed；`run typecheck`、`run build`、`run contracts:check`、`run ui-data:check` 全部通过 |
| 契约生成 | `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/generate_contracts.py --check`：无漂移 |
| 文档与架构 | `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/validate_docs.py`：533 文件、328 来源记录、32 任务，errors=[]；`scripts/check_architecture.py`：errors=[]。历史快照原有失效链接单列保留 |
| 真实 Qt | `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m03_qt.py`：双声道四图、单声道拒绝、选区过期、CSV/三 PNG 实际保存、IF 四图及双 WAV 回读、混合批次失败继续、整批保存、关闭重开记录，通过 |
| 独立 Chrome | `node tests/e2e/m03.cjs`：16 项实际操作与显示检查通过，0 页面异常；含取消/重试、24 px 公共图字号、1920/1440/1280/1000/390 宽度、1.25/1.5/2 倍模拟 DPI、真实本机任务导出与批次保存 |

Chrome 最终证据：`output/validation/m03-ui/chrome-5a40756c5a5e429283e1f50dedb322e6/report.json`，按钮截图 `controls-light.png`；Qt 最终证据：`output/validation/m03-ui/qt-bba5dbc32b744ee58235be0301013dcf/report.json`。两者均在已有合成测试库的一致副本运行，`schema_applied=[]`，没有执行 DDL。Chrome 测试适配接真实本机 FileProvider/TaskBridge/API/兼容子进程，不是生产网页账号验证。

科学 wheel：`output/validation/m03-ui/wheels-inverse/phonetic_core-3.0.0a1-py3-none-any.whl`，仅安装到 `.venv/m03-compatible`。开发 Qt/API 宿主继续使用 `.venv/m09-ui`，未升级其科学包。初次字体测试误放在不含 Matplotlib 的宿主环境产生两项环境失败，已在正确的独立科学环境纳入上述 122 项，未安装依赖绕过隔离约定。

## 既有回归的保留问题

广域后台/桌面/契约回归（排除应在独立科学环境运行的三份文件）有一次 **257 passed、1 failed**：`test_m01_segments.py::test_cancel_timeout_and_failed_heartbeat_terminate_owned_child` 的 Scratch 目录回收出现 WinError 32。此文件及其清理实现本轮未修改；单独复跑该项通过。现有 OwnedProcess 只等待主进程退出，后代当前目录句柄释放时序是待查方向，不能将这一推断写成已修复根因，也不宣称广域回归稳定全绿。该问题单列保留，未降低检查或扩大 M03 数值容差。

## 覆盖与下一项

- 本轮覆盖 A01/A02/A04–A07/A12/A17–A21/A24–A26/A28 的指定合成/显示/文件路径；A03/A08–A16/A22–A23 的既有科学证据继续以 A/B/C 报告为准。屏幕 PSD 缩小不替代原数值对照。
- 批次逐文件持久化，最后一组已提交 ID 仅作本机恢复索引。尚未提交的选择不冒充服务器持久队列。失败文件单列，保存本次结果只保存已成功项。
- 本轮使用 0.8 s 双声道合成、单声道和静音样例；原自然录音核心/任务证据不能直接扩大为本页验收。声卡听感、多屏原生 DPI、生产网页账号/配额/TTL 联合路径、全部旧版交互细节、自然语料页面与新 EXE 仍待 E/F。
- 通用输出命名与旧按输入文件名命名有差异，JSON保留输入哈希，后续 E 再核对全部手册承诺、输出目录体验和 30 项，PENDING-EGG 学术来源/许可继续未决。未改 v2、旧 EXE、现有数据库 schema，也未 push 或发布。
