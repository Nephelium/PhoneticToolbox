# P19-R6 图窗手势与按钮的文本选择修复

日期：2026-10-04。状态：源码定向 verified，限定 Windows Chrome；本机 EXE 构建单列。

## 完成行为

图窗、图像、空图占位、按钮及折叠标题不参与浏览器文字选择。手势从这些位置开始时，临时禁止整个当前文档选择文字，鼠标释放、取消、窗口失焦或页面隐藏后解除。普通说明文字保持可选择，输入框、textarea 与 contenteditable 保留编辑与选择。Shift 从已有文字选区进入图窗时清除旧文字范围，不会扩展到其他图窗或按钮。

统一资源 `frontend/public/interaction-selection.js` 与 `.css` 同时由 Vue 主入口和 M10 独立页面加载，覆盖所有现有 SVG/canvas 图窗，未修改科研算法或各图窗的绘制、拖选、平移事件。鼠标事件继续传播，保留 click、double-click、键盘激活与图窗焦点。原生滑块、复选框等控件保留浏览器默认鼠标动作。文字禁选状态仅在手势期间设置，不写入用户偏好。

## 验证

| 范围 | 结果 |
| --- | --- |
| 前端类型检查 | 通过 |
| 前端测试 | 278 passed，0 skipped；含同期模块新增用例 |
| 前端生产构建 | 通过，既有大 chunk 提示保留 |
| Chrome 定向行为 | 6 组通过。实际 M08 曲线 Shift 绘制、Ctrl 恢复和普通平移，拖出图窗、已有文字选区，按钮单击/双击/键盘，公共波形/科学图/canvas，空图，原生滑块/复选框，输入/编辑区，窗口失焦，M10 入口资源与独立 iframe，深色及 150% 缩放 |
| 上轮选区回归 | 3 组通过。实际 M07/M08 页面加载本轮全局资源后，源/目标/第三条以及原音/合成音的选区互斥与空格播放/停止仍正常 |

Chrome 输入使用真实鼠标/键盘。音频为公开合成正弦，WebAudio 静音；M08 任务适配器沿用 UI 夹具，不作为科学合成验收。M10 检查实际入口引用与独立 iframe 中的公共行为，不声称本轮重新验证整个声道引擎。未运行新 EXE、实体设备、物理 DPI 或 Linux/macOS GUI 检查。

证据：

- `output/validation/interaction-selection/chrome-1791101939545/report.json`
- `output/validation/audio-selection/chrome-1791101708007/report.json`
- `output/validation/interaction-selection-frontend.log`
- `output/validation/interaction-selection-build.log`

全库文档检查读取 1412 份文件，仍有 10 条既有旧 EXE 缺失链接，另列 36 条未改历史快照的未解析链接。本轮报告无缺失链接，未改写旧记录；全库检查未通过，输出保留于 `output/validation/interaction-selection-docs.log`。

## 本机打包

最终新包使用既有 `.venv/m14` 与 `--lean-qt` 构建完成，构建进程退出码为 0：

- PhoneticToolbox-v3-Latest-20261004-R3.exe（历史本地产物，当前工作区不存在；原路径 `../../dist/PhoneticToolbox-v3-Latest-20261004-R3/PhoneticToolbox-v3-Latest-20261004-R3.exe`）
- 311348804 字节，约 296.93 MiB。
- SHA-256：`ee52ee65845e4c65473ebb21012f321ef9845e57d7ee796a4804095502345408`。
- 构建日志：`output/validation/interaction-selection-exe-build.log`。

构建基线为 `0255daa8214f9d885d120d6562792cb8c9db421d`，冻结源码包含本轮未提交修改与工作区同期代码。旧包保留。沿用既有独立科学环境，portable:false，新 EXE 未启动检查，不宣称可搬迁发行或成品功能验收。

无新 push、数据库 DDL、全局依赖安装、公开发布、删除用户文件或修改 V2。保留当前同期模块改动及开始时已有的根图标删除差异。
