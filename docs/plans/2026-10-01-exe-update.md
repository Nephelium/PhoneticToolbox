# P12 本机 EXE 更新

2026-10-01，井井明确要求更新 EXE，之后自行试用并反馈 bug。授权范围为本机试用构建，沿用现有打包方式和运行时，保留旧包与用户数据。

- 产物名称：PhoneticToolbox-v3-LocalPreview-20261001。
- 使用现有 .venv/m14 构建，前端重新生产构建，主程序冻结当前核心/后端/桌面源码。EGG/LPC、M05 和 MFA 继续使用现有独立环境，此包限当前电脑试用。
- 默认任务目录沿用 local-preview-20260927，不因换包清空或迁移已有试用数据。
- 本轮仅更新验证脚本及交付文档，不增加业务功能。成品检查追加 acoustic/2 的实际 M01 任务，以及标注侧栏切换后拖动边界的定位，覆盖刚修复的蓝线问题。
- 命令：npm --prefix frontend run build；.venv/m14/Scripts/python.exe scripts/build_v3_local_preview.py --name PhoneticToolbox-v3-LocalPreview-20261001；python scripts/run_v3_local_preview_check.py --exe dist/PhoneticToolbox-v3-LocalPreview-20261001/PhoneticToolbox-v3-LocalPreview-20261001.exe。最后的检查器使用本机已有、含 psutil 的 Miniconda Python，仅负责启动/观察 EXE，不替换包内科学运行时。
- 实际 EXE 检查在项目外目录、清理继承环境后启动，使用新建的测试独占任务目录和合成输入，检查 10 项代表性任务、15 页加载和退出清理。使用 offscreen Qt，不打开摄像头或麦克风，不将此扩大为人工全功能验收。

状态：verified，限定当前 Windows 本机试用成品及本计划定向检查。10 项任务、15 页、三种标注侧栏状态、退出清理和非法参数拒绝均通过，见[交付报告](../testing/2026-10-01-exe-update-report.md)。等待井井后续人工使用反馈，不扩大为跨平台或所有功能已验证。
