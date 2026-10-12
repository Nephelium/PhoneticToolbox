# Windows 最小化恢复与窗口按钮

`P04-WINDOW-RESTORE` 修复当前 Windows 源码宿主在最大化后最小化、再由 Windows 原生恢复时的客户区尺寸失配。最大化按钮点击与键盘激活后的框线一并移除。最终 EXE 未重建。

## 复现与修复

旧验证只覆盖普通窗口经 showNormal 恢复，未核对最大化窗口原生恢复的实际像素。此次用 ShowWindow(SW_MINIMIZE / SW_RESTORE) 保留 Windows 的恢复路径，首页即可稳定复现，说明不依赖摄像头或音视频处理。失败证据为 output/validation/window-restore/qt-9265601b49374bda84559670592d5d87/report.json：页面仍能执行 JavaScript、没有 renderer termination，恢复后 viewport 从 1707×1019 变为 1693×960，PrintWindow 采样全部黑色，首次最大化遮罩已关闭。

宿主提前消费 WM_NCCALCSIZE 时，Qt 未正常维护其原生框架边距，恢复后出现系统标题栏/边框尺寸偏移。只增加 IsIconic 条件仍失败，证据 qt-0cc46fdee4fb49228c38cece97ea9edb 和原生事件跟踪 qt-2a02b3db02744491bbe0d1bbeb9fe207。移除宿主的客户区计算拦截，由 Qt 已有 FramelessWindowHint 处理无边框客户区与工作区尺寸，同时保留原生缩放样式和边缘命中测试。

这一判断结合本机差异复验与 Qt 6.11.0 实现：[qwindowscontext.cpp](https://github.com/qt/qtbase/blob/v6.11.0/src/plugins/platforms/windows/qwindowscontext.cpp) 在窗口过程末尾记录计算前后的边距，[qwindowswindow.cpp](https://github.com/qt/qtbase/blob/v6.11.0/src/plugins/platforms/windows/qwindowswindow.cpp) 已处理无边框客户区和最大化工作区。未修改 Qt、驱动、GPU 设置或系统配置，未重载网页或清除用户草稿。

移除拦截后的过渡运行 qt-b79b57af28a04b81b11b4ea2acf0e601 已恢复正确尺寸，但仍记录过一次全黑 PrintWindow 帧，该运行保留为失败，不计通过。后续独立实例的 18 次、48 次以及最终浅深模式 38 次恢复未再次出现黑帧。有限次数验证不扩大为长期稳定性或其他平台结论。

## 实际验证

- `.venv/m14/Scripts/python.exe -X utf8 scripts/verify_window_restore_qt.py`：最终证据 output/validation/window-restore/qt-bc58ac4fa2014e1fb0cd0e4940fa52b2/report.json，success=true、closed=true。Windows 11、Qt 6.11.0、屏幕2560×1600、DPR1.5，隔离 profile 与既有 schema 数据库副本，无摄像头/麦克风操作。
- 首页、参数显示、唇形提取在浅深模式与普通/最大化窗口下，各做3次原生最小化恢复，共36次，最小化持续100 / 600 / 2500毫秒。每次恢复核对真实窗口像素、viewport、激活标签、文档标记与最大化状态；全部非黑采样比例1.0，renderer terminations为空。
- 浅深模式各一次未应用字体草稿恢复，共2次，18px编辑值保留。最终退出前取消隔离实例的测试编辑。
- 两种模式通过键盘激活窗口最大化按钮，outline与boxShadow均为none；最大化范围与屏幕工作区一致。字体和科学默认值未改。
- 额外连续48次恢复通过，证据 output/validation/window-restore/qt-a014e69f493e43c48e4fb5da1f46ed42/report.json；初轮18次通过，证据 qt-a0e28e833d1b46019a868d76e8cf77a5。
- `pytest -o addopts='' desktop/tests/test_presentation.py -q`：9项通过。仅取消根目录遗留的 phonetic_toolbox 覆盖率参数以运行v3定向测试，断言与容差未改。
- `node --test frontend/tests/appearance.test.ts`：5项通过；前端typecheck、生产构建通过。生成的前端资源由构建入口维护。
- `scripts/verify_compact_shell_qt.py`：回归报告 output/validation/compact-shell/qt-b7cb857829564ad891d25bcad301651c/report.json，success=true、closed=true。顶边标签、最大化工作区/全屏、原生边缘缩放、可拖动阅读侧栏以及未保存关闭确认通过。
- `scripts/check_architecture.py`：errors=[]。`scripts/validate_docs.py` 全仓库检查发现已有来源清单计数不一致，actual=414、total_records=413、无重复ID；未改 third_party/source-registry.json。当前任务新增文档链接及台账ID没有错误，完整日志保留在 output/validation/window-restore-docs-final.log。

## 交付与边界

保留失败与通过报告、原生跟踪日志和必要截图；9个测试进程退出后按项目清理入口回收隔离 scratch，cleanup.json均为deleted，正式数据和用户偏好不动。正式说明书现有窗口操作说明保持有效，无需新增恢复步骤；本次校正桌面架构与旧验收范围。

当前源码通过 scripts/Start-Research-Workbench.ps1 -Module home 重新启动，启动日志 output/validation/window-restore/workbench-20261011-175425.*.log。实际源码GUI进程33212 / HWND1050910已核对归属，另完成4次最大化原生最小化恢复，真实2560×1528窗口像素全部通过，证据 output/validation/window-restore/source-restore.json；保留该实例供用户继续试用。最终EXE、Windows10、多屏跨DPI、其他平台和长时间休眠唤醒仍未验证。
