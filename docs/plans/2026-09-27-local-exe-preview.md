# P12 本机试用包

2026-09-27，井井明确要求将当前 v3 打成 EXE，在本机逐模块试用并记录修改意见。本轮为本机试用，非公开发行。保留现有源码、运行环境和旧 EXE，不触及服务器或 GitHub main。

## 实现边界

- 新增 `scripts/build_v3_local_preview.py`、`scripts/v3_local_preview_entry.py`、`scripts/verify_v3_local_preview.py` 和 `scripts/run_v3_local_preview_check.py`。
- 使用既有 `.venv/m14` 构建主 EXE，包含当前前端、核心/API/桌面源码快照、原生资源和 Word 导出依赖。
- M03/M04 调用既有锁定 MKL 环境，M05 离线计算调用既有 M05 环境，M11 使用已通过组件导入的独立注册表。具体绝对路径仅在忽略的本机产物中生成，不修改系统环境。
- 源码随主包冻结，外部固定 child 从包内真实源码目录启动，禁止运行时动态导入 V2 或工作区业务代码。
- 使用独立的 `LocalAppData/PhoneticToolbox/v3/local-preview-20260927` 本机任务目录，复用既有新目录初始化流程，不迁移现存数据库。
- 此包依赖当前电脑独立环境，不能当作完整便携版。MFA、物理设备和各模块科学验收仍以各自报告为准。

## 验证与交付

前端执行 `npm --prefix frontend run typecheck`、`npm --prefix frontend test`、`npm --prefix frontend run build`。最终打包使用 `.venv/m14/Scripts/python.exe scripts/build_v3_local_preview.py --name PhoneticToolbox-v3-LocalPreview-20260927-R4`。

真实 EXE 在非项目目录、清除环境路径影响后运行固定 `--verify-preview`，检查全部页面可达、正式本地服务和代表性模块任务、结果文件、正常退出。测试仅使用公开合成输入，不自动调用摄像头或麦克风。新建模块试用记录表，区分已知限制、待测试与实际反馈，不将空白记录填写为通过。

当前状态：本轮本机试用交付 verified，限定 Windows 构建、9 项任务与 15 页检查；完整 P12/跨平台发行保持主计划状态。最终产物、检查范围和剩余限制见 [专项报告](../testing/2026-09-27-local-exe-preview-report.md)。
