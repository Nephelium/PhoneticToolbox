# P12 · 20261001 Windows 本机试用 EXE

井井明确授权更新 EXE，随后自行试用并反馈 bug。本轮构建与成品定向检查通过，范围为当前 Windows 电脑；未扩大为跨电脑便携包、全模块功能验收或跨平台发行。

## 交付

- 文件：`dist/PhoneticToolbox-v3-LocalPreview-20261001/PhoneticToolbox-v3-LocalPreview-20261001.exe`。
- 大小：358,692,781 字节，约 342.1 MiB。
- SHA-256：`f1ea5776ec53db23cc7b2286c4c0744184eb2cc704ad283d010c8e8e78dd95a7`。
- 包含本轮 P16 九项代码修复、侧栏蓝线定位修复，以及工作区已有的 M03/M04/P04/M13 更新。源码基线 HEAD 为 `13b9a28b1d7c1df3ad73f1c46285ef3a3dfd60c4`，当前未提交内容以本次包内快照与哈希清单为准。
- 默认任务目录继续为 `%LOCALAPPDATA%/PhoneticToolbox/v3/local-preview-20260927`，保留原试用任务。没有删除或迁移已有目录。
- 同目录附 `试用说明.txt`、`build-info.json`、本报告和成品验证摘要。

主包冻结本次业务源码和前端。EGG/LPC 继续调用 `.venv/m03-compatible`，M05 调用 `.venv/m05`，MFA 继续使用既有独立组件注册表。未安装依赖或改变这些环境，仍为 `portable:false` 的本机试用包。

## 成品检查

实际 EXE 从系统临时目录启动，清除继承的 PYTHON/PTB/QT 环境变量，PATH 仅保留 Windows 目录。测试使用新建的独占目录与公开合成输入，未访问用户语料或现存任务库。

1. 从 EXE 归档逐项提取并核对 338 个文件哈希，其中 276 份 Python 源码与本轮构建快照一致，前端与最新生产构建一致。
2. 10 项真实任务完成并从本地 API 回读全部产物、核对 SHA-256：M01 参数估计、M08 变速变调、M07 分析/生成、M06 参数生成/合成、M04 LPC、M03 EGG、M14 预览/导出。
3. M01 JSON 确认 `computation_revision=acoustic/2`，实际使用原生 REAPER，输出 JSON、SQLite、XLSX 完整。
4. Qt WebEngine offscreen 加载 M01–M15 全部 15 页并截图；现有 MFA 注册表可见。未启动摄像头或麦克风。
5. 标注页初始、收起和重新展开文件栏的三种状态均检查：布局保持相对定位，可见拖动边界位于所属布局内。人工查看 M12 截图，无贯穿整页的错位蓝线。此项检查为无输入页面的布局反应，完整标注操作沿用 P16 的 Chrome 回归证据。
6. 主程序退出码 0，观察到的 39 个子进程全部退出；三种非法 worker 参数均以退出码 2 拒绝。
7. 旧 R4 EXE 保留，SHA-256 仍为 `61268b4560209da95faedb35741e718c5cf64dc01c9e6f8c3316115d5f1d3a68`。

证据目录：`output/validation/v3-preview-exe-ed811c06039945cb95b573023ec7a013`，包含 `process-report.json`、`results/report.json`、15 张页面截图与输出日志。源码清单和归档比对位于 `output/build-PhoneticToolbox-v3-LocalPreview-20261001/source-manifest.json`、`archive-contents.json`，构建日志为同目录 `build.log`。

## 命令与边界

```powershell
npm --prefix frontend run build
.venv/m14/Scripts/python.exe scripts/build_v3_local_preview.py --name PhoneticToolbox-v3-LocalPreview-20261001
python scripts/run_v3_local_preview_check.py --exe dist/PhoneticToolbox-v3-LocalPreview-20261001/PhoneticToolbox-v3-LocalPreview-20261001.exe
```

最后一条检查器使用本机已有 Miniconda Python 和 psutil 7.2.2 观察进程。首次用构建环境启动检查器时因缺少 psutil 停止，尚未运行 EXE；随后换用现有环境，未安装或绕过检查。成品的科学计算仍由 EXE 及其既有独立环境执行。

前端保留既有大 chunk 提示。PyInstaller 的四项 hidden-import 提示与旧 R4 一致，未为了压制提示添加忽略项。offscreen Qt 日志仍有 GPU context/fallback 消息，页面检查通过不代表普通桌面 GPU/设备已全面验证。

本轮没有重跑全套源码测试、所有自然语料或长时使用。M05 离线算法、M10 录制、M11 真实对齐、M15 物理时序仍按各模块证据与后续人工试用判断。未改现存数据库、V2、CI/CD、系统配置，无 push、服务器部署或公开发布。下一步接收井井对本次 EXE 的具体使用反馈，逐项定位和修改。
