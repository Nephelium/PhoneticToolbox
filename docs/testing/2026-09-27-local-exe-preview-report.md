# P12 当前源码本机 EXE 试用包

日期：2026-09-27。状态：verified，限定 Windows 本机试用包的构建、代表性任务与 offscreen Qt 页面检查。完整发行和各模块全部行为不据此扩大为 verified。

## 最终交付与结果

- 入口：`dist/PhoneticToolbox-v3-LocalPreview-20260927-R4/PhoneticToolbox-v3-LocalPreview-20260927-R4.exe`。
- 大小：358,659,306 字节，约 342 MiB。
- SHA256：`61268b4560209da95faedb35741e718c5cf64dc01c9e6f8c3316115d5f1d3a68`。
- 最终 EXE 9 项任务与 M01–M15 共 15 页检查全部通过，M14 两份 DOCX 与一份 XLSX 均成功回读并核对 SHA256。
- 退出码 0，观察到的 20 个子进程均已退出；3 种非法 worker 参数均以退出码 2 拒绝。
- 证据：`output/validation/v3-preview-exe-3e2f780e3ac245ac8005df43ba0259a3/results/report.json`、同目录上一级 `process-report.json`、15 张页面截图。已人工查看 M11、M14、M15 截图，无整页空白，M11 显示已注册 MFA 3.3.8 和 Mandarin 模型。
- 构建日志和 268 份 Python 源码哈希清单位于 `output/build-PhoneticToolbox-v3-LocalPreview-20260927-R4`，快照与当前工作区逐文件一致。
- Qt offscreen 日志存在 GPU context/fallback 提示；页面加载检查通过，不将此扩大为普通桌面 GPU 设备验证。
- 同目录附 `试用说明.txt`、`逐模块修改记录.md`、本报告和 `build-info.json`。无需启动开发服务器。

## 范围与来源

井井授权将当前 v3 打成 EXE，在本机逐模块试用。应用源码基线为 `13b9a28b1d7c1df3ad73f1c46285ef3a3dfd60c4`，本轮新增打包入口、构建/验证脚本与试用文档。没有改动模块算法、数据库 schema、全局环境、旧 EXE、V2 或 CI。此次未推送或发布，GitHub main 保持原状。

主包冻结当前前端、核心/API/桌面源码和原生资源，M03/M04、M05、MFA 的运行时继续独立。构建不安装依赖。外部子进程执行主包内的源码快照，不从 V2 动态加载算法。本包只用于当前电脑，不代表便携版或正式跨平台发行。

默认双击时使用新的 `%LOCALAPPDATA%/PhoneticToolbox/v3/local-preview-20260927` 任务目录，复用既有首次初始化流程，不操作原有研究任务库。清除该目录会损失此次试用任务，本轮没有执行清理。

## 执行命令与证据

```powershell
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
.venv/m14/Scripts/python.exe scripts/build_v3_local_preview.py --name PhoneticToolbox-v3-LocalPreview-20260927-R4
python scripts/run_v3_local_preview_check.py --exe dist/PhoneticToolbox-v3-LocalPreview-20260927-R4/PhoneticToolbox-v3-LocalPreview-20260927-R4.exe
```

前端类型检查与生产构建通过，178 项测试通过、0 失败、0 跳过。原有大分块提示保留，没有为了打包改动代码拆分。检查器使用已有 Python/psutil 观察本次启动的子进程，没有安装全局包。

实际 EXE 检查在系统临时目录启动，移除继承的 `PYTHON*`、`PTB_*`、`QT_*`，PATH 只保留 Windows 目录。EXE 自身装配本机科学运行时。使用公开合成音频、公开 EGG 数组和表格，仅初始化测试独占任务目录。

实际完成：M08 变速变调、M07 分析及连续统生成、M06 参数生成及合成、M04 LPC、M03 EGG、M14 预览及三文件导出，共 9 项正式任务；所有产物从本地 API 回读并核对 SHA256。实际 Qt/WebEngine 以 offscreen 模式逐一打开 M01–M15 页面并保存截图。MFA 注册表可见性、进程退出及 3 种非法 worker 参数拒绝均通过。

## 打包问题与修正

1. 首个候选在 LPC 外部子进程失败。构建快照过滤了项目 `*.egg-info`，子进程导入 API 时无法读取 `ptb-api` 元数据。已保留源码包元数据，原算法未变。
2. R2 中 EGG/LPC 与另外 5 项任务通过，但 M14 返回 503。模块要求的 `xlrd` 2.0.2 代码已被收集，发行元数据没有进入 EXE，触发原有能力检查。R3 显式收集 `xlrd` 代码与元数据，没有放宽版本门槛。

3. R3 的 M14 预览通过，导出失败。检查构建依赖清单，发现 `docx.api` 和 `phonetic_core.transcription.phonology.export` 等代码未进入分析范围，虽然模板资源已经在包内。仅 `collect-submodules` 未覆盖当前源码的全部模块。R4 对源码快照逐文件枚举 hidden imports，显式收集 docx 子模块，并加入独立 Word 模板检查。

首个候选、R2 和 R3 均保留为诊断产物，不作为试用交付。R2 检查证据：`output/validation/v3-preview-exe-fec084b92efe41c796e7de694c530da6`，退出码 1，观察到 18 个子进程，退出后无本次残留进程。R3 证据：`output/validation/v3-preview-exe-7fd68e738365404ca64f25435aa9fe47`，8 项任务通过，导出失败，16 个子进程均已退出。

源码对照检查 `output/validation/v3-preview-source-7ac6b70261034f75913c9c0664c08888` 已通过全部 9 项任务与 15 页。该早期 JSON 的 scope 文案误标 frozen，本报告明确它只是源码检查；验证脚本现已按 `sys.frozen` 区分来源。最终验收必须以 R4 冻结成品的独立报告为准。

## 未覆盖与人工反馈

- 15 页可达不等于 15 模块所有功能/科研结果已经通过此次 EXE 验收。
- 未启动摄像头或麦克风，未重跑 M05 离线分析、M10 录制、M11 真实对齐、M15 物理声学与反应时验证。
- 未重跑全部旧模块的科学基准、自然语料、长时运行、普通桌面 GPU/声卡/多屏场景。
- MFA 只检查现有注册表可用，完整模型执行沿用独立 M11 报告范围。
- 不覆盖 Linux、远程节点、云服务器、跨电脑安装或公开发行。

人工试用入口：[逐模块修改记录](2026-09-27-local-trial-notes.md)。表中空白不代表通过，井井反馈后按模块、步骤、实际/期望行为及优先级补充。
