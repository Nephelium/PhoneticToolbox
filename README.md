# PhoneticToolbox 3.0 Preview

面向语言学与实验语音学研究的本机工作台，提供录音、声学与 EGG 分析、合成与音频操控、TextGrid 标注、音标输入、音系归纳和感知实验。当前源码的 18 个模块共用侧栏、标签页、浅深色主题与设置。

当前发行目标为 `3.0.0-preview.1`，Windows x64。预览版本适合先用副本试做并核对结果。各模块的科学解释、输入限制和验证范围分别记录，完整使用方法在应用内说明书中。

M18 语音学论文精读已接入当前源码，支持服务器独立分发、原译文连续阅读、信息折叠、全屏、直接复制、批注/荧光笔、PDF 导出、引用格式及按日期获取往期。右上角帮助打开对应说明书。当前保留的 Startup-20261008-R1 免安装 EXE 已包含 M18-R2，也可用根目录的 `打开语音学论文精读.cmd` 启动源码。详见[论文模块说明](frontend/src/modules/paper-reading/README.md)及 [R2 验收](docs/testing/2026-10-08-m18-r2-report.md)。

## 获取与使用

安装版按当前用户安装，默认目录为 `%LOCALAPPDATA%\Programs\PhoneticToolbox`，无需管理员权限。免安装单文件 EXE 双击直接启动，没有安装或解压目录向导。ZIP 仅包含同一个 EXE 与更新身份文件。两种程序包均以 500,000,000 字节为上限，MFA 环境和模型另行配置，其他模块的必要运行时随包提供。

免安装版首次打开时显示准备进度，运行文件保留在当前用户缓存中。后续打开校验并复用，更新后未变化的组件也可复用。安装版在安装阶段完成准备。下载体积和展开占用不同，当前约468 MB的免安装包按文件身份去重后保留约1.80 GB运行缓存，另有约25 MB小型引导文件在正常退出后清理，实际占用随版本及文件系统变化。

设置中可选择清除全部缓存并安全退出，下一次打开会重新准备。未导出的内部处理结果清理后需要重新计算，设置、草稿、原始输入、录音、工程及正式导出文件保留。安装版卸载也会清理可清理缓存，有活动实例时先提示关闭。最新实测与限制见[最新启动与成品验收](docs/testing/2026-10-08-startup-feedback-report.md)。

打开应用时和点击检查更新时，会查询服务器与 GitHub 的版本信息。出现更高版本后先询问下载，校验成功后再询问退出升级。国内优先服务器，国外优先 GitHub，可在更新页手动切换。完整 Preview 发行尚未公开发布；源码曾按授权推送，当前工作树的后续改动并未因存在于本机而自动上传。

程序更新继续使用固定的用户数据位置。录音工程、标注和正式导出文件保留。未导出的内部处理结果默认保留 30 天，已导出结果的内部副本默认保留 7 天，可在设置中调整或关闭。正在运行任务和依赖中的结果受保护。MFA 已完成的诊断临时目录默认保留 7 天，模型与词典不属于清理范围。

## 模块索引

| 分组 | 模块 | 操作与工程说明 |
| --- | --- | --- |
| 分析与采集 | M16 录音 | [README](frontend/src/modules/recording/README.md) |
| 分析与采集 | M05 唇形提取 | [README](frontend/src/modules/lip-extraction/README.md) |
| 分析与采集 | M01 参数估计 | [README](frontend/src/modules/parameter-estimation/README.md) |
| 分析与采集 | M02 参数显示 | [README](frontend/src/modules/parameter-display/README.md) |
| 分析与采集 | M03 EGG 信号分析 | [README](frontend/src/modules/egg-analysis/README.md) |
| 分析与采集 | M04 LPC 谱图 | [README](frontend/src/modules/lpc-spectrum/README.md) |
| 合成与操控 | M06 声学参数合成 | [README](frontend/src/modules/speech-synthesis/README.md) |
| 合成与操控 | M10 生理参数合成 | [README](frontend/src/modules/vocal-tract/README.md) |
| 合成与操控 | M07 发声类型合成 | [README](frontend/src/modules/phonation-synthesis/README.md) |
| 合成与操控 | M08 变速变调 | [README](frontend/src/modules/pitch-manipulation/README.md) |
| 合成与操控 | M09 语谱图转音频 | [README](frontend/src/modules/spectrogram-to-audio/README.md) |
| 标注与实验 | M17 国际音标表Plus | [README](frontend/src/modules/ipa-plus/README.md) |
| 标注与实验 | M13 汉字转国际音标 | [README](frontend/src/modules/mandarin-ipa/README.md) |
| 标注与实验 | M11 MFA 自动标注 | [README](frontend/src/modules/mfa/README.md) |
| 标注与实验 | M12 TextGrid标注 | [README](frontend/src/modules/annotation/README.md) |
| 标注与实验 | M14 音系归纳 | [README](frontend/src/modules/phonology-induction/README.md) |
| 标注与实验 | M15 感知实验 | [README](frontend/src/modules/perception/README.md) |
| 文献与学习 | M18 语音学论文精读 | [README](frontend/src/modules/paper-reading/README.md) |

## 使用说明与作者编辑

模块帮助跳转到应用内对应章节，左侧选择章节与小节，正文包含步骤、控件表、截图、图注和例音。阅读页面跟随主题与字体设置。生理参数合成章节已有正文并持续修订，唇形视频演示待作者补录。

作者通过独立 [说明书编辑器](tools/manual-studio/README.md) 修改内容，该工具不包含在普通应用中。正文与素材工程见 [manual/README.md](manual/README.md)，音标内容另由独立 [音标内容编辑工具](docs/manual/ipa-content-authoring.md) 维护。

自然例音及其处理结果只获准随软件分发，不进入公开 GitHub。公开文档构建过滤受限素材，私人原始路径不写入正文。

## 工程结构

| 目录 | 职责 |
| --- | --- |
| `frontend` | Vue 3 界面、主题、模块与只读说明书 |
| `desktop` | PyQt6 / Qt WebEngine 宿主、本机文件、录音与更新 |
| `packages/phonetic_core` | 科学计算与数据格式 |
| `backend` | 本机任务、受控文件、结果与保留策略 |
| `manual` | 可编辑的说明书工程 |
| `tools/manual-studio` | 作者专用富文本编辑器 |
| `release` | 版本、独立科学环境与安装包工程 |
| `contracts` | 接口与格式契约 |
| [requirements](requirements/README.md) | Python 依赖声明、锁和配套 Conda 清单 |
| `third_party` | 来源登记、版权通知与许可原文 |
| `docs/testing` | 实际验证与明确限制 |

## 开发与验证

使用项目指定的 Python 3.11 与 Node 24.13 环境。科学模块存在独立环境与原生组件要求，按各模块 README 和 [发行工程](release/README.md) 配置。不要把仅依赖开发目录的 EXE 当作可分发包。

```powershell
cd frontend
npm ci
npm run contracts:check
npm run ui-data:check
npm run typecheck
npm test
npm run build
```

源码工作台入口为 [Start-Research-Workbench.ps1](scripts/Start-Research-Workbench.ps1)。Python 检查按各模块 README 选择相关测试，说明书额外执行严格结构与媒体 SHA-256 校验。实际成品、安装升级、物理音频设备、听辨和跨平台验证分别报告。

所有模块启动脚本现在共用上述入口，直接读取当前工程源码。修改 Python 业务代码后重新启动即可，无须另改环境中的安装副本；Vue 与原生组件仍需重新构建。主入口使用已有 `.venv/m14`，计算子进程保留原有依赖环境。详情见 [源码入口与修改方式](docs/development/source-entry.md)。

```powershell
.\scripts\Start-Research-Workbench.ps1 -CheckOnly
.\scripts\Start-Research-Workbench.ps1
.\scripts\Start-Research-Workbench.ps1 -Module M10
```

## 来源与发行状态

科学论文、具体代码、字体、模型和数据分别登记。应用的关于与方法引用入口提供来源和许可原文，完整核查见 [第三方说明](third_party/README.md) 与 [来源分类审计](docs/references/p19-license-classification-audit.md)。公开发行的项目总许可证及 SoE 对应代码许可仍待作者确认，确认前不宣称完整发行许可验收通过。

已完成部分与后续任务见[当前状态](docs/project-status.md)，测试/打包产物管理见[生命周期](docs/development/artifact-lifecycle.md)。开发与规范入口见[文档导航](docs/README.md)，当前构建门槛见[严格打包规则](release/PACKAGING_RULES.md)。具体源码、成品及平台验证以对应报告为准；旧交付记录按需检索，不作为当前可运行文件清单。
