# Preview 1 必要运行时与单文件验收

2026-10-06，Windows x64。状态：本机隔离成品与大小验收 verified，服务器私有上传按单独回执确认，公开发行许可及完整原生对应源码仍 in_progress，GitHub 未上传。

井井要求两种 EXE 不超过 500,000,000 字节，免安装版双击直接启动。MFA 环境和模型排除，其他模块不依赖开发环境。说明书正文、图片、音频与作者工程本轮冻结。

## 最终文件

| 文件 | 实际字节 | SHA-256 |
| --- | ---: | --- |
| portable.exe | 486656313 | 18f1ea2da3bd557a0a3b22de04ddc89c8acd7dead7088a139583d8b617678652 |
| portable.zip | 486656889 | 857ff737c689d95e2b77ff4792eed0d03a9ebdfbe0642f9b515793849f47c580 |
| setup.exe | 488125269 | ed5689e0a9bdb77d746e30d853fc9cf8bd141a91e1afcc11c3a35d25f6ee0f26 |

版本均为 `3.0.0-preview.1`。最终构建位于 `dist/PhoneticToolbox-Preview1-20261006-Compact-R7`，发行三个文件位于 `output/release-staging/compact-delivery-20261006-R7/artifacts`。portable.exe 是应用本身，ZIP 仅含同一应用与 `application.json`，安装包包含同一应用。无 SFX 目录向导。正式安装默认当前用户应用目录、无需管理员权限，保留原 AppId。安装验证使用独立 QA AppId 和目录，未触碰真实应用安装或研究数据。

## 筛选与压缩

- EGG/LPC 最小环境 750077697 字节，唇形环境 422172882 字节。科学压缩包 238184372 字节，原字节及每文件 SHA 保留。
- EGG 使用原 NumPy 2.2.6、SciPy 1.16.3 与 MKL 2025.3.0。唇形保留原 MediaPipe 0.10.14、OpenCV 4.13.0.92、PyAV 16.1.0、FaceMesh 468/478 图及必要检测/关键点模型。
- MFA 环境、声学模型、词典全部排除。唇形排除未使用的 JAX/JAXlib、SciPy、ml_dtypes、非 FaceMesh 模型及开发组件。正常导入所需的 NumPy testing 支持保留。
- 宿主原生文件压缩包 166713200 字节，展开 668887076 字节。20 个 Qt DLL 均与原宿主环境 SHA 一致。科学环境作为不参与 PE 依赖合并的独立数据处理。
- 原说明书 79 MB 包含未引用媒体。当前构建仅选取正文引用的 36 张 PNG 与 12 个 WAV 文件，共 55295176 字节，同一音频可能被多个播放器引用。所有保留媒体原字节不变，作者工程与未引用素材未删除或移动，没有重新压缩图音。

运行时临时展开约 2 GB，与固定用户数据目录分开。首次进入新进程需要解压和校验。最终测试正常退出后无 `_MEI` 目录残留。应用设置、工程、录音和正式导出不受此清理影响。

## 实际验证

| 项目 | 结果与证据 |
| --- | --- |
| 定向桌面检查 | 68 passed，覆盖两套展开逻辑、路径/Windows 别名/链接/哈希失败边界、单文件更新布局与旧布局兼容、更新协调器 |
| 工程外成品 | 只复制一个最终 EXE 到 `D:/PTB-Compact-QA-20261006/compact-r7`，去除 Python/Conda/PTB 开发绑定，仅 Windows PATH，实际 `--verify-distribution` 退出 0，103.5 秒含完整计算与 GUI 巡检 |
| 原环境对照 | 最终包中的 EGG 7 项、唇形 9 项探针通过。计算/导出 signatures 与库 fingerprint 逐项等于完整原环境，导入路径全部位于实际临时包内 |
| 实际任务 | 11 个真实托管任务：M01 原生 REAPER 参数估计、M08、M07 分析与三步合成、M06 Klatt、LPC、EGG、M09 图片与原相位音频、M14 预览及 DOCX/XLSX 导出。全部结果文件逐 SHA 回读 |
| 唇形与媒体 | 两套 FaceMesh 图初始化，合成可变 PTS 视频完整处理、CSV/lip、MP4 AAC/WAV 保存和时间戳回读、MP4/GIF 动画编码解码通过，无自然人脸或摄像头准确率声明 |
| 录音与音标 | 合成双通道采集、精确帧删除/恢复、真实同 EXE spawn 降噪、FLOAT WAV 逐样本及 manifest 回读、关闭保护；625 音标入口与 14 布局、打包字体、输入/撤销/复制验证通过 |
| 其他客户端 | M13 转换与原生 PNG、M15 独立会话及原生 J/F 响应、IndexedDB/JSON/XLSX 回读、M10 原生构形动画与 9600 音频样本通过 |
| TextGrid | 最终 EXE 在独立 `textgrid-r7` 中完成 11 阶段，导入、中文标注编辑、自动保存文件回读、独立唇偏、实际原生 TextGrid 与 lip JSON 导出回读通过，退出 0，34.594 秒 |
| 帮助 | 当前 17 页各一个页内帮助，准确进入章节并返回。M10 空章保留。GPU 可用的隐藏 Windows Qt 窗口完成此检查 |
| 直接启动 | `direct-r7` 中只放一个 EXE，脱离工程、无开发 PATH，DETACHED_PROCESS、无控制台及 IO 重定向，真实 GUI/图音解码通过，退出 0，无临时展开残留 |
| 真实更新 | 最终 ZIP 真实 helper/Windows 父进程句柄/解压/启动通过；同最终应用的独立 QA 安装包真实安装、原目录更新、启动通过。两者均恢复深色主题、本机草稿，工程标记 SHA 不变 |

证据：`D:/PTB-Compact-QA-20261006/compact-r7/{launch-report.json,scientific-comparison.json,distribution/report.json}`、`textgrid-r7/{launch-report.json,textgrid/report.json}`、`direct-r7/{direct-report.json,relaunch-report.json}`，以及 `output/validation/compact-{portable,installed}-update-r7/{handoff.json,relaunch-report.json}`。

此前 R1–R3 超出大小上限或构建空间不足，保留诊断，不交付。R4–R6 曾被旧巡检选择器阻断，未标通过；旧顶部帮助、录音按钮/设备刷新和标注状态选择器已按实际当前界面校正。禁用 GPU 的测试不能完成 M10 WebGL 初始化，完整巡检使用隐藏 Windows Qt 与正常 GPU。最终上述检查均通过，未降低科学输出对照要求。

## 保留与边界

固定数据位置和内部结果保留策略不变：未导出内部结果默认 30 天，成功导出的内部副本默认 7 天。用户原始输入、录音、工程、标注、正式导出和活动依赖链受保护。更新为便于回退保留旧程序及下载包。

本次没有另一台实体干净 Windows/虚拟机验证，没有实体声卡/摄像头、听辨、物理时延/DPI、长期或跨平台验收。因此不宣称任意电脑绝无问题。已验证实际冻结依赖、独立环境绑定、媒体处理和输出，MFA 的单独环境配置仍排除在此次离线验收之外。

旧程序、候选包、原环境和原素材均保留。此前为应对 D 盘不足而移动的四个本轮候选目录已回到原 D 盘位置，后续构建及最终交付均在 D 盘。无新 GitHub 上传、push、现存数据库 DDL、全局依赖或系统运行时设置更改。

公开发布继续受 [AGENTS.md 的具体许可规则](../../AGENTS.md)约束：项目总 LICENSE、SoE 对应实现许可和部分原生 GPL 对应源码未完成确认。服务器私有暂存不发布公共 latest.json。新的应用源码准备快照不能标为完整原生对应源码验收通过。
