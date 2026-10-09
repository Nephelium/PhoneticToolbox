# P10 / Preview 1 说明书、编辑器、更新与发行验收

日期：2026-10-05。Windows x64，`3.0.0-preview.1`。应用内说明书、独立作者编辑器、真实换版和单文件自解压包装在本报告限定范围为 `verified`。私有服务器上传按最终回执核实。公众下载、完整应用对应源码交付与总许可证仍 `in_progress`，未上传 GitHub。

## 说明书与作者工具

17 个模块分别由独立 GPT 6.1 Sol / xhigh 作者处理，统一核对 v2 文档、当前 v3 源码与界面，没有使用 Astra，遵循本机并发上限。主线统一标题、八个二级小节、术语、图注、引用和例音顺序。

- 共 19 章，含入门、17 模块和设置。M10 生理参数合成正文按作者要求暂空，M05 实际视频待录制。
- 正文 215,541 个字符，36 张带图注截图、18 处播放器、113 张表。播放器复用 12 份真实原音与模块处理结果，不能视为 18 份独立录音。
- 100 项素材登记，当前引用 39 项，软件媒体约 74 MB。未删除未引用旧素材。
- 截图来自最大化后的实际隐藏 Qt 窗口，保留完整原图与浅深主题证据。逻辑 1440×900、像素 2160×1350、DPR 1.5。隐藏渲染不代表物理 DPI/DWM 验证。
- 所有模块帮助跳到对应应用内章节。左侧章节/小节、搜索、引用跳转、图片放大、播放器互斥和返回模块上下文已接入，阅读器跟随主题与字体。
- 作者双击 `tools/manual-studio/打开说明书编辑器.cmd`，编辑正文、图音视频、图注、表格、格式、代码、公式、引用与原始结构。原子保存、工程锁、冲突保护、磁盘恢复和工程包已实现。工具不随普通应用分发，改稿后重建阅读资源及应用。
- 首版无图片交互裁切及 DOCX/HTML/Markdown 文件导入，未声称这些已提供。
- 总 README、17 份模块 README、说明书及发行 README 完成。最终编校检查 20 份 README、453 条相对链接，无缺链。

证据：`output/manual-work/final-validation.json`、`final-editorial-audit.json`、17 份 `chapter-audit-mXX.json`。严格校验无错误，仅 M10 主动预留提示。编辑器 16 项磁盘/安全测试和 13 组实际 Chrome 操作通过，报告 `tools/manual-studio/test-output/browser-2026-10-05T11-33-23-372Z/report.json`。

## 例音与素材分发

EGG 用获准 `3.wav`，音频/EGG 两声道角色已核对。单音节和句子使用作者指定自然材料。M08 展示实际 0.8 速度与 1.2 音高处理，M06 为真实提取后 Klatt 输出，M07 为实际九步 F0 对照。原文件摘要不变，没有用假音频代替处理，也未声称完成听辨。

例音处理证据：`output/manual-work/examples/e7ce8bd8bf7340fb96b3cc1635ccacda/report.json` 和 `95cd8b6fa87c4c89ba6dbdab3a6155af/report.json`。

自然材料和相关截图为 `software-only` / `git:false`，只随软件分发。本机原始路径在忽略的作者来源配置，正文不带私人路径。独立公开构建 `output/manual-work/public-20261005-final` 实际媒体文件为 0，正文明确显示占位。GitHub 写操作未执行。

## 更新与固定数据位置

启动默认按日检查，手动检查立即查询服务器 HTTPS 和 GitHub Releases，比较 Preview 次序、忽略 draft，最高可信版本决定提示。同版本摘要矛盾拒绝安装。国内优先服务器，其他地区优先 GitHub，可手动选择，地区查询不保存公网 IP。没有公共最新清单时不伪造更新。

先询问下载，完整大小与 SHA-256 校验成功后再询问退出换版。采集、实验、任务和未保存草稿由统一关闭协调处理。

- `%LOCALAPPDATA%/PhoneticToolbox/v3`：任务、托管媒体与更新状态。
- `%LOCALAPPDATA%/PhoneticToolbox-v3/workbench`：保留原 Qt 工作台名字、设置与草稿。
- 安装默认 `%LOCALAPPDATA%/Programs/PhoneticToolbox`，当前用户，无管理员要求。卸载不删除研究数据。
- 免安装换版创建新版本目录，旧程序保留。单文件自解压 EXE、ZIP 与安装包装的边界见 [ADR](../decisions/ADR-Preview1-packaging.md)。

未导出内部结果默认 30 天，成功导出内部副本默认 7 天，可调整或关闭。旧记录首次可靠登记给予完整宽限。输入、录音、标注、正式导出、实验工程和活动依赖链受保护。MFA 只清理身份可靠且已结束的诊断临时目录，默认 7 天，模型/词典保留。更新缓存清理保护活动换版及身份不明项，失败或部分删除如实显示。

边界报告：[更新执行](2026-10-05-updates-apply-report.md)、[原生更新](2026-10-05-updates-native-report.md)、[MFA 临时目录](2026-10-05-mfa-diagnostic-retention-report.md)。启动资源修复后 helper/cache 37 项定向测试重新通过，与此前整组计数重叠，不累加。

## 打包修复与最终身份

完整包自带 EGG/LPC、M05、MFA 运行时与实际模型/词典。初始目录包在 PyInstaller 分析科学环境时混入旧 Qt。Final2 先冻结主宿主、再复制科学运行时，逐文件比对宿主 Qt，实际启动通过。失败包完整保留，不交付。

首次真实 helper 暴露主入口前 setuptools 数据缺失。现保留小型 setuptools 资源、QtCore 启动钩子和固定源码入口，不复制整套科学环境。冻结程序先运行无请求探针，返回 2 才允许关闭旧界面。实际探针成功，证据 `output/validation/update-preview1-portable-final/bootstrap-probe.json`。

Final2 的最终前端、通知与 helper 源码快照显式复制到 `_internal`，由既有入口优先读取。最后源码修复没有重冻 EXE，EXE 身份不变，最终 ZIP 全文件清单绑定实际快照。

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| 主 EXE | 22,765,853 | `f95a5b8503b841732ae87b3b6fb87248d0ee46d8eddc1a33d970e3b36fa93d4e` |
| 免安装 ZIP r2 | 1,857,308,869 | `73ef291f928b935dfba2859350b0a8a10201f9862dec5010c979b01f318291ad` |
| 安装 EXE r3 | 1,496,151,020 | `aab16cfd770833a35eed94ec2a5fdc9913c62a10baa3177dcf0425c92770e0ad` |
| 免安装自解压 EXE r3 | 1,496,151,439 | `792c158624176c4ff0d2107ebd77cb66abffd657af39d91d37ec5cd52236a4f3` |

展开 46,614 个文件、5,275,700,001 字节。科学环境为固定运行资源，自动结果清理不删除。稳定交付目录 `C:/Users/13680/Desktop/PhoneticToolbox-3.0.0-preview.1`。测试安装包使用独立 QA AppId，不交付或上传。

## 实际成品与换版验证

`output/validation/distribution-preview1-final2/report.json` 通过：包内科学环境绑定、MediaPipe FaceMesh 图初始化、冻结本地服务七个合成输入任务（M01 原生 REAPER、M08、M07 分析/三步生成、M06、LPC、EGG）、实际随包 MFA `a1` 探针、17 模块实际最大化 Qt 帮助跳转。16 章各八个二级小节，M10 为 0，软件媒体无缺失。摄像头未打开。

最终真实换版证据为 `output/validation/update-preview1-portable-final` 与 `update-preview1-installer-final`：

1. 真实冻结 helper 等待真实 Windows 父句柄退出，解包/安装后打开新 EXE。QA AppId 与目录独立。
2. 旧版本为 preview.0 元数据夹具，旧界面是源码 Qt，用隔离用户目录写实际主题与 M13 草稿。这不代表所有历史 EXE 迁移通过。
3. 两种更新均恢复深色主题、四个固定设置/草稿键与实际文本，工程 JSON 摘要不变。
4. 新 Qt 内原音 0.672125 秒和处理音 0.840125 秒实际解码播放，播放器互斥、图片放大、离开章节暂停通过。测试静音，无人的听辨结论。
5. 两个实际更新后目录全部 46,614 个预期文件与最终 ZIP 清单逐文件 SHA-256 一致，额外运行时文件未要求不存在。

验证脚本：`verify_distribution.py`、`verify_update_handoff.py`、`verify_update_relaunch.mjs`、`verify_release_payload.py`。Qt 仅暴露 page CDP，测试直连专用本机 page target，没有减弱产品用户手势要求。

最后的自解压 EXE 验证见 `output/manual-work/portable-exe-refusal-r3.json`、`portable-exe-final-r3-QA.json`、`portable-exe-final-r3-payload-hashes.json` 与 `output/validation/preview1-selfextract-launch-r2/relaunch-report.json`。非空的无关测试目录自动拒绝，退出码 1，唯一原文件摘要不变；新目录展开退出码 0，无安装标记/卸载登记，46,614 个预期文件全部匹配。实际独立 GUI 使用全新隔离数据、DETACHED_PROCESS、无控制台和无 IO 重定向启动，正常显示、保存/恢复草稿并解码例音。此项不冒充历史迁移，历史设置保持由前述两种真实换版证明。

早期自解压 r2 的静默保护测试触发自定义提示框而等待确认。测试拥有的进程停止后保留原文件，再于 r3 增加静默初始化拒绝，实际 r3 自行退出验证通过。旧测试记录和包保留，r2 的 -1 退出码不算正常保护验证通过。另一次自解压验证试图复制仍运行的测试 Qt 完整缓存，因 GPU cache/LOCK 被系统锁定而失败；改为全新隔离测试数据，不复制活动用户存储。

冻结巡检末尾 `manual-frozen-maximized.png` 停留在模块图面，不作为说明书排版图。实际说明书最大化浅深截图见 `output/validation/m08-wiring/33164024dbbd420cb4e19e2bfb4e9587/manual-captures/report.json`，正文插图以各章实际捕获报告和登记摘要为准。

## 服务器与剩余公开发行条件

服务器私有暂存 `/home/admin/phonetictoolbox-preview-staging/20261005-preview1` 及父目录 0700，位于 nginx 公共根目录之外。凭据仅用于原生无回显登录，未写入工程或产物。三个最终软件包、两个源码准备包及七份支持文件按清单在服务器逐文件重算 SHA-256，文件权限 0600。最终回执为 `output/manual-work/upload-receipt-20261005-final.json`，另复制到稳定交付目录 `服务器上传回执.json`。GitHub 未上传，公共 latest.json 尚未发布，现有网页与 nginx 配置保留。

见 [实际发行物盘点](2026-10-05-preview-distribution-inventory.md)：193 个 MFA Conda archive 身份匹配，207 份通知逐字节验证后随包，宿主通知另列。VTL 2.4 API 源码、桥接、补丁和构建说明已随包。

独立 [源码准备报告](2026-10-05-preview-source-preparation.md) 已完成 61/61 源码绑定目标、60 份不同完整源码归档，含两个完整 Qt/Chromium、PyQt、Parselmouth 所用 Praat 6.1.38、两套 FFmpeg 与 codec；193 个 Conda 包的 2,511 份配方/补丁材料已校验。另获准下载的官方 MFA v2.0.0a 模型与汉字词典逐字节一致，具体 CC BY 4.0 身份闭合。

上游源码准备 ZIP 为 2,173,464,982 字节，SHA-256 `0c8f157e84b5b19904fef68d57939b382ef309c0bee0bdd5d5c7b0ddc58d8ebe`，2,773 项 CRC 通过。它只汇集已验证上游源码、配方、补丁和身份/许可资料，排除下载分段、缓存及完整模型比对文件。原材料保留。

应用源码准备 ZIP 另为 20,841,933 字节，SHA-256 `bb7499007d4daeaf63016bc9ff7ef365618e40002d94e2f9ec33ac27d767a76b`，3,464 个成员的 CRC 和 SHA-256 回读通过。最终包的 350 项应用文件绑定均一致，其中 332 项源/资源和 17 项发行元数据收入归档，1 项生成 pyc 排除；319 份当前 Python 源一致。19 章及生成阅读清单一致，从优选 Vue 源在隔离目录重新构建的 259 个前端文件全部逐 SHA-256 等于最终免安装 ZIP。九条扫描误报逐项复核保留，未发现实际凭据，媒体、环境、缓存和用户语料排除。EGG 环境的三份旧 wheel 原稿与实际 bootstrap 选择的新源分别记录。它和上游归档共同为源码准备材料，完整公开许可、编译材料及替换安排仍待验收，不将准备 ZIP 标为已满足全部对应源码义务。

剩余对象为项目总 LICENSE、SoE func_getSoE.m 改写/分发依据、历史 Mandarin 1.0 模型许可、M05 codec 许可方案与滚动 MSYS2 DLL 的精确身份、Qt/PyQt wheel 编译材料及最终同渠道源码/通知/替换安排。已有许可和纯论文引用不泛化为重新求许可。未来 GitHub 软件包也须遵守自然例音不进入公开 GitHub 的授权边界，不能直接上传本次含例音软件包。

项目 [AGENTS.md](../../AGENTS.md) 条款：未明确的再分发许可是对应发行物的验收阻断项。私人暂存已获准，具体缺项核实后才原子公开清单，不把私有服务器文件称为公众最新版本。

未验证物理听辨、摄像头/麦克风、实体 DPI/DWM、自然准确率、长期稳定性、Linux/macOS 桌面、所有真实历史数据迁移或生产 PostgreSQL。无现存库 DDL、用户素材/旧包删除、系统运行时修改、全局安装、Git push、GitHub Release 或外部邮件。
