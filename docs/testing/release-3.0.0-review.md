# 3.0.0 发布前审查

2026-10-11。审查完成，发布条件仍未收口。本轮仅检查现存 EXE、当前源码和公开下载入口，未修产品代码、重打包、安装到用户正式目录或对外发布。

## 结论

当前 MFA-R3 的常规启动、科学任务、所选桌面流程通过本轮复验。发现一个可复现的更新逻辑缺陷、一项尚未上线的必要组件交付，以及正式版本和当前源码尚未冻结的发布准备缺口。建议先处理这些事项，再对新的 3.0.0 最终字节验收。

### F1 / P1：旧 GitHub Release 会使有效的新版本候选丢失

位置：[updates.py](../../desktop/src/ptb_desktop/updates.py)，`UpdateService._github` 对全部可解析版本遍历，某条历史发布没有当前要求的包布局时直接抛出 `CHECKSUM_MISSING`，此前收集的有效版本随之丢失。

现场只读获取的 GitHub 列表含 v2.1.4 等七条历史发布，例如 `PhoneticToolbox_v2.1.4.exe`。其资产虽有 SHA-256，却不符合当前 `windows-x64` 及 portable/setup 命名规则。隔离复现使用实际历史列表和一条合成的、摘要完整的 3.0.1 发布，当前版本设为 3.0.0，模拟官网不可达：

| 输入 | 实际结果 |
| --- | --- |
| 仅有效 3.0.1 | `available`，候选 3.0.1 |
| 同一 3.0.1 加现有七条旧发布 | `incomplete`，候选为空，GitHub `CHECKSUM_MISSING` |

这会破坏 GitHub 作为更新来源的可用性。官网独立提供有效更新时仍可能给出候选，但不能依靠官网正常来掩盖 GitHub 分支失败。修复应区分旧于当前版本的历史资产与应当可校验的新候选，保留对真正的新版本缺摘要、最高版本包缺失及跨来源冲突的保护，增加混合历史列表回归。无需删除历史 Release。

本文件与当前 EXE 的冻结副本逐字节相同，因此此缺陷同时适用于现存包。复现为当前源码的真实解析器加受控网络响应，并非已上传 3.0.1。

### F2 / P1：MFA 下载入口指向 404

[release_catalog.py](../../backend/src/ptb_worker/mfa/release_catalog.py) 固定的 `https://www.phonetictoolbox.com/releases/components/mfa/3.3.8-r1/PhoneticToolbox-MFA-3.3.8-r1-windows-x64.zip` 在本次检查返回 HTTP 404。当前 EXE 中此目录配置与源码相同，主包又明确排除了 MFA 环境、声学模型和词典。新用户无法通过下载并安装组件完成首次准备。已有环境与本地 ZIP 接入是另两条路径，本轮未把它们判为不可用。

发布前应使该固定地址提供与内置大小及 SHA-256 一致的组件并验真实下载、恢复和安装，或调整对用户承诺的交付入口。本次没有上传组件。官网 stable/preview 两个 `latest.json` 同时为 404，属于正式发布时需要补齐的更新清单准备项。

### F3 / 发布准备：现存包仍为 Preview，且落后于当前工作区

[version.json](../../release/version.json)、前端版本及现存包均为 `3.0.0-preview.1`，应用 Python 版本为 `3.0.0rc1`。仅重命名下载文件不能形成正式版身份。

冻结源码清单的 689 个文件中，16 个与当前内容不同，另外当前选择范围新增 20 个文件。包含宿主、图标、反馈、多个模块桥和窗口组件。当前 `frontend/dist/assets` 中 36 个文件相对冻结包身份不同或未包含。说明书媒体有构建时压缩，未将媒体与源文件的差异直接当作遗漏。

当前源码的顶边窗口改造晚于该包，不能据源码的窗口恢复修复记录宣称当前 EXE 已验该功能，也不能据其差异反推旧式 EXE 必有同一黑屏问题。需将预定正式内容冻结并重建，再验证新包。源码测试和当前 Preview EXE 测试分别成立。

## 本轮实测

现存应用 `dist/PhoneticToolbox-v3-MFA-20261011-R3/PhoneticToolbox-windows-x64.exe`：473,722,527 字节，SHA-256 `5c7f0688493e7f926505066b959ffd6c7cf2089a17d3ec81c13975f64f5c0468`。安装包 475,639,106 字节，更新 ZIP 473,723,103 字节，均未超 500,000,000 字节。重新计算三包摘要与已有记录一致，ZIP 内应用与免安装 EXE 同 SHA。冻结清单 SHA 为 `3ed67e87dd3c064c40d41a0f8d44ad1682567be00dd681af36238aa930a4a95c`。

| 层次 | 实际检查与结果 |
| --- | --- |
| 最终 EXE | 拷贝单文件到工程外 `D:/PTB-QA-300-7cd42f16`，隔离 LOCALAPPDATA/TEMP，移除 Python/Conda/PTB 开发绑定，Windows-only PATH，`--verify-distribution` 与 `--verify-m12` 均退出 0 |
| 科学与依赖 | EGG/M05 两组随包科学探针、MediaPipe 图初始化、主宿主恢复文件摘要及 73 组共享路径身份通过 |
| 实际任务 | 11 项托管任务及产物回读通过，涉及声学分析、变速变调、发声合成分析/生成、Klatt、LPC、EGG、语谱图转音频两模式、音系预览/导出 |
| 实际 Qt | 18 个模块帮助打开/返回，TextGrid 11 阶段及保存回读，M13 PNG、M15 文本刺激会话/按键/JSON、M16 合成 PCM 录制/编辑/导出/关闭协商、M17 输入/字体/布局、M10 原生采样通过。复看 TextGrid 浅色与 VoQS 1366×768 深色截图 |
| MFA 清理 | 实际 EXE `--ptb-remove-managed-mfa` 在独立合成目录退出 0，管理组件及下载残片删除，外部环境、设置和对应注册记录保留；未操作实际 MFA 或真实卸载确认界面 |
| 当前源码 | 桌面/发行/反馈/MFA 等 180 项定向 Python 检查通过；M03-R8 指定科学环境 11 项通过；前端 366 项通过，typecheck、ui-data:check、contracts:check、文档及架构检查通过 |
| 退出 | 两个成品检查正常退出，隔离 TEMP 无 `_MEI` 残留，工程外 QA 所属进程已结束 |

Python 首轮误把 M03-R8 放入 `.venv/m14`，187 passed / 4 failed，四个失败均为 `egg_runtime_mismatch`。随后按 `Invoke-M03-Python.ps1` 的固定环境及 DLL 搜索路径重跑完整 M03-R8，11/11 通过。保留首轮错误日志，未删除断言或改变容差。180 为首轮扣除 M03-R8 后的检查数量。

最初深层隔离目录把 LOCALAPPDATA 设为 135 字符，成品 EGG 探针的 Pillow `_imaging` 加载报文件名过长。改用上述短的工程外目录，同一 SHA 的 EXE 全部检查通过。这是自定义超长用户数据路径下已复现的兼容性边界，普通用户目录未复现。产品未来如支持这类路径，需要路径预算检查或较短的缓存布局；本轮没有修改系统长路径配置。

## 公开发行与未验范围

现有[来源登记](../../third_party/source-registry.json)仍将 SoE 实现的具体改写/分发许可记为待确认，根目录未发现总 LICENSE。对应源码是否随最终公开渠道交付也需按[已有准备报告](2026-10-05-preview-source-preparation.md)收口。[来源规则](../../third_party/AGENTS.md)明确规定“未决许可阻断对应公开发行”。这是项目既有发布门槛，本轮未作新的法律判断或代选许可。

另一台干净 Windows、Windows 10、实体麦克风/摄像头与同步、热插拔、多屏混合 DPI、实体声音/反应时、长时间运行及完整科研参数矩阵未测。安装/升级的当前字节证据复核了[本日 R3 报告](windows-current-release.md)，本轮未重新运行完整安装/升级循环。重新构建正式版后须按[严格打包规则](../../release/PACKAGING_RULES.md)针对最终字节验收。

## 复现证据与清理

本机证据入口：`output/validation/release-review-20261011/`。`review_checks.py` 为源码/身份检查，`reproduce.py` 为 GitHub 混合历史列表和 EGG 重跑，`frozen_short.py` 为工程外成品检查，`mfa_cleanup_probe.py` 为清理边界探针。

- `frozen-short/`：最终成功的 distribution/TextGrid 报告、截图与命令结果。
- `targeted-9fc4e83fd3224bff9eb5b65b907e46da/`：GitHub 三组对照、M03 11 项通过。
- `source-601f711b8da349e6b77688336ecd1047/`：首轮 Python、前端、文档、架构日志。
- `frozen-4460662e7c9742d8905065345d53d1c0/`：成品身份/源码差异、长路径失败与成功清理回执。
- `mfa-remove-1b1b1f405fb045b0b54434a91e6e8413/`：最终 EXE MFA 清理保护结果，scratch 已按受管入口删除。
- `network.json`、`github-public.json`：仅公开端点的状态与 Release 列表。

工程外 QA 目录和源码单元测试 scratch 的后续清理被自动审批拒绝，原文仅为 `blocked by policy`。前者未发现联接或存活所属进程，逻辑文件长度约 4.38 GB，含硬链接重复计数；后者保留安全测试故意创建的联接，其目标也在同一测试 scratch 内。两者均保留，回执为 retained，未沿联接删除或绕过拒绝。其余受管探针 scratch 已清理。原始资料、正式设置和运行中的用户实例均未清理。
