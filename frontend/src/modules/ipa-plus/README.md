# 国际音标表Plus · M17

本目录实现 IPA、extIPA、VoQS 的本机查找、Unicode 输入、介绍与可配置演示。模块注册见 [registry.ts](../../app/registry.ts)，完整操作正文见[结构化手册章节](../../../../manual/chapters/m17.json)。手册八节按用途、界面、流程、控件、进阶、输出、恢复和来源组织；其中旧 `m17-section-01` 至 `-05` 锚点保留。

## 架构与职责

M17 运行在公共前端工作台内，不提交声学计算任务。目录、解释、字体随前端资源读取；编辑器操作在内存中执行，草稿写入本机 IndexedDB。外部来源只由用户主动打开。桌面纯文本复制复用公共 Qt 通道，TXT 下载复用桌面下载桥接；网页使用浏览器机制。

普通工作台的帮助按钮由 [AppShell.vue](../../app/AppShell.vue) 捕获并导向 M17 说明书，阅读器提供返回模块按钮。`IpaPlusPage.vue` 仍保留早期内联帮助实现，不能据此写成当前普通用户入口。字体许可的实际入口为左下关于 → 内置字体版权与许可 → PTB IPA Plus。

| 文件 | 职责与约束 |
| --- | --- |
| [IpaPlusPage.vue](IpaPlusPage.vue) | 体系、搜索、显示设置、光标、历史、自动保存、复制/下载、介绍/演示的生命周期。 |
| [ChartView.vue](ChartView.vue)、[SymbolButton.vue](SymbolButton.vue) | IPA 两页、extIPA 四页、VoQS 分类，矩阵/元音图/分组，按当前模式输入或播放；右键及 Alt+Enter 看详情。 |
| [editor.ts](editor.ts)、[unicode.ts](unicode.ts) | 字素边界保护、literal/combining/bridge/paired-span 插入，统一撤销/重做，码位和异常字符提示。 |
| [storage.ts](storage.ts) | 草稿版本恢复、旧默认字号迁移、修订号比较与冲突停止覆盖。 |
| [SymbolDetails.vue](SymbolDetails.vue)、[hover-position.ts](hover-position.ts) | 自主解释、实际输入、例示、别名、短来源；浮窗在符号附近自动避让指针及视口边界。 |
| [content.ts](content.ts)、[SymbolPlayback.vue](SymbolPlayback.vue)、[playback.ts](playback.ts) | 内容覆盖、受限素材相对路径、原生 audio/video、实时动画注册与退出清理。 |
| [catalog.ts](catalog.ts)、[types.ts](types.ts)、[fonts.css](fonts.css) | 类型化目录读取、当前体系内包含检索、固定音标字体与塑形设置。 |

## 资产与数据

| 数据 | 格式与身份 | 内容和边界 |
| --- | --- | --- |
| [catalog.json](data/catalog.json) | `Catalog`，当前 `m17/1.2.0-20261003` | 625 稳定入口，IPA 351 / extIPA 209 / VoQS 65。含基础字符、组合、例示和工具，数目不等于独立音标数。保存实际输入、码位、插入模式和来源定位。 |
| [sources.json](data/sources.json) | 来源 ID 对象映射 | 公开书目、URL、图表版次和对象许可。VoQS 的旧稳定键保留，显示作者与 URL 已用王天恒 Zenodo 正式记录。 |
| [voqs-zh-names.json](data/voqs-zh-names.json) | 独立中文名称数据 | 56 个原表译名，与新增程度/范围/例句入口分开。 |
| [symbol-content.json](data/symbol-content.json) | `{version: 1, entries: {...}}` | 按目录 ID 覆盖介绍及媒体配置；不改原 ID、输入序列、插入模式和来源定位。当前 `entries` 为空，无真实条目演示素材。 |
| [PTBIPAPlus-Regular.ttf](../../assets/ipa-plus/PTBIPAPlus-Regular.ttf) | PTB IPA Plus 1.000 | Doulos SIL 7.000 派生字体，补充必要组合；OFL 和 Noto 原文随模块提供。中文说明沿用应用字体。 |

IPA 辅音是 14 部位 × 13 方式、186 入口的合并矩阵，另有元音 28、其他符号 9、附加符号 80、超音段 14、声调与词重调 34。IPA 空格表示未收录当前入口。extIPA 保留原表阴影语义，并把 2025 内容、补充组合、2002 旧版记号区分显示。VoQS 原表 56 项另加 9 个程度/范围/例句入口。旧私用区资料仅辅助检索，现代字符映射与来源材料本身分别保留。

## 输入、输出与持久格式

| 类型 | 内容 | 限制 |
| --- | --- | --- |
| 输入 | 手动文本、粘贴、目录按钮、组合工具 | 不读取音频，不判定转写是否适合语料，不自动把整串文字改成另一体系。 |
| 纯文本输出 | 全文剪贴板、UTF-8 `.txt` | TXT 默认 `国际音标-YYYY-MM-DD.txt`，不内嵌字体、解释、素材或历史；下载提示不能替代实际文件检查。 |
| 草稿 | `Draft.version=1`；数据库 `phonetic-toolbox-m17` / `drafts`；默认键 `ipa-plus.v1.M17` | 正文、记录中的选区、体系、悬停开关、字号、高度、修订号和写入者。`stateKey` 区分工作台记录。 |
| 独立草稿 | 冲突时追加写入者身份的键 | 保留当前内存文字的分支，默认入口仍读原键；必须同时导出文本。 |
| 媒体配置 | `audio` / `video` 相对路径，或 `animation={renderer,version,config}` | 仅接受站点/包内素材，禁止绝对路径、外站 URL、路径穿越；允许的扩展名见 `mediaUrl()`，扩展名允许不代表全部编码已验证。 |

首次为空文本、IPA 基础图表、悬停介绍开启、音标字号 26px、编辑高度 120px、点击播放关闭。字号 18–54 取整，普通表项默认 20px；高度 96–360px。编辑后约 450ms 自动保存。旧字号默认 28px 一次迁移到 26px，其他显式值按范围恢复。搜索、分区页、点击播放、详情状态和撤销栈不持久。选择变化本身不一定触发写入。历史最多 120 步、累计约 800 万 UTF-16 单元，只淘汰较早记录，不截断正文。

## 操作语义

1. 默认点符号插入光标处或替换选区。虚线圆只用于显示，独立附加号只写入附加字符；浅色按钮写入完整例示。
2. bridge 在选中恰好两个字素时插入连接字符，字母及已有附加号通常算一个字素。paired-span 在选区前后加标签，空选区时光标在范围内部。
3. 手动输入、粘贴、点击和清空共享历史。输入法组合先提交，切表/检索/详情不改正文。
4. 点击播放模式只激活演示，保持正文、选区及撤销/重做；没有素材、解码失败、动画渲染器未注册都有具体提示，失败不回退为输入。
5. 独立音轨存在时视频静音，两路共同请求播放。收起、模式/体系切换和离开模块终止媒体与渲染器；动画通过 AbortSignal 和清理函数停止。
6. 帮助打开本模块说明书，返回国际音标表Plus恢复原文本、选区和已开工作区。字体许可窗口使用公共 Doulos SIL、JetBrains Mono、PTB IPA Plus 三页签，默认 Doulos SIL，派生字体及 Noto 原文在 PTB IPA Plus 页签。

## 独立内容维护工具

此工具供源码维护者使用，不进入普通用户手册操作流。入口为[独立启动器](../../../../scripts/Start-M17-Content-Editor.ps1)，详见[内容维护说明](../../../../docs/manual/ipa-content-authoring.md)。它选择 625 个稳定入口，写入介绍、包内素材路径和动画配置到 `symbol-content.json`。实际素材需维护者另行提供，并核对来源与分发许可。

维护服务只在单独启动的本机 Vite 会话中提供，含回环地址、会话、同源、路径/格式和版本冲突校验。普通产品构建排除维护页面和写入服务。动态接口已提供，具体声道实时动画仍待接入；配置文件并不证明科学动画完成。维护时保留目录输入字符、稳定 ID、原始译名及非目标条目。

## 方法来源与许可

- IPA：官方 2026 重印，内容修订 2015/2005，交互衍生图表按 CC BY-SA 4.0。中文术语按中国语言学会语音学分会 2007 中文版和江荻 2008 译本定位，原图/书籍不随包。
- extIPA：ICPLA 2025 表。资源页列 CC BY-SA 3.0，同时保留原图复制不作变更的要求。当前自主交互重排与解释不复制原图；2013 中文论文讨论的 2002 版单独处理。
- VoQS 原表：Ball、Esling、Dickson 的 2016 修订表，论文在线 2017、卷期 2018。原论文/图版版权独立保留，未打包原 PDF。
- VoQS 译名：王天恒. *VoQS：音质符号（2016 中英双语版）*. Zenodo(2020-08-30)[2023-12-26]. [DOI 10.5281/zenodo.10206204](https://doi.org/10.5281/zenodo.10206204)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。保留译者、原表作者、来源、许可和交互改写说明，不将译表许可扩大到原论文或图像。
- 字体：PTB IPA Plus 1.000 / Doulos SIL 7.000 和两个 Noto 分隔符来源按 OFL 1.1 保留版权、许可和派生字体更名要求。字体许可与论文、图表及用户文本分别处理。

证据入口：[术语定位](../../../../docs/references/m17-r2-terminology-sources.md)、[资料审计](../../../../docs/references/m17-source-audit.md)、[字体审计](../../../../docs/references/m17-font-audit.md)、[当前 VoQS 许可核查](../../../../docs/references/p19-license-classification-audit.md)和[统一来源登记](../../../../third_party/source-registry.json)。早期审计的 500 入口与来源登记中的旧覆盖数为历史基线，当前以目录 625 为准。原书、论文 PDF、原始 CIN 与私有来源路径不作为公开仓库媒体。

## 验证入口与实际限度

从仓库根目录运行以下只读检查，使用项目已有的 Python 和 Node 环境，不新增全局依赖：

```powershell
python scripts/verify_m17_catalog.py
python scripts/verify_m17_fonts.py
node --test frontend/tests/m17-ipa-plus.test.ts frontend/tests/m17-r2-sources.test.ts frontend/tests/m17-r3.test.ts
python scripts/manual/validate.py --project manual --strict
```

公共前端检查为 `npm --prefix frontend run typecheck`、`npm --prefix frontend test`、`npm --prefix frontend run build`。工作台启动与有界原生验证入口为 [Start-M16-M17-Workbench.ps1](../../../../scripts/Start-M16-M17-Workbench.ps1) 和 [verify_m17_r3_qt.py](../../../../scripts/verify_m17_r3_qt.py)。它们会启动任务拥有的服务或隐藏 Qt 进程，执行前需沿用项目运行环境与报告所列约束。

说明书截图入口为 [capture_m17_states.py](../../../../scripts/manual/capture_m17_states.py)，使用隔离任务库、缓存和 Qt profile，核对输入/选区、撤销、缺媒体提示及 TXT 回读。复制适配器检查与实际系统剪贴板检查分别记录，截图遵循当前作者规范。取证留在本机 `output/manual-work/chapter-audit-m17.json`，结束清理测试副本。

[R3 报告](../../../../docs/testing/2026-10-05-m17-r3-report.md)限定 Windows 源码、Chrome、隐藏 Qt、静态发行前端媒体解码与独立维护保存。报告只证明其列出的对象与范围。实体扬声器听辨、真实录制、精确音视频同步、任意编码兼容、具体声道动画、实体 DPI/DWM、近期 EXE 和跨平台 GUI 均需独立验收。介绍是转写说明，VoQS 标签不对应统一声学阈值或病理诊断；多字符圈围 `⟅…⟆` 为显式文本替代。外部软件需要 Unicode/字体塑形支持，浏览器关闭后的离线冷启动无 PWA 保证。
