# M18-R1 语音学论文精读验收

2026-10-07。状态 **verified**，范围为 Windows 当前源码、实际 Qt 原生阅读器和生产服务器论文内容分发。现有 EXE 未更新，不将本报告作为成品安装包或整个项目发行许可验收。

## 已交付行为

- 公共工作台新增 M18 语音学论文精读，沿用导航、按钮、字体和主题变量。左栏按栏目发布日期列论文，右栏显示导读、作者、原稿日期、固定版本、原文和许可链接以及译文说明。
- 原文与中文译文切换，分别保留当前标签内的页码。支持翻页、页码、适宽、缩放和可复制的本页文本。正文 PDF 保持白纸背景。
- 宿主首次启动包含此能力的版本时在持久用户目录记录日期，升级不重置。默认获取首次日期当日及以后发布的论文。历史窗口按包含当日在内的起始日期显示总篇数、待下载篇数和大小，主动下载不改变默认起点。
- 打开模块及前台每 15 分钟检查目录，可手动检查。下载后离线可读，失败保留有效目录，损坏文件重试，支持取消。固定 HTTPS 源、目录与文件大小上限、内容摘要校验和原子替换用于限制原生读取范围。
- PDF 位于服务器及用户数据目录，未进入前端资源和发行快照。复用已有 QtPdf；PyQt6 绑定 6.11.0、实际 Qt 6.11.2。M18 来源登记覆盖论文和既有 Qt 依赖，完整软件发行义务仍沿用项目未决项。

源码入口为根目录 `打开语音学论文精读.cmd`，内部调用统一 `Start-Research-Workbench.ps1 -Module M18`。[模块说明](../../frontend/src/modules/paper-reading/README.md)、[分发维护说明](../deployment/paper-reading.md)、[决定](../decisions/ADR-M18-paper-distribution.md)已补齐。

## 首篇内容与许可

Jesuraj Bandekar、Shinji Watanabe、Prasanta Kumar Ghosh，*Articulatory Source-Filter TTS: Physically Grounded Control through Vocal Tract Kinematics*，[arXiv:2610.00735v1](https://arxiv.org/abs/2610.00735v1)，原稿 2026-09-30，栏目发布日期 2026-10-07，许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。许可及署名同时进入界面、目录和译文首页。译文注明 AI 辅助翻译、改编及未经原作者审校，不暗示原作者认可。

原文 13 页，译文 11 页，174 个翻译单元，保留双栏结构、图表、公式和文献。23 个公式 TeX 原样一致，7 表数值一致，4 图来源一致，引用键次数和首次出现顺序一致。基线段落内两处重复引用按中文语序调整，未改变文献编号。图内英文标注和文献条目保留。内容检查不能替代独立专家对译文逐句审校。

导读说明伪构音目标与真实 EMA 的区别、作者定义的功能评分、MOS 样本与预印本边界，并指出正文全尺度 F0 优势陈述与表 VI 部分值不一致。译文忠实保留原陈述。

制作与检查证据在 `output/paper-reading/2610.00735v1`：源项目、来源映射、译文、`content-audit.json`、`pdf-verification.json` 及 24 页渲染图。已回看全部页面联系表、译文首页和表格页高清图。现有 XeLaTeX/BibTeX 编译通过，无依赖安装；有少量字体替换及小于 1.2pt 的表格垂直盒警告，回看未见内容重叠。

## 服务器发布与清理

[公共目录](https://www.phonetictoolbox.com/papers/catalog.json)及两份 PDF 已实际发布并通过 HTTPS 回读。目录 SHA-256：`d0fbf3c021f483da24d7983bf576af93ca9e523fb2d54ce4683ecf4b6d028dee`。

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| 原文 | 891216 | b565b46f09931c821e7104091c678fca3c8609243fb869323a7b043c76cd95fc |
| 译文 | 1003025 | b7308f109c75f5e5cbaefc393c4ac18e24402c3bc8317c4660bb93330d2a7532 |

保留 `/var/www/phonetictoolbox-coming-soon/index.html`，清理前后和 HTTPS 回读 SHA-256 均为 `744ece83cdfbcb53e59126193f1de0a7dd1a09b280620b9ab1a0516ee86f69d7`。nginx 配置、HTTPS 证书、SSH 和系统服务保留。

用户已授权清理旧部署。在确认无对应活动服务、逐项验证目标为 `/home/admin` 下非符号链接后，删除 `phonetictoolbox-preview-staging`、`ptb-m14-20260927`、`ptb-p11-20260926` 以及 `ptb-m14-stage1-20260927.tar.gz`、`ptb-m14-stage2-20260927.tar.gz`、`ptb-m14-evidence-20260927.tar.gz`。记录为逻辑文件大小，不据此推算实际释放磁盘量。回执：`output/paper-reading/server-cleanup-receipt.json`、`server-verification.json`。当前公共渠道没有重新发布软件 EXE。

## 实际验证

- Python 34 项通过，覆盖首次日期持久与并发、损坏记录、日期边界、许可/路径/摘要、下载失败/取消/重试、旧更新桥与统一入口及打包排除。
- 前端类型检查、344 项测试、生产构建、生成来源一致性通过。构建保留既有大分块提示。登记 396 条，界面显示 376 条。
- 统一源码入口 M18 `-CheckOnly` 通过。前端构建产物无 PDF，发行内容策略排除外部论文制作目录；本轮没有完整打包。
- 实际 Qt 从公共 HTTPS 获取目录和两份 PDF，原生指针切换语言、翻页和文本，离线缓存、语言独立页码通过。模拟下一日首次启动不自动获取前日论文，历史窗显示 1 篇待下载，真实下载后首次日期不变。
- 实际 Qt 原文/译文、浅色/深色、1440×900/1100×720 共 8 组布局通过。最终证据 `output/validation/m18/qt-e55669f3b7964224aaa81259931c9742/report.json` 与截图，服务退出 0。
- 首轮工具返回退出码 1，但内部检查成功且无异常堆栈，原因未确定。随后三轮原生退出码均为 0，最终两轮包含历史下载验证。保留全部记录，不将首轮异常抹除。Qt 日志含既有 PNG 色彩资料警告。
- 全库架构检查仍有 477 条既有问题，未涉及本次 M18 文件。文档检查剩余 214 条历史失效链接，本轮报告与新增引用无错误。

最终日志在 `output/paper-reading`：`frontend-tests-final.log`、`frontend-build-final.log`、`python-tests-final.log`、`qt-final.log` 和 `docs-check.json`。

## 未验证范围

未重打 EXE，未验另一实体电脑、全部配色、实体 DPI 切换、跨平台 GUI、长期在线与全部科研模块。译文尚未由独立领域专家逐句审校。当前阅读器逐页显示，不包含全文搜索、批注或连续双页模式。普通浏览器预览显示桌面能力提示。没有 push、数据库迁移、全局安装、系统运行时修改或用户研究数据清理；保留同期工作树改动。
