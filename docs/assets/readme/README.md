# README 配图来源

本目录用于根 [README](../../../README.md) 的 GitHub 展示。当前主页展示五张 **Windows 3.0.0 最终 EXE** 的实际截图，以及本次绘制的静态官网按钮。此前源码工作台的六张配图保留为历史来源证据。

## 当前主页：Windows 3.0.0

五张 PNG 来自同一份实际运行的 Windows x64 应用，版本为 3.0.0，文件大小为 496,815,155 字节，SHA-256 为 `162c8fa542b7b3b1203ce535e3021235e261f0dfaeb095659f9946baec055edb`。本轮候选标识为 R7，源码快照 SHA-256 为 `3d1fd7bb865ca389df96ca0b90a3a39327267e90ca76df3fd01abc30e4ffa880`。这些标识用于核对图片对应的应用字节，不表示工作区其他改动已公开。

截图使用任务独立的 Windows Qt 进程、C 盘临时配置与缓存，通过该进程自己的 Qt WebEngine 应用目标取得。页面视口为 1600 × 1050 CSS 像素，DPR 为 1.5，应用页面缩放为 120%，输出原始 PNG 为 **2400 × 1575 像素**。没有裁剪、缩放、拼贴或重绘图片，也没有用源码网页代替 EXE。

| 图片 | 实际展示内容 |
| --- | --- |
| [release-300-home.png](release-300-home.png) | 18 个模块的统一工作台，默认 W1 黄蓝图标，Everforest 浅色配色 |
| [release-300-ipa.png](release-300-ipa.png) | 国际音标表Plus 的 IPA 表与文本编辑区，Catppuccin 深色配色，未播放媒体 |
| [release-300-mandarin.png](release-300-mandarin.png) | 默认转换标准下的通用示例文本与字音排版，Rose Pine 浅色配色 |
| [release-300-vocal.png](release-300-vocal.png) | 默认 VTL `/a/` 构形、声道面积与传递函数，129 个模型截面，Solarized 浅色配色，未播放音频 |
| [release-300-settings.png](release-300-settings.png) | 外观页的配色、深浅模式、页面缩放和波形颜色，Tokyo Night 深色配色 |

汉字转换示例由本次任务编写，仅含“语音学工具箱”“采集、分析、标注、合成”“普通话国际音标”。其他视图使用应用内默认内容。截图没有载入私有研究数据、原始音频、私人文件名或参与者信息。M10 默认模型已完成构形计算，图片记录其当时的展示状态，不作为标准普通话准确率、音频试听或设备验证结果。

[website-button.svg](website-button.svg) 为本次绘制的 300 × 58 静态 SVG，使用 W1 图标对应的黄蓝配色，不依赖脚本、外部字体下载或动态图片服务。根 README 将其链接到 `https://www.phonetictoolbox.com/`，替代普通文字官网入口。

## 历史配图：源码工作台

以下六张 PNG 保留原始 2561 × 1601 像素，页面缩放为 100%，屏幕 DPR 为 1.5，记录此前实际运行的源码 Windows Qt 工作台。它们当前不出现在根 README 中。原有输入授权、摘要和使用范围保留如下。

### 输入与使用范围

用户在本次任务中明确要求配图使用真实音频，随后指定本机语句录音、配套 TextGrid 和 EGG 测试数据。最终配图使用这两组本机素材，未使用合成音频或先前考察的公开朗读样本。

| 公开代号 | 输入情况 | 使用位置 |
| --- | --- | --- |
| `语句示例.wav` | 真人语句录音，44,100 Hz，2 声道，206,009 帧，4.6714059 秒 | 参数估计、LPC、TextGrid 标注 |
| `语句示例.TextGrid` | 同一录音的原有 words / phones 标注，8,310 字节 | 参数估计、LPC 标签、TextGrid 编辑视图 |
| `EGG示例.wav` | 真实 EGG 与音频双声道同步记录，44,100 Hz，646,699 帧，14.6643764 秒 | EGG 分析 |

仅在任务专属临时目录复制输入并采用通用文件名，WAV 与 TextGrid 的字节未更改。截图前后核对原文件 SHA-256 一致。原始录音、TextGrid、处理结果表和参与者信息不进入本目录或公开 Git；临时输入及缓存使用后已清理。私有来源映射保留在本机忽略目录中。

本次使用范围为作者指定的 GitHub 主页静态分析配图。该范围不赋予原始语料公开下载或开放再分发许可。统一来源登记 ID 为 `DOC-README-NATURAL-AUDIO`，见[来源登记](../../../third_party/source-registry.json)。界面涉及的软件、算法、字体和音标表仍遵循各自已有来源登记。

### 图片与计算范围

| 图片 | 展示内容 |
| --- | --- |
| [home.png](home.png) | 统一工作台首页与 18 个模块入口，无音频输入 |
| [acoustic-analysis.png](acoustic-analysis.png) | 同源真人录音的波形、实际 Praat 语谱图、TextGrid 与已完成的 pF0 / pF1 / pF2 任务；分析声道采用界面默认的全部声道混合，试听及语谱预览为声道 1 |
| [egg-analysis.png](egg-analysis.png) | 原录音 3.00–3.50 秒，默认左 EGG / 右音频，25–2,000 Hz 滤波，50 ms 微观窗口，实际 CQ / SQ、声门事件及语谱图；此图未启用 F0 叠加 |
| [textgrid-annotation.png](textgrid-annotation.png) | 同源真人录音与原有 TextGrid，视窗 0–3.2 秒；未在本轮改写原标注或重新运行 MFA |
| [lpc-analysis.png](lpc-analysis.png) | 同源真人录音 1.80–1.95 秒，50 阶 LPC，频率显示上限 8,000 Hz；实际分析结果 |
| [ipa-chart.png](ipa-chart.png) | 国际音标表Plus，未播放或合成音频 |

两组图片的采集日期均为 2026-10-11。输入摘要、截图摘要、实际尺寸和采集方式见 [provenance.json](provenance.json)，验证范围见 [README 检查报告](../../testing/github-readme.md)。历史六图没有断言单一冻结 EXE 身份，不能用于替代本次新包截图。图片也不外推科学准确率、物理设备、长期运行或跨平台验收。
