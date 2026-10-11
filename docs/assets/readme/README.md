# README 配图来源

本目录用于根 [README](../../../README.md) 的 GitHub 展示。六张 PNG 均由实际运行的 PhoneticToolbox Windows Qt 工作台直接截图，保留原始 2561 × 1601 像素，页面缩放为 100%，屏幕 DPR 为 1.5。没有裁剪、缩放、拼贴或重绘截图。

## 输入与使用范围

用户在本次任务中明确要求配图使用真实音频，随后指定本机语句录音、配套 TextGrid 和 EGG 测试数据。最终配图使用这两组本机素材，未使用合成音频或先前考察的公开朗读样本。

| 公开代号 | 输入情况 | 使用位置 |
| --- | --- | --- |
| `语句示例.wav` | 真人语句录音，44,100 Hz，2 声道，206,009 帧，4.6714059 秒 | 参数估计、LPC、TextGrid 标注 |
| `语句示例.TextGrid` | 同一录音的原有 words / phones 标注，8,310 字节 | 参数估计、LPC 标签、TextGrid 编辑视图 |
| `EGG示例.wav` | 真实 EGG 与音频双声道同步记录，44,100 Hz，646,699 帧，14.6643764 秒 | EGG 分析 |

仅在任务专属临时目录复制输入并采用通用文件名，WAV 与 TextGrid 的字节未更改。截图前后核对原文件 SHA-256 一致。原始录音、TextGrid、处理结果表和参与者信息不进入本目录或公开 Git；临时输入及缓存使用后已清理。私有来源映射保留在本机忽略目录中。

本次使用范围为作者指定的 GitHub 主页静态分析配图。该范围不赋予原始语料公开下载或开放再分发许可。统一来源登记 ID 为 `DOC-README-NATURAL-AUDIO`，见[来源登记](../../../third_party/source-registry.json)。界面涉及的软件、算法、字体和音标表仍遵循各自已有来源登记。

## 图片与计算范围

| 图片 | 展示内容 |
| --- | --- |
| [home.png](home.png) | 统一工作台首页与 18 个模块入口，无音频输入 |
| [acoustic-analysis.png](acoustic-analysis.png) | 同源真人录音的波形、实际 Praat 语谱图、TextGrid 与已完成的 pF0 / pF1 / pF2 任务；分析声道采用界面默认的全部声道混合，试听及语谱预览为声道 1 |
| [egg-analysis.png](egg-analysis.png) | 原录音 3.00–3.50 秒，默认左 EGG / 右音频，25–2,000 Hz 滤波，50 ms 微观窗口，实际 CQ / SQ、声门事件及语谱图；此图未启用 F0 叠加 |
| [textgrid-annotation.png](textgrid-annotation.png) | 同源真人录音与原有 TextGrid，视窗 0–3.2 秒；未在本轮改写原标注或重新运行 MFA |
| [lpc-analysis.png](lpc-analysis.png) | 同源真人录音 1.80–1.95 秒，50 阶 LPC，频率显示上限 8,000 Hz；实际分析结果 |
| [ipa-chart.png](ipa-chart.png) | 国际音标表Plus，未播放或合成音频 |

采集日期为 2026-10-11。输入摘要、截图摘要、实际尺寸和采集方式见 [provenance.json](provenance.json)，验证范围见 [README 检查报告](../../testing/github-readme.md)。这些图片记录本机源码工作台当时的界面与计算结果，不作为新安装包、科学准确率、物理设备或跨平台验收的替代证据。
