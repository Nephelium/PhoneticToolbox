# README 配图来源

根 [README](../../../README.md) 当前展示18个模块、首页和设置，共20张实际Windows工作台截图，统一采用 **Absolutely浅色配色**。图片原尺寸为2561 × 1601，页面缩放100%，屏幕DPR为1.5。当前配图采于2026-10-12的源码工作台。

## 当前展示图片

| 页面 | 图片尺寸 |
| --- | --- |
| [工作台首页](absolutely-home.png) | 2561 × 1601 |
| [参数估计](absolutely-acoustic-analysis.png) | 2561 × 1601 |
| [参数显示](absolutely-parameter-display.png) | 2561 × 1601 |
| [LPC 谱图](absolutely-lpc-analysis.png) | 2561 × 1601 |
| [EGG 信号分析](absolutely-egg-analysis.png) | 2561 × 1601 |
| [TextGrid 标注](absolutely-textgrid-annotation.png) | 2561 × 1601 |
| [声学参数合成](absolutely-speech-synthesis.png) | 2561 × 1601 |
| [发声类型合成](absolutely-phonation-synthesis.png) | 2561 × 1601 |
| [变速变调](absolutely-pitch-manipulation.png) | 2561 × 1601 |
| [语谱图转音频](absolutely-spectrogram-to-audio.png) | 2561 × 1601 |
| [生理参数合成](absolutely-vocal-tract.png) | 2561 × 1601 |
| [唇形提取](absolutely-lip-extraction.png) | 2561 × 1601 |
| [MFA 自动标注](absolutely-mfa-alignment.png) | 2561 × 1601 |
| [汉字转国际音标](absolutely-mandarin-ipa.png) | 2561 × 1601 |
| [国际音标表Plus](absolutely-ipa-chart.png) | 2561 × 1601 |
| [音系归纳](absolutely-phonology-induction.png) | 2561 × 1601 |
| [感知实验](absolutely-perception-experiment.png) | 2561 × 1601 |
| [录音](absolutely-recording.png) | 2561 × 1601 |
| [语音学论文精读](absolutely-paper-reading.png) | 2561 × 1601 |
| [工作台设置](absolutely-settings.png) | 2561 × 1601 |

每张图均由任务专属的实际Qt工作台采集，公开副本与采集原图逐字节一致。界面通过自己的设置选择配色，图片没有裁剪、缩放、拼贴或重绘。图片摘要、输入摘要及采集范围见 [provenance.json](provenance.json)。

## 示例输入与公开范围

- 参数估计、参数显示、LPC、声学参数合成等保留正式说明书自然语句示例。TextGrid为任务编写的单层演示标注。
- EGG保留说明书的双声道原始媒体，显示40–41秒有声段与微观窗口。发声类型合成保留自然单音的F0提取对照。
- 变速变调使用本轮指定目录中的单音节ma，实际导入230→110 Hz序列并完成整段合成；截图包含输出波形与重新提取的F0。
- 语谱图转音频使用该单音节经STFT得到的灰度频谱PNG，正常图片模式实际重建，显示输入/输出频谱与重建波形。Hann窗512点、10 kHz、步长64点、0–5000 Hz、灰度0至−80 dB；实际应用重建参数见图及provenance。
- 音系归纳使用用户指定的《测试.xlsx》，第1行起导入全部5453条记录，生成两份DOCX与一份XLSX。
- 感知实验展示实际ABX设计，36份分组文件来自12段授权单音节，生成12试次并按种子20260927全局洗牌。项目已保存，未建立被试会话。
- 汉字转国际音标使用张若虚《春江花月夜》全文，36句、252汉字，加标点和换行为305字符。文本按[维基文库](https://zh.wikisource.org/zh-hans/春江花月夜_(張若虛))核对，采用简体字。
- 论文阅读使用项目已发布的协同发音论文栏目缓存，普通工作台内50%缩放，完整显示第一页面。原文、译文和解读均按已登记PDF摘要核对；未使用全屏，未展示本机路径。
- 唇形提取保留采集准备界面，录音保留任务清单，没有启动摄像头或麦克风。

公开文件只包含原字节截图、匿名输入摘要和来源说明。作者提供的录音、字表、原始频谱图片、处理输出、数据库与任务配置均留在本机。输入原件摘要不变，两轮调整的临时副本与缓存按归属清理，报告和独有截图保留。

## 保留的既有证据

2026-10-11的五张 `release-300-*.png` 来自已公开的Windows 3.0.0 R7最终EXE，应用SHA-256为 `162c8fa542b7b3b1203ce535e3021235e261f0dfaeb095659f9946baec055edb`。此前六张源码工作台图也保留。它们当前不在根README展示，其原输入授权、尺寸、摘要和EXE身份保存在provenance的历史节点中。

[website-button.svg](website-button.svg) 为300 × 58的静态黄蓝官网按钮，不依赖脚本、外部字体或动态图片服务。

当前图片记录源码工作台的界面和示例计算，已公开R7成品的字节身份单独记录于 [发行检查](../../testing/windows-current-release.md)。本轮文档和配图没有重建EXE。检查结果见 [README检查报告](../../testing/github-readme.md)。
