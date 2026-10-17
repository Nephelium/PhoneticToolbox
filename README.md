<h1 align="center">PhoneticToolbox 3.0.3</h1>

<p align="center"><strong>语音学工具箱</strong><br>把录音、分析、标注、合成与感知实验放进同一个工作台。</p>

<p align="center">
<a href="https://www.phonetictoolbox.com/"><img src="docs/assets/readme/website-button.svg" alt="访问软件官网 ↗" width="300" height="58"></a>
</p>

<p align="center"><a href="#下载与开始使用">下载与开始使用</a> · <a href="#全部模块">全部模块</a> · <a href="#文档与开发">开发文档</a> · <a href="#来源许可与历史版本">来源与许可</a></p>

**Windows x64 正式版 · macOS Apple Silicon 测试版 · 18 个工具**

PhoneticToolbox 是韶子为语言学、实验语音学研究与教学制作的一只语音学工具箱。希望把准备材料、分析信号、制作刺激和整理结果这些日常工作放到一起，让大家在不同工具之间切换时，少花些找窗口和搬文件的力气。

到了 v3，十八个工具共用侧栏、标签页、设置和使用说明。从一段录音出发，可以查看波形与语谱图、提取参数、整理标注，再继续做语音合成、音系分析或感知实验。也可以打开音标表、设计实验，或切到论文阅读页读几页文献。

下面先介绍下载与使用，再逐个展示工具和实际示例。3.0.3 减少了绘图、阅读和预览时的重复计算，语谱图转音频新增离线窗长与动态范围预测，并默认采用动量加速的 Griffin–Lim 重建。完整变化见[这一版的发行说明](https://github.com/Nephelium/PhoneticToolbox/releases/tag/v3.0.3)。

![PhoneticToolbox 的 Windows 工作台首页，按分析与采集、合成与操控、标注与实验、文献与学习组织十八个工具](docs/assets/readme/absolutely-home.png)

## 下载与开始使用

### 选择 Windows 版本

可以在[官方网站](https://www.phonetictoolbox.com/)获取当前 Windows 版本。日常使用选安装版即可；想放在自己的目录里直接运行，可以选免安装版。两类应用提供相同的工具，按自己的使用习惯选择就好。

| 版本 | 文件名 | 使用方式 |
| --- | --- | --- |
| 免安装版 | `PhoneticToolbox-windows-x64.exe` | 双击启动，无需运行安装向导 |
| 安装版 | `PhoneticToolbox-3.0.3-windows-x64-setup.exe` | 按当前用户安装，默认不要求管理员权限，创建应用入口 |

Windows 版本要求 **Windows 10 1809 及以上或 Windows 11、64 位 x64 系统**。可在 Windows **设置 → 系统 → 关于 → 系统类型**中核对。

安装版默认放在 `%LOCALAPPDATA%\Programs\PhoneticToolbox`。安装完成后，从开始菜单或已创建的桌面快捷方式启动应用。主程序下载包小于 500 MB，第一次打开会准备本机运行文件，展开后的磁盘占用更大。**MFA 运行环境、声学模型和发音词典独立下载，主包不包含这些组件。**

也可以从 [GitHub Releases](https://github.com/Nephelium/PhoneticToolbox/releases/tag/v3.0.3)下载 Windows 版，旧版本仍然保留，查阅或对照时都能找到。当前成品的实际检查范围见[3.0.3 发行验收](docs/testing/package-trimming.md)与[同版本更新验收](docs/testing/release-3.0.3-replacement.md)。

### macOS 测试版

[官方网站](https://www.phonetictoolbox.com/#macos-download)已提供 **3.0.3 r6 测试版**，适用于 **Apple Silicon（M 系列芯片）、macOS 14 及以上**。下载文件为 `PhoneticToolbox-3.0.3-macos14-arm64-preview-r6.zip`，解压后将 `PhoneticToolbox.app` 拖入“应用程序”，再双击打开。

测试包尚未经过 Apple 公证，首次打开请参照官网 Mac 下载按钮内的安装说明。Windows 与 Mac 的下载入口均在官网，按自己的系统选择即可。

### 第一次使用

1. 打开应用，先从首页或侧栏选一个需要的工具。工具会留在标签页里，后面可以随时切回来。
2. 导入对应的音频、视频、TextGrid、参数表或工程文件。
3. 核对输入声道、时间范围、分析方法与参数，再开始分析、编辑或合成。
4. 完成后，用工具中的保存或导出入口保留成果，再按需要进入下一个研究环节。
5. 遇到不熟悉的操作，点工具里的**帮助**即可查看对应章节，左下角的**使用说明**则可以打开整本说明书。

## 用一个工作台连接研究环节

做语音学研究时，同一份材料往往要经过好几个步骤。韶子把常见的工作整理在下面，大家可以按手头的任务选择入口。

| 研究任务 | 可以这样开始 | 常见成果 |
| --- | --- | --- |
| 采集与声学分析 | 录音或导入已有音频，提取参数，再查看时间轨迹、EGG 与局部频谱 | 音频、参数表、分析图 |
| 标注与音系整理 | 导入 TextGrid，编辑标注边界和标签，或用已有转写进行 MFA 对齐 | TextGrid、调查字表、同音字表 |
| 语音操控与合成 | 编辑基频和时长，调整声学参数、发声类型或声道构形 | 合成音频、参数工程、动作视频 |
| 教学与感知研究 | 查询音标、转换汉字读音，设计感知实验，阅读和批注语音学论文 | 音标文本与图片、实验配置和记录、阅读批注 |

波形、语谱图、参数轨迹与 TextGrid 在相应工具中共享时间位置。找到需要观察的一小段语音后，就可以围着这一段核对信号、标签和数值。MFA 得到的自动对齐结果，也可以接着在 TextGrid 标注工具里人工复核。

## 全部模块

想先看工具能做什么，可以从下面的表格和示例图开始。每个模块还有独立 README，记录输入输出、计算方法与验证范围；具体怎么操作，应用内说明书里有图文步骤。

| 方向 | 模块与独立说明 | 主要用途 |
| --- | --- | --- |
| 分析与采集 | [录音](frontend/src/modules/recording/README.md) | 本机采集、任务录音与可恢复剪辑 |
| 分析与采集 | [唇形提取](frontend/src/modules/lip-extraction/README.md) | 视频采集、唇形识别与时间对齐 |
| 分析与采集 | [参数估计](frontend/src/modules/parameter-estimation/README.md) | 批量提取声学与唇形参数，关联 TextGrid |
| 分析与采集 | [参数显示](frontend/src/modules/parameter-display/README.md) | 多轨参数对照、时间选区与图像导出 |
| 分析与采集 | [EGG 信号分析](frontend/src/modules/egg-analysis/README.md) | 音频与 EGG 联动、接触商、声门事件与逆滤波 |
| 分析与采集 | [LPC 谱图](frontend/src/modules/lpc-spectrum/README.md) | 波形、标注与局部 LPC 频谱联动 |
| 合成与操控 | [声学参数合成](frontend/src/modules/speech-synthesis/README.md) | Klatt 参数化语音合成 |
| 合成与操控 | [生理参数合成](frontend/src/modules/vocal-tract/README.md) | 声道建模、动态发音与动作视频 |
| 合成与操控 | [发声类型合成](frontend/src/modules/phonation-synthesis/README.md) | 探索发声方式连续统 |
| 合成与操控 | [变速变调](frontend/src/modules/pitch-manipulation/README.md) | 调整语速与基频，试听和批量生成 |
| 合成与操控 | [语谱图转音频](frontend/src/modules/spectrogram-to-audio/README.md) | 从谱图重建声音，进行频谱编辑 |
| 标注与实验 | [国际音标表Plus](frontend/src/modules/ipa-plus/README.md) | IPA、extIPA 与 VoQS 查找、介绍和输入 |
| 标注与实验 | [汉字转国际音标](frontend/src/modules/mandarin-ipa/README.md) | 汉字读音转换、字音对照与图片导出 |
| 标注与实验 | [MFA 自动标注](frontend/src/modules/mfa/README.md) | 依据已有转写进行强制对齐 |
| 标注与实验 | [TextGrid 标注](frontend/src/modules/annotation/README.md) | 多层标注编辑、边界调整与唇形对齐 |
| 标注与实验 | [音系归纳](frontend/src/modules/phonology-induction/README.md) | 调查字表、音系分析与同音字表 |
| 标注与实验 | [感知实验](frontend/src/modules/perception/README.md) | 实验设计、配置与运行 |
| 文献与学习 | [语音学论文精读](frontend/src/modules/paper-reading/README.md) | 论文库、PDF 阅读、译读对照与批注 |

为了便于比较，下面的配图统一采用 **Windows 工作台的 Absolutely 日间配色**。日常使用时可以换成自己喜欢的主题。点击图片可查看完整尺寸，示例与采集来源见[配图说明](docs/assets/readme/README.md)。

### 分析与采集

#### 录音

可以自由录音，也可以把要读的材料整理成任务清单。录完以后，继续查看音频与 EGG 声道，剪辑并保存需要的片段。

![录音工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-recording.png)

#### 唇形提取

需要观察发音时的唇部动作，可以在这里选择设备、采集模式和帧率，完成录制，再进行唇形分析。

![唇形提取工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-lip-extraction.png)

#### 参数估计

先看波形与语谱图，选好需要的声学参数，再批量分析。已有 TextGrid 时，也可以把参数与标注关联起来。

![参数估计工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-acoustic-analysis.png)

#### 参数显示

提取完参数，可以把几条轨迹放在一起比较。图窗共享时间选区，方便回到同一段语音检查，也可以导出图像。

![参数显示工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-parameter-display.png)

#### EGG 信号分析

把音频、EGG、接触商与声门事件放在一起查看，再到局部窗口核对波形。观察细节时，时间位置也能对应起来。

![EGG 信号分析工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-egg-analysis.png)

#### LPC 谱图

想仔细看某一小段的频谱，可以从波形中选取片段，比较 LPC 包络与频谱，也可以参考标注来定位。

![LPC 谱图工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-lpc-analysis.png)

### 合成与操控

#### 声学参数合成

可以从音频提取 Klatt 参数，再编辑轨迹、设置合成与淡入淡出参数，按需要制作合成声音。

![声学参数合成工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-speech-synthesis.png)

#### 生理参数合成

调整发音器官的构形，看看声道截面、面积函数与传递函数怎样变化，再把不同构形组织成动态发音。

![生理参数合成工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-vocal-tract.png)

#### 发声类型合成

选好源音与目标音，比较它们的参数，再生成不同插值位置的声音，制作发声类型的连续统。

![发声类型合成工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-phonation-synthesis.png)

#### 变速变调

想调整一条调型，或为实验制作变调刺激，可以在这里编辑基频、调整语速，合成后再查看和试听结果。

![变速变调工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-pitch-manipulation.png)

示例使用单音节 ma，导入 230→110 Hz 的下降基频序列后合成整段，查看原音、合成波形和输出 F0。

#### 语谱图转音频

可以导入图片重建声音，也可以在图片或音频频谱上绘制信息，再观察生成的声音与频谱。填写时长、频率范围和坐标类型后，可离线预测窗长与动态范围，预测值也能手动调整。新任务默认使用 0.99 动量加速的 Griffin–Lim，迭代次数默认为 32 次。

![语谱图转音频工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-spectrogram-to-audio.png)

示例把单音节的灰度语谱图作为图片输入，完成重建后对照输入频谱、输出频谱和实际音频波形。

### 标注与实验

#### 国际音标表Plus

需要查符号、组合音标时，可以在这里查看 IPA、extIPA 与 VoQS 的介绍和例音，输入音标并保存草稿。

![国际音标表Plus工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-ipa-chart.png)

#### 汉字转国际音标

把汉字文本放进来，就可以查看字音对照。多音字和儿化形式可逐个选择，排版也能按讲义或材料的需要调整。

![汉字转国际音标工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-mandarin-ipa.png)

示例输入张若虚《春江花月夜》全文，共 36 句、252 个汉字，展示字音同显的转换结果。

#### MFA 自动标注

准备好音频和已有转写，核对模型与词典后进行强制对齐。结果保存为 TextGrid，可以接着人工复核。

![MFA 自动标注工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-mfa-alignment.png)

#### TextGrid 标注

对照波形编辑标注层、边界和标签，也可以把唇形数据关联进来，让声音与动作一起帮助判断。

![TextGrid 标注工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-textgrid-annotation.png)

#### 音系归纳

把调查字表导入后，逐步整理调类与声韵分类，再生成两种排列的同音字表和二维声韵表。记录比较多时，可以直接在结果里搜索字头或 IPA。

![音系归纳工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-phonology-induction.png)

示例使用《测试.xlsx》的全部 5,453 条记录，从第一行导入，生成两份同音字表和二维声韵表。

#### 感知实验

把准备好的刺激分组导入，选择实验范式，安排试次、呈现与响应流程，再保存设计，供后续实验使用。

![感知实验工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-perception-experiment.png)

示例为单音节 ABX 辨别设计：36 份分组素材、12 个试次，已生成并洗牌保存。

### 文献与学习

#### 语音学论文精读

论文也有独立的阅读空间。可以整理本机 PDF 和收藏夹，阅读原文与译读内容，把需要回头看的地方标亮或写下批注。

![语音学论文精读工作台，Absolutely 浅色配色](docs/assets/readme/absolutely-paper-reading.png)

示例在普通工作台内以 50% 缩放阅读协同发音论文，完整显示首页，可切换原文、中文译文与论文解读。

### 工作台设置

看得舒服也很重要。外观、字体、图标、本机存储与自动清理都放在设置中，支持浅色、深色和跟随系统。页面缩放、正文与图表字体、按钮高亮和波形颜色，可以按屏幕和使用习惯调整。

![工作台设置，选用 Absolutely 浅色配色](docs/assets/readme/absolutely-settings.png)

## MFA 独立组件

第一次使用 MFA，可以在模块里点**下载并安装组件**，按进度提示完成安装。已经准备好组件 ZIP 或解压后的文件夹，也可以直接选择导入，应用会校验并检查是否可运行。

Windows 独立组件为 **MFA 3.3.8-r1 / Windows x64 / CPU**，包含默认普通话声学模型和发音词典。组件通过应用内入口从官网独立下载，也支持同版 ZIP 离线安装。其他语言可从模块内的官方资源按钮取得匹配的模型和词典，再导入检查。

组件下载约 813 MB，安装后约 2.81 GB，解压过程中还需要临时空间。

MFA 根据已有文本寻找音频中的对齐位置。请准备同名 WAV 与 LAB、TXT 或适用的 TextGrid 转写，并核对词典覆盖与模型音素。结果保存到独立目录，原有 TextGrid 保留。详细流程、兼容要求和方法来源见[MFA 模块说明](frontend/src/modules/mfa/README.md)。

## 帮助、反馈与数据保存

**不熟悉的操作，可以先翻翻说明书。** 说明书随应用提供，模块内的帮助会直接打开对应章节。图片可以放大，音频和演示媒体也能边读边播放，便于对照自己的材料试一试。

**遇到问题，欢迎从关于页提交反馈。** 第一次提交时验证联系邮箱即可，无需注册账号。带上问题描述、操作步骤和截图，会更方便定位问题；后续回复保存在应用内，也可以选择接收邮件提醒。关于页还提供官网、软件与代码来源、字体许可等入口。

**需要长期保留的成果，请导出到自己的研究目录。** 本机存储与自动清理页提供当前占用和缓存清理入口。运行缓存清理后，下次启动会重新准备；未导出的内部处理结果被清理后，需要重新计算。用户导入的原始录音和保存到研究目录的成果由用户管理。

应用通过**检查更新**获取新版信息，当前更新下载完整新版应用包。也可以关闭应用后，在原目录安装或替换新版。设置、草稿、研究文件及已安装的 MFA 组件沿用原存储位置。卸载安装版时，可选择删除应用管理的 MFA 组件，默认保留；自行指定的外部环境保留。

## 文档与开发

| 想做什么 | 从这里开始 |
| --- | --- |
| 学习操作 | 应用内**使用说明**，或各模块**帮助** |
| 查某个模块的功能、格式与限制 | 上方模块表中的独立 README |
| 了解源码结构与职责 | [总架构](ARCHITECTURE.md)、[文档导航](docs/README.md) |
| 配置环境并启动源码 | [开发说明](docs/development.md)、[统一源码入口](docs/development/source-entry.md)、[依赖清单](requirements/README.md) |
| 修改正式说明书 | [说明书工程](manual/README.md)、[独立说明书编辑器](tools/manual-studio/README.md) |
| 构建 Windows 发行包 | [发行工程](release/README.md)、[打包要求](release/PACKAGING_RULES.md) |
| 查看进展与验证 | [当前状态](docs/project-status.md)、[验证策略](docs/testing/verification-plan.md) |

界面使用 Vue 3，桌面宿主使用 PyQt6 / Qt WebEngine。科学计算与文件格式位于 `packages/phonetic_core`，本机任务由 `backend` 管理；浏览器适配与受限服务器计算共用这些核心。网页版目前处于私测阶段，进展见[当前状态](docs/project-status.md)。

Windows 开发环境需要 Python、Node.js、各科学模块环境及相应原生组件，按[统一源码入口](docs/development/source-entry.md)配置后，在工程根目录启动：

```powershell
.\scripts\Start-Research-Workbench.ps1 -CheckOnly
.\scripts\Start-Research-Workbench.ps1
```

前端命令在 `frontend` 目录运行：

```powershell
npm ci
npm run contracts:check
npm run ui-data:check
npm run typecheck
npm test
npm run build
```

如果准备参与开发，建议先阅读对应模块的 README，再按改动范围运行验证。报告会记录实际检查的平台、设备与范围，科学准确率、实体音视频设备、长期运行和其他系统分别核验。

源码仓库保留加载器、算法适配和构建脚本。运行环境、MFA 组件及语谱图预测模型等资源按[依赖清单](requirements/README.md)与[来源登记](third_party/README.md)另行准备。

## 来源、许可与历史版本

科学方法、代码、字体、模型、词典和数据分别登记。应用内**方法与引用**及**关于**可查相关信息，详细记录见[第三方来源与许可](third_party/README.md)。项目整体发行许可证及部分第三方代码的再分发条件仍在确认，当前不作统一的商用许可承诺。

公开仓库提供维护源码、必要资源和可公开文档。用户原始录音、研究数据、下载的论文全文，以及说明书中仅限随软件分发的媒体不随公开源码提供。本页当前截图使用任务专属的隔离配置与示例文本，来源见[配图说明](docs/assets/readme/README.md)。

默认分支 `main` 承载 v3，旧版源码保留在独立的 [v2 分支](https://github.com/Nephelium/PhoneticToolbox/tree/v2)。旧版本可通过[历史发布](https://github.com/Nephelium/PhoneticToolbox/releases)和 [v2.1.4 标签](https://github.com/Nephelium/PhoneticToolbox/tree/v2.1.4)查阅。

韶子希望这只工具箱能为大家的研究与教学帮上一点忙。欢迎使用，也欢迎带着具体材料和操作步骤提出建议，让它更贴近实际的语音学工作。
