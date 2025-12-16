# PhoneticToolbox 语音学工具箱

[![GitHub release](https://img.shields.io/github/v/release/Nephelium/PhoneticToolbox)](https://github.com/Nephelium/PhoneticToolbox/releases)
[![Documentation](https://img.shields.io/badge/docs-查看完整说明书-blue)](https://nephelium.github.io/PhoneticToolbox/)

## [点击这里查看完整图文说明书 / Click here for full documentation](https://nephelium.github.io/PhoneticToolbox/)

---

## 简介 (Introduction)

PhoneticToolbox 是一个集成了多种语音分析功能的工具箱，旨在为语音学研究者提供便捷的分析工具。本项目基于 MATLAB 版 **VoiceSauce** 及 Python 版 **opensauce-python** 深度开发，整合了最新的语音处理算法与模型，致力于构建高效、易用且功能强大的语音学研究平台。

### 主要功能 (Key Features)

#### 1. 声学参数提取 (Acoustic Parameter Extraction)
-   提供与 **VoiceSauce** 高度相似的声学参数提取功能。
-   支持提取基频 (F0)、共振峰 (F1-F4)、能量 (Energy)、H1-H2、H1-A1、H1-A2、H1-A3、CPP、HNR、SHR 等多种声学参数。
-   特别优化了**嘎裂声 (Creaky Voice)** 等非模态发声类型的基频提取及参数估算准确度。

#### 2. EGG 信号分析 (EGG Signal Analysis)
-   提供专业的电声门图 (EGG) 信号处理与分析功能，自动计算CQ（闭合商）及SQ（速度商）值。
-   **信号预处理**：支持双声道归一化、高通/低通滤波及双声道信号交换，适应不同采集设备。
-   **逆滤波 (Inverse Filtering)**：支持对语音信号进行逆滤波分析，获取声门波形。
-   支持批量处理与数据导出。

#### 3. 唇形特征提取 (Lip Feature Extraction) **[NEW]**
-   基于 **MediaPipe Face Mesh** 的高精度唇形追踪。
-   **实时指标计算**：实时计算并显示唇面积、唇圆度、唇宽度、唇开度 (Opening) 等关键发音运动指标。
-   **同步录音**：支持在采集唇形数据的同时进行高保真音频录制，确保音视频数据严格对齐。
-   **数据可视化**：录制后在文件夹中提供数据曲线显示，方便回顾监控实验过程。

#### 4. 自动标注 (Auto Annotation) **[NEW]**
-   集成了 **Montreal Forced Aligner (MFA)** 强制对齐引擎。
-   **一键对齐**：能够根据音频和文本脚本，自动生成包含音素 (Phone) 和字/词 (Word) 层级的 TextGrid 标注文件。
-   **便携化设计**：解决了 MFA 在打包环境下的依赖问题，无需繁琐配置即可在本地运行。
-   **文件名一键转换**：解决了 MFA 路径不支持中文字符的问题，支持将所选文件夹内的所有文件一键转换为拼音名称，可在对齐结束后再一键转换回来。

#### 5. 语谱图转音频 (Spectrogram to Audio) **[NEW]**
-   利用 **Griffin-Lim** 算法，实现从语谱图 (Spectrogram) 图像到音频信号的重构。
-   支持隐藏窗口截图，调整重构参数，通过相位恢复技术还原出可听的语音波形。

#### 6. 普通话转国际音标 (Mandarin to IPA) **[NEW]**
-   内置多套注音标准数据库，支持将汉字文本转换为国际音标 (IPA)。
-   **多标准支持**：涵盖“standard Beijing（Chinese） ”、“胡裕树《现代汉语》”、“黄伯荣、廖序东《现代汉语》”、“赵元任《汉语口语语法》”等多种学术标准。
-   支持多音字手动消歧。

#### 7. 感知实验 (Perception Experiment) **[UPDATED]**
-   内置灵活的听辨实验设计模块，支持 **AX**、**ABX**、**AXB** 等经典实验范式。
-   **实验流程优化**：新增了**间歇** 阶段，支持在试次之间插入指导语。
-   **自动化控制**：支持试次间的自动推进 (Auto-advance) 或手动触发，满足不同实验设计的需求。
-   提供完整的数据收集与导出功能。

#### 8. Klatt 语音合成 (Klatt Synthesis)
-   内置图形化的界面。
-   支持精细调整共振峰频率、带宽、基频等几十项合成参数。
-   支持直接输入元音参数进行快速合成。

#### 9. 基频修改与变速变调 (F0 Modification & Time-Scale Modification)
-   **基频重合成**：支持对音频文件的基频 (F0) 曲线进行修改（如拉平、升降调）并重合成音频。
-   **变速变调**：支持对多个音频文件进行批量的变速（不变调）或变调（不变速）处理。

## 下载与安装 (Download & Installation)

请访问 [Releases 页面](https://github.com/Nephelium/PhoneticToolbox/releases) 下载最新版本的 `PhoneticToolbox.exe`。
本软件为绿色免安装版，下载后双击即可运行。

## 使用说明 (Usage)

详细的使用教程、参数说明及参考文献，请参阅我们的在线文档：
**[https://nephelium.github.io/PhoneticToolbox/](https://nephelium.github.io/PhoneticToolbox/)**
