# 说明书草案 · 方法、参考文献与第三方致谢

D0.3。本文用于 v3 说明书编排和软件来源页的内容基线；当前描述迁移来源，新的软件入口尚未实现。模块操作说明在逐模块实现与验收后补入，不以设计稿代替真实教程。

PhoneticToolbox 集成多项开源工具和研究方法，并提供统一工作流、参数编辑、可视化、文件管理与平台适配。算法、实现、模型与素材的贡献按下列来源分别归属，不将这些工作统称为 PhoneticToolbox 独立原创。

## 如何查看或引用方法
软件计划在各模块的“方法与来源”中显示本次使用的算法/版本与引用，结果 manifest 保存 source_ids；“关于 → 参考文献与第三方组件”集中列出依赖和许可证。桌面离线可查看引用文字，打开外部原文需要网络。

研究报告还应写明采样率、分析后端、窗长/帧移、参数范围、缺失值处理和预处理步骤。引用某篇方法论文不自动证明与论文实验完全相同；本工具的具体实现差异见相关条目。

## 声道模块的 PDF 入口
- [VocalTractLab 2.4 官方手册（对应当前引擎）](https://www.vocaltractlab.de/download-vocaltractlab/VTL2.4-manual.pdf)
- [VocalTractLab 2.3 官方手册（补充旧版）](https://www.vocaltractlab.de/download-vocaltractlab/VTL2.3-manual.pdf)
- [Birkholz 等 2006 三维声道模型论文](https://www.vocaltractlab.de/publications/birkholz-2006-icassp.pdf)

外部头部形状与平均鼻腔是独立参考几何；不能视为 VTL JD2 说话人完整实测解剖。鼻腔 v4 在线元数据本轮尚未重新取得，以本地来源记录为审计线索，并在发布前再次核对。

## Praat / Parselmouth
来源编号：`SRC-PRAAT`；用于 M01, M02, M03, M04, M06, M07, M08, M12。

作者/机构：Yannick Jadoul and Bill Thompson and Bart de Boer。

本地 acoustic/f0_praat.py、formants_praat.py、manipulation 使用 parselmouth；环境 praat-parselmouth 0.4.7。

软件注明 Praat 作者 Paul Boersma / David Weenink 与 Parselmouth 作者；使用者导出方法记录实际版本。本文的 jitter/shimmer 主路径不据此称为 Praat 实现。

参考：Yannick Jadoul and Bill Thompson and Bart de Boer. (2018). Introducing Parselmouth: A Python Interface to Praat. 10.1016/j.wocn.2018.07.001
[项目主页](https://www.praat.org/) · [文档](https://parselmouth.readthedocs.io/en/stable/) · [论文/出版页](https://doi.org/10.1016/j.wocn.2018.07.001) · [原 PDF](https://cris.vub.be/ws/files/38628284/jadoul_introducing_parselmouth_a_python_interface_to_praat.pdf)

## REAPER (Robust Epoch And Pitch EstimatoR)
来源编号：`SRC-REAPER`；用于 M01, M02, M07。

作者/机构：David Talkin and REAPER contributors。

core/acoustic/f0_reaper.py 调用随带 reaper.exe；upstream README/LICENSE 已读取。

当前 exe 与哪次源码构建对应尚不明确，不能以本轮远端 HEAD 代替。非同名 DAW。

[代码仓库](https://github.com/google/REAPER)

## IRAPT
来源编号：`SRC-IRAPT`；用于 M01。

作者/机构：Elias Azarov and Maxim Vashkevich and Alexander A. Petrovsky。

core/acoustic/f0_irapt.py；Sinc_hash_1000.mat 与上游同名文件 SHA-256 完全一致：e3e2fb01d67f722b7f13c1c9a0f4d62559a1ece77167343dfa42de7203f4d860。

需要保留原代码/数据版权和移植说明；精确初始复制 commit 未知。

参考：Elias Azarov and Maxim Vashkevich and Alexander A. Petrovsky. (2012). Instantaneous pitch estimation based on RAPT framework.
[代码仓库](https://github.com/Mak-Sim/IRAPT)

## Troparion / WM-PC jitter and shimmer
来源编号：`SRC-WMPC`；用于 M01。

作者/机构：Maxim Vashkevich and Alexander Petrovsky and Yuliya Rushkevich。

jitter_shimmer.py 的 _wm_phase_const、bias 搜索、amp_extract、PPQ/RAP/APQ 对应上游 MATLAB；F0 路径先 IRAPT 后回退 Parselmouth。

arXiv 2020 与 SPA 2019 版本分别标识；上游 README 与 arXiv 作者顺序有出入，正式出版引用顺序仍待对照。不可宣称工具获得医疗诊断有效性。

参考：Maxim Vashkevich and Alexander Petrovsky and Yuliya Rushkevich. (2020). Bulbar ALS Detection Based on Analysis of Voice Perturbation and Vibrato.
[代码仓库](https://github.com/Mak-Sim/Troparion) · [论文/出版页](https://arxiv.org/abs/2003.10806) · [原 PDF](https://arxiv.org/pdf/2003.10806) · [published_doi](https://doi.org/10.23919/SPA.2019.8936657)

## VoiceSauce
来源编号：`SRC-VOICESAUCE`；用于 M01, M02。

作者/机构：Yen-Liang Shue and Patricia Keating and Chad Vicenik and Kristine Yu。

本地 CPP/HNR/SHR/SoE/谐波计算的注释、函数结构与 VoiceSauce MATLAB 参照；本地原工程 func_getCPP/HNR 标有 Shue/UCLA SPAPL 版权，SoE 标有 Soo Jin Park。

不能把 OpenSauce Apache 许可证直接套给 VoiceSauce 原始 MATLAB；2011 官方指定引文优先。每个函数的来源链还需 P10 全量精查。

参考：Yen-Liang Shue and Patricia Keating and Chad Vicenik and Kristine Yu. (2011). VoiceSauce: A program for voice analysis.
[项目主页](https://www.phonetics.ucla.edu/voicesauce/) · [文档](https://www.phonetics.ucla.edu/voicesauce/documentation/index.html) · [原 PDF](https://www.phonetics.ucla.edu/voiceproject/Publications/Shue-etal_2011_ICPhS.pdf)

## OpenSauce Python
来源编号：`SRC-OPENSAUCE`；用于 M01。

作者/机构：Terri M. Yu and R. David Murray and Kate Silverstein and Kristine M. Yu。

本机 opensauce-python-master 及上游 harmonics.py/shrp.py 与本项目方法对应；LICENSE/CITATION.txt 已核验。

该项目是 VoiceSauce/OpenSauce 的 Python 实现；数学公式对应不等于已证明每个本地函数直接复制于该 commit。

参考：Terri M. Yu and R. David Murray and Kate Silverstein and Kristine M. Yu. (2019). OpenSauce: Open source software for voice analysis. 10.5281/zenodo.2638411
[代码仓库](https://github.com/voicesauce/opensauce-python) · [论文/出版页](https://doi.org/10.5281/zenodo.2638411)

## CPP: Hillenbrand, Cleveland & Erickson
来源编号：`REF-CPP`；用于 M01。

作者/机构：James Hillenbrand and Ronald A. Cleveland and Robert L. Erickson。

core/acoustic/cpp.py 的倒谱峰显著性；论文出版元数据已核验。

本地实现基于 VoiceSauce 风格处理，数值设置不因引用此论文就视为完全复现原实验。

参考：James Hillenbrand and Ronald A. Cleveland and Robert L. Erickson. (1994). Acoustic Correlates of Breathy Vocal Quality. 10.1044/jshr.3704.769
[论文/出版页](https://pubs.asha.org/doi/abs/10.1044/jshr.3704.769)

## HNR: de Krom
来源编号：`REF-HNR`；用于 M01。

作者/机构：Guus de Krom。

core/acoustic/hnr.py 与 VoiceSauce HNR 路径；出版元数据已核验。

HNR05/15/25/35 与其他软件 HNR 定义不可混为一项。

参考：Guus de Krom. (1993). A Cepstrum-Based Technique for Determining a Harmonics-to-Noise Ratio in Speech Signals. 10.1044/jshr.3602.254
[论文/出版页](https://pubs.asha.org/doi/10.1044/jshr.3602.254)

## SHR: Xuejing Sun
来源编号：`REF-SHR`；用于 M01。

作者/机构：Xuejing Sun。

SHR 本地实现和 OpenSauce/VoiceSauce shrp.py 对应；IEEE 搜索元数据核对。

不要引用同年 ISCA 另一篇无关文章；具体实现设置和引用来源分开记录。

参考：Xuejing Sun. (2002). Pitch determination and voice quality analysis using Subharmonic-to-Harmonic Ratio.
[论文/出版页](https://ieeexplore.ieee.org/document/5743722/)

## 谐波幅度校正: Iseli & Alwan
来源编号：`REF-ISELI`；用于 M01。

作者/机构：Markus Iseli and Abeer Alwan。

corrections.py 的 r=exp(-πB/fs) 与数字共振器公式对应 2004 论文；官方作者机构 PDF 已阅读。

旧文档写成 Iseli 1999 不准确，v3 方法页改为本条；本轮未更改算法。

参考：Markus Iseli and Abeer Alwan. (2004). An improved correction formula for the estimation of harmonic magnitudes and its application to open quotient estimation.
[原 PDF](https://www.seas.ucla.edu/spapl/paper/iseli04.pdf)

## 共振峰带宽估计: Hawks & Miller
来源编号：`REF-HAWKS`；用于 M01。

作者/机构：John W. Hawks and James D. Miller。

本地校正/带宽模型与 OpenSauce harmonics.py 的系数和引用对应；作者机构出版记录核验。

估计带宽和 Praat 直接测得带宽分别说明。

参考：John W. Hawks and James D. Miller. (1995). A formant bandwidth estimation procedure for vowel synthesis. 10.1121/1.412986
[论文/出版页](https://profiles.wustl.edu/en/publications/a-formant-bandwidth-estimation-procedure-for-vowel-synthesis-4372) · [DOI](https://doi.org/10.1121/1.412986)

## Strength of Excitation / epoch extraction
来源编号：`REF-SOE`；用于 M01。

作者/机构：K. Sri Rama Murty and B. Yegnanarayana。

本机原 MATLAB func_getSoE.m 标 Soo Jin Park (2015)，并引用 K. Sri Rama Murty / B. Yegnanarayana (2008) Epoch Extraction From Speech Signals；本地 core/acoustic/soe.py。

本轮确认方法链与源注释；原始论文完整出版字段和本地改动待核验，不能只留下论文而漏掉实现作者。

参考：K. Sri Rama Murty and B. Yegnanarayana. (2008). Epoch Extraction From Speech Signals.
[文档](https://www.phonetics.ucla.edu/voicesauce/documentation/index.html)

## tdklatt / TrackDraw
来源编号：`SRC-TDKLATT`；用于 M06。

作者/机构：Adrian Y. Cho and Daniel R Guest。

本地 core/synthesis/klatt/tdklatt.py 与上游行结构/类/文档字符串高度对应；归一化逐行匹配约 91.87%，并有本地播放/处理改动。

原版权：Copyright (c) 2017 Adrian Y. Cho and Daniel R Guest。保留原文和改动说明；不声称 Klatt 引擎原创。

[代码仓库](https://github.com/guestdaniel/tdklatt)

## Klatt formant synthesizer
来源编号：`REF-KLATT`；用于 M06。

作者/机构：Dennis H. Klatt。

Klatt 方法论文，已通过原论文 PDF 和 tdklatt 研究背景核对；具体代码作者见 tdklatt。

方法引用不取代 tdklatt 实现版权。

参考：Dennis H. Klatt. (1980). Software for a cascade/parallel formant synthesizer. 10.1121/1.383940
[原 PDF](https://sail.usc.edu/~lgoldste/Ling582/Week%2012/klatt1980.pdf) · [DOI](https://doi.org/10.1121/1.383940)

## 载瓦语 F0 与发声类型研究代码/数据
来源编号：`SRC-ZAIWA`；用于 M07。

作者/机构：Yao Lu and Changwei Liang and Jiangping Kong。

本机 python_replication 明确写为论文复现；原 Section 2/main.m 等包含合成流程和原始音节，v2 phonation_synthesis.py 从该 Python 工作衍生。

公开代码仓库没有发现明确 LICENSE，不能当作公共领域；代码、录音、统计数据和论文许可需分别判断。当前 v2 不必加载原论文统计/音节数据，勿为演示而新增分发。

[代码仓库](https://github.com/Luyao2025/Contribution-of-F0-and-phonation-to-tone-perception-in-the-Zaiwa-language)

## F0 与发声类型对载瓦语声调感知的贡献
来源编号：`REF-ZAIWA`；用于 M07。

作者/机构：Yao Lu and Changwei Liang and Jiangping Kong。

作者/期刊/卷号 110/文章 101413 已由出版方索引与本机论文原文核对。

v2 Python 采用 Parselmouth/REAPER 替换原 STRAIGHT MulticueF0 等步骤，为近似重建；不要声称精确复现原论文。

参考：Yao Lu and Changwei Liang and Jiangping Kong. (2025). Contribution of F0 and phonation to tone perception in the Zaiwa language. 10.1016/j.wocn.2025.101413
[论文/出版页](https://www.sciencedirect.com/science/article/pii/S0095447025000245) · [DOI](https://doi.org/10.1016/j.wocn.2025.101413)

## Griffin–Lim 重建
来源编号：`REF-GRIFFINLIM`；用于 M09。

作者/机构：Daniel W. Griffin and Jae S. Lim。

core/spec2wav/griffin_lim.py；原论文 PDF 已核对。

相位估计是重建，不保证恢复原始录音；固定随机种子与迭代设置进入结果。

参考：Daniel W. Griffin and Jae S. Lim. (1984). Signal estimation from modified short-time Fourier transform. 10.1109/TASSP.1984.1164317
[论文/出版页](https://ieeexplore.ieee.org/document/1164317) · [原 PDF](https://dub.ucsd.edu/CATbox/Reader/GriffinLimMSTFT.pdf) · [DOI](https://doi.org/10.1109/TASSP.1984.1164317)

## VocalTractLab API 2.4
来源编号：`SRC-VTL`；用于 M10。

作者/机构：Peter Birkholz and VocalTractLab contributors。

resources/vocal_tract/ 源码 ZIP、GPL 文本、sources.lock.json、native DLL 和 bridge 源码；已核验官方 2.4 下载页面。

VTL 与本地 bridge/adapter 的版权/修改声明随包；VocalTractLabAnalysis.dll 同字节副本用于状态隔离，非独立自研引擎。

[项目主页](https://www.vocaltractlab.de/) · [官方下载](https://www.vocaltractlab.de/download-vocaltractlab/VTL2.4.zip) · [文档](https://www.vocaltractlab.de/index.php?page=vocaltractlab-download)

## VocalTractLab 2.4 官方手册
来源编号：`DOC-VTL24`；用于 M10。

作者/机构：见原来源；缺失信息在发行前补证。

官方 PDF 已打开，30 页；对应当前 2.4 引擎版本。

软件提供打开官方手册/下载原 PDF 链接；不默认把 PDF 镜像打包。

[原 PDF](https://www.vocaltractlab.de/download-vocaltractlab/VTL2.4-manual.pdf)

## VocalTractLab 2.3 官方手册（补充）
来源编号：`DOC-VTL23`；用于 M10。

作者/机构：见原来源；缺失信息在发行前补证。

井井指定链接，已打开官方 PDF，30 页。

明确标为 2.3 版补充资料，不能冒充当前 2.4 API 文档。

[原 PDF](https://www.vocaltractlab.de/download-vocaltractlab/VTL2.3-manual.pdf)

## 三维声道模型
来源编号：`REF-VTL2006`；用于 M10。

作者/机构：Peter Birkholz and Dietmar Jackèl and Bernd J. Kröger。

官方作者网页与 4 页论文 PDF 核验。

声道几何/控制模型的基础引用；2.4 实际引擎增量以当前手册/源码说明为准。

参考：Peter Birkholz and Dietmar Jackèl and Bernd J. Kröger. (2006). Construction and Control of a Three-Dimensional Vocal Tract Model.
[原 PDF](https://www.vocaltractlab.de/publications/birkholz-2006-icassp.pdf) · [publications](https://www.vocaltractlab.de/index.php?page=birkholz-publications)

## Three.js r180 / OrbitControls
来源编号：`SRC-THREE`；用于 M10。

作者/机构：见原来源；缺失信息在发行前补证。

gui/resources/vocal_tract/vendor/ 有三个脚本、THREE-LICENSE；sources.lock.json 记录准确 hash。

v3 可迁移既有三维显示，但保留版本、版权。

[代码仓库](https://github.com/mrdoob/three.js/tree/r180)

## 3D human parts pack / FACE2
来源编号：`ASSET-HEAD`；用于 M10。

作者/机构：byzmod3d。

本地 sources.lock.json 记录 face2_0.obj 原 hash 和 head.json 衍生 hash，官方页面作者 byzmod3d/CC0 已核验。

只取 FACE2，去掉 CABELO，三角化；配准为外观参考，不是受试者 MRI，不参与声学边界。

[项目主页](https://opengameart.org/content/3d-human-parts-pack)

## Healthy nasal cavities — averaged geometry
来源编号：`ASSET-NASAL`；用于 M10。

作者/机构：Jan Brüning and Thomas Hildebrandt and Werner Heppt and Nora Schmidt and Hans Lamecker and Angelika Szengel and Natalja Amiridze and Heiko Ramm and Matthias Bindernagel and Stefan Zachow and Leonid Goubergrits。

本地 lock/notice 含 v4 下载 hash、作者全表和修改说明；本轮 DOI 直接获取未成功，相关论文检索可定位。

继承证据，不冒充本轮已完成全部在线验证；须保留原作者、v4 DOI、坐标变换及平均模型≠JD2 个体说明。

参考：Jan Brüning and Thomas Hildebrandt and Werner Heppt and Nora Schmidt and Hans Lamecker and Angelika Szengel and Natalja Amiridze and Heiko Ramm and Matthias Bindernagel and Stefan Zachow and Leonid Goubergrits. (2020). Healthy nasal cavities - averaged geometry. 10.6084/m9.figshare.9585410.v4
[DOI](https://doi.org/10.6084/m9.figshare.9585410.v4) · [related_paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7048824/)

## VocalTractLab-Python
来源编号：`SRC-VTLWRAPPER`；用于 M10。

作者/机构：见原来源；缺失信息在发行前补证。

本地 lock 明确：Read as reference; not installed, not imported；本轮确认官方仓库可读。

保留参考来源；不能在依赖表中声称当前引擎使用其 Python wrapper。

[代码仓库](https://github.com/paul-krug/VocalTractLab-Python)

## MediaPipe Face Mesh
来源编号：`SRC-MEDIAPIPE`；用于 M05。

作者/机构：见原来源；缺失信息在发行前补证。

lip_gui.py 使用 legacy Face Mesh；phonetic_311 安装 mediapipe 0.10.14，PyPI 对应发布元数据已核验。

官方当前 Face Landmarker 文档不能冒充本地使用新 API；具体 tflite 模型 hash/许可单独补齐。

[文档](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) · [代码仓库](https://github.com/google-ai-edge/mediapipe)

## FFmpeg
来源编号：`SRC-FFMPEG`；用于 M05。

作者/机构：见原来源；缺失信息在发行前补证。

run.spec/视频处理涉及 FFmpeg；当前分发二进制 build flags 和确切来源未完成核验。

GPL/LGPL 取决于实际构建；需要记录每平台具体二进制、许可证/源代码获取方式，不能笼统标同一种许可。

[项目主页](https://ffmpeg.org/) · [license](https://ffmpeg.org/legal.html)

## Montreal Forced Aligner / Kaldi / models
来源编号：`SRC-MFA`；用于 M11。

作者/机构：见原来源；缺失信息在发行前补证。

mfa_alignment_service / pipeline 调用外部环境；官方文档已查阅。

MFA 程序、Kaldi 依赖、声学模型与词典分别列许可；本轮未安装/运行 MFA，版本待环境专项探测。

[文档](https://montreal-forced-aligner.readthedocs.io/en/latest/)

## Montreal Forced Aligner 论文
来源编号：`REF-MFA`；用于 M11。

作者/机构：Michael McAuliffe and Michaela Socolof and Sarah Mihuc and Michael Wagner and Morgan Sonderegger。

ISCA 原始论文记录核验。

当前采用的 MFA 版本/模型应另记，引用 2017 论文不代表使用 2017 二进制。

参考：Michael McAuliffe and Michaela Socolof and Sarah Mihuc and Michael Wagner and Morgan Sonderegger. (2017). Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi. 10.21437/Interspeech.2017-1386
[论文/出版页](https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html) · [DOI](https://doi.org/10.21437/Interspeech.2017-1386)

## Doulos SIL 字体
来源编号：`ASSET-DOULOS`；用于 M13。

作者/机构：SIL International。

IPA HTML 随附字体；官方说明 SIL OFL；当前网站 7.000 不可当成本地字体版本。

正式包装提取字体 name 表版本和 hash，随字体放准确 OFL 文本；其他系统字体不擅自复制。

[项目主页](https://software.sil.org/doulos/) · [官方下载](https://software.sil.org/doulos/download/)

## EGG 事件分析/简化逆滤波方法
来源编号：`PENDING-EGG`；用于 M03。

作者/机构：见原来源；缺失信息在发行前补证。

core/egg 的算法出处没有完整引用链，需逐函数查 GCI/GOI、CQ/SQ、滤波和简化 CPIF 的实现来源。

列入 P10 待核验清单；不能用无来源占位条目通过发行审查。


## 内置普通话发音词典
来源编号：`PENDING-DICTIONARY`；用于 M12。

作者/机构：见原来源；缺失信息在发行前补证。

web_praat_editor/default.dict 的初始来源、授权和修改史尚未定位。

列入 P10 待核验清单；不能用无来源占位条目通过发行审查。


## 11 套普通话转换规则及映射数据
来源编号：`PENDING-IPA`；用于 M13。

作者/机构：见原来源；缺失信息在发行前补证。

原 HTML/生成脚本内标准名称已核对；字表/多音规则/各标准原始出版信息尚未完整核实。

列入 P10 待核验清单；不能用无来源占位条目通过发行审查。


## 音系归纳规则和帮助材料
来源编号：`PENDING-PHONOLOGY`；用于 M14。

作者/机构：见原来源；缺失信息在发行前补证。

本地转换/排序/归并代码已定位；未找到足以断言全部原创的历史记录，需审查复用数据和参考规则。

列入 P10 待核验清单；不能用无来源占位条目通过发行审查。


## 本地 pitch_perception 的 web_praat_editor 来源链
来源编号：`ORIGIN-WEBEDITOR`；用于 M12。

作者/机构：见原来源；缺失信息在发行前补证。

v2 AGENTS 明确由本地 pitch_perception 项目通用化而来；需追溯其内部第三方脚本，不据项目同属用户就认为没有外部来源。

列入 P10 待核验清单；不能用无来源占位条目通过发行审查。


## K2 蓝色波形团子与 U2 设计参考
来源编号：`ASSET-MASCOT`；用于 global。

作者/机构：见原来源；缺失信息在发行前补证。

本次任务此前生成并由用户暂定；保存原图用于设计，不是正式多尺寸图标发行物。

列入 P10 待核验清单；不能用无来源占位条目通过发行审查。


## html2canvas
来源编号：`SRC-HTML2CANVAS`；用于 M13。

作者/机构：见原来源；缺失信息在发行前补证。

原 HTML/CDN import 已确认存在；具体锁定版本/脚本正文 license 尚需在前端迁移阶段逐个核验。

v3 构建本地静态资源并保存许可证，禁止发行时靠未锁定 CDN 加载运行。

[代码仓库](https://github.com/niklasvh/html2canvas)

## React
来源编号：`SRC-REACT`；用于 M15。

作者/机构：见原来源；缺失信息在发行前补证。

原 HTML/CDN import 已确认存在；具体锁定版本/脚本正文 license 尚需在前端迁移阶段逐个核验。

v3 构建本地静态资源并保存许可证，禁止发行时靠未锁定 CDN 加载运行。

[代码仓库](https://github.com/facebook/react)

## Babel standalone
来源编号：`SRC-BABEL`；用于 M15。

作者/机构：见原来源；缺失信息在发行前补证。

原 HTML/CDN import 已确认存在；具体锁定版本/脚本正文 license 尚需在前端迁移阶段逐个核验。

v3 构建本地静态资源并保存许可证，禁止发行时靠未锁定 CDN 加载运行。

[代码仓库](https://github.com/babel/babel)

## Tailwind CSS
来源编号：`SRC-TAILWIND`；用于 M15。

作者/机构：见原来源；缺失信息在发行前补证。

原 HTML/CDN import 已确认存在；具体锁定版本/脚本正文 license 尚需在前端迁移阶段逐个核验。

v3 构建本地静态资源并保存许可证，禁止发行时靠未锁定 CDN 加载运行。

[代码仓库](https://github.com/tailwindlabs/tailwindcss)

## SheetJS
来源编号：`SRC-SHEETJS`；用于 M15。

作者/机构：见原来源；缺失信息在发行前补证。

原 HTML/CDN import 已确认存在；具体锁定版本/脚本正文 license 尚需在前端迁移阶段逐个核验。

v3 构建本地静态资源并保存许可证，禁止发行时靠未锁定 CDN 加载运行。

[代码仓库](https://git.sheetjs.com/sheetjs/sheetjs)

## Lucide
来源编号：`SRC-LUCIDE`；用于 M15。

作者/机构：见原来源；缺失信息在发行前补证。

原 HTML/CDN import 已确认存在；具体锁定版本/脚本正文 license 尚需在前端迁移阶段逐个核验。

v3 构建本地静态资源并保存许可证，禁止发行时靠未锁定 CDN 加载运行。

[代码仓库](https://github.com/lucide-icons/lucide)

## 运行依赖与完整许可证
NumPy、SciPy、Pandas、Matplotlib、PyQt、PyQtGraph、Parselmouth、PyWavelets、SoundDevice、SoundFile、OpenPyXL、python-docx、OpenCV、MediaPipe、Pillow、PyInstaller 等版本见 [登记表](../../third_party/package-source-audit.json)。这些版本来自原开发环境，v3 最终依赖将在迁移验证后锁定。
[全部来源](../../third_party/README.md) · [可复制 BibTeX](../../third_party/references.bib)

## 本轮未解决的来源
EGG 方法链、IPA 字表/多音规则、内置词典、部分原始 VoiceSauce 代码条款、载瓦语代码/录音授权、实际 FFmpeg 构建许可、MediaPipe/MFA 模型、旧 HTML 依赖版本及历史说明书示例素材，仍需逐项补证。公开仓库不等于已允许任何再分发；该状态要在开发追踪中保持，不能直接作为完整正式说明书发布。
