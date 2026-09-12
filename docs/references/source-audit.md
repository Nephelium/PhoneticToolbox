# 第三方代码、论文与素材来源核验 · D0.3

核验日期：2026-09-09。范围是当前 v2 源码、随带资源及可定位的原始项目，服务于 v3 迁移和软件/说明书引用。**这是有明确待处理项的审计基线，不是“全项目许可已经审核通过”的证明。**

## 证据与状态
- 本地真实文件/注释/调用：证明项目目前使用什么；数学公式相似只能支持方法对应，不能单独证明哪份代码最初被复制。
- 官方仓库 README/许可证/commit、作者或期刊页面、官方 PDF：核验上游是谁、当前可追溯版本和声明。
- `observed_upstream_commit` 是本轮查看的远端版本；`actual_included_version` 才是本地实际使用版本，两者不能混填。
- 每条记录包含原作者、用途、模块、证据、链接和风险。找不到作者/版本时保留待查，不能编造“原创”或随意套用 MIT。
- [来源注册表](../../third_party/source-registry.json)、[上游观测与校验信息](../../third_party/upstream-observations.json)、[BibTeX](../../third_party/references.bib) 是可追踪数据。16 个已装 Python 包的发布元数据另保存于 [环境依赖审计](../../third_party/package-source-audit.json)，不能把这 16 个包当成完整传递依赖 SBOM。

## 1. Jitter / Shimmer 不是简单的一组 Praat 输出
已读本地 `core/acoustic/jitter_shimmer.py` 的 WM-PC 相位/振幅路径与上游 MATLAB。IRAPT 优先给出 F0；失败时才用 Parselmouth 回退。相位累计、偏移搜索、周期和幅度扰动的实现线索对应 [Troparion](https://github.com/Mak-Sim/Troparion)，同时还有 [IRAPT](https://github.com/Mak-Sim/IRAPT) 的算法和查表数据。

特别强的资产证据：本地 `Sinc_hash_1000.mat` 与 IRAPT 上游文件 SHA-256 相同：
`e3e2fb01d67f722b7f13c1c9a0f4d62559a1ece77167343dfa42de7203f4d860`。

软件应显示 jitter/shimmer 的实际算法路径、F0 后端、参数、缺失处理和原实现来源。相关 WM-PC 论文作者为 **Maxim Vashkevich、Alexander Petrovsky、Yuliya Rushkevich**，本次 BibTeX 对应 arXiv 2020 版本；SPA 2019 出版条目及作者顺序须继续对照。[作者论文页及 PDF](https://arxiv.org/abs/2003.10806)。

Troparion 仓库声明 MPL-2.0；IRAPT 为 GPL-3.0。两个来源的许可分别保留，不能因主项目名称变化就消失。

## 2. VoiceSauce / OpenSauce 和声学论文要一起体现
本机原始 MATLAB 参照文件有 Shue/UCLA SPAPL 版权，SoE 文件另外署名 Soo Jin Park。OpenSauce Python 的说明表明其与 VoiceSauce 的移植关系，LICENSE 为 Apache-2.0；这不能直接覆盖原始 MATLAB 文件的权限。

软件/说明书应注明：
- VoiceSauce：Shue、Keating、Vicenik、Yu，2011，ICPhS。使用 [官方引用说明](https://www.phonetics.ucla.edu/voicesauce/) 与 [原论文 PDF](https://www.phonetics.ucla.edu/voiceproject/Publications/Shue-etal_2011_ICPhS.pdf)。
- OpenSauce：Terri M. Yu、R. David Murray、Kate Silverstein、Kristine M. Yu，2019，[仓库及引文](https://github.com/voicesauce/opensauce-python)、[存档 DOI](https://doi.org/10.5281/zenodo.2638411)。
- CPP 对应 Hillenbrand 等 1994；HNR 对应 de Krom 1993；SHR 对应 Sun 2002。分别提供参数含义和实现设置，不把不同软件同名指标自动当成数值等价。
- 谐波校正公式对应 **Iseli & Alwan 2004**，[作者机构 PDF](https://www.seas.ucla.edu/spapl/paper/iseli04.pdf)。旧注释/文档的“1999”不应照抄。带宽估计另引 Hawks & Miller 1995。
- SoE 要保留实现作者和其引用链；原始论文完整字段尚需补证，不能将“已见源注释”冒充论文全文核验。

P10 要逐个函数登记“直接移植/参考算法/独立重写/调用依赖”。本轮已定位关键链，不代表全部函数的版权判断都结束。

## 3. Klatt 合成：方法作者和 Python 实现作者分列
本地 tdklatt 与 [guestdaniel/tdklatt](https://github.com/guestdaniel/tdklatt) 的结构及大量行一致，归一化逐行匹配约 91.87%，且存在本地修改。这个比例是文本比对线索，不是整个模块的“原创比例”。

保留 MIT 原文中的 **Adrian Y. Cho 与 Daniel R Guest（2017）**。算法背景引用 Dennis H. Klatt 1980，提供 [原论文 PDF](https://sail.usc.edu/~lgoldste/Ling582/Week%2012/klatt1980.pdf)。软件可说明 PhoneticToolbox 增加了界面、参数编辑与集成，但不能声称自行发明/独立编写整个合成器。

## 4. 发声类型连续统：已找到载瓦语论文与代码
本机 `python_replication` 文档明确说明以载瓦语研究为基础；原研究仓库 Section 2 提供合成代码与原始音节。对应论文：

Lu, Y., Liang, C., & Kong, J. (2025). *Contribution of F0 and phonation to tone perception in the Zaiwa language*. *Journal of Phonetics*, 110, 101413。[论文 DOI](https://doi.org/10.1016/j.wocn.2025.101413)、[作者代码仓库](https://github.com/Luyao2025/Contribution-of-F0-and-phonation-to-tone-perception-in-the-Zaiwa-language)。

本地 Python 路径改用了 Parselmouth/REAPER 等步骤，不能称为原 STRAIGHT 流程的精确重现；LPC 残差处理也不等于完整的生理逆滤波模型。说明书需要列出这些实现差异，以及当前保留的三种连续统、两种方向/对齐与参数单位。

本轮查看仓库树没有发现明确 LICENSE。论文开放访问或文章许可不能自动授权代码/录音再分发。先保留来源证据，后续确认许可或选取可合法复用的替代实现；与作者发信需要另获授权。本轮没有对外发信，也没新增分发论文实验录音。

## 5. 声道工作台：四种来源分开写
| 来源 | 实际作用 | 已核验/待核验 |
| --- | --- | --- |
| VocalTractLab 2.4 | 原生声道/声源/声学计算 | 官方下载与本地源码、GPL、DLL/hash 可追溯 |
| Three.js r180 / OrbitControls | 浏览器三维渲染 | 本地 vendor 和 MIT 文本、版本记录可定位 |
| byzmod3d FACE2 | 外部头部外观 | 官方资源页 CC0；本地三角化/删发型说明 |
| Brüning 等平均鼻腔 v4 | 独立解剖参考几何 | 本地作者/DOI/hash 完整；本轮 v4 在线元数据获取未成功，保留待复核 |

当前 API 是 **2.4**。软件提供 [2.4 官方手册 PDF](https://www.vocaltractlab.de/download-vocaltractlab/VTL2.4-manual.pdf)，同时保留井井指定的 [2.3 官方手册 PDF](https://www.vocaltractlab.de/download-vocaltractlab/VTL2.3-manual.pdf)，清楚注明版本。两份均已打开确认。基础模型论文：[Birkholz、Jackèl 与 Kröger（2006）原 PDF](https://www.vocaltractlab.de/publications/birkholz-2006-icassp.pdf)。

头模不是当前说话人的 MRI；平均鼻腔不是 VTL JD2 个体的原声学边界。配准变换、外观缩放、鼻腔出处必须写在“模型与来源”，避免把展示几何说成同一个真实个体。[头部原资源](https://opengameart.org/content/3d-human-parts-pack)、[鼻腔 v4 DOI](https://doi.org/10.6084/m9.figshare.9585410.v4)。

本地还记有 [VocalTractLab-Python](https://github.com/paul-krug/VocalTractLab-Python) 的阅读参考 commit；它未被安装/导入，按“参考”列出，不能声称软件依赖该 wrapper。

## 6. 其他实际依赖与待补项
Parselmouth/Praat、REAPER、MediaPipe、MFA、FFmpeg、Doulos SIL、React/Babel/Tailwind/SheetJS/Lucide/html2canvas 均应出现于对应模块或全局组件页。安装版本与发行包中真正存在的文件要一致：
- MediaPipe 的当前本地代码使用 legacy Face Mesh，不能把最新 Face Landmarker 文档说成旧代码所用 API。模型文件要单列 hash 与许可。
- MFA 的程序、Kaldi、声学模型与发音词典是不同来源；用户自己提供模型时仍需记录实际使用名称/版本。
- FFmpeg 二进制许可依构建选项判断；只知道工具名不够。
- Doulos 字体须核对本地 name 表与 OFL；当前官网下载版本不等于已内嵌版本。
- EGG 的具体方法链、IPA 十一套映射数据、内置普通话词典、旧感知页各 CDN 锁定版本和说明书中示例音频/图片的出处仍需逐项补齐。数据库样例、论文附录数据、字体和模型不能漏在“代码依赖”之外。

另外，v2 的 MIT 徽章/元数据不能证明整个发行包可一律用 MIT；本机未找到足以覆盖全部组件的根许可证。PyQt 的官方许可是 GPLv3/商业双轨之一，VTL/IRAPT 等还各有要求。[PyQt 官方许可](https://www.riverbankcomputing.com/software/pyqt/intro)。本轮登记事实与待办，不改写所有权或替用户作最终发行授权判断。

## 7. 软件和说明书的落点
- 侧栏“关于 → 参考文献与第三方组件”：按模块、类型、作者搜索，可复制引用，打开仓库/DOI/官方 PDF，查看随包许可。
- 各模块“方法与来源”：解释实际方法、版本、实现差异、参数单位与可用范围。
- 结果 manifest：附所用算法 ID/版本/后端/配置及 source_ids，用户可复现。
- 说明书同表生成；不重复手写一套容易过期的引文。
- 用户离线也能阅读随包的引用文本和许可证；外部 PDF 链接需网络，明确标识。直接下载链接保留官方地址，发布前测试可达性；未获许可不把全文重新镜像打包。
- P10 在每个平台的实际构建产物上生成完整 SBOM/许可清单，当前 42 条重点来源和 16 包元数据只是起点。

## M01-C适配补充

2026-09-09：新增2个导出依赖和3个API/格式文档参考，登记共323条；[C清单](../../third_party/m01-io-inventory.json)保留40个实际包的来源映射与新增包安装文件hash。原生REAPER无新复制或下载，确切hash/PE架构见 [资源清单](../../resources/manifests/acoustic.json)。原commit、编译选项、完整原生依赖及再分发审计仍待核实；MIT元数据不等于整包发行许可审查完成。

## 2026-09-11 M10 Windows 资源迁入

同日 M02/M09：统一登记补充表读取、Griffin–Lim 核心迁入及受限图像/音频适配位置，前端致谢重新生成。独立运行时精确锁定原已审计版本 OpenCV 4.13.0.92、SoundFile 0.13.1。论文属于方法参考，v2零起点数值/字节对照与非零频带修复分开记录；详见 [联合报告](../testing/m02-m09-report.md)。没有打包论文PDF，也没有将原生依赖许可待审项改为通过。

VTL API 2.4 原始二进制、2.3 参考手册、几何桥接 m10/2、Three.js、头壳和平均鼻腔分别保留来源。迁移/当前哈希见 [M10 迁移清单](../modules/evidence/M10-migration.json)，数值和显示适配边界见 [M10 报告](../testing/m10-report.md)。显示读数补丁与实际侧缘拟合源码随包；独立运行时新增 sounddevice 0.5.3/PyInstaller 锁，未把全项目许可未决项标成通过。

R4 增加 WebCodecs 与 WebM 规范参考，已有 Qt WebEngine 提供 VP8/Opus 编码，本项目按规范写容器，没有移植第三方 muxer。耳语试听补偿、后移约束和共享显示适配见 [R4 报告](../testing/m10-recording-features-report.md)。原生 DLL、原鼻腔资源与来源锁保留，资源包标记 `3.0.0-m10.4`。

## 2026-09-12 M03-D 显示路径迁移

PENDING-EGG增加原egg_widget.update_zoom_plots和InverseFilteringResultDialog的纯数组显示路径，记录于third_party/egg-migration.json。原v2 GUI独立双轮捕获的公开合成数组位于tests/fixtures/m03/ui-display.npz，源码SHA见ui-source.json，逐样本对照通过。没有增加第三方库或改变未确立的代码/学术许可状态；同一来源登记继续生成工作台致谢。

M03-E3新增REF-YIN-EGG-THESIS馆藏书目及REF-HENRICH-2004-DEGG方法审阅参考，共330条登记；两者仅引用与原站链接。阈值/DECOM差异、原手册截图哈希和未决链见[M03方法核查](m03-method-audit.md)。PENDING-EGG许可不变。
