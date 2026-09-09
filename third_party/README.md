# 第三方组件、方法与素材

D0.3 · 58 条记录（42 条重点方法/代码/素材来源，16 条已装 Python 包发布元数据，部分相互关联）。尚未完成全传递依赖与实际发行包审查，不得将全部记录状态改为已通过。

[查验报告](../docs/references/source-audit.md) · [机器可读注册表](source-registry.json) · [引用 BibTeX](references.bib) · [已装依赖元数据](package-source-audit.json) · [上游观测](upstream-observations.json)

登记原则：代码、论文、模型、实验数据、字体和图标分列。论文的引用/在线 PDF 链接不意味着把论文全文重新分发的许可。软件界面尚未接入本表，P10 负责实现。

## 来源索引

| ID | 名称 | 类型 | 作用模块 | 原作者/机构 | 状态 |
| --- | --- | --- | --- | --- | --- |
| SRC-PRAAT | Praat / Parselmouth | runtime-and-method | M01, M02, M03, M04, M06, M07, M08, M12 | Yannick Jadoul and Bill Thompson and Bart de Boer | review-required |
| SRC-REAPER | REAPER (Robust Epoch And Pitch EstimatoR) | bundled-code | M01, M02, M07 | David Talkin and REAPER contributors | review-required |
| SRC-IRAPT | IRAPT | ported-code-and-data | M01 | Elias Azarov and Maxim Vashkevich and Alexander A. Petrovsky | review-required |
| SRC-WMPC | Troparion / WM-PC jitter and shimmer | ported-method | M01 | Maxim Vashkevich and Alexander Petrovsky and Yuliya Rushkevich | review-required |
| SRC-VOICESAUCE | VoiceSauce | reference-code-and-method | M01, M02 | Yen-Liang Shue and Patricia Keating and Chad Vicenik and Kristine Yu | review-required |
| SRC-OPENSAUCE | OpenSauce Python | reference-implementation | M01 | Terri M. Yu and R. David Murray and Kate Silverstein and Kristine M. Yu | review-required |
| REF-CPP | CPP: Hillenbrand, Cleveland & Erickson | paper | M01 | James Hillenbrand and Ronald A. Cleveland and Robert L. Erickson | review-required |
| REF-HNR | HNR: de Krom | paper | M01 | Guus de Krom | review-required |
| REF-SHR | SHR: Xuejing Sun | paper | M01 | Xuejing Sun | review-required |
| REF-ISELI | 谐波幅度校正: Iseli & Alwan | paper | M01 | Markus Iseli and Abeer Alwan | review-required |
| REF-HAWKS | 共振峰带宽估计: Hawks & Miller | paper | M01 | John W. Hawks and James D. Miller | review-required |
| REF-SOE | Strength of Excitation / epoch extraction | method-chain | M01 | K. Sri Rama Murty and B. Yegnanarayana | review-required |
| SRC-TDKLATT | tdklatt / TrackDraw | adapted-code | M06 | Adrian Y. Cho and Daniel R Guest | review-required |
| REF-KLATT | Klatt formant synthesizer | paper | M06 | Dennis H. Klatt | review-required |
| SRC-ZAIWA | 载瓦语 F0 与发声类型研究代码/数据 | adapted-method-and-source | M07 | Yao Lu and Changwei Liang and Jiangping Kong | review-required |
| REF-ZAIWA | F0 与发声类型对载瓦语声调感知的贡献 | paper | M07 | Yao Lu and Changwei Liang and Jiangping Kong | review-required |
| REF-GRIFFINLIM | Griffin–Lim 重建 | paper | M09 | Daniel W. Griffin and Jae S. Lim | review-required |
| SRC-VTL | VocalTractLab API 2.4 | bundled-native-code | M10 | Peter Birkholz and VocalTractLab contributors | review-required |
| DOC-VTL24 | VocalTractLab 2.4 官方手册 | manual | M10 | 见上游/待补 | review-required |
| DOC-VTL23 | VocalTractLab 2.3 官方手册（补充） | manual | M10 | 见上游/待补 | review-required |
| REF-VTL2006 | 三维声道模型 | paper | M10 | Peter Birkholz and Dietmar Jackèl and Bernd J. Kröger | review-required |
| SRC-THREE | Three.js r180 / OrbitControls | bundled-code | M10 | 见上游/待补 | review-required |
| ASSET-HEAD | 3D human parts pack / FACE2 | derived-asset | M10 | byzmod3d | review-required |
| ASSET-NASAL | Healthy nasal cavities — averaged geometry | derived-research-data | M10 | Jan Brüning and Thomas Hildebrandt and Werner Heppt and Nora Schmidt and Hans Lamecker and Angelika Szengel and Natalja Amiridze and Heiko Ramm and Matthias Bindernagel and Stefan Zachow and Leonid Goubergrits | local-provenance-verified; live-dataset-metadata-pending |
| SRC-VTLWRAPPER | VocalTractLab-Python | reference-only | M10 | 见上游/待补 | review-required |
| SRC-MEDIAPIPE | MediaPipe Face Mesh | runtime-and-model | M05 | 见上游/待补 | review-required |
| SRC-FFMPEG | FFmpeg | native-tool | M05 | 见上游/待补 | local-use-confirmed; exact-binary-pending |
| SRC-MFA | Montreal Forced Aligner / Kaldi / models | external-tool-and-models | M11 | 见上游/待补 | review-required |
| REF-MFA | Montreal Forced Aligner 论文 | paper | M11 | Michael McAuliffe and Michaela Socolof and Sarah Mihuc and Michael Wagner and Morgan Sonderegger | review-required |
| ASSET-DOULOS | Doulos SIL 字体 | font | M13 | SIL International | review-required |
| PENDING-EGG | EGG 事件分析/简化逆滤波方法 | provenance-pending | M03 | 见上游/待补 | local-evidence-only |
| PENDING-DICTIONARY | 内置普通话发音词典 | provenance-pending | M12 | 见上游/待补 | local-evidence-only |
| PENDING-IPA | 11 套普通话转换规则及映射数据 | provenance-pending | M13 | 见上游/待补 | local-evidence-only |
| PENDING-PHONOLOGY | 音系归纳规则和帮助材料 | provenance-pending | M14 | 见上游/待补 | local-evidence-only |
| ORIGIN-WEBEDITOR | 本地 pitch_perception 的 web_praat_editor 来源链 | provenance-pending | M12 | 见上游/待补 | local-evidence-only |
| ASSET-MASCOT | K2 蓝色波形团子与 U2 设计参考 | provenance-pending | global | 见上游/待补 | local-evidence-only |
| SRC-HTML2CANVAS | html2canvas | legacy-web-dependency | M13 | 见上游/待补 | local-use-confirmed; upstream-artifact-review-pending |
| SRC-REACT | React | legacy-web-dependency | M15 | 见上游/待补 | local-use-confirmed; upstream-artifact-review-pending |
| SRC-BABEL | Babel standalone | legacy-web-dependency | M15 | 见上游/待补 | local-use-confirmed; upstream-artifact-review-pending |
| SRC-TAILWIND | Tailwind CSS | legacy-web-dependency | M15 | 见上游/待补 | local-use-confirmed; upstream-artifact-review-pending |
| SRC-SHEETJS | SheetJS | legacy-web-dependency | M15 | 见上游/待补 | local-use-confirmed; upstream-artifact-review-pending |
| SRC-LUCIDE | Lucide | legacy-web-dependency | M15 | 见上游/待补 | local-use-confirmed; upstream-artifact-review-pending |
| PKG-NUMPY | numpy | installed-runtime-package | shared-runtime | Travis E. Oliphant et al. | actual-wheel-and-native-audit-required |
| PKG-SCIPY | scipy | installed-runtime-package | shared-runtime | 见上游/待补 | actual-wheel-and-native-audit-required |
| PKG-PANDAS | pandas | installed-runtime-package | shared-runtime | 见上游/待补 | actual-wheel-and-native-audit-required |
| PKG-MATPLOTLIB | matplotlib | installed-runtime-package | shared-runtime | John D. Hunter, Michael Droettboom | actual-wheel-and-native-audit-required |
| PKG-PYQT6 | PyQt6 | installed-runtime-package | shared-runtime | Riverbank Computing Limited | actual-wheel-and-native-audit-required |
| PKG-PYQTGRAPH | pyqtgraph | installed-runtime-package | shared-runtime | Luke Campagnola | actual-wheel-and-native-audit-required |
| PKG-PRAAT-PARSELMOUTH | praat-parselmouth | installed-runtime-package | shared-runtime | Yannick Jadoul | actual-wheel-and-native-audit-required |
| PKG-PYWAVELETS | PyWavelets | installed-runtime-package | shared-runtime | 见上游/待补 | actual-wheel-and-native-audit-required |
| PKG-SOUNDDEVICE | sounddevice | installed-runtime-package | shared-runtime | Matthias Geier | actual-wheel-and-native-audit-required |
| PKG-SOUNDFILE | soundfile | installed-runtime-package | shared-runtime | Bastian Bechtold | actual-wheel-and-native-audit-required |
| PKG-OPENPYXL | openpyxl | installed-runtime-package | shared-runtime | See AUTHORS | actual-wheel-and-native-audit-required |
| PKG-PYTHON-DOCX | python-docx | installed-runtime-package | shared-runtime | 见上游/待补 | actual-wheel-and-native-audit-required |
| PKG-OPENCV-CONTRIB-PYTHON | opencv-contrib-python | installed-runtime-package | shared-runtime | 见上游/待补 | actual-wheel-and-native-audit-required |
| PKG-MEDIAPIPE | mediapipe | installed-runtime-package | shared-runtime | The MediaPipe Authors | actual-wheel-and-native-audit-required |
| PKG-PILLOW | Pillow | installed-runtime-package | shared-runtime | 见上游/待补 | actual-wheel-and-native-audit-required |
| PKG-PYINSTALLER | PyInstaller | installed-runtime-package | shared-runtime | Hartmut Goebel, Giovanni Bajo, David Vierra, David Cortesi, Martin Zibricky | actual-wheel-and-native-audit-required |

## 许可原文与证据
上游许可证文件按 source ID 保存于 licenses/；PyPI license 字段按准确包版本保留。后者可能带有多个 bundled libraries 的声明，也可能缺项。发布前应读取实际 wheel/原生包中的 LICENSE/NOTICE，不能只靠 PyPI 字段。
现有 VTL 与 Three.js 的对应原文在迁移源码中：[VTL](../phonetic_toolbox/resources/vocal_tract/VTL-LICENSE.txt)、[Three.js](../phonetic_toolbox/gui/resources/vocal_tract/vendor/THREE-LICENSE.txt)、[VTL 综合声明](../phonetic_toolbox/resources/vocal_tract/THIRD_PARTY_NOTICES.md)。P10 将其迁至最终资源位置并验证随包。
引用文字、说明书和 UI 从同一表派生；更新来源时记录日期、证据和与当前代码对应关系。不要把本轮上游 HEAD 当作原始复制 commit。
此目录保存来源材料而非可执行代码；README 中来自上游的操作说明是证据内容，不覆盖项目 AGENTS.md。
