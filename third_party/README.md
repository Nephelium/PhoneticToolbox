# 第三方组件、方法与素材

D0.3 · 58 条记录（42 条重点方法/代码/素材来源，16 条已装 Python 包发布元数据，部分相互关联）。尚未完成全传递依赖与实际发行包审查，不得将全部记录状态改为已通过。

P01 更新：注册表现有 150 条记录，新增探针 Python/前端锁定依赖与运行时、候选绑定记录。详细安装元数据与 npm integrity 见 [P01 依赖清单](p01-dependency-inventory.json)。70 条 npm 锁定项含可选平台包，并非全部安装或进入 EXE。Doulos SIL 已从本地字体确认版本 7.000，并提取内嵌版权与 OFL；这不代表 Chromium/Qt 原生传递许可或历史来源缺口已经闭合。

[查验报告](../docs/references/source-audit.md) · [机器可读注册表](source-registry.json) · [引用 BibTeX](references.bib) · [已装依赖元数据](package-source-audit.json) · [上游观测](upstream-observations.json)

登记原则：代码、论文、模型、实验数据、字体和图标分列。论文的引用/在线 PDF 链接不意味着把论文全文重新分发的许可。P04 已接入分组来源浏览；P10 继续完善方法说明、说明书与随包许可。

P02 更新：新增正式开发环境与前端锁定项，注册表现有 282 条记录。完整 31 个 Python 包和 100 条 npm lock 条目（含未安装的可选平台项）见 [P02 依赖清单](p02-dependency-inventory.json)。未把历史无明确许可的材料迁入新包。

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

P05 更新：新增 8 个隔离环境依赖、2 条 OWASP 方法参考和 PostgreSQL 测试运行时，共 294 条；见 [P05 依赖清单](p05-dependency-inventory.json)与 [PostgreSQL 运行时清单](p05-postgres-runtime.json)。论文/语言学学术组仍在前，工程安全参考与代码依赖归软件组。psycopg-binary/Argon2 原生传递许可证待发行物逐项审计，不据此宣称发行许可已通过。

P06 更新：现有 CPython SQLite 3.50.4 与 PostgreSQL 17 锁定文档单独登记，总数 296；新增项归软件组。原始流程探针仅计算确定性测试字节摘要，不是语音学算法或已迁移科研模块。

P07 迁移前更新：来源共 299 条；新增标准库文件持久化/锁、Starlette 响应接口参考及现有独立浏览器测试工具记录。均属软件组，未新增项目运行依赖；这是当时的迁移前记录；随后 003/004 已获具体授权并通过受控路径验收。

P07 联合更新：来源共 300 条，新增 P07-PYTHON-ZIP 标准库依赖/API 来源，复用既有 CPython 3.11.14；没有新安装或声学算法移植。受控 writer、ZIP 限制与原生工具未开放边界见 ../docs/testing/p07-job-files-report.md。

M01-C更新：323条来源登记；新增openpyxl3.1.5/et-xmlfile2.0.0及Win32/WAVE/pickle文档参考。40包完整C环境映射见 [C依赖清单](m01-io-inventory.json)，方法/代码署名不变，真实REAPER来源及发行缺口见 [原生资源](../resources/manifests/acoustic.json)。

## M10 Windows 本地录制资源

M02/M09 更新（2026-09-11）：来源条目仍为326条，既有 Griffin–Lim 方法条目、NumPy/SciPy、OpenCV、SoundFile、openpyxl、SQLite 补充实际使用位置与独立运行时锁。`requirements-m09-ui.lock` 固定 OpenCV 4.13.0.92、SoundFile 0.13.1；未修改旧环境。方法引用与代码迁入的关系见 `docs/modules/evidence/M09-source-map.md`，实际 wheel/native 的完整再分发审计仍保留未决，不宣称发行许可完成。

本轮迁入路径见统一登记的 `m10_migration_evidence` 与 [M10 报告](../docs/testing/m10-report.md)。VTL 2.4 API 保留，几何桥接 m10/2 和读数补丁分别提供源码。声音与几何适配继续遵循 VTL GPL 条款，Three.js/头壳/平均鼻腔分别为 MIT/CC0/CC BY 4.0。当前产物供 Windows 本机验证与录制，未公开发布。

2026-09-12：新增REF-PNG，W3C PNG第三版规范参考，用于M02整幅PNG的300dpi元数据；编码器沿用现有浏览器，不增加运行依赖。来源登记和软件致谢同步，当前共327条记录。
