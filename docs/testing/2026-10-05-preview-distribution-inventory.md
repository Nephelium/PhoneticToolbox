# Windows Preview 1 实际发行物盘点

任务：PREVIEW-DISTRIBUTION-INVENTORY。核查日期：2026-10-05。状态：`verified` 仅限本报告列出的只读文件盘点与摘要比对；完整公开发行许可验收仍为 `in_progress`。

初始盘点对象：`dist/PhoneticToolbox-Preview1-20261005-Final/PhoneticToolbox`。后续成品为 `dist/PhoneticToolbox-Preview1-20261005-Final2/PhoneticToolbox`，其宿主 DLL 以 Final2 的 `host-qt-identity` 逐文件验证为准。主线发现 Final 构建分析曾将 MFA 内 Qt6.6.1 的 WebChannel/Qml 混入宿主，Final2 改为构建结束后再复制独立 runtimes，已修正该问题。本报告的环境、模型与许可盘点先取样 Final，后续变更状态另行记录，不能把 Final 的混合宿主当作可交付成品。本报告只新增于 `docs/testing`，没有修改成品、来源登记、总许可证或环境，没有删除、下载源码、上传或对外联络。说明书资产由主线最后更新，文件总量和成品摘要应在最终压缩前重算。自然例音、截图作者与说明书章节内容不在本子任务范围。

## 结论

1. 当前包是真实随带 EGG/LPC、M05 和 MFA 独立运行环境的文件夹发行物。`desktop-bundle.json` 标记 `portable:true`，版本 `3.0.0-preview.1`。程序 EXE 必须与 `_internal` 一同交付。
2. VocalTractLab API 2.4 完整 API 源码 ZIP、桥接源码、补丁、构建说明及 GPL 原文已经随包。不能将此项与其他缺少原生源码的组件混为一谈。
3. 当前包有自有及适配 Python 源码快照，有许多 Python wheel 的许可证，也有方法/引用登记。它没有形成 PyQt6、Qt/Chromium、Parselmouth/Praat、FFmpeg 及其他实际 GPL/LGPL 原生组件的完整对应源码交付。通用上游链接、可执行文件、已安装 wheel 或 Conda 二进制包不等于对应源码。
4. MFA 的 193 个 Conda 包有精确本机缓存，184 个保存 `info/licenses`，合计 207 份文件。全部 193 份二进制 archive SHA-256 与实际发行元数据一致。主线随后已经将 207 份原文复制到准备阶段的 `mfa/notices-20261005`，并收集 56 条宿主环境候选元数据通知；这 56 条不能直接宣称全部已经打入宿主。最终压缩物仍要核对该新增通知树已被纳入。具体清单和摘要见报告末尾。
5. 项目总 LICENSE 尚未存在；VoiceSauce `func_getSoE.m` 的具体许可仍按作者确认状态保留。这两项不能擅自标为公开发行通过。私有服务器暂存与对公众提供下载分别记录，服务器来源不免除分发义务。

## 实际程序与环境

首次只读快照为 46,615 个文件、6,002,988,886 字节，约 5.590 GiB。该数值包含当时说明书资产，不作为随后更新后的最终包大小。

| 对象 | 随包路径或身份 | 核查结果 |
| --- | --- | --- |
| 主 EXE | `PhoneticToolbox.exe` | 22,765,853 字节；文件夹版入口 |
| 绑定清单 | `_internal/desktop-bundle.json` | `desktop-bundle/1`、win32、x86_64、`portable:true` |
| EGG/LPC | `_internal/runtimes/egg/python.exe` | 实际存在；整个环境 6,984 文件、767,700,220 字节 |
| M05 | `_internal/runtimes/m05/python.exe` | 实际存在；整个环境 8,132 文件、835,770,877 字节 |
| MFA | `_internal/runtimes/mfa/runtime/python.exe` | 实际存在；MFA 目录共 28,636 文件、2,820,257,660 字节，包含两组模型与词典 |
| MFA 登记与回执 | `runtimes/mfa/registry-bundled.json`、`receipt.json` | 相对路径已绑定；登记 `validated:true` 只是既有测试回执，本次未重跑科学任务 |
| MFA Conda 元数据 | `runtimes/mfa/runtime/conda-meta/*.json` | 193 条，包含实际包版本、build、文件列表、下载身份与摘要 |
| EGG Conda 元数据 | `runtimes/egg/conda-meta/*.json` | 保留科学构建凭据，例如 SciPy `1.16.3-py311hf127856_1` |
| Python 安装元数据 | 主宿主/EGG/M05/MFA | 主宿主 19 条、EGG 22 条、M05 33 条、MFA 93 条 `.dist-info/METADATA`；这些数字不等于全部原生依赖数 |

未发现名为 `runtime-registration.json` 的发行文件。本包的真实对应材料是 `desktop-bundle.json`、MFA `registry-bundled.json`、`receipt.json` 与各环境 `conda-meta`。MFA runtime 自有 `ptb-component.json` 也存在。运行时准备阶段另有工程内 `output/release-staging/runtimes-preview1/stage-report.json`，它没有作为顶层发行审计清单随包。

主要原生入口还包括 QtWebEngineProcess、REAPER、MFA、FFmpeg、ffprobe 和 SoX。MFA 带入完整环境，出现其中的 PyQt6、开发/构建工具和未使用字体，仍属于实际分发文件，不能仅因应用没有主动使用就忽略其通知。

## 精确版本与对应源码

| 实际组件 | 实际身份与本轮证据 | 已随包内容 | 具体待补内容 |
| --- | --- | --- | --- |
| 主宿主 PyQt6 / WebEngine | 各 6.11.0；构建 `COLLECT-00.toc` 绑定 `.venv/m09-ui`，QtCore/QtWebEngineCore `.pyd` 与原安装文件 SHA-256 一致 | 两份 GPLv3 通用全文在 `third_party/licenses/p01` | 精确 6.11.0 bindings 对应源码、构建说明、版权归属及 GPL 兼容的应用对应源码交付 |
| 主宿主 Qt / QtWebEngine | 各 6.11.2；Qt6Core/Qt6WebEngineCore DLL 与 `.venv/m09-ui` 原文件 SHA-256 一致 | Qt DLL、插件、翻译、WebEngine 资源 | 精确 Qt 6.11.2 / Chromium 组件通知和对应源码；主宿主未保留 Qt `.dist-info/LICENSE` |
| 主宿主 PyQt6_sip | 13.12.0，`.pyd` 摘要与原安装一致 | 二进制 | 精确 BSD-2-Clause 版权许可，可直接复制 `.venv/m09-ui/Lib/site-packages/pyqt6_sip-13.12.0.dist-info/licenses/LICENSE` |
| MFA 内 PyQt6 / Qt | PyQt6 6.6.1、PyQt6-Qt6 6.6.1；PyQt6_sip 13.10.3 | Qt `.dist-info/LICENSE`、sip LICENSE；PyQt6 本体元数据目录缺许可证，但全包另有 GPLv3 全文 | 与主宿主的 6.11 系列分开绑定 6.6.1 对应源码和原生通知；不能用另一版本代替 |
| Parselmouth / Praat | `praat-parselmouth 0.4.7`，实际 `.pyd` 与 wheel GPL 原文在主宿主、EGG、MFA | GPL 原文、Python API 文件 | Parselmouth 0.4.7 对应 C++/绑定源码、其内嵌 Praat 对应源码及构建材料；Python API 文件不替代编译扩展源码 |
| VocalTractLab | API 2.4、原 DLL 与独立状态副本、geometry_p2 桥接 | `resources/vocal_tract/sources/VTL2.4-API-source.zip`、`geometry_bridge.cpp`、补丁、构建说明、`VTL-LICENSE.txt`、第三方通知 | 本轮未发现源归档缺失；最后仍需把原版与修改、DLL 摘要及构建材料绑定到最终源码交付清单 |
| IRAPT / Troparion WM-PC | 源登记 GPL-3.0 / MPL-2.0，实际适配源码位于 `packages/phonetic_core/src/phonetic_core/acoustic` | Python 源码、acoustic NOTICE、GPL/MPL 原文 | 源码交付中保留版权、移植修改与原始 MATLAB/数据身份说明；MPL 按适用文件范围处理 |
| MFA FFmpeg | 8.0.0 `gpl_he3062b8_906`，Conda 元数据与回执一致 | FFmpeg/ffprobe 与 AV DLL，Conda 许可标识 | GPLv2/GPLv3 原文、精确 FFmpeg 源码、recipe/补丁/配置与其实际 codec 依赖源码和通知 |
| PyAV 视频库 | PyAV 16.1.0；另有带摘要后缀 AV 62/60 系列 DLL与 x264/x265；avcodec 内含 `LGPL version 3 or later` 声明 | PyAV 自己的 BSD 全文 | FFmpeg 库及 codec 的独立版权许可、精确源码与构建依据。此 avcodec 的配置有 `--enable-version3 --enable-libx264 --enable-libx265`；不能直接套 MFA 的另一个 build 或顶层 PyAV BSD |
| MFA 其他 GPL/LGPL | SoX 14.4.2、libmad 0.15.1b、x264 `1!164.3095`、x265 3.5；libiconv、libsndfile、GLib、Pango、Fribidi、Gdk-pixbuf、librsvg、soxr 等 | 二进制、Conda 元数据，部分包内许可证 | 按本报告实际 Conda 版本/build 提供适用源码、通知和必要重链接/替换说明；不能用顶层 NumPy/SciPy BSD 覆盖 |
| MKL / TBB | MKL 2025.3.0、TBB 2022.3.0，EGG 与 MFA 真实文件及凭据 | 原生 DLL | MKL `license.txt` 与 `tpp.txt`、TBB LICENSE 与第三方通知均可从精确缓存补；不擅自给 MKL 加 GPL 标签 |
| GCC 运行库 | libgcc/libgomp 15.2.0，元数据 GPL-3.0 WITH GCC-exception-3.1 | DLL、元数据及实际 `Library/share/licenses/gcc-libs/RUNTIME.LIBRARY.EXCEPTION` / `.gomp_copy` | 两份例外已随包；补 GPLv3 全文的显式归属引用，保留 GCC 运行库例外，不据此断言整个应用必须 GPL |

Qt 官方要求包括精确库源码或合法获取安排、用户替换/重新链接能力，以及 LGPL 原文与醒目通知。Qt 还包含单独许可的第三方组件。[Qt 官方义务说明](https://www.qt.io/development/open-source-lgpl-obligations)。FFmpeg 官方明确要求源码与实际二进制匹配并保留编译配置；使用 GPL 部分的构建与 LGPL 构建分开判断。[FFmpeg 官方法律与许可说明](https://ffmpeg.org/legal.html)。本报告据实际构建列出对象，没有把纯论文引用泛化为作者授权需求。

## 字体、模型和许可原文

| 对象 | 实际存在与已有通知 | 缺项或下一步 |
| --- | --- | --- |
| Doulos SIL 7.000 | 前端字体、后端字体及 `backend/src/ptb_worker/assets/Doulos-OFL.txt`，OFL 4,442 字节 | 保留该原文即可，不因位于后端 assets 而误判缺失 |
| JetBrains Mono 2.304 | 前端 WOFF2 与 `third_party/licenses/JetBrainsMono-OFL.txt` | 已有独立 OFL |
| PTB IPA Plus 1.000 / Noto | 前端 TTF、`OFL-PTBIPAPlus.txt`、`OFL-Noto.txt` | 已有独立 OFL 与派生说明 |
| KaTeX / Tiptap | 实际新增前端资源及 `third_party/licenses/manual/{katex-0.19.0,tiptap-3.31.4}/LICENSE` | 软件 MIT 原文存在；本轮不将作者编辑器加入发行范围 |
| MFA Conda 字体 | DejaVu 2.37、Inconsolata 3.000、Source Code Pro 2.038、Ubuntu 0.83，实际 31 个 TTF | 对应四套许可证未随环境拷入，可从下列缓存原样补；Ubuntu 独立字体许可不能用 OFL 概括 |
| Mandarin MFA 模型 | SHA-256 `bc16c3278295fad0d742399b849e5a8375a65bcbe5b7c7612ff1e3ffb81dcef7`、92,275,957 字节；ZIP 含 `mandarin_mfa/meta.json` | ZIP 无 LICENSE/NOTICE；metadata 有科学训练身份，未提供该具体模型的版权许可。绑定官方具体模型记录与许可 |
| Mandarin 拼音模型 | SHA-256 `beca5c15f6c15bee4435a37bd2904e15b12fcc56342013d253260f31fd2c3f33`、14,709,255 字节；ZIP 含 `mandarin/meta.yaml`，version 1.0.0 | ZIP 无 LICENSE/NOTICE；不能由 MFA 程序 Apache-2.0 直接推导模型许可。应补精确原始下载记录与该模型许可 |
| 两份词典 | SHA-256 分别为 `6b71538ba4cd48d40f92561a129e4531ee256951fcf218ac1198da72ca13076e`、`c6a9b62905917be3f21a9138503ccf9a5fb1ca080d5a7eed75c8a662c80770cb` | 本包登记只写 see component notices，MFA 目录实际没有对应独立组件通知。拼音自有映射与外部模型/词典分别绑定，不扩大既有自有词典作者确认 |
| MediaPipe | M05 wheel 0.10.14 实际带模型与 Apache 原文；前端 Web 模型依 `resources/m05/resources.json` 另有身份 | 延续已查模型卡和实际 hash，不把网页 footer 的内容许可当模型许可；本轮未重新判定旧模型科学效果 |

MFA 官方也分别管理和下载模型与词典，软件程序版本不自动构成每一模型的身份或许可依据。[MFA 3.3.8 模型文档](https://montreal-forced-aligner.readthedocs.io/en/v3.3.8/user_guide/models/index.html)。

### 初始 Final 盘点发现的简短许可缺项

- `third_party/licenses/pypi/openpyxl-3.1.5-metadata.txt` 内容只有 `MIT\n`，主宿主与 MFA 的 openpyxl dist-info 均缺完整版权与许可原文。不能将这份标签文件标为完整 MIT 通知。
- M05 `flatbuffers 25.12.19` dist-info 没有许可原文，需要补该版本的 Apache-2.0 与版权归属。
- MFA `et_xmlfile 2.0.0`、`ordered-set 4.1.0`、`tqdm 4.67.1` 等 dist-info 缺独立许可，优先使用精确 Conda 缓存中现有原文。tqdm 允许的 MPL/MIT 范围以原 `LICENCE` 为准。
- 主宿主缺 sip13.12.0 BSD 原文；Qt6 6.11.2 的完整包级 LICENSE 可以从已验证 `.venv/m09-ui` 补，但 Chromium/Qt 第三方通知还需要精确版本材料。
- 初始 Final 的 `third_party/source-registry.json` 为 395 条来源登记，`SRC-MFA.actual_included_version` 仍有 not bundled in main EXE 的历史描述。应在发行层清单明确实际随包，不覆盖历史来源版本。来源登记条数不等于实际独立组件数量。

### 九项没有 info/licenses 的精确处理

| 缓存包 | 实际文件与现有证据 | 最小处理 |
| --- | --- | --- |
| libfreetype 2.14.1 `h57928b3_0` | 实际 0 文件，仅依赖 libfreetype6 的 metapackage | 记录空包即可，不虚构新增代码通知 |
| libfreetype6 2.14.1 `hdbac1cb_0` | 仅 `Library/bin/freetype.dll` | 绑定同源同版本 `freetype-2.14.1-h57928b3_0/info/licenses/docs/FTL.TXT` 与 `GPLv2.TXT`，注明拆包归属和实际选用条款 |
| libgcc 15.2.0 `h8ee18e1_16` | 8 文件；实际 payload 有 GCC runtime exception | 保留包内 `Library/share/licenses/gcc-libs/RUNTIME.LIBRARY.EXCEPTION`，并明确绑定 GPLv3 全文与例外 |
| libgomp 15.2.0 `h8ee18e1_16` | 3 文件；实际 payload 有 `.gomp_copy` runtime exception | 同上，使用实际 `.gomp_copy` 并保留单独组件归属 |
| librosa 0.11.0 `pyhd8ed1ab_0` | 实际已有 `Lib/site-packages/librosa-0.11.0.dist-info/LICENSE.md`，SHA-256 `1746ca32b53fa31941c38b63a986e4126ab998e81a2f73a566000847a588e2a3` | 无许可原文缺失，登记现有 payload 路径即可 |
| libsqlite 3.51.1 `hf5d6505_0` | 4 文件；recipe 明示 public-domain/blessing，包含 sqlite3.h | 记录 SQLite public-domain/blessing 和精确源码/recipe 身份，不强加 MIT 或额外邮件授权 |
| sqlite 3.51.1 `hdb435a2_0` | 仅 sqlite3.exe；与上述精确 libsqlite build 绑定 | 同上，保留公有领域声明 |
| vc 14.3 `h2b53caa_33` | 实际 0 文件，仅依赖 vc14_runtime | 记录空 metapackage，其实际运行库许可由已收集 vc14_runtime 的 LICENSE.RTF/TXT 覆盖 |
| vs2015_runtime 14.44.35208 `h38c0c73_33` | 实际 0 文件，仅依赖 vc14_runtime | 同上，不将空包当成缺失代码许可 |

GCC 两份 runtime exception 原文均为 3,324 字节，SHA-256 `9d6b43ce4d8de0c878bf16b54d8e7a10d9bd42b75178153e3af6a815bdc90f74`，与精确缓存同路径一致。libgcc/libgomp 的本机 `info/recipe/meta.yaml` 还保留 GCC 15.2.0 源压缩包 SHA-256 `7294d65cc1a0558cb815af0ca8c7763d86f7a31199794ede3f630c0d1b0a5723`、conda-forge feedstock commit `6a6995549693ce0e316c3e71d2041e0f17e68a07` 与补丁列表。这是构建依据，不能将其误称为已下载完整 GCC 源码。

## 可直接补入的本机路径与操作边界

精确 Conda 缓存根目录：`C:\Users\13680\Miniconda3_broken_backup_20260317_200156\pkgs`。按照本包各 `conda-meta/*.json` 的 `fn` 去掉 `.conda` 或 `.tar.bz2` 后，得到缓存目录名 `<stem>`，对应完整许可位于 `<stem>/info/licenses`。应复制整个树，保留嵌套归属，例如 MKL 的多层 `mkl/info/licenses`，不能只挑文件名为 LICENSE 的一份。

建议新建随包 `third_party/licenses/distribution/conda/<stem>/`，放入各组件全部 `info/licenses`，附 `component-inventory.json`，记录 name/version/build、原 archive 摘要、原文相对路径与 SHA-256。此处只是主代理的可补清单，本子任务没有复制或改动包。

主宿主现有精确材料可直接取自 `.venv/m09-ui/Lib/site-packages/`：

- `pyqt6-6.11.0.dist-info/licenses/LICENSE`，SHA-256 `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903`。
- `pyqt6_webengine-6.11.0.dist-info/licenses/LICENSE`，相同 GPLv3 原文摘要。
- `pyqt6_qt6-6.11.2.dist-info/LICENSE` 及 `pyqt6_webengine_qt6-6.11.2.dist-info/LICENSE`，SHA-256 `6c671e2912ec69c0832f32066c0313cee5fdc3bbcbce237822e89f3da58edd4e`。
- `pyqt6_sip-13.12.0.dist-info/licenses/LICENSE`，SHA-256 `3e6f5b427c36f94ecf86bc01698af7030a1ed6eb3748110d5dbb8d142d804611`。

已有 `output/validation/m11/component-a4159343dbc44af180e2f2e361666ef9/mfa-3.3.8-windows-x86_64-candidate.zip` 含环境文件及部分包内许可证，缺完整 Conda `info/licenses` 分发树。它是二进制环境归档，不能用作上述 GPL/LGPL 对应源码归档。

本地源码准备应同时保留构建脚本、版本锁与前端可编辑源码，并过滤只获随软件分发的自然材料及作者工具。GitHub 暂停不影响在同一服务器另提供依法应交付的源码下载。对应源码地址、范围、摘要及获取方式应成为发行清单的一部分；公开下载开放前再次核查，不能将本报告的只读盘点直接写为公开发行许可全部通过。

## 本轮验证边界

实际执行了文件存在/大小盘点、JSON 元数据解析、模型 ZIP 内部列表与 meta 读取、主宿主 5 个 Qt/SIP 文件与构建环境 SHA-256 比较，以及缓存/许可摘要核查。没有执行 UI、科学任务、安装、更新、实体音频、用户素材作者核查、全平台验证或服务器上传。本报告只新增一份文档。

下面的自动附录按实际 Conda 记录生成，列出缓存 archive 与 metadata 的摘要是否相符，以及每份 `info/licenses` 原文的精确路径和 SHA-256。

## A. MFA Conda cache verification

Generated 2026-10-05T21:16:02+08:00. Read-only cache archive hashing plus streamed archive info/licenses comparison. No extraction or package mutation. Both info- and pkg- tar streams were checked: many Windows Conda archives store info/licenses in the pkg stream.

Packages: 193. With info/licenses: 184. License files: 207. SHA-256 archive matches: 193. License-byte matches to cached archives: 207. Mismatches: 0.

Cache root: `C:\Users\13680\Miniconda3_broken_backup_20260317_200156\pkgs`. Every stem below is relative to this cache root. Exact binary archive URL and expected hash remain in the bundled conda-meta JSON.

| Package / version / build | License label in actual metadata | Cached archive SHA-256 | Matches bundled identity | Notice files |
| --- | --- | --- | --- | ---: |
| `_openmp_mutex-4.5-2_gnu` | BSD-3-Clause | `1a62cd1f215fe0902e7004089693a78347a30ad687781dfda2289cab000e652d` | True | 1 |
| `aom-3.9.1-he0c23c2_0` | BSD-2-Clause | `0524d0c0b61dacd0c22ac7a8067f977b1d52380210933b04141f5099c5b6fec7` | True | 1 |
| `audioread-3.0.1-py311h1ea47a8_3` | MIT | `bce73827b53d285934c01969bf837532e2d76f0378c57f71c546be1bd357da8e` | True | 1 |
| `backports.zstd-1.2.0-py311h71c1bcc_0` | BSD-3-Clause AND MIT AND EPL-2.0 | `28984981f212813c0bfec0688d3c34937488ab060f9b16602ef4e7b6a0c3bfe1` | True | 1 |
| `baumwelch-0.3.11-hd620369_0` | Apache-2.0 | `449d7444b1b53eaf6ed4f3b660479434ba10d485ff1011a51fea35c0d8437e21` | True | 1 |
| `biopython-1.86-py311h3485c13_0` | LicenseRef-Biopython | `f1e4767f7cb3f6995e00a232387ce69073a42d7fe72bce8ca8cdb9dbed63dc3a` | True | 1 |
| `brotli-1.2.0-h2d644bc_1` | MIT | `a4fffdf1c9b9d3d0d787e20c724cff3a284dfa3773f9ce609c93b1cfd0ce8933` | True | 1 |
| `brotli-bin-1.2.0-hfd05255_1` | MIT | `e76966232ef9612de33c2087e3c92c2dc42ea5f300050735a3c646f33bce0429` | True | 1 |
| `brotli-python-1.2.0-py311hc5da9e4_1` | MIT | `1803c838946d79ef6485ae8c7dafc93e28722c5999b059a34118ef758387a4c9` | True | 1 |
| `bzip2-1.0.8-h0ad9c76_8` | bzip2-1.0.6 | `d882712855624641f48aa9dc3f5feea2ed6b4e6004585d3616386a18186fe692` | True | 1 |
| `ca-certificates-2025.11.12-h4c7d964_0` | ISC | `686a13bd2d4024fc99a22c1e0e68a7356af3ed3304a8d3ff6bb56249ad4e82f0` | True | 1 |
| `cairo-1.18.4-h5782bbf_0` | LGPL-2.1-only or MPL-1.1 | `b9f577bddb033dba4533e851853924bfe7b7c1623d0697df382eef177308a917` | True | 3 |
| `certifi-2025.11.12-pyhd8ed1ab_0` | ISC | `083a2bdad892ccf02b352ecab38ee86c3e610ba9a4b11b073ea769d55a115d32` | True | 1 |
| `cffi-2.0.0-py311h3485c13_1` | MIT | `c9caca6098e3d92b1a269159b759d757518f2c477fbbb5949cb9fee28807c1f1` | True | 1 |
| `charset-normalizer-3.4.4-pyhd8ed1ab_0` | MIT | `b32f8362e885f1b8417bac2b3da4db7323faa12d5db62b7fd6691c02d60d6f59` | True | 2 |
| `click-8.3.1-pyha7b4d00_1` | BSD-3-Clause | `c3bc9a49930fa1c3383a1485948b914823290efac859a2587ca57a270a652e08` | True | 1 |
| `colorama-0.4.6-pyhd8ed1ab_1` | BSD-3-Clause | `ab29d57dc70786c1269633ba3dff20288b81664d3ff8d21af995742e2bb03287` | True | 1 |
| `contourpy-1.3.3-py311h3fd045d_3` | BSD-3-Clause | `ca1bde6f4afec87945c1186a307727ba7e151aabb46fc67683562319987b1088` | True | 1 |
| `cycler-0.12.1-pyhcf101f3_2` | BSD-3-Clause | `bb47aec5338695ff8efbddbc669064a3b10fe34ad881fb8ad5d64fbfa6910ed1` | True | 1 |
| `dataclassy-1.0.1-pyhd8ed1ab_0` | MPL-2.0 | `c536392f4e464dc3d845d69293b58c3e399785c8e7b6041dc104e16be33e394a` | True | 1 |
| `dav1d-1.2.1-hcfcfb64_0` | BSD-2-Clause | `2aa2083c9c186da7d6f975ccfbef654ed54fff27f4bc321dbcd12cee932ec2c4` | True | 1 |
| `decorator-5.2.1-pyhd8ed1ab_0` | BSD-2-Clause | `c17c6b9937c08ad63cb20a26f403a3234088e57d4455600974a0ce865cb14017` | True | 1 |
| `dlfcn-win32-1.4.2-hac47afa_0` | MIT | `2a9baf44fbbdc1fcd45f24dd81c79240c040925687772355a5e240d5bdd14dbf` | True | 1 |
| `ffmpeg-8.0.0-gpl_he3062b8_906` | GPL-2.0-or-later | `c35f51336ae9ccc4f30b85559537d3dea3208b8baf602a051ad8679872783a76` | True | 2 |
| `font-ttf-dejavu-sans-mono-2.37-hab24e00_0` | BSD-3-Clause | `58d7f40d2940dd0a8aa28651239adbf5613254df0f75789919c4e6762054403b` | True | 1 |
| `font-ttf-inconsolata-3.000-h77eed37_0` | OFL-1.1 | `c52a29fdac682c20d252facc50f01e7c2e7ceac52aa9817aaf0bb83f7559ec5c` | True | 1 |
| `font-ttf-source-code-pro-2.038-h77eed37_0` | OFL-1.1 | `00925c8c055a2275614b4d983e1df637245e19058d79fc7dd1a93b8d9fb4b139` | True | 1 |
| `font-ttf-ubuntu-0.83-h77eed37_3` | LicenseRef-Ubuntu-Font-Licence-Version-1.0 | `2821ec1dc454bd8b9a31d0ed22a7ce22422c0aef163c59f49dfdf915d0f0ca14` | True | 1 |
| `fontconfig-2.15.0-h765892d_1` | MIT | `ed122fc858fb95768ca9ca77e73c8d9ddc21d4b2e13aaab5281e27593e840691` | True | 1 |
| `fonts-conda-ecosystem-1-0` | BSD-3-Clause | `a997f2f1921bb9c9d76e6fa2f6b408b7fa549edd349a77639c9fe7a23ea93e61` | True | 1 |
| `fonts-conda-forge-1-hc364b38_1` | BSD-3-Clause | `54eea8469786bc2291cc40bca5f46438d3e062a399e8f53f013b6a9f50e98333` | True | 1 |
| `fonttools-4.61.0-py311h3f79411_0` | MIT | `b58748a3fb357b7baedd10a1e577e9235b367d5b84a1763cc9767e3ade1737a2` | True | 1 |
| `freetype-2.14.1-h57928b3_0` | GPL-2.0-only OR FTL | `a9b3313edea0bf14ea6147ea43a1059d0bf78771a1336d2c8282891efc57709a` | True | 2 |
| `fribidi-1.0.16-hfd05255_0` | LGPL-2.1-or-later | `15011071ee56c216ffe276c8d734427f1f893f275ef733f728d13f610ed89e6e` | True | 1 |
| `gdk-pixbuf-2.44.4-h1f5b9c4_0` | LGPL-2.1-or-later | `24189e4615a0aa574ab2bd5c270fff999da6951e3cd391f1e807c7e4fafd5cdc` | True | 1 |
| `getopt-win32-0.1-h6a83c73_3` | LGPL-3.0-only | `d04c4a6c11daa72c4a0242602e1d00c03291ef66ca2d7cd0e171088411d57710` | True | 1 |
| `glslang-16.1.0-h5b34520_0` | BSD-3-Clause | `ab8cca5c5b8aba98f83d8732a3fca71a246a05525d00dce7e528089348cd64ec` | True | 1 |
| `graphite2-1.3.14-hac47afa_2` | LGPL-2.0-or-later | `5f1714b07252f885a62521b625898326ade6ca25fbc20727cfe9a88f68a54bfd` | True | 1 |
| `graphviz-14.1.0-h4c50273_0` | EPL-1.0 | `c14e28d2dc405c58dc2094d98961bc6c0aab591fb074d36f7c9fefbd420ebfd6` | True | 1 |
| `greenlet-3.3.0-py311h3e6a449_0` | MIT | `3b6323752e9bdd652e4a4fd1e0abf7d891462687d2765f6a8e3aa35a1ef571c4` | True | 1 |
| `gts-0.7.6-h6b5321d_4` | LGPL-2.0-or-later | `b79755d2f9fc2113b6949bfc170c067902bc776e2c20da26e746e780f4f5a2d4` | True | 1 |
| `h2-4.3.0-pyhcf101f3_0` | MIT | `84c64443368f84b600bfecc529a1194a3b14c3656ee2e832d15a20e0329b6da3` | True | 1 |
| `harfbuzz-12.2.0-h5f2951f_0` | MIT | `db73714c7f7e0c47b3b9db9302a83f2deb6f8d6081716d35710ef3c6756af6c3` | True | 1 |
| `hdbscan-0.8.39-py311h17033d2_1` | BSD-3-Clause | `ca5fc4e36e337efd1871c1fbd292e3f3ca7f3c20868d8d3fb7f407fb8f731f6b` | True | 1 |
| `hpack-4.1.0-pyhd8ed1ab_0` | MIT | `6ad78a180576c706aabeb5b4c8ceb97c0cb25f1e112d76495bff23e3779948ba` | True | 1 |
| `hyperframe-6.1.0-pyhd8ed1ab_0` | MIT | `77af6f5fe8b62ca07d09ac60127a30d9069fdc3c68d6b256754d0ffb1f7779f8` | True | 1 |
| `icu-75.1-he0c23c2_0` | MIT | `1d04369a1860a1e9e371b9fc82dd0092b616adcf057d6c88371856669280e920` | True | 1 |
| `idna-3.11-pyhd8ed1ab_0` | BSD-3-Clause | `ae89d0299ada2a3162c2614a9d26557a92aa6a77120ce142f8e0109bbf0342b0` | True | 1 |
| `importlib-metadata-8.7.0-pyhe01879c_1` | Apache-2.0 | `c18ab120a0613ada4391b15981d86ff777b5690ca461ea7e9e49531e8f374745` | True | 1 |
| `joblib-1.5.2-pyhd8ed1ab_0` | BSD-3-Clause | `6fc414c5ae7289739c2ba75ff569b79f72e38991d61eb67426a8a4b92f90462c` | True | 1 |
| `kaldi-5.5.1172-cpu_hb4f072a_2` | Apache-2.0 | `379d33e7d0a1b17a7fc98e6e127825bafdea5deec562a605d992834e62362239` | True | 1 |
| `kalpy-0.8.2-py311h3fd045d_0` | MIT | `6c4f5aff7b9ab214d9e2388fb16d6de96ed4c1194eb22fad25960f23b20dce05` | True | 1 |
| `kiwisolver-1.4.9-py311h275cad7_2` | BSD-3-Clause | `29a932673249b8c821c3074223296aa1fd3934474fadad2b2daa5ebb4830f420` | True | 1 |
| `kneed-0.8.5-pyhd8ed1ab_1` | MIT | `dea0274cf6408001a9f9a7fe2e952254d7bab553d5a5dee125ed7aa01e9b172c` | True | 1 |
| `krb5-1.21.3-hdf4eb48_0` | MIT | `18e8b3430d7d232dad132f574268f56b3eb1a19431d6d5de8c53c29e6c18fa81` | True | 1 |
| `lame-3.100-hcfcfb64_1003` | LGPL-2.0-only | `824988a396b97bb9138823a1b3aabd8326e06da5834b3011253d72bb45fd3a88` | True | 1 |
| `lazy-loader-0.4-pyhd8ed1ab_2` | BSD-3-Clause | `d7ea986507090fff801604867ef8e79c8fda8ec21314ba27c032ab18df9c3411` | True | 1 |
| `lazy_loader-0.4-pyhd8ed1ab_2` | BSD-3-Clause | `e26803188a54cd90df9ce1983af70b287c4918c0fd178a9aabd9f1580f657a2b` | True | 1 |
| `lcms2-2.17-hbcf6048_0` | MIT | `7712eab5f1a35ca3ea6db48ead49e0d6ac7f96f8560da8023e61b3dbe4f3b25d` | True | 1 |
| `lerc-4.0.0-h6470a55_1` | Apache-2.0 | `868a3dff758cc676fa1286d3f36c3e0101cca56730f7be531ab84dc91ec58e9d` | True | 1 |
| `libblas-3.11.0-4_hf2e6a31_mkl` | BSD-3-Clause | `0c6ecdabcd3c5b92c7be68a65c30c29983040dd81f502d2e9ad3763fdbbabdef` | True | 1 |
| `libbrotlicommon-1.2.0-hfd05255_1` | MIT | `5097303c2fc8ebf9f9ea9731520aa5ce4847d0be41764edd7f6dee2100b82986` | True | 1 |
| `libbrotlidec-1.2.0-hfd05255_1` | MIT | `3239ce545cf1c32af6fffb7fc7c75cb1ef5b6ea8221c66c85416bb2d46f5cccb` | True | 1 |
| `libbrotlienc-1.2.0-hfd05255_1` | MIT | `3226df6b7df98734440739f75527d585d42ca2bfe912fbe8d1954c512f75341a` | True | 1 |
| `libcblas-3.11.0-4_h2a3cdd5_mkl` | BSD-3-Clause | `4cd0f2ec9823995a74b73c0119201dcf9a28444bdc2f0a824dfa938b5bdd5601` | True | 1 |
| `libdeflate-1.25-h51727cc_0` | MIT | `834e4881a18b690d5ec36f44852facd38e13afe599e369be62d29bd675f107ee` | True | 1 |
| `libexpat-2.7.3-hac47afa_0` | MIT | `844ab708594bdfbd7b35e1a67c379861bcd180d6efe57b654f482ae2f7f5c21e` | True | 1 |
| `libffi-3.5.2-h52bdfb6_0` | MIT | `ddff25aaa4f0aa535413f5d831b04073789522890a4d8626366e43ecde1534a3` | True | 1 |
| `libflac-1.4.3-h63175ca_0` | BSD-3-Clause | `965d1b9c957956a50797db24c031bdb3a604ef0e9a03713965513419aa1f99df` | True | 1 |
| `libfreetype-2.14.1-h57928b3_0` | GPL-2.0-only OR FTL | `2029702ec55e968ce18ec38cc8cf29f4c8c4989a0d51797164dab4f794349a64` | True | 0 |
| `libfreetype6-2.14.1-hdbac1cb_0` | GPL-2.0-only OR FTL | `223710600b1a5567163f7d66545817f2f144e4ef8f84e99e90f6b8a4e19cb7ad` | True | 0 |
| `libgcc-15.2.0-h8ee18e1_16` | GPL-3.0-only WITH GCC-exception-3.1 | `24984e1e768440ba73021f08a1da0c1ec957b30d7071b9a89b877a273d17cae8` | True | 0 |
| `libgd-2.3.3-h7208af6_11` | GD | `485a30af9e710feeda8d5b537b2db1e32e41f29ef24683bbe7deb6f7fd915825` | True | 1 |
| `libglib-2.86.3-h0c9aed9_0` | LGPL-2.1-or-later | `84b74fc81fff745f3d21a26c317ace44269a563a42ead3500034c27e407e1021` | True | 1 |
| `libgomp-15.2.0-h8ee18e1_16` | GPL-3.0-only WITH GCC-exception-3.1 | `9c86aadc1bd9740f2aca291da8052152c32dd1c617d5d4fd0f334214960649bb` | True | 0 |
| `libhwloc-2.12.1-default_h4379cf1_1003` | BSD-3-Clause | `2d534c09f92966b885acb3f4a838f7055cea043165a03079a539b06c54e20a49` | True | 1 |
| `libiconv-1.18-hc1393d2_2` | LGPL-2.1-only | `0dcdb1a5f01863ac4e8ba006a8b0dc1a02d2221ec3319b5915a1863254d7efa7` | True | 1 |
| `libintl-0.22.5-h5728263_3` | LGPL-2.1-or-later | `c7e4600f28bcada8ea81456a6530c2329312519efcf0c886030ada38976b0511` | True | 1 |
| `libjpeg-turbo-3.1.2-hfd05255_0` | IJG AND BSD-3-Clause AND Zlib | `795e2d4feb2f7fc4a2c6e921871575feb32b8082b5760726791f080d1e2c2597` | True | 1 |
| `liblapack-3.11.0-4_hf9ab0e9_mkl` | BSD-3-Clause | `d820333e9bac8381fb69e857d673c12d034bb45d0fe4818a1d12e1ec7a39e7df` | True | 1 |
| `liblapacke-3.11.0-4_h3ae206f_mkl` | BSD-3-Clause | `036953687e8fd9a7d39a3b20b59f0772d5c3b3c861f3404af8c91bd73512895e` | True | 1 |
| `liblzma-5.8.1-h2466b09_2` | 0BSD | `55764956eb9179b98de7cc0e55696f2eff8f7b83fc3ebff5e696ca358bca28cc` | True | 2 |
| `liblzma-devel-5.8.1-h2466b09_2` | 0BSD | `1ccff927a2d768403bad85e36ca3e931d96890adb4f503e1780c3412dd1e1298` | True | 2 |
| `libmad-0.15.1b-hcfcfb64_1001` | GPL-2.0-only | `466af4bb68679fd2c37b5811601cd80dabd7ba53b27e18a5a9ccab5138eb9aed` | True | 1 |
| `libogg-1.3.5-h2466b09_1` | BSD-3-Clause | `c63e5fb169dbd192aacdcee6e37235407f106b8ca9c9036942a25e0366cbc73c` | True | 1 |
| `libopus-1.5.2-h2466b09_0` | BSD-3-Clause | `4c5e04de758450f9427a75095a54957de521b57234711374fac1cdc89fc7a9ca` | True | 1 |
| `libpng-1.6.53-h7351971_0` | zlib-acknowledgement | `e5d061e7bdb2b97227b6955d1aa700a58a5703b5150ab0467cc37de609f277b6` | True | 1 |
| `libpq-16.10-h18d9880_2` | PostgreSQL | `f92be880643d5607eeee164f9eacfd06cb2eb16812389a6e2590e3545b3366ad` | True | 1 |
| `librosa-0.11.0-pyhd8ed1ab_0` | ISC | `e791136e2254d73fb0c9624823eaf78778415e15c26e3950916c502141871c38` | True | 0 |
| `librsvg-2.60.0-hd5e4115_0` | LGPL-2.1-or-later | `a0e8d89c36e555149f3ba2d58bb96f1b77e8ed7924db8a242ee0b0fb613c588d` | True | 1 |
| `libsndfile-1.2.2-h81429f1_1` | LGPL-2.1-or-later | `e73454d76600243cd8dc6a64a3121a393c03871ee2d044859db027998920a94e` | True | 1 |
| `libsqlite-3.51.1-hf5d6505_0` | blessing | `a976c8b455d9023b83878609bd68c3b035b9839d592bd6c7be7552c523773b62` | True | 0 |
| `libtiff-4.7.1-h8f73337_1` | HPND | `f1b8cccaaeea38a28b9cd496694b2e3d372bb5be0e9377c9e3d14b330d1cba8a` | True | 1 |
| `libusb-1.0.29-h1839187_0` | LGPL-2.1-or-later | `9837f8e8de20b6c9c033561cd33b4554cd551b217e3b8d2862b353ed2c23d8b8` | True | 1 |
| `libvorbis-1.3.7-h5112557_2` | BSD-3-Clause | `429124709c73b2e8fae5570bdc6b42f5418a7551ba72e591bb960b752e87b365` | True | 1 |
| `libvulkan-loader-1.4.328.1-h477610d_0` | Apache-2.0 | `934d676c445c1ea010753dfa98680b36a72f28bec87d15652f013c91a1d8d171` | True | 1 |
| `libwebp-base-1.6.0-h4d5522a_0` | BSD-3-Clause | `7b6316abfea1007e100922760e9b8c820d6fc19df3f42fb5aca684cfacb31843` | True | 1 |
| `libwinpthread-12.0.0.r4.gg4f2fc60ca-h57928b3_10` | MIT AND BSD-3-Clause-Clear | `0fccf2d17026255b6e10ace1f191d0a2a18f2d65088fd02430be17c701f8ffe0` | True | 2 |
| `libxcb-1.17.0-h0e4246c_0` | MIT | `08dec73df0e161c96765468847298a420933a36bc4f09b50e062df8793290737` | True | 1 |
| `libxml2-16-2.15.1-h06f855e_0` | MIT | `3f65ea0f04c7738116e74ca87d6e40f8ba55b3df31ef42b8cb4d78dd96645e90` | True | 1 |
| `libxml2-2.15.1-ha29bfb0_0` | MIT | `fb51b91a01eac9ee5e26c67f4e081f09f970c18a3da5231b8172919a1e1b3b6b` | True | 1 |
| `libzlib-1.3.1-h2466b09_2` | Zlib | `ba945c6493449bed0e6e29883c4943817f7c79cbff52b83360f7b341277c6402` | True | 1 |
| `llvm-openmp-21.1.7-h4fa8253_0` | Apache-2.0 WITH LLVM-exception | `79121242419bf8b485c313fa28697c5c61ec207afa674eac997b3cb2fd1ff892` | True | 1 |
| `llvmlite-0.45.1-py311h4f568be_0` | BSD-2-Clause | `39656b282a8cd5ad9c435931eae6e3d217935b6992cdca0c9d9fae6f50740ae7` | True | 1 |
| `markdown-it-py-4.0.0-pyhd8ed1ab_0` | MIT | `7b1da4b5c40385791dbc3cc85ceea9fad5da680a27d5d3cb8bfaa185e304a89e` | True | 1 |
| `matplotlib-base-3.10.8-py311h1675fdf_0` | PSF-2.0 | `afc65698c2d3e884890c708c68b45cdd0f46c0373fa73561f4ea0bd9606bb90e` | True | 11 |
| `mdurl-0.1.2-pyhd8ed1ab_1` | MIT | `78c1bbe1723449c52b7a9df1af2ee5f005209f67e40b6e1d3c7619127c43b1c7` | True | 1 |
| `mkl-2025.3.0-hac47afa_454` | LicenseRef-IntelSimplifiedSoftwareOct2022 | `3c432e77720726c6bd83e9ee37ac8d0e3dd7c4cf9b4c5805e1d384025f9e9ab6` | True | 2 |
| `montreal-forced-aligner-3.3.8-pyhd8ed1ab_0` | MIT | `68372ccfdc382ef4e7e9716ee160a62d67cca86dbd243e9e9bca3285e8825bf8` | True | 1 |
| `mpg123-1.32.9-h01009b0_0` | LGPL-2.1-only | `a1d7d25f2c448f5c47d1678cca1f6ae5deadb38e176ea0c76ea5c688589dfd7a` | True | 1 |
| `msgpack-python-1.1.2-py311h3fd045d_1` | Apache-2.0 | `9883b64dea87c50e98fabc05719ff0fdc347f57d7bacda19bcd69b80d8c436d4` | True | 1 |
| `munkres-1.1.4-pyhd8ed1ab_1` | Apache-2.0 | `d09c47c2cf456de5c09fa66d2c3c5035aa1fa228a1983a433c47b876aa16ce90` | True | 1 |
| `ngram-1.3.17-hc790b64_0` | Apache-2.0 | `4156cabdb7885581a4eac0fc0ac4b39da6dee0f42f8b1ff353c71afb6eef7878` | True | 1 |
| `numba-0.62.1-py311h5e69a0e_1` | BSD-2-Clause | `06d33f749b45024ea9e3a79fb7015554cc9f33c527068c87742976922e051dc1` | True | 1 |
| `openfst-1.8.4-hc790b64_1` | Apache-2.0 | `09a1e1fc8d96b146b653782955175815b19091105fd5f653cd4114ebf0ca12c2` | True | 1 |
| `openh264-2.6.0-hb17fa0b_0` | BSD-2-Clause | `914702d9a64325ff3afb072c8bc0f8cbea3f19955a8395a8c190e45604f83c76` | True | 1 |
| `openjpeg-2.5.4-h24db6dd_0` | BSD-2-Clause | `226c270a7e3644448954c47959c00a9bf7845f6d600c2a643db187118d028eee` | True | 1 |
| `openssl-3.6.0-h725018a_0` | Apache-2.0 | `6d72d6f766293d4f2aa60c28c244c8efed6946c430814175f959ffe8cab899b3` | True | 1 |
| `packaging-25.0-pyh29332c3_1` | Apache-2.0 | `289861ed0c13a15d7bbb408796af4de72c2fe67e2bcb0de98f4c3fce259d7991` | True | 1 |
| `pango-1.56.4-h03d888a_0` | LGPL-2.1-or-later | `dcda7e9bedc1c87f51ceef7632a5901e26081a1f74a89799a3e50dbdc801c0bd` | True | 1 |
| `pcre2-10.47-hd2b5f0e_0` | BSD-3-Clause | `3e9e02174edf02cb4bcdd75668ad7b74b8061791a3bc8bdb8a52ae336761ba3e` | True | 1 |
| `pgvector-0.8.1-h2466b09_1` | PostgreSQL | `825bfe0ad0da8f2106bb1c6077af8e84b5cf1c65a6be25434eb41ac7c74aa801` | True | 1 |
| `pgvector-python-0.4.1-pyhc48fcf7_0` | MIT | `f433f955f123740b3ced88ab9e298a9458994a00c0537c55ac15207ceb647ecf` | True | 1 |
| `pillow-12.0.0-py311h17b8079_2` | HPND | `4f6ca7bd966a50a989187e91552788d04c00529a56fcdd0f22177042cf1e11b8` | True | 1 |
| `pip-25.3-pyh8b19718_0` | MIT | `b67692da1c0084516ac1c9ada4d55eaf3c5891b54980f30f3f444541c2706f1e` | True | 1 |
| `pixman-0.46.4-h5112557_1` | MIT | `246fce4706b3f8b247a7d6142ba8d732c95263d3c96e212b9d63d6a4ab4aff35` | True | 1 |
| `platformdirs-4.5.1-pyhcf101f3_0` | MIT | `04c64fb78c520e5c396b6e07bc9082735a5cc28175dbe23138201d0a9441800b` | True | 1 |
| `pooch-1.8.2-pyhd8ed1ab_3` | BSD-3-Clause | `032405adb899ba7c7cc24d3b4cd4e7f40cf24ac4f253a8e385a4f44ccb5e0fc6` | True | 1 |
| `postgresql-16.10-h08ea38a_2` | PostgreSQL | `2783d58346a72d2b8316a02969efc015931a9da0615505d803b6b72cff2b7352` | True | 1 |
| `praatio-6.2.0-pyhd8ed1ab_1` | MIT | `2662e62d899b92abc463a7833f34e52cf71953ee093d05fd84f3f4ee0a01ef6b` | True | 1 |
| `psycopg2-2.9.9-py311haf3d11b_1` | LGPL-3.0-or-later | `c58cd095d9ec2a2c830d7c0bf2709accc34d1a6f0c0b9007bbadbe10315fe7fa` | True | 1 |
| `pthread-stubs-0.4-h0e40799_1002` | MIT | `7e446bafb4d692792310ed022fe284e848c6a868c861655a92435af7368bae7b` | True | 1 |
| `pycparser-2.22-pyh29332c3_1` | BSD-3-Clause | `79db7928d13fab2d892592223d7570f5061c192f27b9febd1a418427b719acc6` | True | 1 |
| `pygments-2.19.2-pyhd8ed1ab_0` | BSD-2-Clause | `5577623b9f6685ece2697c6eb7511b4c9ac5fb607c9babc2646c811b428fd46a` | True | 1 |
| `pynini-2.1.7-py311h3fd045d_2` | Apache-2.0 | `839b098409f568cb3e902baf6bf5ffd2d33c23ef13f1e0efd2ec2ae26fa03094` | True | 1 |
| `pyparsing-3.2.5-pyhcf101f3_0` | MIT | `6814b61b94e95ffc45ec539a6424d8447895fef75b0fec7e1be31f5beee883fb` | True | 1 |
| `pysocks-1.7.1-pyh09c184e_7` | BSD-3-Clause | `d016e04b0e12063fbee4a2d5fbb9b39a8d191b5a0042f0b8459188aedeabb0ca` | True | 1 |
| `pysoundfile-0.13.1-pyhd8ed1ab_0` | BSD-3-Clause | `09379ca52a8b119013acc325df3c609526094f923a9b3e3ac297404ff24dbba1` | True | 1 |
| `python-3.11.14-h0159041_2_cpython` | Python-2.0 | `d5f455472597aefcdde1bc39bca313fcb40bf084f3ad987da0441f2a2ec242e4` | True | 1 |
| `python-dateutil-2.9.0.post0-pyhe01879c_2` | Apache-2.0 | `d6a17ece93bbd5139e02d2bd7dbfa80bee1a4261dced63f65f679121686bf664` | True | 1 |
| `python_abi-3.11-8_cp311` | BSD-3-Clause | `fddf123692aa4b1fc48f0471e346400d9852d96eeed77dbfdd746fa50a8ff894` | True | 1 |
| `pyyaml-6.0.3-py311h3f79411_0` | MIT | `22dcc6c6779e5bd970a7f5208b871c02bf4985cf4d827d479c4a492ced8ce577` | True | 1 |
| `qhull-2020.2-hc790b64_5` | LicenseRef-Qhull | `887d53486a37bd870da62b8fa2ebe3993f912ad04bd755e7ed7c47ced97cbaa8` | True | 1 |
| `requests-2.32.5-pyhd8ed1ab_0` | Apache-2.0 | `8dc54e94721e9ab545d7234aa5192b74102263d3e704e6d0c8aa7008f2da2a7b` | True | 1 |
| `rich-14.2.0-pyhcf101f3_0` | MIT | `edfb44d0b6468a8dfced728534c755101f06f1a9870a7ad329ec51389f16b086` | True | 1 |
| `rich-click-1.9.4-pyhd8ed1ab_0` | MIT | `8e2d63c18811c2f6a9dac8aaf67628b33080882d03371f001bd1d47d4be4afe4` | True | 1 |
| `scikit-learn-1.7.2-py311h8a15ebc_0` | BSD-3-Clause | `6b7db7a33e44b2ef36b77054f3f939a6bb7722e5a1e9a1b55bfe022eda0045a8` | True | 1 |
| `scipy-1.16.3-py311hf127856_1` | BSD-3-Clause | `13938664a19955586a988e144d592440f903326f6f9cca8216e045401a2b59b1` | True | 1 |
| `sdl2-2.32.56-h5112557_0` | Zlib | `d17da21386bdbf32bce5daba5142916feb95eed63ef92b285808c765705bbfd2` | True | 1 |
| `sdl3-3.2.28-h5112557_0` | Zlib | `d4e0d53652a8087d2aa2607491c6ed8689b0fb72e1e66e1c012ef8e01f579e64` | True | 1 |
| `setuptools-80.9.0-pyhff2d567_0` | MIT | `972560fcf9657058e3e1f97186cc94389144b46dbdf58c807ce62e83f977e863` | True | 1 |
| `shaderc-2025.4-haa9a63f_0` | Apache-2.0 | `20d5a983dcf43af980fbdaddf4de9c2509a9ccae9a0b41bbdbdd96cfc937e750` | True | 1 |
| `six-1.17.0-pyhe01879c_1` | MIT | `458227f759d5e3fcec5d9b7acce54e10c9e1f4f4b7ec978f3bfd54ce4ee9853d` | True | 1 |
| `sox-14.4.2-hd84d653_1020` | GPL-2.0-only | `5ac49659414f58dbc80c977f4470a51f5a82e367795a4788c52836333f85bff3` | True | 1 |
| `soxr-0.1.3-hcfcfb64_3` | LGPL-2.1-or-later | `167201931c6be6355fa79e22fe5b8993f67a2efd0fc94acbb15393918d9a7578` | True | 1 |
| `soxr-python-1.0.0-py311h3e6a449_1` | LGPL-2.1-or-later | `7174508be8a38779d63c0723e0192104361d2cbf5dcc93fc52162d5607edb722` | True | 2 |
| `spirv-tools-2025.4-h49e36cd_0` | Apache-2.0 | `952a88cb050d8b21c020c03181af4ae8d89dd586631438cefbe66be6c15d6b92` | True | 1 |
| `sqlalchemy-2.0.44-py311h3485c13_0` | MIT | `115edbda3466956616d2ef68a393ebe22fc27f15b92d67444cce5a90a74ed395` | True | 1 |
| `sqlite-3.51.1-hdb435a2_0` | blessing | `87284f2f3c5da52fa00d694fea32656b9616fcdd425b970cef46c5de0ac636e8` | True | 0 |
| `standard-aifc-3.13.0-py311h1ea47a8_3` | BSD-4-Clause | `01d92357adb89917f3b3c0db49325ac84bc8bdb88a726f6560abd6d63068c2a9` | True | 1 |
| `standard-sunau-3.13.0-py311h1ea47a8_3` | BSD-4-Clause | `e8064a3c930ff2a369d6016a0736ce91d7af42ba38a5d9e990ef71597ac853fc` | True | 1 |
| `svt-av1-3.1.2-hac47afa_0` | BSD-2-Clause | `444c94a9c1fcb2cdf78b260472451990257733bcf89ed80c73db36b5047d3134` | True | 1 |
| `tbb-2022.3.0-hd094cb3_1` | Apache-2.0 | `c31cac57913a699745d124cdc016a63e31c5749f16f60b3202414d071fc50573` | True | 2 |
| `threadpoolctl-3.6.0-pyhecae5ae_0` | BSD-3-Clause | `6016672e0e72c4cf23c0cf7b1986283bd86a9c17e8d319212d78d8e9ae42fdfd` | True | 1 |
| `tk-8.6.13-h2c6b04d_3` | TCL | `4581f4ffb432fefa1ac4f85c5682cc27014bcd66e7beaa0ee330e927a7858790` | True | 1 |
| `tqdm-4.67.1-pyhd8ed1ab_1` | MPL-2.0 or MIT | `11e2c85468ae9902d24a27137b6b39b4a78099806e551d390e394a8c34b48e40` | True | 1 |
| `typing-extensions-4.15.0-h396c80c_0` | PSF-2.0 | `7c2df5721c742c2a47b2c8f960e718c930031663ac1174da67c1ed5999f7938c` | True | 1 |
| `typing_extensions-4.15.0-pyhcf101f3_0` | PSF-2.0 | `032271135bca55aeb156cee361c81350c6f3fb203f57d024d7e5a1fc9ef18731` | True | 1 |
| `tzdata-2025b-h78e105d_0` | LicenseRef-Public-Domain | `5aaa366385d716557e365f0a4e9c3fca43ba196872abbbe3d56bb610d131e192` | True | 1 |
| `ucrt-10.0.26100.0-h57928b3_0` | LicenseRef-MicrosoftWindowsSDK10 | `3005729dce6f3d3f5ec91dfc49fc75a0095f9cd23bab49efb899657297ac91a5` | True | 1 |
| `unicodedata2-17.0.0-py311h3485c13_1` | Apache-2.0 | `1b1bda3e9eca513cda58e9a3f1d112839bd56c9a1f6e0bf35035acbf028b0f4f` | True | 1 |
| `urllib3-2.6.1-pyhd8ed1ab_0` | MIT | `a66fc716c9dc6eb048c40381b0d1c5842a1d74bba7ce3d16d80fc0a7232d8644` | True | 1 |
| `vc-14.3-h2b53caa_33` | BSD-3-Clause | `7036945b5fff304064108c22cbc1bb30e7536363782b0456681ee6cf209138bd` | True | 0 |
| `vc14_runtime-14.44.35208-h818238b_33` | LicenseRef-MicrosoftVisualCpp2015-2022Runtime | `7e8f7da25d7ce975bbe7d7e6d6e899bf1f253e524a3427cc135a79f3a79c457c` | True | 2 |
| `vcomp14-14.44.35208-h818238b_33` | LicenseRef-MicrosoftVisualCpp2015-2022Runtime | `f79edd878094e86af2b2bc1455b0a81e02839a784fb093d5996ad4cf7b810101` | True | 2 |
| `vs2015_runtime-14.44.35208-h38c0c73_33` | BSD-3-Clause | `93fc61d05770f4c6b66214ed3494f632bf6e0e6ee7fcb0fb0a847a4bed131953` | True | 0 |
| `wheel-0.45.1-pyhd8ed1ab_1` | MIT | `1b34021e815ff89a4d902d879c3bd2040bc1bd6169b32e9427497fa05c55f1ce` | True | 1 |
| `win_inet_pton-1.1.0-pyh7428d3b_8` | LicenseRef-Public-Domain | `93807369ab91f230cf9e6e2a237eaa812492fe00face5b38068735858fba954f` | True | 1 |
| `x264-1!164.3095-h8ffe710_2` | GPL-2.0-or-later | `97166b318f8c68ffe4d50b2f4bd36e415219eeaef233e7d41c54244dc6108249` | True | 1 |
| `x265-3.5-h2d74725_3` | GPL-2.0-or-later | `02b9874049112f2b7335c9a3e880ac05d99a08d9a98160c5a98898b2b3ac42b2` | True | 1 |
| `xorg-libice-1.1.2-h0e40799_0` | MIT | `bf1d34142b1bf9b5a4eed96bcc77bc4364c0e191405fd30d2f9b48a04d783fd3` | True | 1 |
| `xorg-libsm-1.2.6-h0e40799_0` | MIT | `065d49b0d1e6873ed1238e962f56cb8204c585cdc5c9bd4ae2bf385cadb5bd65` | True | 1 |
| `xorg-libx11-1.8.12-hf48077a_0` | MIT | `3f0854bc592d31a5742c6c4550914a976c89d73b74d052545b418521d21b3043` | True | 1 |
| `xorg-libxau-1.0.12-hba3369d_1` | MIT | `156a583fa43609507146de1c4926172286d92458c307bb90871579601f6bc568` | True | 1 |
| `xorg-libxdmcp-1.1.5-hba3369d_1` | MIT | `366b8ae202c3b48958f0b8784bbfdc37243d3ee1b1cd4b8e76c10abe41fa258b` | True | 1 |
| `xorg-libxext-1.3.6-h0e40799_0` | MIT | `7fdc3135a340893aa544921115c3994ef4071a385d47cc11232d818f006c63e4` | True | 1 |
| `xorg-libxpm-3.5.17-h0e40799_1` | MIT | `a605b43b2622a4cae8df6edc148c02b527da4ea165ec67cabb5c9bc4f3f8ef13` | True | 1 |
| `xorg-libxt-1.3.1-h0e40799_0` | MIT | `c940a6b71a1e59450b01ebfb3e21f3bbf0a8e611e5fbfc7982145736b0f20133` | True | 1 |
| `yaml-0.2.5-h6a83c73_3` | MIT | `80ee68c1e7683a35295232ea79bcc87279d31ffeda04a1665efdb43cbd50a309` | True | 1 |
| `zipp-3.23.0-pyhcf101f3_1` | MIT | `b4533f7d9efc976511a73ef7d4a2473406d7f4c750884be8e8620b0ce70f4dae` | True | 1 |
| `zlib-1.3.1-h2466b09_2` | Zlib | `8c688797ba23b9ab50cef404eca4d004a948941b6ee533ead0ff3bf52012528c` | True | 1 |
| `zlib-ng-2.3.2-h5112557_0` | Zlib | `331e63a801efc9aa47e0a7f7be5becc81d9c52c1163308182078108e003c12e5` | True | 1 |
| `zstd-1.5.7-h534d264_6` | BSD-3-Clause | `368d8628424966fd8f9c8018326a9c779e06913dd39e646cf331226acc90e5b2` | True | 1 |

## B. Full notice file paths and SHA-256

For each row, the complete local source path is cache root + package stem + notice path. Nested files retain their original ownership. `Archive bytes match` compares the notice with the exact cached archive whose hash was verified above.

| Cache package stem | Notice path | SHA-256 | Archive bytes match |
| --- | --- | --- | --- |
| `_openmp_mutex-4.5-2_gnu` | `info/licenses/LICENSE` | `344c439ba60e5db75f0c894fdd22e3445db089b3bd936806bdf128c0016c12c7` | True |
| `aom-3.9.1-he0c23c2_0` | `info/licenses/LICENSE` | `4764a286d8b2faeaf42f4418e7d7a28d58fc8fd4d00a3d0a7f44b0a4099de7f2` | True |
| `audioread-3.0.1-py311h1ea47a8_3` | `info/licenses/LICENSE` | `e00fff68a75a582132842e33426730699255986f1dec362252f7423db027019d` | True |
| `backports.zstd-1.2.0-py311h71c1bcc_0` | `info/licenses/LICENSE.txt` | `b0e25a78cffb43f4d92de8b61ccfa1f1f98ecbc22330b54b5251e7b6ba010231` | True |
| `baumwelch-0.3.11-hd620369_0` | `info/licenses/LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | True |
| `biopython-1.86-py311h3485c13_0` | `info/licenses/LICENSE.rst` | `d4bbb0c4e5345e6f2508c3b93fb247fcb2db95783fb8a584c23cd88e3c03df6c` | True |
| `brotli-1.2.0-h2d644bc_1` | `info/licenses/LICENSE` | `3d180008e36922a4e8daec11c34c7af264fed5962d07924aea928c38e8663c94` | True |
| `brotli-bin-1.2.0-hfd05255_1` | `info/licenses/LICENSE` | `3d180008e36922a4e8daec11c34c7af264fed5962d07924aea928c38e8663c94` | True |
| `brotli-python-1.2.0-py311hc5da9e4_1` | `info/licenses/LICENSE` | `3d180008e36922a4e8daec11c34c7af264fed5962d07924aea928c38e8663c94` | True |
| `bzip2-1.0.8-h0ad9c76_8` | `info/licenses/LICENSE` | `c6dbbf828498be844a89eaa3b84adbab3199e342eb5cb2ed2f0d4ba7ec0f38a3` | True |
| `ca-certificates-2025.11.12-h4c7d964_0` | `info/licenses/LICENSE` | `e93716da6b9c0d5a4a1df60fe695b370f0695603d21f6f83f053e42cfc10caf7` | True |
| `cairo-1.18.4-h5782bbf_0` | `info/licenses/COPYING` | `67228a9f7c5f9b67c58f556f1be178f62da4d9e2e6285318d8c74d567255abdf` | True |
| `cairo-1.18.4-h5782bbf_0` | `info/licenses/COPYING-LGPL-2.1` | `9e9e8608c4cdda51a78cc3a385f4ec9a2e4c96d5ecad74ac8bca5fca3e563b7d` | True |
| `cairo-1.18.4-h5782bbf_0` | `info/licenses/COPYING-MPL-1.1` | `53692a2ed6c6a2c6ec9b32dd0b820dfae91e0a1fcdf625ca9ed0bdf8705fcc4f` | True |
| `certifi-2025.11.12-pyhd8ed1ab_0` | `info/licenses/certifi/LICENSE` | `e93716da6b9c0d5a4a1df60fe695b370f0695603d21f6f83f053e42cfc10caf7` | True |
| `cffi-2.0.0-py311h3485c13_1` | `info/licenses/LICENSE` | `5ba24ddc57067f9249add644c3afc41a5d6dc37e23433ef759d95df370b0af63` | True |
| `charset-normalizer-3.4.4-pyhd8ed1ab_0` | `info/licenses/data/NOTICE.md` | `0cb3efcfd8f7a02a337e98dc3de4b0b57424d7208a332b26e7deb8cb94c13922` | True |
| `charset-normalizer-3.4.4-pyhd8ed1ab_0` | `info/licenses/LICENSE` | `6d0d41bfe170ac6c7dc248c9a63e254d0fb45a60d50a8257d0af92c6e249b887` | True |
| `click-8.3.1-pyha7b4d00_1` | `info/licenses/LICENSE.txt` | `9a8ad106a394e853bfe21f42f4e72d592819a22805d991b5f3275029292b658d` | True |
| `colorama-0.4.6-pyhd8ed1ab_1` | `info/licenses/LICENSE.txt` | `cac35c02686e5d04a5a7140bfb3b36e73aed496656e891102e428886d7930318` | True |
| `contourpy-1.3.3-py311h3fd045d_3` | `info/licenses/LICENSE` | `34170979fc64f4f5e6dfa66ef27dec314ffffc5852000c60f4836ec1dfbf156e` | True |
| `cycler-0.12.1-pyhcf101f3_2` | `info/licenses/LICENSE` | `f1218143d766da3fea66f13396b7f15df46a83303f29bf96ba6e98eb4d42f408` | True |
| `dataclassy-1.0.1-pyhd8ed1ab_0` | `info/licenses/LICENSE.md` | `4d51155b35e4c804365b4c8a133aed634ab520296c9e559bd0c91128980ccb1d` | True |
| `dav1d-1.2.1-hcfcfb64_0` | `info/licenses/COPYING` | `b327887de263238deaa80c34cdd2ff3e0ba1d35db585ce14a37ce3e74ee389e9` | True |
| `decorator-5.2.1-pyhd8ed1ab_0` | `info/licenses/LICENSE.txt` | `914ee6ed78a5efc173bda698f2708444ca9d140fcc4080c60d0503d40db39da6` | True |
| `dlfcn-win32-1.4.2-hac47afa_0` | `info/licenses/COPYING` | `4cc7ac997b9293db5919baf630100cc09b3508efdfe6a6611c95511fb863b3c7` | True |
| `ffmpeg-8.0.0-gpl_he3062b8_906` | `info/licenses/COPYING.GPLv2` | `8177f97513213526df2cf6184d8ff986c675afb514d4e68a404010521b880643` | True |
| `ffmpeg-8.0.0-gpl_he3062b8_906` | `info/licenses/COPYING.GPLv3` | `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903` | True |
| `font-ttf-dejavu-sans-mono-2.37-hab24e00_0` | `info/licenses/LICENSE` | `7a083b136e64d064794c3419751e5c7dd10d2f64c108fe5ba161eae5e5958a93` | True |
| `font-ttf-inconsolata-3.000-h77eed37_0` | `info/licenses/OFL.txt` | `5d362a6f8690517fd9a5573128a081d8bbbb2f92714cf00556e08fbbe9600426` | True |
| `font-ttf-source-code-pro-2.038-h77eed37_0` | `info/licenses/LICENSE.md` | `1036668c56392d012c81aa1c84cd8aacea3e767c52190f789d14857b3d0a5765` | True |
| `font-ttf-ubuntu-0.83-h77eed37_3` | `info/licenses/LICENCE.txt` | `2f0015108d68627bd788d313f529c21ff4da2c2c42a5e1f3883acc83480f9002` | True |
| `fontconfig-2.15.0-h765892d_1` | `info/licenses/COPYING` | `51a51aa9823704fd90bccc616cdd17ebabb5b2b3e9cbde886ca02c7002288067` | True |
| `fonts-conda-ecosystem-1-0` | `info/licenses/LICENSE.txt` | `e7e41b9bdcc1e5a3b3f897a084ae48ac2a1a50172edf1a9a29ad482e0c5e8341` | True |
| `fonts-conda-forge-1-hc364b38_1` | `info/licenses/LICENSE.txt` | `e7e41b9bdcc1e5a3b3f897a084ae48ac2a1a50172edf1a9a29ad482e0c5e8341` | True |
| `fonttools-4.61.0-py311h3f79411_0` | `info/licenses/LICENSE` | `6787208f83f659ccbc2223b2fde952ffa6f7e8aca62f1a8a2bf5bc51bb1b2383` | True |
| `freetype-2.14.1-h57928b3_0` | `info/licenses/docs/FTL.TXT` | `5a5ee54c5001bbad1cdc1a57cc3dd4c42199b2da09d39c7ee41fab002d02967f` | True |
| `freetype-2.14.1-h57928b3_0` | `info/licenses/docs/GPLv2.TXT` | `c4120c6752c910c299e3bd9cb3a46ff262c268303ca2069b61f92f10a5656c18` | True |
| `fribidi-1.0.16-hfd05255_0` | `info/licenses/COPYING` | `32434afcc8666ba060e111d715bfdb6c2d5dd8a35fa4d3ab8ad67d8f850d2f2b` | True |
| `gdk-pixbuf-2.44.4-h1f5b9c4_0` | `info/licenses/COPYING` | `dc626520dcd53a22f727af3ee42c770e56c97a64fe3adb063799d8ab032fe551` | True |
| `getopt-win32-0.1-h6a83c73_3` | `info/licenses/LICENSE` | `a5681bf9b05db14d86776930017c647ad9e6e56ff6bbcfdf21e5848288dfaf1b` | True |
| `glslang-16.1.0-h5b34520_0` | `info/licenses/LICENSE.txt` | `17e70c676e1521ff3e4686f04a2053d93a7e28a33be8de7ec37ab0ff72feb677` | True |
| `graphite2-1.3.14-hac47afa_2` | `info/licenses/COPYING` | `2c272829ff23f182c8bf7ea12fedea69b0ae148663123ed5240c117cb129c9c1` | True |
| `graphviz-14.1.0-h4c50273_0` | `info/licenses/COPYING` | `77d36f258d2abcc3b2fd03db32ca0abfeb71226e9160873bed066f05906b82dd` | True |
| `greenlet-3.3.0-py311h3e6a449_0` | `info/licenses/LICENSE` | `769831d6e5dfaf2c20802faccff1fafb4c2025dd8f6253dfa47fcad59d4d0979` | True |
| `gts-0.7.6-h6b5321d_4` | `info/licenses/COPYING` | `d245807f90032872d1438d741ed21e2490e1175dc8aa3afa5ddb6c8e529b58e5` | True |
| `h2-4.3.0-pyhcf101f3_0` | `info/licenses/LICENSE` | `7a65a5af0cbabf1c16251c7c6b2b7cb46d16a7222e79975b9b61fcd66a2e3f28` | True |
| `harfbuzz-12.2.0-h5f2951f_0` | `info/licenses/COPYING` | `ba8f810f2455c2f08e2d56bb49b72f37fcf68f1f4fade38977cfd7372050ad64` | True |
| `hdbscan-0.8.39-py311h17033d2_1` | `info/licenses/LICENSE` | `d5be17fef04ce2213c09a3fffec11021065eaff565733c46ef287248b353a081` | True |
| `hpack-4.1.0-pyhd8ed1ab_0` | `info/licenses/LICENSE` | `763a9342a04df62046c9dc748a5287934eb0a5331c6863b3ca0aee20e18cb4ed` | True |
| `hyperframe-6.1.0-pyhd8ed1ab_0` | `info/licenses/LICENSE` | `763a9342a04df62046c9dc748a5287934eb0a5331c6863b3ca0aee20e18cb4ed` | True |
| `icu-75.1-he0c23c2_0` | `info/licenses/LICENSE` | `3ae033da0bfd60f4609c5c99fe50c0c840685889d3b705d3f527fd7b52032f5c` | True |
| `idna-3.11-pyhd8ed1ab_0` | `info/licenses/LICENSE.md` | `b7a336abf3b04e180ec065cdd16e705d079e1cc7a14f910aa6e9187f36b9cd87` | True |
| `importlib-metadata-8.7.0-pyhe01879c_1` | `info/licenses/LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | True |
| `joblib-1.5.2-pyhd8ed1ab_0` | `info/licenses/LICENSE.txt` | `42612911c1872c5e4b43f6ae0e8ee59467cd350332241cf72ce90640264fae6a` | True |
| `kaldi-5.5.1172-cpu_hb4f072a_2` | `info/licenses/COPYING` | `e68f06f3248553b47aa0014d176697c3cabab88395dd073f2887312742fa80b9` | True |
| `kalpy-0.8.2-py311h3fd045d_0` | `info/licenses/LICENSE` | `f905e16aae782998844a64949cede943629bd51ec845937b97fe6e71c6ccca96` | True |
| `kiwisolver-1.4.9-py311h275cad7_2` | `info/licenses/LICENSE` | `cf20799d32de0eefa2ea904e3ac5122f47aed5d352930751eb61724762c49d90` | True |
| `kneed-0.8.5-pyhd8ed1ab_1` | `info/licenses/LICENSE` | `262b008491cfbdd69f323a2d1e712344665c63b741c54f8adc496e01fc502b78` | True |
| `krb5-1.21.3-hdf4eb48_0` | `info/licenses/doc/notice.rst` | `f0201fc3511b96d5fffc59a6bba3c25dcc25e19c76f657049475ef5604499d5c` | True |
| `lame-3.100-hcfcfb64_1003` | `info/licenses/COPYING` | `bfe4a52dc4645385f356a8e83cc54216a293e3b6f1cb4f79f5fc0277abf937fd` | True |
| `lazy-loader-0.4-pyhd8ed1ab_2` | `info/licenses/LICENSE.md` | `797b6937a4f976836efbbbb3ae333d7866ddc3eb3cb152972b2be348d3529372` | True |
| `lazy_loader-0.4-pyhd8ed1ab_2` | `info/licenses/LICENSE.md` | `797b6937a4f976836efbbbb3ae333d7866ddc3eb3cb152972b2be348d3529372` | True |
| `lcms2-2.17-hbcf6048_0` | `info/licenses/LICENSE` | `6dbd60437f8ef91d8de1f08ad75882547fd4931bfcc3566a0735f28db1484d31` | True |
| `lerc-4.0.0-h6470a55_1` | `info/licenses/LICENSE` | `77a8b761727c75e2167b15bfdf61b2c0bcf8792271228bebe80779106ad00671` | True |
| `libblas-3.11.0-4_hf2e6a31_mkl` | `info/licenses/LICENSE.txt` | `c52691f010a91c8ef71062f01d62dcef465fd14bd5ff8b47155cee2480c1d844` | True |
| `libbrotlicommon-1.2.0-hfd05255_1` | `info/licenses/LICENSE` | `3d180008e36922a4e8daec11c34c7af264fed5962d07924aea928c38e8663c94` | True |
| `libbrotlidec-1.2.0-hfd05255_1` | `info/licenses/LICENSE` | `3d180008e36922a4e8daec11c34c7af264fed5962d07924aea928c38e8663c94` | True |
| `libbrotlienc-1.2.0-hfd05255_1` | `info/licenses/LICENSE` | `3d180008e36922a4e8daec11c34c7af264fed5962d07924aea928c38e8663c94` | True |
| `libcblas-3.11.0-4_h2a3cdd5_mkl` | `info/licenses/LICENSE.txt` | `c52691f010a91c8ef71062f01d62dcef465fd14bd5ff8b47155cee2480c1d844` | True |
| `libdeflate-1.25-h51727cc_0` | `info/licenses/COPYING` | `4ad69099cb4374836fd27583d9991e2838cd86b6e4666ab26b2d32582c91e73a` | True |
| `libexpat-2.7.3-hac47afa_0` | `info/licenses/COPYING` | `31b15de82aa19a845156169a17a5488bf597e561b2c318d159ed583139b25e87` | True |
| `libffi-3.5.2-h52bdfb6_0` | `info/licenses/LICENSE` | `d5699fa516968e3a1550e6c902b7441c78856f2603d61bedd0b0662a26655366` | True |
| `libflac-1.4.3-h63175ca_0` | `info/licenses/COPYING.Xiph` | `12600ea1a7affcbf469bd0d8b2cd725e4167114a2ee834b88f5d2857bfd7ddbf` | True |
| `libgd-2.3.3-h7208af6_11` | `info/licenses/COPYING` | `005f4b6b0141d1bd11d371bbf7d4f67947f85a4906b7f5465f942204cf918ba3` | True |
| `libglib-2.86.3-h0c9aed9_0` | `info/licenses/COPYING` | `fa6f36630bb1e0c571d34b2bbdf188d08495c9dbf58f28cac112f303fc1f58fb` | True |
| `libhwloc-2.12.1-default_h4379cf1_1003` | `info/licenses/COPYING` | `d79a936a42f3c6cb7c8375a023d43f4435f4664d3a5a2ea6b4623cff83c7fc06` | True |
| `libiconv-1.18-hc1393d2_2` | `info/licenses/COPYING.LIB` | `20e50fe7aae3e56378ebf0417d9de904f55a0e61e4df315333e632a4d3555d95` | True |
| `libintl-0.22.5-h5728263_3` | `info/licenses/COPYING` | `e79e9c8a0c85d735ff98185918ec94ed7d175efc377012787aebcf3b80f0d90b` | True |
| `libjpeg-turbo-3.1.2-hfd05255_0` | `info/licenses/LICENSE.md` | `2189dc45a8fe96204069f8124caa53a148dfdc193f50f584c6ca7849a6072872` | True |
| `liblapack-3.11.0-4_hf9ab0e9_mkl` | `info/licenses/LICENSE.txt` | `c52691f010a91c8ef71062f01d62dcef465fd14bd5ff8b47155cee2480c1d844` | True |
| `liblapacke-3.11.0-4_h3ae206f_mkl` | `info/licenses/LICENSE.txt` | `c52691f010a91c8ef71062f01d62dcef465fd14bd5ff8b47155cee2480c1d844` | True |
| `liblzma-5.8.1-h2466b09_2` | `info/licenses/COPYING` | `616a3ad264ce29b8f1cb97e53037b139d406899ca8d1f799651e17bfa09830b8` | True |
| `liblzma-5.8.1-h2466b09_2` | `info/licenses/COPYING.0BSD` | `0b01625d853911cd0e2e088dcfb743261034a091bb379246cb25a14cc4c74bf1` | True |
| `liblzma-devel-5.8.1-h2466b09_2` | `info/licenses/COPYING` | `616a3ad264ce29b8f1cb97e53037b139d406899ca8d1f799651e17bfa09830b8` | True |
| `liblzma-devel-5.8.1-h2466b09_2` | `info/licenses/COPYING.0BSD` | `0b01625d853911cd0e2e088dcfb743261034a091bb379246cb25a14cc4c74bf1` | True |
| `libmad-0.15.1b-hcfcfb64_1001` | `info/licenses/COPYING` | `32b1062f7da84967e7019d01ab805935caa7ab7321a7ced0e30ebe75e5df1670` | True |
| `libogg-1.3.5-h2466b09_1` | `info/licenses/COPYING` | `d2ab5758336489da61c12cc5bb757da5339c4ae9001f9bb0562b4370249af814` | True |
| `libopus-1.5.2-h2466b09_0` | `info/licenses/COPYING` | `01e1167d54a096d123cf6dfbbeb19587278845c6481d2d66d545669846079551` | True |
| `libpng-1.6.53-h7351971_0` | `info/licenses/LICENSE` | `16d9daaafbf63a31a5bdc91d4600972548fef5aaa1244202393288dbd079c49a` | True |
| `libpq-16.10-h18d9880_2` | `info/licenses/COPYRIGHT` | `5ed3ce5c9373dff7f98b1fae7a6c7ccd98df7d734d46d24c1bcebf1240be8307` | True |
| `librsvg-2.60.0-hd5e4115_0` | `info/licenses/COPYING.LIB` | `dc626520dcd53a22f727af3ee42c770e56c97a64fe3adb063799d8ab032fe551` | True |
| `libsndfile-1.2.2-h81429f1_1` | `info/licenses/COPYING` | `ad01ea5cd2755f6048383c8d54c88459cd6fcb17757c5c8892f8c5ea060f6140` | True |
| `libtiff-4.7.1-h8f73337_1` | `info/licenses/LICENSE.md` | `0e27c2382d7b8147972bbb746e04059a1152c8d0fda9d03ef1399d1a433c4ade` | True |
| `libusb-1.0.29-h1839187_0` | `info/licenses/COPYING` | `5df07007198989c622f5d41de8d703e7bef3d0e79d62e24332ee739a452af62a` | True |
| `libvorbis-1.3.7-h5112557_2` | `info/licenses/COPYING` | `ec1815db59fcd302846df949d7424876cb2e2dc5ed1606c5fb0b36787b1cf43a` | True |
| `libvulkan-loader-1.4.328.1-h477610d_0` | `info/licenses/LICENSE.txt` | `43c0a37e6a0fa7ff3c843b3ec5a4fac84b712558ddac103fbd4c1649662a9ece` | True |
| `libwebp-base-1.6.0-h4d5522a_0` | `info/licenses/COPYING` | `5aec868f669e384a22372a4e8a1a6cd7d44c64cd451f960ca69cc170d1e13acf` | True |
| `libwinpthread-12.0.0.r4.gg4f2fc60ca-h57928b3_10` | `info/licenses/COPYING.MinGW-w64/COPYING.MinGW-w64.txt` | `f38e6194bd3bfa1b654f118e5acefe0aead437bbe669eee43957ccc65a7127f1` | True |
| `libwinpthread-12.0.0.r4.gg4f2fc60ca-h57928b3_10` | `info/licenses/mingw-w64-libraries/winpthreads/COPYING` | `63263614cdd29f2f93cba85e992f041b31f9fc7b4033692f31269489a8a1b177` | True |
| `libxcb-1.17.0-h0e4246c_0` | `info/licenses/COPYING` | `c5ffbfeaa501071ceeb97b7de2c0d703fdaa35de01c0fb6cbac1c28453a3e9fd` | True |
| `libxml2-16-2.15.1-h06f855e_0` | `info/licenses/Copyright` | `5d4873884a890122a4b9b20ad56ac6f7da1d796a5bfcf04a427970ac96217626` | True |
| `libxml2-2.15.1-ha29bfb0_0` | `info/licenses/Copyright` | `5d4873884a890122a4b9b20ad56ac6f7da1d796a5bfcf04a427970ac96217626` | True |
| `libzlib-1.3.1-h2466b09_2` | `info/licenses/LICENSE` | `845efc77857d485d91fb3e0b884aaa929368c717ae8186b66fe1ed2495753243` | True |
| `llvm-openmp-21.1.7-h4fa8253_0` | `info/licenses/openmp/LICENSE.TXT` | `fdad1758a9e1f9d5a81e18879b3406772115edc92c24bfa36b70c654f325e8e4` | True |
| `llvmlite-0.45.1-py311h4f568be_0` | `info/licenses/LICENSE` | `4b9a7264b0113a7b326ee84fc244b73991b535b4833e4a56d58750f2a72250dc` | True |
| `markdown-it-py-4.0.0-pyhd8ed1ab_0` | `info/licenses/LICENSE` | `4a2260d6e2cd0f5a151a1e86dbfe7d3ed552b1e2beabf9941c1ba5c49cbce484` | True |
| `matplotlib-base-3.10.8-py311h1675fdf_0` | `info/licenses/LICENSE/LICENSE` | `5a1a81ea301728c8bba2933da832c0cd62229daf20893a024ab3d53244468dbc` | True |
| `matplotlib-base-3.10.8-py311h1675fdf_0` | `info/licenses/LICENSE/LICENSE_AMSFONTS` | `155141d59877f338f6e1c0805de9b79964dce71fe5d2a56c44e38b2c0f7e35ee` | True |
| `matplotlib-base-3.10.8-py311h1675fdf_0` | `info/licenses/LICENSE/LICENSE_BAKOMA` | `93adb2b53c02b7ce06b26bff41358c21248f42fed287b633ec4c31038377cdbb` | True |
| `matplotlib-base-3.10.8-py311h1675fdf_0` | `info/licenses/LICENSE/LICENSE_CARLOGO` | `61902d5eef34dd24870b7287a96260d3328233b96f7202bf70ad6e2a0da2de3f` | True |
| `matplotlib-base-3.10.8-py311h1675fdf_0` | `info/licenses/LICENSE/LICENSE_COLORBREWER` | `01173653fe745ce2124ec156d78b460582d2a719d2d9212804267e95fd2c97e7` | True |
| `matplotlib-base-3.10.8-py311h1675fdf_0` | `info/licenses/LICENSE/LICENSE_COURIERTEN` | `ac81f717aaff9456b88010fc55382a8dff199e4c4bcc5d61e040c3f7063beb43` | True |
| `matplotlib-base-3.10.8-py311h1675fdf_0` | `info/licenses/LICENSE/LICENSE_JSXTOOLS_RESIZE_OBSERVER` | `597756adcb51f243ef4fb386920377f61d012ace0904364e1a8ee9aaec6afc84` | True |
| `matplotlib-base-3.10.8-py311h1675fdf_0` | `info/licenses/LICENSE/LICENSE_QT4_EDITOR` | `b2b50ca8b6172aca23095adf15db89d3727e9ef2d6ef0178e427234011bed3cd` | True |
| `matplotlib-base-3.10.8-py311h1675fdf_0` | `info/licenses/LICENSE/LICENSE_SOLARIZED` | `12d5327fbc4df845a86883de99ed5fdf4198445d56dbacdf5fad8f0efdc97513` | True |
| `matplotlib-base-3.10.8-py311h1675fdf_0` | `info/licenses/LICENSE/LICENSE_STIX` | `bab3d31dfef07f483624f2f65f2711e76065b8e7273278b1c071ede1041c9959` | True |
| `matplotlib-base-3.10.8-py311h1675fdf_0` | `info/licenses/LICENSE/LICENSE_YORICK` | `cab753d38c093651e8deb5abb684e3ed68020e0e418034f94d79e890dc7ae84d` | True |
| `mdurl-0.1.2-pyhd8ed1ab_1` | `info/licenses/LICENSE` | `7c605df6e28667a9603118e98274f64a49ce3eed0d26fccce9534a345e0ef955` | True |
| `mkl-2025.3.0-hac47afa_454` | `info/licenses/mkl/info/licenses/license.txt` | `7721633d0ddff43fae25ebfd405f8166a0ce730cbcec44f2f3ad9d5eb8ac9a6f` | True |
| `mkl-2025.3.0-hac47afa_454` | `info/licenses/mkl/info/licenses/tpp.txt` | `6fb57367c2107b8b3347e0e9db10d368c7b9ea898a08132b2f841b24b86f076f` | True |
| `montreal-forced-aligner-3.3.8-pyhd8ed1ab_0` | `info/licenses/LICENSE` | `ffaac053ec76f9df77885df9102299bf0950f3ca5b3dfd04b898aee0c77bc605` | True |
| `mpg123-1.32.9-h01009b0_0` | `info/licenses/COPYING` | `c22482728a634a8dfdb4ff72a96d4c1ed64cd8f3e79335c401751ac591609366` | True |
| `msgpack-python-1.1.2-py311h3fd045d_1` | `info/licenses/COPYING` | `492dedba85da5872f78e6091bcd1fea474d660d35acb4dee964b8aab3f007427` | True |
| `munkres-1.1.4-pyhd8ed1ab_1` | `info/licenses/LICENSE.md` | `5e146a6732ad1351686c5eca491bf569065462a0cdd44157bb2e65cd4eadb75d` | True |
| `ngram-1.3.17-hc790b64_0` | `info/licenses/LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | True |
| `numba-0.62.1-py311h5e69a0e_1` | `info/licenses/LICENSE` | `3a70e0a21cad03147246f6363d07bf76fae216c72779707f4ae5db06bcacf067` | True |
| `openfst-1.8.4-hc790b64_1` | `info/licenses/COPYING` | `7c477b6dfa08be060cf8a747f1c4b8392783027ee7fb2234837ddc5b3a3abc9d` | True |
| `openh264-2.6.0-hb17fa0b_0` | `info/licenses/LICENSE` | `dd5c1c9668512530fa5a96e4c29ac4033d70a7eeb0eed7a42fddb6dd794ebdbb` | True |
| `openjpeg-2.5.4-h24db6dd_0` | `info/licenses/LICENSE` | `a6af136f3e15038a666b61f376612a07d9a4e48cb7c01adbf3e33b3f14ab49b6` | True |
| `openssl-3.6.0-h725018a_0` | `info/licenses/LICENSE.txt` | `7d5450cb2d142651b8afa315b5f238efc805dad827d91ba367d8516bc9d49e7a` | True |
| `packaging-25.0-pyh29332c3_1` | `info/licenses/LICENSE` | `cad1ef5bd340d73e074ba614d26f7deaca5c7940c3d8c34852e65c4909686c48` | True |
| `pango-1.56.4-h03d888a_0` | `info/licenses/COPYING` | `d245807f90032872d1438d741ed21e2490e1175dc8aa3afa5ddb6c8e529b58e5` | True |
| `pcre2-10.47-hd2b5f0e_0` | `info/licenses/COPYING` | `99272c55f3dcfa07a8a7e15a5c1a33096e4727de74241d65fa049fccfdd59507` | True |
| `pgvector-0.8.1-h2466b09_1` | `info/licenses/LICENSE` | `3959f8e3ea8b08e35d1c26fc25bc7b6dcf7c2d80f372e4fbff6f6b0412758456` | True |
| `pgvector-python-0.4.1-pyhc48fcf7_0` | `info/licenses/LICENSE.txt` | `8d899495773282daf11e0a1ff203967c71abda500e4fd246dc197d7b3ca37638` | True |
| `pillow-12.0.0-py311h17b8079_2` | `info/licenses/LICENSE` | `17f240ae101143707e5e7303a5d800450d9ccc7475b463cedb555cefdb3c6ece` | True |
| `pip-25.3-pyh8b19718_0` | `info/licenses/LICENSE.txt` | `634300a669d49aeae65b12c6c48c924c51a4cdf3d1ff086dc3456dc8bcaa2104` | True |
| `pixman-0.46.4-h5112557_1` | `info/licenses/COPYING` | `fac9270f0987b96ff4533fca3548c633e02083cbba4a0172a3b149b2e4019793` | True |
| `platformdirs-4.5.1-pyhcf101f3_0` | `info/licenses/LICENSE` | `29e0fd62e929850e86eb28c3fdccf0cefdf4fa94879011cffb3d0d4bed6d4db6` | True |
| `pooch-1.8.2-pyhd8ed1ab_3` | `info/licenses/LICENSE.txt` | `af0482d59fc7a9d68d9f64682c74ce6e8a4684406bb3a21a191ae329950916b4` | True |
| `postgresql-16.10-h08ea38a_2` | `info/licenses/COPYRIGHT` | `5ed3ce5c9373dff7f98b1fae7a6c7ccd98df7d734d46d24c1bcebf1240be8307` | True |
| `praatio-6.2.0-pyhd8ed1ab_1` | `info/licenses/LICENSE` | `e3916b107f7039986b72c5b0b6a4afff9e6f13a8b20e904de999a9c782f02642` | True |
| `psycopg2-2.9.9-py311haf3d11b_1` | `info/licenses/LICENSE` | `9614b85dfc9a72c5b2ca33144c1d7e1ed3b1c297459d9fb28a6a5762c2e8d71b` | True |
| `pthread-stubs-0.4-h0e40799_1002` | `info/licenses/COPYING` | `78c20706e799f2b8f445e71d3d2ade6ba23b3388fd6cbeed7d71796623febde8` | True |
| `pycparser-2.22-pyh29332c3_1` | `info/licenses/LICENSE` | `0c846399369ea76ddd7b5c44fe6d16497415fcf015f5cbb508c24bf98b81c5b1` | True |
| `pygments-2.19.2-pyhd8ed1ab_0` | `info/licenses/LICENSE` | `a9d66f1d526df02e29dce73436d34e56e8632f46c275bbdffc70569e882f9f17` | True |
| `pynini-2.1.7-py311h3fd045d_2` | `info/licenses/LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | True |
| `pyparsing-3.2.5-pyhcf101f3_0` | `info/licenses/LICENSE` | `10d5120a16805804ffda8b688c220bfb4e8f39741b57320604d455a309e01972` | True |
| `pysocks-1.7.1-pyh09c184e_7` | `info/licenses/LICENSE` | `7027e214e014eb78b7adcc1ceda5aca713a79fc4f6a0c52c9da5b3e707e6ffe9` | True |
| `pysoundfile-0.13.1-pyhd8ed1ab_0` | `info/licenses/LICENSE` | `0dd2e411fed553ba891845077e5a32c5b726d1bbc46ff31a2118ae1ccf816752` | True |
| `python-3.11.14-h0159041_2_cpython` | `info/licenses/LICENSE` | `3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf` | True |
| `python-dateutil-2.9.0.post0-pyhe01879c_2` | `info/licenses/LICENSE` | `ba00f51a0d92823b5a1cde27d8b5b9d2321e67ed8da9bc163eff96d5e17e577e` | True |
| `python_abi-3.11-8_cp311` | `info/licenses/LICENSE` | `44ef42d743263ebacf7f3d175a940c62959d75af9afd05ef016861e9b4bf2859` | True |
| `pyyaml-6.0.3-py311h3f79411_0` | `info/licenses/LICENSE` | `8d3928f9dc4490fd635707cb88eb26bd764102a7282954307d3e5167a577e8a4` | True |
| `qhull-2020.2-hc790b64_5` | `info/licenses/COPYING.txt` | `106d55c931fd6a84822e5345d900273d059f1c27310d02567ccb313c5d18c55d` | True |
| `requests-2.32.5-pyhd8ed1ab_0` | `info/licenses/LICENSE` | `09e8a9bcec8067104652c168685ab0931e7868f9c8284b66f5ae6edae5f1130b` | True |
| `rich-14.2.0-pyhcf101f3_0` | `info/licenses/dist/LICENSE` | `deed7c17a4318158190a3ea239cc879a5a50271cebb98ae7025f48fbe58dca15` | True |
| `rich-click-1.9.4-pyhd8ed1ab_0` | `info/licenses/LICENSE` | `d460cfe666617a2f86cb7c66f9041f0a87611c8b0c1f2f3f674a2022adc1f2af` | True |
| `scikit-learn-1.7.2-py311h8a15ebc_0` | `info/licenses/COPYING` | `75684d16f4662ab842ed5af9482ac01939aee83c8243d89993616d959d3053f8` | True |
| `scipy-1.16.3-py311hf127856_1` | `info/licenses/LICENSE.txt` | `221e59f5e910fd7f94e44f0dac77436a11338c285c6346232e4a850a50da0e94` | True |
| `sdl2-2.32.56-h5112557_0` | `info/licenses/LICENSE.txt` | `e3dadca51e6517be4a59b41e50f572ed0553b662e11fe4150d3b50360e3069a0` | True |
| `sdl3-3.2.28-h5112557_0` | `info/licenses/LICENSE.txt` | `97f35b302b361680ec1e891e95d2d52097bb95abff361434916d99dc1305f127` | True |
| `setuptools-80.9.0-pyhff2d567_0` | `info/licenses/LICENSE` | `86da0f01aeae46348a3c3d465195dc1ceccde79f79e87769a64b8da04b2a4741` | True |
| `shaderc-2025.4-haa9a63f_0` | `info/licenses/LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | True |
| `six-1.17.0-pyhe01879c_1` | `info/licenses/LICENSE` | `4375ba20e2b9c6c4e7cad2940a628fd90e95cc3d50ee92aae755715d8ba1fbd0` | True |
| `sox-14.4.2-hd84d653_1020` | `info/licenses/COPYING` | `7d03a542ce0b414c27b248b553bc49d45a7a4dcbcc8c08aeb5ebdca204bf1226` | True |
| `soxr-0.1.3-hcfcfb64_3` | `info/licenses/LICENCE` | `dc98676341fdcd29d9f279c9679d6a75288785b174ded8d1b2e316c366166135` | True |
| `soxr-python-1.0.0-py311h3e6a449_1` | `info/licenses/COPYING.LGPL` | `f2f118b9029ec1871b953639ecc46651b2fc7b62e295e6cf3ef2ac4c9a058b33` | True |
| `soxr-python-1.0.0-py311h3e6a449_1` | `info/licenses/LICENSE.txt` | `aeeb7bce59a3dcc7bb08a48a2f012f340c6b28ec50cd22f0ad93194a7d254420` | True |
| `spirv-tools-2025.4-h49e36cd_0` | `info/licenses/LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` | True |
| `sqlalchemy-2.0.44-py311h3485c13_0` | `info/licenses/LICENSE` | `9821720b58d4a565b61321007a8ad43ab6991a3959e66cd5db4dd9eae69c7dfc` | True |
| `standard-aifc-3.13.0-py311h1ea47a8_3` | `info/licenses/LICENSE` | `14949dab294d8379384238869abc035f4ef95bb1968b10f88554afa001ead28e` | True |
| `standard-sunau-3.13.0-py311h1ea47a8_3` | `info/licenses/LICENSE` | `14949dab294d8379384238869abc035f4ef95bb1968b10f88554afa001ead28e` | True |
| `svt-av1-3.1.2-hac47afa_0` | `info/licenses/LICENSE.md` | `0acc2fcb27472bdc9aaf8b71f37055bbdac4f54671b7d922f241bd7fcd0dd3e6` | True |
| `tbb-2022.3.0-hd094cb3_1` | `info/licenses/LICENSE.txt` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` | True |
| `tbb-2022.3.0-hd094cb3_1` | `info/licenses/third-party-programs.txt` | `1634857eb058dce5278c57a832b8974190785145f4be9acdbdb511f3a0c4091c` | True |
| `threadpoolctl-3.6.0-pyhecae5ae_0` | `info/licenses/LICENSE` | `81ac619075248b06e53660b652d10e485f4675f5d0ae0f97ea22370da1f7e23b` | True |
| `tk-8.6.13-h2c6b04d_3` | `info/licenses/tcl8.6.13/license.terms` | `c0a69a2bfd757361ec7e6143973b103c90409316b49e9c88db26ad6388e79f16` | True |
| `tqdm-4.67.1-pyhd8ed1ab_1` | `info/licenses/LICENCE` | `dc33252e829015e3b150086fb9b3a40f6ad6fb32c2f4610ce812fa677d35986a` | True |
| `typing-extensions-4.15.0-h396c80c_0` | `info/licenses/LICENSE` | `3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf` | True |
| `typing_extensions-4.15.0-pyhcf101f3_0` | `info/licenses/LICENSE` | `3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf` | True |
| `tzdata-2025b-h78e105d_0` | `info/licenses/LICENSE` | `0613408568889f5739e5ae252b722a2659c02002839ad970a63dc5e9174b27cf` | True |
| `ucrt-10.0.26100.0-h57928b3_0` | `info/licenses/LICENSE.txt` | `f4589c739597890dd40bccb6bf20632a99c228bdd20cfbf585b5b30c9e61b3fb` | True |
| `unicodedata2-17.0.0-py311h3485c13_1` | `info/licenses/LICENSE` | `cb5e8e7e5f4a3988e1063c142c60dc2df75605f4c46515e776e3aca6df976e14` | True |
| `urllib3-2.6.1-pyhd8ed1ab_0` | `info/licenses/LICENSE.txt` | `130e3a64d5fdd5d096a752694634a7d9df284469de86e5732100268041e3d686` | True |
| `vc14_runtime-14.44.35208-h818238b_33` | `info/licenses/LICENSE.RTF` | `8099dc3cf9502c335da829e5c755948a12e3e6de490eb492a99deb673d883d8b` | True |
| `vc14_runtime-14.44.35208-h818238b_33` | `info/licenses/LICENSE.TXT` | `cf89b37107c22cf8ce21043da1da143087b86ce7fd4556f91c25bc429533807e` | True |
| `vcomp14-14.44.35208-h818238b_33` | `info/licenses/LICENSE.RTF` | `8099dc3cf9502c335da829e5c755948a12e3e6de490eb492a99deb673d883d8b` | True |
| `vcomp14-14.44.35208-h818238b_33` | `info/licenses/LICENSE.TXT` | `cf89b37107c22cf8ce21043da1da143087b86ce7fd4556f91c25bc429533807e` | True |
| `wheel-0.45.1-pyhd8ed1ab_1` | `info/licenses/LICENSE.txt` | `30c23618679108f3e8ea1d2a658c7ca417bdfc891c98ef1a89fa4ff0c9828654` | True |
| `win_inet_pton-1.1.0-pyh7428d3b_8` | `info/licenses/LICENSE` | `2688ebd0bad28507e6f4881d89acc60ad3f27c71da229042cd92cb871e7f1774` | True |
| `x264-1!164.3095-h8ffe710_2` | `info/licenses/COPYING` | `32b1062f7da84967e7019d01ab805935caa7ab7321a7ced0e30ebe75e5df1670` | True |
| `x265-3.5-h2d74725_3` | `info/licenses/COPYING` | `d8afb1bcc7a2cfc603683b168d6987ef0a48e59e0da3693bf55c5d33b67e2b49` | True |
| `xorg-libice-1.1.2-h0e40799_0` | `info/licenses/COPYING` | `60105b7ea93cb07a67fee8443b092b727e3db7f0dff4fbe05bc6cd7747fb53c8` | True |
| `xorg-libsm-1.2.6-h0e40799_0` | `info/licenses/COPYING` | `136fb258a67a2810302c6cd99525b068e6f875b8f1031a93b34edad3ffc740a5` | True |
| `xorg-libx11-1.8.12-hf48077a_0` | `info/licenses/COPYING` | `2e7012a140f000735a7172674a2d314398d79622444fba65d108b029b29ab283` | True |
| `xorg-libxau-1.0.12-hba3369d_1` | `info/licenses/COPYING` | `56abe29bb1d9806a9e04fa9f80fed2c0f18027594df3f098148d814aef6bddfa` | True |
| `xorg-libxdmcp-1.1.5-hba3369d_1` | `info/licenses/COPYING` | `8a3c3f35b0dbcb60a4e242b9e4394a352a65bb27deb2938ea1e2e62a626e16e9` | True |
| `xorg-libxext-1.3.6-h0e40799_0` | `info/licenses/COPYING` | `fd62910be4b13829d94e76c1447cf840953f0e225c4dc6c79349c84dd0557f22` | True |
| `xorg-libxpm-3.5.17-h0e40799_1` | `info/licenses/COPYING` | `a80d706759624a04aa90fd62bc644a360fc3d72e08dcbfb129f167c11ca285de` | True |
| `xorg-libxt-1.3.1-h0e40799_0` | `info/licenses/COPYING` | `75c5574ca04731d739b1420f55f2b7b47f30df895817f1b03d0d7f5c1fbee534` | True |
| `yaml-0.2.5-h6a83c73_3` | `info/licenses/License` | `c40112449f254b9753045925248313e9270efa36d226b22d82d4cc6c43c57f29` | True |
| `zipp-3.23.0-pyhcf101f3_1` | `info/licenses/LICENSE` | `5a57cb4db85e2a2dd88c290628908add57e3451449e0a9a71fdfb38776fd759d` | True |
| `zlib-1.3.1-h2466b09_2` | `info/licenses/LICENSE` | `845efc77857d485d91fb3e0b884aaa929368c717ae8186b66fe1ed2495753243` | True |
| `zlib-ng-2.3.2-h5112557_0` | `info/licenses/LICENSE.md` | `6c9f0d975b41afaa34d22f55bb8986ce69e5cb7ad327cb2b28820cd425edf5ee` | True |
| `zstd-1.5.7-h534d264_6` | `info/licenses/LICENSE` | `7055266497633c9025b777c78eb7235af13922117480ed5c674677adc381c9d8` | True |

## C. Packages without info/licenses

- `libfreetype-2.14.1-h57928b3_0`
- `libfreetype6-2.14.1-hdbac1cb_0`
- `libgcc-15.2.0-h8ee18e1_16`
- `libgomp-15.2.0-h8ee18e1_16`
- `librosa-0.11.0-pyhd8ed1ab_0`
- `libsqlite-3.51.1-hf5d6505_0`
- `sqlite-3.51.1-hdb435a2_0`
- `vc-14.3-h2b53caa_33`
- `vs2015_runtime-14.44.35208-h38c0c73_33`

These are individually classified in the prose above. Empty metapackages do not create nonexistent code. Existing payload licenses (for example librosa), split-package licenses, SQLite public-domain statements and the GCC runtime exception need their own matching provenance.

## D. Bound manifest and model hashes (initial Final sample)

| File | SHA-256 | Manifest match |
| --- | --- | --- |
| `runtimes/egg/python.exe` | `927eff8bdc63468d3bc66a6083257ada8f1a42304c3f308be599d1b3c90c6148` | True |
| `runtimes/m05/python.exe` | `927eff8bdc63468d3bc66a6083257ada8f1a42304c3f308be599d1b3c90c6148` | True |
| `runtimes/mfa/registry-bundled.json` | `3c15089591d97da6d71a394df6fad59395d458b08061e6a72ae0a61b8155a5ce` | True |
| `runtimes/mfa/resources/model/bc16c3278295fad0d742399b849e5a8375a65bcbe5b7c7612ff1e3ffb81dcef7.zip` | `bc16c3278295fad0d742399b849e5a8375a65bcbe5b7c7612ff1e3ffb81dcef7` | True |
| `runtimes/mfa/resources/dictionary/6b71538ba4cd48d40f92561a129e4531ee256951fcf218ac1198da72ca13076e.dict` | `6b71538ba4cd48d40f92561a129e4531ee256951fcf218ac1198da72ca13076e` | True |
| `runtimes/mfa/resources/model/beca5c15f6c15bee4435a37bd2904e15b12fcc56342013d253260f31fd2c3f33.zip` | `beca5c15f6c15bee4435a37bd2904e15b12fcc56342013d253260f31fd2c3f33` | True |
| `runtimes/mfa/resources/dictionary/c6a9b62905917be3f21a9138503ccf9a5fb1ca080d5a7eed75c8a662c80770cb.dict` | `c6a9b62905917be3f21a9138503ccf9a5fb1ca080d5a7eed75c8a662c80770cb` | True |
| `resources/vocal_tract/sources/VTL2.4-API-source.zip` | `aa696bf4cf5c44cbd433304943aebb241e5ea3064b949daf5ea09f96567cff9b` | recorded |
| `third_party/source-registry.json` | `83e2f49dd17ab377b393d8ec759111792b17e8e379290bd92bf60134e5d46005` | recorded |
| `../PhoneticToolbox.exe` | `fb5753fbe002101d456eaad2b3efbfc6ec6771cf96e1893d0f03c345dc8a062f` | recorded |

## E. Final2 targeted identity checks

Only the files listed below were repeated against Final2 in this audit. The full host DLL identity and seven scientific tasks/MFA/UI validation were performed by the main release task and remain in its own report.

| Final2 file | Compared with | SHA-256 equal |
| --- | --- | --- |
| `desktop-bundle.json` | Final sample | True |
| `runtimes/egg/python.exe` | Final sample | True |
| `runtimes/egg/conda-meta/scipy-1.16.3-py311hf127856_1.json` | Final sample | True |
| `runtimes/m05/python.exe` | Final sample | True |
| `runtimes/mfa/registry-bundled.json` | Final sample | True |
| `runtimes/mfa/receipt.json` | Final sample | True |
| `resources/vocal_tract/sources/VTL2.4-API-source.zip` | Final sample | True |
| `Qt6Core.dll` | host m09-ui exact installed file | True |
| `Qt6WebEngineCore.dll` | host m09-ui exact installed file | True |
| `Qt6WebChannel.dll` | host m09-ui exact installed file | True |
| `Qt6Qml.dll` | host m09-ui exact installed file | True |
