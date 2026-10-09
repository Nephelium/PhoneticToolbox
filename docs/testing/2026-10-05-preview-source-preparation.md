# Preview 1 精确源码与组件来源准备

日期：2026-10-05。状态：本轮明确源码与模型比对下载已完成。此报告依据实际 Final2 科学运行环境与主宿主 DLL 身份，不修改任何当前 app 发行包。主线之后形成的 portable r2、installer r3、自解压 EXE 与私有服务器暂存属于主代理工作，本报告不代替它们的运行验收。

## 结论与当前可用材料

所有本轮新增源码和来源材料集中在 `C:\Users\13680\AppData\Local\Temp\PTB-Preview1-20261005-corresponding-source`。**61/61 个源码绑定目标全部通过，对应 60 个不同的本地完整归档。**两个 Qt 完整源码归档均已通过官方 SHA-256。两个另行获准下载的 MFA v2.0.0a 官方模型及词典已与当前包 SHA-256 完全一致，并绑定到具体版本的 CC BY 4.0 原文、归属和引用。此处的 60 个归档不包括两个模型/词典比对文件。

- 主宿主 PyQt6 6.11.0、PyQt6-WebEngine 6.11.0、PyQt6-sip 13.12.0 和 MFA 独立环境 PyQt6 6.6.1、sip 13.10.3 的官方 sdist 均通过 PyPI 所公布的 SHA-256。
- Qt 6.11.2 的完整源码归档已通过 Qt 官方 SHA-256，含完整 QtWebEngine/Chromium，已经逐成员核实内容，补齐先前盘点的上游源码归档缺项。
- Parselmouth 0.4.7 的官方 sdist 已通过 PyPI SHA-256，实际含其所用 Praat 6.1.38 源码及 CMake、pybind11/fmt 来源，毋须拿现行 Praat 版本替代。
- MFA 实际 193 个 Conda 归档对应的构建配方、补丁、元数据已保存在 `conda-recipes/`，共 2,511 个文件逐项记录 SHA-256。全部 193 个本地缓存归档的 SHA-256 已在发行盘点阶段与包内 `conda-meta` 一致。已识别 32 条 GPL/LGPL/MPL 配方 source 绑定，包括现代 `rendered_recipe.yaml` 格式，未漏掉 libusb 和 getopt-win32。
- MFA FFmpeg 8.0.0 `gpl_he3062b8_906` 与 M05 的 FFmpeg 8.0.1 分别准备。两者的 x264/x265 版本和补丁有实质差异，不能合并声明。

这些材料是可复核的源码准备成果，尚未作为最终成品的同渠道对应源码交付。项目总 LICENSE、VoiceSauce `func_getSoE.m` 的具体许可仍待井井答复，本轮不代选或修改。纯论文方法引用、已有明确授权和自有工作均不新增许可阻断。

## 身份与摘要的证据级别

`source-status-manifest.json` 记录每个目标的实际路径、版本、URL、来源摘要、实际 SHA-256 和状态。`download-targets.json` 为官方 PyPI/Qt 目标，`conda-source-targets.json`、`conda-codec-source-targets.json` 为精确 Conda 配方目标，`pyav-codec-source-targets.json` 为精确 PyAV 原生依赖构建目标。JSON 与本文末尾清单应一并保留。

1. **官方独立摘要**：PyPI JSON 给出对应 sdist 的 SHA-256，Qt 官方 `.sha256` 与 MirrorBrain 文件信息给出完整源码摘要和长度。下载后重新计算并比较。
2. **精确构建配方摘要**：Conda source 的摘要来自已校验过的实际二进制归档中的配方。它不是全部上游各自另行公布的摘要，报告保留二者区别。源码通过这个摘要及随存的补丁与实际 build 绑定。
3. **固定 Git 对象**：PyAV 的原生依赖构建仓库按 tag 固定 commit，16 个原始文件重新计算 Git blob SHA-1 与官方 Git API 对象一致，同时记录 SHA-256。获取的是来源代码及补丁，没有执行下载的代码。
4. **模型实际内容比对**：官方旧 release 没有公布 SHA-256 时，单独下载获准模型与词典，计算其 SHA-256 与当前包比较。下载前文件大小匹配只构成身份线索。

较早单流 Qt 下载保留为 `.part`，分段文件保留在 `qt-ranges/`，没有删除已有下载。重跑自有下载器会复用已校验完整文件并续传分段。下载异常不会被写作校验通过。源码准备不安装或执行第三方下载脚本，也不修改当前虚拟环境。

## PyQt6 与 Qt / Chromium

| 对象 | 实际分发身份 | 本轮准备 |
| --- | --- | --- |
| 主宿主 PyQt6 | 6.11.0 | `archives/pyqt6-6.11.0.tar.gz`，SHA-256 `45dd60aa69976de1918b5ced6b4e7b6a25abd2a919ecef5fd5826ecc76718889` |
| 主宿主 PyQt6-WebEngine | 6.11.0 | `archives/pyqt6_webengine-6.11.0.tar.gz`，SHA-256 `15cf49efbbbd4c6bc87653b2c4ae80d6049f800e31620b336734ae2e37cbedae` |
| 主宿主 sip | 13.12.0 | `archives/pyqt6_sip-13.12.0.tar.gz`，SHA-256 `a7ad45c1e3cec3a2473d37ea9870b6c3baeccc560298623c8eb59265714c06e2` |
| 主宿主 Qt / QtWebEngine | 6.11.2 | `archives/qt-everywhere-src-6.11.2.tar.xz`，1,019,661,552 字节，SHA-256 `6dcfbca271d76a6502741a2c0dc6fc98ef7dd0b7b4cfd0abcebb285a86a26f33` |
| MFA PyQt6 | 6.6.1 | `archives/PyQt6-6.6.1.tar.gz`，SHA-256 `9f158aa29d205142c56f0f35d07784b8df0be28378d20a97bcda8bd64ffd0379` |
| MFA sip | 13.10.3 | `archives/pyqt6_sip-13.10.3.tar.gz`，SHA-256 `630895b3827e2c3b4e072089157985691fe4210d64340e71141f93775ea4ae51` |
| MFA Qt | 6.6.1 | `archives/qt-everywhere-src-6.6.1.tar.xz`，814,132,652 字节，官方与实际 SHA-256一致：`dd3668f65645fe270bc615d748bd4dc048bd17b9dc297025106e6ecc419ab95d` |

PyPI 官方身份与摘要已保存于 `metadata/`，可分别追溯 [PyQt6 6.11.0](https://pypi.org/pypi/PyQt6/6.11.0/json)、[PyQt6-WebEngine 6.11.0](https://pypi.org/pypi/PyQt6-WebEngine/6.11.0/json)、[MFA PyQt6 6.6.1](https://pypi.org/pypi/PyQt6/6.6.1/json)。Qt 完整源码及其独立摘要来源为 [Qt 6.11.2 官方完整源码](https://download.qt.io/archive/qt/6.11/6.11.2/single/qt-everywhere-src-6.11.2.tar.xz)、[Qt 6.11.2 官方 SHA-256](https://download.qt.io/archive/qt/6.11/6.11.2/single/qt-everywhere-src-6.11.2.tar.xz.sha256)、[Qt 6.6.1 官方 SHA-256](https://download.qt.io/archive/qt/6.6/6.6.1/single/qt-everywhere-src-6.6.1.tar.xz.sha256)。

对已完整校验的 Qt 6.11.2 源码归档逐成员扫描得到 390,976 个文件，QtWebEngine 269,594 个，Chromium 子树 266,964 个。已抽取并记录 QtBase/QtWebEngine 许可证集合、Chromium `LICENSE`、其 FFmpeg `LICENSE.md`、模块 `.cmake.conf`、`chrome/VERSION`。Chromium VERSION 为 **140.0.7339.225**。记录在 `qt-6.11.2-source-content.json`，原文存于 `metadata/qt-6.11.2-source-members/`。

Qt 6.6.1 官方当前未列可用镜像，最终从官方直连分段取得完整文件。逐成员扫描确认 326,346 个文件，其中 QtWebEngine 219,009 个，Chromium 216,477 个。实际 QtBase/QtWebEngine 版本文件与许可原文保存在 `metadata/qt-6.6.1-source-members/`，清单为 `qt-6.6.1-source-content.json`。这一完整归档涵盖的模块多于 MFA 实际 Qt 使用范围，不由归档内存在某模块推断该模块被 app 使用。

已获取的是精确版本上游完整源树。本轮未重建 Riverbank wheel，也未证明每个第三方二进制 wheel 的全部编译选项及额外构建补丁均已穷尽。因此完整源码取得与最终对应源码交付、通知及 LGPL 替换安排的完成状态分开记录，不宣称公开发行已全面通过。参见 [Qt 官方 LGPL 义务说明](https://www.qt.io/development/open-source-lgpl-obligations)。

## Parselmouth 与其内嵌 Praat

`archives/praat_parselmouth-0.4.7.tar.gz` 为 22,526,491 字节。其 PyPI 与实际 SHA-256 均为 `6dd81d246ce1eef5fd93d8cbdaf1bef61ca40ef1d2fc12aa23996a28071181e6`。归档中有 3,627 个文件，其中 3,090 个位于 `praat/`。`praat/sys/praat_version.h` 明确 `PRAAT_VERSION_STR 6.1.38` 和日期 2021-01-02。

已核实其项目 `LICENSE`、`praat/main/GNU_General_Public_License.txt`、pybind11 许可、fmt 许可，以及根 CMake/构建材料。Parselmouth 自身 GPL-3.0-or-later，README 将内嵌 Praat 指向 GPL-2.0-or-later，保留两种身份。上游明确提到为了 Parselmouth 对 Praat 有少量修改，因此本轮保留包含这些修改的 sdist。下载独立现行 Praat 源码不能代替这个精确源树。[Parselmouth 0.4.7 官方 PyPI 元数据](https://pypi.org/pypi/praat-parselmouth/0.4.7/json)。

源码归档带有项目文档/测试中的自然例音和其原始归属说明。本轮仅取得完整上游源码，不将这些内容另作公开音频传播，也不重新审计音频、截图内容作者章节。

## MFA Conda FFmpeg 及其他组件

保留的 `conda-recipe-inventory.json` 对每个实际包记录 name/version/build、原二进制归档 URL/SHA-256、所有配方与补丁的路径/SHA-256。配方来源来自本机精确缓存 `C:\Users\13680\Miniconda3_broken_backup_20260317_200156\pkgs`，不是从现行 feedstock 主页推测历史 build。现代配方的 `rendered_recipe.yaml` 与传统 `meta.yaml` 均读取。

关键 MFA 绑定为：

| 对象 | 实际版本/build | 本轮对应源 |
| --- | --- | --- |
| FFmpeg | 8.0.0 `gpl_he3062b8_906` | `ffmpeg-8.0.tar.gz`，SHA-256 `cce1136d38c389e6baaa452d6babc384cb2d3a9406ebe48c36a48f3ee115d8df`，五项配方补丁和 `build.sh` 全部保存 |
| x264 | `1!164.3095` `h8ffe710_2` | 精确 commit `baee400fa9ced6f5481a728138fed6e867b0ff7f`，SHA-256 `436a2be54d8bc0cb05dd33ecbbcb7df9c3b57362714fcdaa3a5991189a33319b` |
| x265 | 3.5 `h2d74725_3` | `x265_3.5.tar.gz`，SHA-256 `e70a3335cacacbba0b3a20ec6fecd6783932288ebc8163ad74bcc9606477cae8` |
| LAME | 3.100 `hcfcfb64_1003` | `lame-3.100.tar.gz`，SHA-256 `ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e`，Windows 补丁保留 |
| libiconv | 1.18 `hc1393d2_2` | `libiconv-1.18.tar.gz`，SHA-256 `3b08f5f4f9b4eb82f151a7040bfd6fe6c6fb922efe4b1659c66ea933276965e8`，三项 CMake/config 补丁保留 |
| libmad | 0.15.1b `hcfcfb64_1001` | `libmad-0.15.1b.tar.gz`，SHA-256 `bbfac3ed6bfbc2823d3775ebb931087371e142bb0e9bb1bee51a76a6e0078690` |
| libusb | 1.0.29 `h1839187_0` | 官方 `libusb-1.0.29.tar.bz2`，源自实际 rendered recipe，而非概括主页 |
| getopt-win32 | 0.1 `h6a83c73_3` | 精确 tag 源码 `0.1.tar.gz`、两项配方补丁、实际 LGPL-3.0-only 归属 |

Conda FFmpeg 配方确实选择 GPL build，`build.sh`/配方测试要求 `enable-gpl`。五项补丁包含 Chromium first-dts 适配与 Windows/LLVM 构建修正。配方固定的 feedstock commit 为 `c59d67b5764c92a7757d2c6c7fcf8fb9508385f7`。本轮没有执行配方或构建脚本，也不把包内 API 文件视为原生库完整源码。官方 source 为 [FFmpeg 8.0](https://ffmpeg.org/releases/ffmpeg-8.0.tar.gz)，许可判断参见 [FFmpeg 官方说明](https://ffmpeg.org/legal.html)。

其余已经取得并逐摘要绑定的 GPL/LGPL/MPL 原生与 Python 源包括 cairo、dataclassy、fribidi、gdk-pixbuf、graphite2、gts、GLib、gettext/libintl、librsvg、libsndfile、mpg123、pango、psycopg2、SoX、soxr、soxr-python、tqdm。mpg123 官方 download URL 原始 basename 为 `download`，因此本地完整归档亦为 `archives/download`，其实际版本 1.32.9 与摘要已绑定，不是网页占位文件。SoX 的历史精确配方只有 MD5，下载匹配 `d04fba2d9245e661f245de0577f48a33`，另记录实际 SHA-256作为完整性凭据，不宣称上游曾公布该 SHA-256。

为保留具体 codec 来源，同时准备 aom、dav1d、libogg、libopus、libvorbis、OpenH264、SVT-AV1 的精确源码及实际版本补丁。它们各有独立许可，不能被 PyAV 顶层 BSD 标签或 FFmpeg 单条 LGPL 标签合并覆盖。

FreeType 2.14.1 精确源已取得。其 FTL/GPL 双许可不等于强制把整个工程改为 GPL。MFA libgcc/libgomp 15.2.0 的精确配方、全部补丁和 GCC Runtime Library Exception 已定位并保留来源记录。本轮未另下载 15.2.0 的大型 GCC 完整源归档，按实际使用的运行库例外单列，不把 compiler/runtime 同整个 GUI 工程混为一类。

## M05 PyAV 16.1.0 / FFmpeg 8.0.1

实际 M05 包中的 `avutil-60-a23272e4a2631ae7398afbababd42d44.dll` SHA-256 为 `01fb0e1c724930ac7514e4b067464ee082f0670481521268c6da62a79db697d0`，调用其现有 `av_version_info` 返回 **8.0.1**。此前已审计 avcodec 配置包含 `--enable-version3 --enable-libx264 --enable-libx265`，许可字符串为 LGPL version 3 or later。它独立于 MFA FFmpeg 8.0.0。

PyAV 16.1.0 的 [精确 tag CI](https://github.com/PyAV-Org/PyAV/blob/v16.1.0/.github/workflows/tests.yml) 和 [vendor 配置](https://github.com/PyAV-Org/PyAV/blob/v16.1.0/scripts/ffmpeg-latest.json) 指向 `PyAV-Org/pyav-ffmpeg` **8.0.1-3**。对应构建仓库固定 commit `9e63121cf6ef9bc1744956c7a4cb772589c80806` 的完整 16 个原始文件已按 Git blob SHA 检查，保存在 `pyav-ffmpeg-build-8.0.1-3/`。SHA-256和官方 Git 对象记录在 `pyav-ffmpeg-build-manifest.json`。

该构建实际固定：FFmpeg 8.0.1、LAME 3.100、ogg 1.3.6、opus 1.6、speex 1.2.1、vorbis 1.3.7、aom 3.13.1、dav1d 1.5.3、SVT-AV1 3.1.2、vpx 1.15.2、png 1.6.53、webp 1.5.0、OpenH264 2.6.0、opencore-amr 0.1.6、x264 commit `32c3b801191522961102d4bea292cdb61068d0dd`、x265 4.1，以及 NV codec/AMF/libvpl headers。Windows 原生依赖共 19 条已分别取得并匹配该固定脚本的 SHA-256。LAME 源与 MFA 所用归档摘要完全相同，因此复用精确内容另存 build 所期望的名字，未以 Debian 镜像主页替代原始源码证据。

**具体待判事项**：[精确 `ffmpeg.patch`](https://github.com/PyAV-Org/pyav-ffmpeg/blob/9e63121cf6ef9bc1744956c7a4cb772589c80806/patches/ffmpeg.patch) 把 `libx264`/`libx265` 从 `EXTERNAL_LIBRARY_GPL_LIST` 移到 `EXTERNAL_LIBRARY_VERSION3_LIST`。该代码变更解释了启用两库而仍报告 LGPLv3 的配置。它没有自动证明 x264/x265 的独立许可权利已经改变。两个版本源码与其 COPYING 已完整保留，应结合工程最终许可和实际选用的 GPL/商业授权安排核实，不能据 DLL 的单一字符串宣称这些 codec 许可通过。

精确 CI/tag、实际 FFmpeg 版本与 codec 源绑定已经形成。没有重新下载 PyAV 二进制 wheel，没有完成 byte-for-byte 原生重建，因此该 CI 链是对应源码来源证据，不能替代本轮未做的完全可复现重建验证。构建脚本从滚动 MSYS2 安装并复制的 `libgcc_s_seh-1`、`libstdc++`、`libwinpthread`、`libiconv`、zlib 仍需保留各自实际 DLL 的精确二进制版本来源，不能将 MFA 的 15.2.0 GCC recipe 套给它们。

## 模型与词典材料

`model-materials/model-dictionary-identity.json` 已重新计算包中两个模型与两个词典的 SHA-256，并保存原 ZIP 元数据、登记材料。`official-text-manifest.json` 保留官方固定 Git tree 对象及 10 个模型/词典说明与许可证原文。模型许可独立于 MFA 软件 MIT。

| 分发对象 | 本轮已核实身份 | 已取得材料与当前边界 |
| --- | --- | --- |
| `mandarin_mfa.zip` | 包 SHA `bc16c3278295fad0d742399b849e5a8375a65bcbe5b7c7612ff1e3ffb81dcef7`，92,275,957 字节 | 官方 `acoustic-mandarin_mfa-v2.0.0a` release 实际完整下载与包内 SHA-256 一致，已绑定具体版本 CC BY 4.0 原文与归属 |
| 关联汉字词典 | 包 SHA `6b71538ba4cd48d40f92561a129e4531ee256951fcf218ac1198da72ca13076e`，8,709,372 字节 | 官方 `dictionary-mandarin_mfa-v2.0.0a` release 实际完整下载与包内 SHA-256 一致，已绑定具体版本 CC BY 4.0原文与归属 |
| `mandarin.zip` | 包 SHA `beca5c15f6c15bee4435a37bd2904e15b12fcc56342013d253260f31fd2c3f33`，14,709,255 字节，归档内 version 1.0.0 | 对本地完整 ZIP 计算 Git blob SHA-1，**73ef69640583290af61e4a1d85646d3800901b9f**，与官方仓库 `acoustic/mandarin.zip` 对象完全一致。官方 v1 release 不存在，尚未取得直接涵盖此历史 ZIP 的明确许可原文，不能套 v2 模型卡 |
| `mandarin_pinyin_tab.dict` | 包 SHA `c6a9b62905917be3f21a9138503ccf9a5fb1ca080d5a7eed75c8a662c80770cb`，25,750 字节，2,003 行 | 与官方旧 `dictionary/mandarin_pinyin.dict` 的全部行仅空白格式不同。与明确 CC BY 4.0 的 v2.0.0 内容逐条相同，新版仅多 `<unk> spn` 一行，已保存原文/许可/比对。可准备署名、来源、格式改写及缺行说明，不需泛化成新的作者联系需求 |

官方材料来自 [MFA mandarin v2.0.0a 模型 release](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-mandarin_mfa-v2.0.0a)、[对应词典 release](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/dictionary-mandarin_mfa-v2.0.0a)、[历史 mandarin ZIP](https://github.com/MontrealCorpusTools/mfa-models/blob/d6eff86a42c6a90b641e17dfdf7a16555b934483/acoustic/mandarin.zip)、[拼音词典 v2.0.0 明确许可材料](https://github.com/MontrealCorpusTools/mfa-models/tree/d6eff86a42c6a90b641e17dfdf7a16555b934483/dictionary/mandarin/pinyin/v2.0.0)。当前固定 Git tree 为 `d6eff86a42c6a90b641e17dfdf7a16555b934483`。不拿现行 Hugging Face 3.3.0 模型卡替代这些旧分发对象。

两个官方旧 release 没有公布 SHA-256。本轮的通过依据为从明确官方 release 下载全部实际字节，计算 SHA-256 后与当前包一致，记录于 `official-model-byte-comparison.json`，没有把本地摘要写成上游曾公布的摘要。拼音词典的 2,003 行逐项等于明确 CC BY v2 词典去掉 `<unk> spn` 后的内容，严格比对记录为 `pinyin-licensed-subset-proof.json`。既有 MediaPipe、鼻腔、OpenSauce 项目来源材料只读副本与其摘要亦存于 `model-materials/existing-project-evidence/`，没有重新改判它们。

具体模型/词典可复用归属材料另存为 `model-materials/BOUND-NOTICES.md`。原始 README 中的作者及 BibTeX 保留，不用软件作者条目替换模型/词典作者。

## 还需具体完成的项

1. 明确历史 `mandarin.zip` 1.0.0 ZIP 的许可依据。其精确官方身份已解决，但 v2 许可文件不能自动替代这个具体老模型。
2. 按实际最终许可方案处理 M05 x264/x265 与 FFmpeg patch、滚动 MSYS2 库的精确来源，保留 Qt/PyQt wheel 编译配置和需要的补丁证据。
3. 工程总 LICENSE 与 `func_getSoE.m` 具体复用许可按井井回复处理。这两项没有被本轮代替确认。
4. 最终形成同发行渠道可取得的源码与构建/补丁材料、实际第三方 notices 与许可、必要的替换或重新链接说明。独立 Temp 已备好的源码不自动等于已随 app 或服务器交付。

VTL 2.4 API 原始源 ZIP、项目适配源、构建材料原已随包，不重复列为缺源。自有 Python 源快照与完整 native 对应源码分别记录。已有科学方法论文引用、载瓦语邮件授权、鼻腔/VoQS 明确 CC BY来源、自有 EGG/TextGrid/字典/IPA历史工作不新增本轮阻断。本报告不对内容作者章节作新的审计。

## 独立源码准备归档

本轮另按主代理授权制作 `C:\Users\13680\AppData\Local\Temp\PTB-Preview1-20261005-source-preparation-archive\PhoneticToolbox-Preview1-20261005-source-preparation.zip`。其名称和 README 明确这是源码准备归档，没有宣称完整应用对应源码已经交付。

归档包含 60 个已通过摘要的源码归档、193 包精确 Conda 配方/补丁/元数据、固定 PyAV build 源文件、许可证抽取、模型身份/比对及 CC BY材料、机器清单与本报告快照。排除 `.part`、分段 `qt-ranges`、下载器、运行日志等下载缓存。获准下载的两个完整模型/词典仅用于比对，保留在原 Temp，源码准备 ZIP 只收它们的摘要证据和来源/许可文本，不将它们冒充模型训练源码。

ZIP 的实际文件清单、归档摘要及验证结果保存在上述新目录的外置 sidecar，避免在 ZIP 自身内嵌自己的摘要。原始准备材料和所有下载分段均未删除。此归档没有上传到服务器或 GitHub，也不修改 app 成品。

## 机器清单及后续更新

快照与源码目录只在本轮授权的独立 Temp 内写入。实际清单如下，完整摘要全部通过，早期部分下载与分段也保持原样保留。目录仍保留构建工具链精确来源/最终许可方案的上述边界，没有将全项目公开许可写为通过。

<!-- SOURCE_STATUS_TABLE -->

快照：`2026-10-05T22:22:56.095397+08:00`。源码绑定通过 **61/61**，不同完整源码归档 **60**。模型/词典官方内容比对通过 **2/2**。

| 对象/版本 | 本地归档 | 状态/字节 | 实际 SHA-256 | 精确取得 URL |
| --- | --- | --- | --- | --- |
| PyQt6 6.11.0 | `archives/pyqt6-6.11.0.tar.gz` | verified / 1087430 | `45dd60aa69976de1918b5ced6b4e7b6a25abd2a919ecef5fd5826ecc76718889` | [pyqt6-6.11.0.tar.gz](https://files.pythonhosted.org/packages/8b/47/b25c13eca5bebc6505394d0223e46d7ebf0c57dcac2ed908d7d19b18ab6b/pyqt6-6.11.0.tar.gz) |
| PyQt6-WebEngine 6.11.0 | `archives/pyqt6_webengine-6.11.0.tar.gz` | verified / 37331 | `15cf49efbbbd4c6bc87653b2c4ae80d6049f800e31620b336734ae2e37cbedae` | [pyqt6_webengine-6.11.0.tar.gz](https://files.pythonhosted.org/packages/9a/c6/b4f777c46ff42a759180dc65ad49a207748ea2e83ac4df21e89eaf4834c3/pyqt6_webengine-6.11.0.tar.gz) |
| PyQt6_sip 13.12.0 | `archives/pyqt6_sip-13.12.0.tar.gz` | verified / 93979 | `a7ad45c1e3cec3a2473d37ea9870b6c3baeccc560298623c8eb59265714c06e2` | [pyqt6_sip-13.12.0.tar.gz](https://files.pythonhosted.org/packages/d1/23/16c583dbb6b53e0494dfcf7d1a44778c82e324edda62326727b37f1a5b34/pyqt6_sip-13.12.0.tar.gz) |
| praat-parselmouth 0.4.7 | `archives/praat_parselmouth-0.4.7.tar.gz` | verified / 22526491 | `6dd81d246ce1eef5fd93d8cbdaf1bef61ca40ef1d2fc12aa23996a28071181e6` | [praat_parselmouth-0.4.7.tar.gz](https://files.pythonhosted.org/packages/4f/28/2c1204fe3e7aeb6942051ff6776e31da52c7ab5b7df2ca438f371bd60d8a/praat_parselmouth-0.4.7.tar.gz) |
| PyQt6 6.6.1 | `archives/PyQt6-6.6.1.tar.gz` | verified / 1043203 | `9f158aa29d205142c56f0f35d07784b8df0be28378d20a97bcda8bd64ffd0379` | [PyQt6-6.6.1.tar.gz](https://files.pythonhosted.org/packages/8c/2b/6fe0409501798abc780a70cab48c39599742ab5a8168e682107eaab78fca/PyQt6-6.6.1.tar.gz) |
| PyQt6_sip 13.10.3 | `archives/pyqt6_sip-13.10.3.tar.gz` | verified / 92621 | `630895b3827e2c3b4e072089157985691fe4210d64340e71141f93775ea4ae51` | [pyqt6_sip-13.10.3.tar.gz](https://files.pythonhosted.org/packages/0d/e9/d1b97154cec1d6c8a3d93fb6565d1463bc528fa5103491d626d07a451c7c/pyqt6_sip-13.10.3.tar.gz) |
| av 16.1.0 | `archives/av-16.1.0.tar.gz` | verified / 4285203 | `a094b4fd87a3721dacf02794d3d2c82b8d712c85b9534437e82a8a978c175ffd` | [av-16.1.0.tar.gz](https://files.pythonhosted.org/packages/78/cd/3a83ffbc3cc25b39721d174487fb0d51a76582f4a1703f98e46170ce83d4/av-16.1.0.tar.gz) |
| Qt including QtWebEngine/Chromium 6.11.2 | `archives/qt-everywhere-src-6.11.2.tar.xz` | verified / 1019661552 | `6dcfbca271d76a6502741a2c0dc6fc98ef7dd0b7b4cfd0abcebb285a86a26f33` | [qt-everywhere-src-6.11.2.tar.xz](https://download.qt.io/archive/qt/6.11/6.11.2/single/qt-everywhere-src-6.11.2.tar.xz) |
| Qt including QtWebEngine/Chromium 6.6.1 | `archives/qt-everywhere-src-6.6.1.tar.xz` | verified / 814132652 | `dd3668f65645fe270bc615d748bd4dc048bd17b9dc297025106e6ecc419ab95d` | [qt-everywhere-src-6.6.1.tar.xz](https://download.qt.io/archive/qt/6.6/6.6.1/single/qt-everywhere-src-6.6.1.tar.xz) |
| cairo 1.18.4 | `archives/cairo-1.18.4.tar.xz` | verified / 32578804 | `445ed8208a6e4823de1226a74ca319d3600e83f6369f99b14265006599c32ccb` | [cairo-1.18.4.tar.xz](https://cairographics.org/releases/cairo-1.18.4.tar.xz) |
| dataclassy 1.0.1 | `archives/dataclassy-1.0.1.tar.gz` | verified / 28786 | `e5a08a304f5f31b35983d3d14e60f00240fbcb38b4aea49db7d19160958b9a3d` | [dataclassy-1.0.1.tar.gz](https://pypi.io/packages/source/d/dataclassy/dataclassy-1.0.1.tar.gz) |
| ffmpeg 8.0.0 | `archives/ffmpeg-8.0.tar.gz` | verified / 17183045 | `cce1136d38c389e6baaa452d6babc384cb2d3a9406ebe48c36a48f3ee115d8df` | [ffmpeg-8.0.tar.gz](https://ffmpeg.org/releases/ffmpeg-8.0.tar.gz) |
| fribidi 1.0.16 | `archives/fribidi-1.0.16.tar.xz` | verified / 1098260 | `1b1cde5b235d40479e91be2f0e88a309e3214c8ab470ec8a2744d82a5a9ea05c` | [fribidi-1.0.16.tar.xz](https://github.com/fribidi/fribidi/releases/download/v1.0.16/fribidi-1.0.16.tar.xz) |
| gdk-pixbuf 2.44.4 | `archives/gdk-pixbuf-2.44.4.tar.xz` | verified / 6541244 | `93a1aac3f1427ae73457397582a2c38d049638a801788ccbd5f48ca607bdbd17` | [gdk-pixbuf-2.44.4.tar.xz](https://ftp.gnome.org/pub/gnome/sources/gdk-pixbuf/2.44/gdk-pixbuf-2.44.4.tar.xz) |
| graphite2 1.3.14 | `archives/graphite2-1.3.14.tgz` | verified / 6630061 | `f99d1c13aa5fa296898a181dff9b82fb25f6cc0933dbaa7a475d8109bd54209d` | [graphite2-1.3.14.tgz](https://github.com/silnrsi/graphite/releases/download/1.3.14/graphite2-1.3.14.tgz) |
| gts 0.7.6 | `archives/gts-0.7.6.tar.gz` | verified / 948847 | `059c3e13e3e3b796d775ec9f96abdce8f2b3b5144df8514eda0cc12e13e8b81e` | [gts-0.7.6.tar.gz](https://downloads.sourceforge.net/gts/gts-0.7.6.tar.gz) |
| lame 3.100 | `archives/lame-3.100.tar.gz` | verified / 1524133 | `ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e` | [lame-3.100.tar.gz](https://downloads.sourceforge.net/sourceforge/lame/lame-3.100.tar.gz) |
| libglib 2.86.3 | `archives/glib-2.86.3.tar.xz` | verified / 5674820 | `b3211d8d34b9df5dca05787ef0ad5d7ca75dec998b970e1aab0001d229977c65` | [glib-2.86.3.tar.xz](https://download.gnome.org/sources/glib/2.86/glib-2.86.3.tar.xz) |
| libiconv 1.18 | `archives/libiconv-1.18.tar.gz` | verified / 5822590 | `3b08f5f4f9b4eb82f151a7040bfd6fe6c6fb922efe4b1659c66ea933276965e8` | [libiconv-1.18.tar.gz](https://ftp.gnu.org/pub/gnu/libiconv/libiconv-1.18.tar.gz) |
| libintl 0.22.5 | `archives/gettext-0.22.5.tar.xz` | verified / 10270724 | `fe10c37353213d78a5b83d48af231e005c4da84db5ce88037d88355938259640` | [gettext-0.22.5.tar.xz](https://ftp.gnu.org/pub/gnu/gettext/gettext-0.22.5.tar.xz) |
| libmad 0.15.1b | `archives/libmad-0.15.1b.tar.gz` | verified / 502379 | `bbfac3ed6bfbc2823d3775ebb931087371e142bb0e9bb1bee51a76a6e0078690` | [libmad-0.15.1b.tar.gz](https://downloads.sourceforge.net/project/mad/libmad/0.15.1b/libmad-0.15.1b.tar.gz) |
| librsvg 2.60.0 | `archives/librsvg-2.60.0.tar.xz` | verified / 6742880 | `0b6ffccdf6e70afc9876882f5d2ce9ffcf2c713cbaaf1ad90170daa752e1eec3` | [librsvg-2.60.0.tar.xz](https://download.gnome.org/sources/librsvg/2.60/librsvg-2.60.0.tar.xz) |
| libsndfile 1.2.2 | `archives/libsndfile-1.2.2.tar.xz` | verified / 730760 | `3799ca9924d3125038880367bf1468e53a1b7e3686a934f098b7e1d286cdb80e` | [libsndfile-1.2.2.tar.xz](https://github.com/libsndfile/libsndfile/releases/download/1.2.2/libsndfile-1.2.2.tar.xz) |
| mpg123 1.32.9 | `archives/download` | verified / 1118388 | `03b61e4004e960bacf2acdada03ed94d376e6aab27a601447bd4908d8407b291` | [download](https://sourceforge.net/projects/mpg123/files/mpg123/1.32.9/mpg123-1.32.9.tar.bz2/download) |
| pango 1.56.4 | `archives/pango-1.56.4.tar.xz` | verified / 1883988 | `17065e2fcc5f5a5bdbffc884c956bfc7c451a96e8c4fb2f8ad837c6413cb5a01` | [pango-1.56.4.tar.xz](https://download.gnome.org/sources/pango/1.56/pango-1.56.4.tar.xz) |
| psycopg2 2.9.9 | `archives/psycopg2-2.9.9.tar.gz` | verified / 384926 | `d1454bde93fb1e224166811694d600e746430c006fbb031ea06ecc2ea41bf156` | [psycopg2-2.9.9.tar.gz](https://pypi.io/packages/source/p/psycopg2/psycopg2-2.9.9.tar.gz) |
| sox 14.4.2 | `archives/sox-14.4.2.tar.gz` | verified / 1134299 | `b45f598643ffbd8e363ff24d61166ccec4836fea6d3888881b8df53e3bb55f6c` | [sox-14.4.2.tar.gz](https://sourceforge.net/projects/sox/files/sox/14.4.2/sox-14.4.2.tar.gz) |
| soxr 0.1.3 | `archives/soxr-0.1.3-Source.tar.xz` | verified / 94384 | `b111c15fdc8c029989330ff559184198c161100a59312f5dc19ddeb9b5a15889` | [soxr-0.1.3-Source.tar.xz](https://downloads.sourceforge.net/project/soxr/soxr-0.1.3-Source.tar.xz) |
| soxr-python 1.0.0 | `archives/soxr-1.0.0.tar.gz` | verified / 171415 | `e07ee6c1d659bc6957034f4800c60cb8b98de798823e34d2a2bba1caa85a4509` | [soxr-1.0.0.tar.gz](https://pypi.org/packages/source/s/soxr/soxr-1.0.0.tar.gz) |
| tqdm 4.67.1 | `archives/tqdm-4.67.1.tar.gz` | verified / 169737 | `f8aef9c52c08c13a65f30ea34f4e5aac3fd1a34959879d7e59e63027286627f2` | [tqdm-4.67.1.tar.gz](https://pypi.org/packages/source/t/tqdm/tqdm-4.67.1.tar.gz) |
| x264 1!164.3095 | `archives/x264-baee400fa9ced6f5481a728138fed6e867b0ff7f.tar.gz` | verified / 942829 | `436a2be54d8bc0cb05dd33ecbbcb7df9c3b57362714fcdaa3a5991189a33319b` | [x264-baee400fa9ced6f5481a728138fed6e867b0ff7f.tar.gz](https://code.videolan.org/videolan/x264/-/archive/baee400fa9ced6f5481a728138fed6e867b0ff7f/x264-baee400fa9ced6f5481a728138fed6e867b0ff7f.tar.gz) |
| x265 3.5 | `archives/x265_3.5.tar.gz` | verified / 1537044 | `e70a3335cacacbba0b3a20ec6fecd6783932288ebc8163ad74bcc9606477cae8` | [x265_3.5.tar.gz](https://bitbucket.org/multicoreware/x265_git/downloads/x265_3.5.tar.gz) |
| aom 3.9.1 | `archives/libaom-3.9.1.tar.gz` | verified / 5524048 | `dba99fc1c28aaade28dda59821166b2fa91c06162d1bc99fde0ddaad7cecc50e` | [libaom-3.9.1.tar.gz](https://storage.googleapis.com/aom-releases/libaom-3.9.1.tar.gz) |
| dav1d 1.2.1 | `archives/dav1d-1.2.1.tar.gz` | verified / 1477079 | `2dd85860d213479672b1c708e31593446e8c2b53ff41e2ca25a2eafb718424e2` | [dav1d-1.2.1.tar.gz](https://code.videolan.org/videolan/dav1d/-/archive/1.2.1/dav1d-1.2.1.tar.gz) |
| freetype 2.14.1 | `archives/freetype-2.14.1.tar.gz` | verified / 4135293 | `174d9e53402e1bf9ec7277e22ec199ba3e55a6be2c0740cb18c0ee9850fc8c34` | [freetype-2.14.1.tar.gz](https://download.savannah.gnu.org/releases/freetype/freetype-2.14.1.tar.gz) |
| getopt-win32 0.1 | `archives/0.1.tar.gz` | verified / 12479 | `7e9653ecd58ce4149959bf6a905f4ab2f7889856fe1218afbf84284074f9e549` | [0.1.tar.gz](https://github.com/libimobiledevice-win32/getopt/archive/0.1.tar.gz) |
| libogg 1.3.5 | `archives/libogg-1.3.5.tar.gz` | verified / 593071 | `0eb4b4b9420a0f51db142ba3f9c64b333f826532dc0f48c6410ae51f4799b664` | [libogg-1.3.5.tar.gz](https://downloads.xiph.org/releases/ogg/libogg-1.3.5.tar.gz) |
| libopus 1.5.2 | `archives/libopus-1.5.2.tar.gz` | verified / 4183352 | `9480e329e989f70d69886ded470c7f8cfe6c0667cc4196d4837ac9e668fb7404` | [libopus-1.5.2.tar.gz](https://github.com/xiph/opus/archive/v1.5.2.tar.gz) |
| libusb 1.0.29 | `archives/libusb-1.0.29.tar.bz2` | verified / 645381 | `5977fc950f8d1395ccea9bd48c06b3f808fd3c2c961b44b0c2e6e29fc3a70a85` | [libusb-1.0.29.tar.bz2](https://github.com/libusb/libusb/releases/download/v1.0.29/libusb-1.0.29.tar.bz2) |
| libvorbis 1.3.7 | `archives/libvorbis-1.3.7.tar.gz` | verified / 1234573 | `270c76933d0934e42c5ee0a54a36280e2d87af1de3cc3e584806357e237afd13` | [libvorbis-1.3.7.tar.gz](https://github.com/xiph/vorbis/archive/v1.3.7.tar.gz) |
| openh264 2.6.0 | `archives/openh264-2.6.0.tar.gz` | verified / 60302243 | `558544ad358283a7ab2930d69a9ceddf913f4a51ee9bf1bfb9e377322af81a69` | [openh264-2.6.0.tar.gz](https://github.com/cisco/openh264/archive/v2.6.0.tar.gz) |
| svt-av1 3.1.2 | `archives/SVT-AV1-v3.1.2.tar.gz` | verified / 10909754 | `d0d73bfea42fdcc1222272bf2b0e2319e9df5574721298090c3d28315586ecb1` | [SVT-AV1-v3.1.2.tar.gz](https://gitlab.com/AOMediaCodec/SVT-AV1/-/archive/v3.1.2/SVT-AV1-v3.1.2.tar.gz) |
| nv-codec-headers | `archives/n13.0.19.0.tar.gz` | verified / 83385 | `86d15d1a7c0ac73a0eafdfc57bebfeba7da8264595bf531cf4d8db1c22940116` | [n13.0.19.0.tar.gz](https://github.com/FFmpeg/nv-codec-headers/archive/refs/tags/n13.0.19.0.tar.gz) |
| amf-headers | `archives/AMF-headers-v1.5.0.tar.gz` | verified / 82755 | `d569647fa26f289affe81a206259fa92f819d06db1e80cc334559953e82a3f01` | [AMF-headers-v1.5.0.tar.gz](https://github.com/GPUOpen-LibrariesAndSDKs/AMF/releases/download/v1.5.0/AMF-headers-v1.5.0.tar.gz) |
| libvpl | `archives/v2.16.0.tar.gz` | verified / 12968334 | `d60931937426130ddad9f1975c010543f0da99e67edb1c6070656b7947f633b6` | [v2.16.0.tar.gz](https://github.com/intel/libvpl/archive/refs/tags/v2.16.0.tar.gz) |
| ffmpeg | `archives/ffmpeg-8.0.1.tar.xz` | verified / 11388848 | `05ee0b03119b45c0bdb4df654b96802e909e0a752f72e4fe3794f487229e5a41` | [ffmpeg-8.0.1.tar.xz](https://ffmpeg.org/releases/ffmpeg-8.0.1.tar.xz) |
| lame | `archives/lame_3.100.orig.tar.gz` | verified / 1524133 | `ddfe36cab873794038ae2c1210557ad34857a4b6bdc515785d1da9e175b1da1e` | [lame_3.100.orig.tar.gz](https://deb.debian.org/debian/pool/main/l/lame/lame_3.100.orig.tar.gz) |
| ogg | `archives/libogg-1.3.6.tar.gz` | verified / 604469 | `83e6704730683d004d20e21b8f7f55dcb3383cdf84c0daedf30bde175f774638` | [libogg-1.3.6.tar.gz](https://downloads.xiph.org/releases/ogg/libogg-1.3.6.tar.gz) |
| opus | `archives/opus-1.6.tar.gz` | verified / 36317446 | `b7637334527201fdfd6dd6a02e67aceffb0e5e60155bbd89175647a80301c92c` | [opus-1.6.tar.gz](https://ftp.osuosl.org/pub/xiph/releases/opus/opus-1.6.tar.gz) |
| speex | `archives/speex-1.2.1.tar.gz` | verified / 1043278 | `4b44d4f2b38a370a2d98a78329fefc56a0cf93d1c1be70029217baae6628feea` | [speex-1.2.1.tar.gz](https://downloads.xiph.org/releases/speex/speex-1.2.1.tar.gz) |
| vorbis | `archives/libvorbis-1.3.7.tar.xz` | verified / 1203792 | `b33cc4934322bcbf6efcbacf49e3ca01aadbea4114ec9589d1b1e9d20f72954b` | [libvorbis-1.3.7.tar.xz](https://ftp.osuosl.org/pub/xiph/releases/vorbis/libvorbis-1.3.7.tar.xz) |
| aom | `archives/libaom-3.13.1.tar.gz` | verified / 6253958 | `19e45a5a7192d690565229983dad900e76b513a02306c12053fb9a262cbeca7d` | [libaom-3.13.1.tar.gz](https://storage.googleapis.com/aom-releases/libaom-3.13.1.tar.gz) |
| dav1d | `archives/dav1d-1.5.3.tar.bz2` | verified / 1217030 | `e099f53253f6c247580c554d53a13f1040638f2066edc3c740e4c2f15174ce22` | [dav1d-1.5.3.tar.bz2](https://code.videolan.org/videolan/dav1d/-/archive/1.5.3/dav1d-1.5.3.tar.bz2) |
| libsvtav1 | `archives/SVT-AV1-v3.1.2.tar.bz2` | verified / 10203273 | `802e9bb2b14f66e8c638f54857ccb84d3536144b0ae18b9f568bbf2314d2de88` | [SVT-AV1-v3.1.2.tar.bz2](https://gitlab.com/AOMediaCodec/SVT-AV1/-/archive/v3.1.2/SVT-AV1-v3.1.2.tar.bz2) |
| vpx | `archives/vpx-1.15.2.tar.gz` | verified / 5630368 | `26fcd3db88045dee380e581862a6ef106f49b74b6396ee95c2993a260b4636aa` | [vpx-1.15.2.tar.gz](https://github.com/webmproject/libvpx/archive/refs/tags/v1.15.2.tar.gz) |
| png | `archives/libpng-1.6.53.tar.xz` | verified / 1063432 | `1d3fb8ccc2932d04aa3663e22ef5ef490244370f4e568d7850165068778d98d4` | [libpng-1.6.53.tar.xz](https://downloads.sourceforge.net/project/libpng/libpng16/1.6.53/libpng-1.6.53.tar.xz) |
| webp | `archives/webp-1.5.0.tar.gz` | verified / 3821241 | `668c9aba45565e24c27e17f7aaf7060a399f7f31dba6c97a044e1feacb930f37` | [webp-1.5.0.tar.gz](https://github.com/webmproject/libwebp/archive/refs/tags/v1.5.0.tar.gz) |
| openh264 | `archives/openh264-2.6.0.tar.gz` | verified / 60302243 | `558544ad358283a7ab2930d69a9ceddf913f4a51ee9bf1bfb9e377322af81a69` | [openh264-2.6.0.tar.gz](https://github.com/cisco/openh264/archive/refs/tags/v2.6.0.tar.gz) |
| opencore-amr | `archives/opencore-amr-0.1.6.tar.gz` | verified / 939179 | `483eb4061088e2b34b358e47540b5d495a96cd468e361050fae615b1809dc4a1` | [opencore-amr-0.1.6.tar.gz](https://downloads.sourceforge.net/project/opencore-amr/opencore-amr/opencore-amr-0.1.6.tar.gz) |
| x264 | `archives/x264-32c3b801191522961102d4bea292cdb61068d0dd.tar.bz2` | verified / 845986 | `d7748f350127cea138ad97479c385c9a35a6f8527bc6ef7a52236777cf30b839` | [x264-32c3b801191522961102d4bea292cdb61068d0dd.tar.bz2](https://code.videolan.org/videolan/x264/-/archive/32c3b801191522961102d4bea292cdb61068d0dd/x264-32c3b801191522961102d4bea292cdb61068d0dd.tar.bz2) |
| x265 | `archives/x265_4.1.tar.gz` | verified / 1725279 | `a31699c6a89806b74b0151e5e6a7df65de4b49050482fe5ebf8a4379d7af8f29` | [x265_4.1.tar.gz](https://bitbucket.org/multicoreware/x265_git/downloads/x265_4.1.tar.gz) |
| MFA Mandarin model 2.0.0a | `archives/mandarin_mfa-v2.0.0a.zip` | verified / 92275957 | `bc16c3278295fad0d742399b849e5a8375a65bcbe5b7c7612ff1e3ffb81dcef7` | [mandarin_mfa-v2.0.0a.zip](https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-mandarin_mfa-v2.0.0a/mandarin_mfa.zip) |
| MFA Mandarin dictionary 2.0.0a | `archives/mandarin_mfa-v2.0.0a.dict` | verified / 8709372 | `6b71538ba4cd48d40f92561a129e4531ee256951fcf218ac1198da72ca13076e` | [mandarin_mfa-v2.0.0a.dict](https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-mandarin_mfa-v2.0.0a/mandarin_mfa.dict) |

<!-- SOURCE_STATUS_TABLE_END -->

## 当前应用源码准备包（2026-10-05，独立本机成果）

按追加授权，以最终 portable r2 清单和 Final2 随包原始项目源码准备应用源码 ZIP。未修改工程代码、现有许可、任何当前软件成品或运行环境，未上传。本成果须与上文的上游来源准备 ZIP 配套查阅，不声明完整对应源码/GPL 验收或公开发行许可通过。

文件：`C:\Users\13680\AppData\Local\Temp\PTB-Preview1-20261005-application-source-preparation\PhoneticToolbox-Preview1-20261005-application-source-preparation.zip`。

- ZIP 大小 **20,841,933 字节**，SHA-256 **`bb7499007d4daeaf63016bc9ff7ef365618e40002d94e2f9ec33ac27d767a76b`**。
- **3,464 个成员**全部通过 CRC 和逐文件 SHA-256 解压回读，展开 72,337,841 字节。外部 `archive-files.json` 覆盖全部成员，`archive-member-crc.json` 记录每个 CRC，`archive-validation.json` 保存最终校验结果。
- Final2 的 **350 个包内项目文件**均与 portable r2 原始清单 SHA-256 一致。332 个源码/科学查表/字体许可等文件及 17 个发行元数据保留，1 个 pyc 缓存排除。其中 **319 个 Python 文件**也与当前工程一致。portable r2 ZIP SHA-256 为 `73ef291f928b935dfba2859350b0a8a10201f9862dec5010c979b01f318291ad`。
- Vue 优选源码在独立 C Temp 目录使用现有 Node 24.13.0/锁定依赖复建，**259/259 前端文件**逐 SHA-256 等于 portable r2，缺失、差异、额外均为 0。复建所需媒体仅恢复到 ZIP 外的验证目录。未安装依赖或改动工程；前端配置只在验证副本改变缓存/输出位置。
- **19 个说明书章节**完整保留，包括开始页、设置及 M10 预留章。严格校验 0 错误、2 项内容状态警告。原稿 project.json 未改为 public。现有 `reading_project(result, software)` 生成的清单摘要 `337313eccf54cc002d78b8ab9870e546cf72d4ed75c5a717de148c0311688a3a` 与 portable r2 完全一致，原始差异来自生成的 distribution、sections、searchIndex。

包内同时保留 Vue/Tiptap、普通应用阅读器与 schema、M17 与说明书作者编辑器、当前版本锁、构建/打包脚本、原生 VTL/C++ 桥接源码和补丁、通知及来源材料。最新版 release README 的唯一时间戳输出名称说明已收录。根旧 2.2 的 pyproject/run 仅作为历史元数据放入 provenance，其中 MIT 字段不作为本轮总 LICENSE 决定。

EGG 运行时另有 15 个项目 Python 原稿，单独按实际字节保留。12 个与主快照相同，export_series.py/f0.py/model.py 为较早的已安装 wheel 文件。现有 egg_bootstrap 固定调用 `use_matching_core_source` 前置同级主快照核心路径，已执行该纯路径选择函数核对，不运行科研任务，也不用旧 wheel 文件覆盖主源码。三者的不同身份写入 SOURCE-BINDINGS，不能合并宣称所有存量文件相同。

隐私过滤排除 .env/私钥等凭据文件、.git/.venv/node_modules、用户语料、全部截图/音视频/软件专属媒体、local-data/output/应用 dist、测试缓存及下载分段。选入文本和已知 VTL 源 ZIP 文本成员已扫描明显密钥模式。初步 **9 条命中均为安全误报**，分别是动态 Vue 令牌绑定和认证、URL 拒绝、TLS、模拟 API 测试常量，逐项核查后全部恢复必要源码。最终实际凭据排除 **0 条**，候选值原文没有输出。EXCLUSIONS 保留排除范围及误报复核。实际包许可证树中 Rich 的原文历史路径 dist/LICENSE 只保留许可证文档，不是应用成品目录。

README 明确媒体通过获准的软件包取得，作者编辑器与普通应用入口分开标明。本包不含未来的主线报告或服务器回执。当前报告的上述新增小节位于 ZIP 外，ZIP 内保留本轮冻结前的原报告及独立准备 README、证明文件。总 LICENSE、SoE、历史 mandarin v1 模型许可、M05 MSYS2 精确身份、wheel/native 构建与最终对应源码交付安排仍按上文具体缺口保留。所有原始材料及旧记录均未删除。

## 主线私有交付记录

前文未上传描述源码准备子任务结束时的状态。主线随后已将两个原样源码准备 ZIP 上传到服务器的任务专属私有目录，并复制到桌面 Preview 交付目录。最终服务器大小、SHA-256 和 0600 权限见主线 `output/manual-work/upload-receipt-20261005-final.json`。未开放公共下载或发布 latest.json，未上传 GitHub。此交付阶段的记录位于归档外，不反写归档中既有快照，也不改变仍待核实的具体许可范围。
