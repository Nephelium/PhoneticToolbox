# 声道工作台第三方来源与修改

本模块随本地 EXE 携带的资源并非全部使用 MIT 许可。主仓库 LICENSE 与下列组件的许可分别保留；这里不对整个组合分发物作仅适用 MIT 的声明。

| 组件 | 来源与版本 | 条款与本项目处理 |
| --- | --- | --- |
| VocalTractLab | Peter Birkholz 等，2.4，https://vocaltractlab.de/download-vocaltractlab/VTL2.4.zip | GPL-3.0-or-later，全文见 VTL-LICENSE.txt；原版 API DLL、JD2/M01/W02 speaker 保留。分析 DLL 是 API DLL 同字节副本，用不同文件名隔离原生状态。 |
| 几何桥接与原生适配 | 本项目的 sources/geometry_bridge.cpp / packages/phonetic_core/src/phonetic_core/vocal_tract/engine.py | GPL-3.0-or-later；当前 M10/3 桥接公开器官表面、截面、鼻腔分支，支持手动舌根、前部实际侧通路拟合、唇宽、扩展舌位与独立声门高低，重新计算管道与传递函数。补丁及修改说明见 sources/M10-build.md。 |
| Three.js / OrbitControls | 0.180.0，https://github.com/mrdoob/three.js/tree/r180 | MIT，全文随网页 vendor/THREE-LICENSE.txt 保留。 |
| 外部头壳 | byzmod3d，3D Human Parts Pack，https://opengameart.org/content/3d-human-parts-pack | CC0；转换为 JSON 并作显示配准，不用于声学计算，也不是 JD2 个体扫描。 |
| 平均鼻腔 | Brüning 等，2020，Healthy nasal cavities – averaged geometry，v4，https://doi.org/10.6084/m9.figshare.9585410.v4 | CC BY 4.0，https://creativecommons.org/licenses/by/4.0/ ；保留全部网格，坐标变换并与 JD2 作参考配准。鼻咽连接和软腭局部显示为本项目重建，未从同一个体 MRI 分割。 |

M10-R11 引入的数据布局版本为 `m10/3`：增加 `sources/m10_r11_patch.py` 中可重建的舌尖/舌叶边界调整，独立下喉部高低和完整闭塞纠正。界面鼻咽连接采用连续位置插值，鼻腔原网格、speaker、原版 API 与 Analysis DLL 保留。新增自由度属于本项目的参数模型扩展，没有据此建立个体医学模型或新的生理测量标准。

M10-R12 通过 `sources/m10_r12_patch.py` 修正舌叶下压与舌尖后卷的原生构造限制，管道计算使用同一修正表面。牙齿仍为原生刚体，显示层 `rigid-contact.mjs` 裁切重建舌腹与封盖中的牙齿体积。声门控制点独立常驻，软腭内部封闭线不作游离组织轮廓绘制。修改继续保留对应源码和上述许可，数据结构版本保持 `m10/3`，见 `sources/M10-build.md`。

M10-R13 通过 `sources/m10_r13_patch.py` 修正下压舌叶的端点切线、辅助切线高度和后卷舌尖采样，使原生尖端形成圆弧。显示软腭背侧延伸到腭咽口前缘，消除组织连接处的假凹口。原始鼻腔网格及声学鼻腔参数保留，原生舌位几何和声管同步重算；仍为参数模型修复，未作医学或卷舌音听辨验证。

`sources.lock.json` 保存原型阶段的下载 URL 与 SHA-256，其相对路径描述原型输入目录；集成后的实际资源清单另见 `bundle-manifest.json`。`sources/VTL2.4-API-source.zip` 包含原版 API C++ 源码、工程和许可；`sources/geometry_bridge.cpp`、`sources/m10_r11_patch.py` 和项目构建脚本包含本地修改。重建方法见 `sources/README.md`。公开再分发仍应保留适用的源码、许可与署名材料，本轮未重打 EXE。

R4 使用现有 Qt WebEngine 的 VP8/Opus 编码能力。API 参考 [W3C WebCodecs](https://www.w3.org/TR/webcodecs/)，本项目容器写入器按 [WebM Container Guidelines](https://www.webmproject.org/docs/container/) 实现，未复制外部封装器代码。Qt/PyQt 及其自带第三方组件仍遵守各自许可。独立验证使用本机 FFmpeg 解码，FFmpeg 可执行文件不作为本功能的运行依赖，也未随本次资源打包。
## 当前原生修改与媒体编码

VTL 2.4 原版压缩包、API DLL 与 speaker 保留。`m10_r14_patch.py` 记录连续舌尖厚度、后侧圆弧接入与表面站位。`m10_r15_patch.py` 及桥接使原生计算可选 65、129、257 纵向截面，当前默认 129。对应构建入口、编译选项及数值变化见 `sources/M10-build.md` 与项目 `docs/testing/2026-10-10-m10-r15-report.md`。这些衍生修改继续采用 GPL-3.0-or-later。

R16 在 R15 后加入 `sources/m10_r16_patch.py`，扩展舌尖侧缘的真实横向构形。几何方法标记为 `m10/r16`，数组布局保持 `m10/3`。气腔按开放域构造并避让牙齿，显示口底附着使用下颌相对参考点。Mills 等 2019（doi:10.1002/ca.23343）和 2020（doi:10.1155/2020/1820978）仅作组织附着解释的文献依据，不提供本模型坐标或代码许可，未复制文章图片或数据。

R17 继续采用上述许可，加入 `sources/m10_r17_patch.py`，将口底参考后移、约束舌叶内凹，并让窄缝几何剖面供声学积分共同使用。方法为 `m10/r17`、交换文件版本 4，数组布局仍为 `m10/3`。0.3 cm 内凹尺度为工程约束，没有新增解剖测量来源。原版 VTL 噪声发生器保持不变，验证入口为项目 `docs/testing/2026-10-12-m10-r17-report.md`。

动作视频默认保存 MP4，复用既有 M05 隔离环境中的 PyAV 16.1.0 和 FFmpeg 库编码 H.264/AAC。PyAV 绑定采用 BSD-3-Clause，FFmpeg 具体二进制与编解码器条款保持单独审阅边界。Qt WebCodecs 的 VP8/Opus 数据只作中间输入，本机 FFmpeg/ffprobe 可执行文件只用于独立验证。来源登记为 `PKG-PYAV`、`SRC-FFMPEG`，本轮未新增全局运行依赖或重打 EXE。


R18 加入 `sources/m10_r18_patch.py`，在原生接触约束前施加平滑局部舌面偏移并导出材料站位，沿用 GPL-3.0-or-later。方法为 `m10/r18`，交换文件版本 5，数组布局保持 `m10/3`。工程参数范围不作为解剖测量，未新增外部素材、模型或依赖。


2026-10-18 当前几何更新为 R20。`m10_r19_patch.py` 与 `m10_r20_patch.py` 分别实现舌部材料站位/连接修正和上下唇独立偏移，`geometry_bridge.cpp` 在原生计算中处理唇齿接触和截面积。`m10_source_perturbation.py` 为声源加入可选种子化 jitter/shimmer，Windows、Wasm 与 Mac 使用相同源码。JD2/W02 原始解剖文件不变，界面新姿态 F0 按用户要求为 120/200 Hz。以上派生代码继续采用 GPL-3.0-or-later；Wasm 候选同时携带对应源码包。当前 Windows 两份 API DLL 字节一致，macOS 因独立 install_name/签名使用 native-pair 收据。可重建步骤与平台验收见 `sources/README.md` 和工程 `docs/testing/m10-shared-controls.md`。
