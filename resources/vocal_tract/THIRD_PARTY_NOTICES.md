# 声道工作台第三方来源与修改

本模块随本地 EXE 携带的资源并非全部使用 MIT 许可。主仓库 LICENSE 与下列组件的许可分别保留；这里不对整个组合分发物作仅适用 MIT 的声明。

| 组件 | 来源与版本 | 条款与本项目处理 |
| --- | --- | --- |
| VocalTractLab | Peter Birkholz 等，2.4，https://vocaltractlab.de/download-vocaltractlab/VTL2.4.zip | GPL-3.0-or-later，全文见 VTL-LICENSE.txt；原版 API DLL、JD2/M01/W02 speaker 保留。分析 DLL 是 API DLL 同字节副本，用不同文件名隔离原生状态。 |
| 几何桥接与原生适配 | 本项目的 sources/geometry_bridge.cpp / packages/phonetic_core/src/phonetic_core/vocal_tract/engine.py | GPL-3.0-or-later；当前 M10/3 桥接公开器官表面、截面、鼻腔分支，支持手动舌根、前部实际侧通路拟合、唇宽、扩展舌位与独立声门高低，重新计算管道与传递函数。补丁及修改说明见 sources/M10-build.md。 |
| Three.js / OrbitControls | 0.180.0，https://github.com/mrdoob/three.js/tree/r180 | MIT，全文随网页 vendor/THREE-LICENSE.txt 保留。 |
| 外部头壳 | byzmod3d，3D Human Parts Pack，https://opengameart.org/content/3d-human-parts-pack | CC0；转换为 JSON 并作显示配准，不用于声学计算，也不是 JD2 个体扫描。 |
| 平均鼻腔 | Brüning 等，2020，Healthy nasal cavities – averaged geometry，v4，https://doi.org/10.6084/m9.figshare.9585410.v4 | CC BY 4.0，https://creativecommons.org/licenses/by/4.0/ ；保留全部网格，坐标变换并与 JD2 作参考配准。鼻咽连接和软腭局部显示为本项目重建，未从同一个体 MRI 分割。 |

M10-R11 的当前几何版本为 `m10/3`：增加 `sources/m10_r11_patch.py` 中可重建的舌尖/舌叶边界调整，独立下喉部高低和完整闭塞纠正。界面鼻咽连接采用连续位置插值，鼻腔原网格、speaker、原版 API 与 Analysis DLL 保留。新增自由度属于本项目的参数模型扩展，没有据此建立个体医学模型或新的生理测量标准。

M10-R12 通过 `sources/m10_r12_patch.py` 修正舌叶下压与舌尖后卷的原生构造限制，管道计算使用同一修正表面。牙齿仍为原生刚体，显示层 `rigid-contact.mjs` 裁切重建舌腹与封盖中的牙齿体积。声门控制点独立常驻，软腭内部封闭线不作游离组织轮廓绘制。修改继续保留对应源码和上述许可，数据结构版本保持 `m10/3`，见 `sources/M10-build.md`。

M10-R13 通过 `sources/m10_r13_patch.py` 修正下压舌叶的端点切线、辅助切线高度和后卷舌尖采样，使原生尖端形成圆弧。显示软腭背侧延伸到腭咽口前缘，消除组织连接处的假凹口。原始鼻腔网格及声学鼻腔参数保留，原生舌位几何和声管同步重算；仍为参数模型修复，未作医学或卷舌音听辨验证。

`sources.lock.json` 保存原型阶段的下载 URL 与 SHA-256，其相对路径描述原型输入目录；集成后的实际资源清单另见 `bundle-manifest.json`。`sources/VTL2.4-API-source.zip` 包含原版 API C++ 源码、工程和许可；`sources/geometry_bridge.cpp`、`sources/m10_r11_patch.py` 和项目构建脚本包含本地修改。重建方法见 `sources/README.md`。公开再分发仍应保留适用的源码、许可与署名材料，本轮未重打 EXE。

R4 使用现有 Qt WebEngine 的 VP8/Opus 编码能力。API 参考 [W3C WebCodecs](https://www.w3.org/TR/webcodecs/)，本项目容器写入器按 [WebM Container Guidelines](https://www.webmproject.org/docs/container/) 实现，未复制外部封装器代码。Qt/PyQt 及其自带第三方组件仍遵守各自许可。独立验证使用本机 FFmpeg 解码，FFmpeg 可执行文件不作为本功能的运行依赖，也未随本次资源打包。
# M10-R14 修改说明（2026-10-07）

VTL 2.4 原版压缩包、API DLL 与 speaker 保留。`m10_r14_patch.py` 记录连续舌尖厚度、后侧圆弧接入与表面站位；桥接新增站位/截面数量查询，默认 257 纵向截面。对应构建入口、编译选项及数值变化见 `sources/M10-build.md` 与项目 `docs/testing/2026-10-07-m10-r14-report.md`。这些衍生修改继续采用 GPL-3.0-or-later。

