# 声道工作台第三方来源与修改

本模块随本地 EXE 携带的资源并非全部使用 MIT 许可。主仓库 LICENSE 与下列组件的许可分别保留；这里不对整个组合分发物作仅适用 MIT 的声明。

| 组件 | 来源与版本 | 条款与本项目处理 |
| --- | --- | --- |
| VocalTractLab | Peter Birkholz 等，2.4，https://vocaltractlab.de/download-vocaltractlab/VTL2.4.zip | GPL-3.0-or-later，全文见 VTL-LICENSE.txt；原版 API DLL、JD2/M01/W02 speaker 保留。分析 DLL 是 API DLL 同字节副本，用不同文件名隔离原生状态。 |
| 几何桥接与原生适配 | 本项目的 sources/geometry_bridge.cpp / packages/phonetic_core/src/phonetic_core/vocal_tract/engine.py | GPL-3.0-or-later；M10/2 桥接公开器官表面、截面、鼻腔分支，支持手动舌根、前部实际侧通路拟合和唇宽缩放后重新计算管道与传递函数。显示截面窄缝读数补丁及修改说明见 sources/M10-build.md。 |
| Three.js / OrbitControls | 0.180.0，https://github.com/mrdoob/three.js/tree/r180 | MIT，全文随网页 vendor/THREE-LICENSE.txt 保留。 |
| 外部头壳 | byzmod3d，3D Human Parts Pack，https://opengameart.org/content/3d-human-parts-pack | CC0；转换为 JSON 并作显示配准，不用于声学计算，也不是 JD2 个体扫描。 |
| 平均鼻腔 | Brüning 等，2020，Healthy nasal cavities – averaged geometry，v4，https://doi.org/10.6084/m9.figshare.9585410.v4 | CC BY 4.0，https://creativecommons.org/licenses/by/4.0/ ；保留全部网格，坐标变换并与 JD2 作参考配准。鼻咽连接和软腭局部显示为本项目重建，未从同一个体 MRI 分割。 |

`sources.lock.json` 保存原型阶段的下载 URL 与 SHA-256，其相对路径描述原型输入目录；集成后的实际资源清单另见 `bundle-manifest.json`。`sources/VTL2.4-API-source.zip` 包含原版 API C++ 源码、工程和许可；`sources/geometry_bridge.cpp` 包含本地修改。重建方法见 `sources/README.md`。它们均随 EXE 的资源目录打包；公开再分发仍应保留适用的源码、许可与署名材料。

R4 使用现有 Qt WebEngine 的 VP8/Opus 编码能力。API 参考 [W3C WebCodecs](https://www.w3.org/TR/webcodecs/)，本项目容器写入器按 [WebM Container Guidelines](https://www.webmproject.org/docs/container/) 实现，未复制外部封装器代码。Qt/PyQt 及其自带第三方组件仍遵守各自许可。独立验证使用本机 FFmpeg 解码，FFmpeg 可执行文件不作为本功能的运行依赖，也未随本次资源打包。
