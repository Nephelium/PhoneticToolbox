# M10/2 对应源码说明

原版 VTL 2.4 C++ 保留在 `VTL2.4-API-source.zip`，SHA-256 为 `aa696bf4cf5c44cbd433304943aebb241e5ea3064b949daf5ea09f96567cff9b`。原版 API 与 Analysis DLL 未改动。

`geometry_bridge.cpp` 是实际桥接源码，增加手动舌根及前部侧缘实际拟合。拟合强度随 TS3 从 -0.15 至 -1.0 增大，后部不变，前部中央最多抬高 0.08 cm，左右最外缘最多降低 0.60 cm，然后重新计算中线与截面。默认启用，`p3_lateral_fit(0)` 仅供基准对照。

`M10-geometric-readout.patch` 对原版 `VocalTract.h/cpp` 新增默认关闭的 geometric 参数。仅 `p3_profile` 的显示读数启用 1e-6 cm 阈值，避免窄缝被原生 0.1/0.01 cm 显示过滤抹去；默认的声学管道计算阈值不变。该变化与实际侧缘拟合分别记录。

项目中的 `scripts/build_m10_geometry.py` 从原压缩包提取必要源文件，逐处断言匹配后应用同一补丁，用已有 MSVC x64 工具链编译桥接。不要直接套用旧原型构建命令。完整目录重建步骤见 README.md；这些修改继续适用 GPL-3.0-or-later。
