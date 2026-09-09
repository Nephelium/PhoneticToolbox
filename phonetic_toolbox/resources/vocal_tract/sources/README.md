# 原生资源重建

1. 解压本目录 `VTL2.4-API-source.zip`，得到 `Developer/Sources` 及 Visual Studio 工程。VocalTractLabApi.dll 可由原版工程的 x64 Release 配置重建；已有分发 DLL 的 SHA-256 见上级清单。
2. 在已有 MSVC x64 Developer PowerShell 中创建独立构建目录。以 C++14、`/LD /O2 /EHsc /DWIN32 /D_USE_MATH_DEFINES /MD` 编译本目录 `geometry_bridge.cpp`，include 目录设置为解压的 `Developer/Sources`。
3. 同时编译该目录内 `Geometry Surface VocalTract Tube Splines XmlNode XmlHelper Dsp Signal IirFilter Constants TlModel Matrix2x2` 这些同名 `.cpp`，输出名 `geometry_p2.dll`。原型阶段的完整命令生成方式保留在 `build_geometry_prototype.py`，其中原型相对目录需按本说明改为当前解压/输出目录。
4. 将同一个原版 API DLL 分别复制为 `VocalTractLabApi.dll` 和 `VocalTractLabAnalysis.dll`。不要在运行时向 EXE 资源目录复制 DLL。
5. 运行正式模块的原生、几何和 HTTP 回归，更新资源清单。不要用重建后的文件冒充先前校验过的二进制哈希。
