# 原生资源重建

1. 解压本目录 `VTL2.4-API-source.zip`，得到 `Developer/Sources` 及 Visual Studio 工程。VocalTractLabApi.dll 可由原版工程的 x64 Release 配置重建；已有分发 DLL 的 SHA-256 见上级清单。
2. M10/2 桥接需要先对原版 VocalTract.h/cpp 应用 `M10-geometric-readout.patch`，修改说明见 `M10-build.md`。推荐在 v3 项目根运行 `.venv/m10-ui/Scripts/python.exe scripts/build_m10_geometry.py`，脚本使用已有 MSVC x64 工具链，不安装或修改全局环境。手动编译时在独立输出目录使用 C++14、`/LD /O2 /EHsc /DWIN32 /D_USE_MATH_DEFINES /MD`，include 设置为已应用补丁的 `Developer/Sources`。
3. 同时编译该目录内 `Geometry Surface VocalTract Tube Splines XmlNode XmlHelper Dsp Signal IirFilter Constants TlModel Matrix2x2` 这些同名 `.cpp`，输出名 `geometry_p2.dll`。`build_geometry_prototype.py` 仅保留历史来源，不能直接重建 M10/2。
4. 将同一个原版 API DLL 分别复制为 `VocalTractLabApi.dll` 和 `VocalTractLabAnalysis.dll`。不要在运行时向 EXE 资源目录复制 DLL。
5. 运行 `tests/parity/test_vocal_tract.py`、`test_vocal_process.py`、`scripts/verify_m10_qt.py` 及录制 EXE 自测，更新资源清单。M10 使用应用拥有的私有进程，不新建 HTTP 服务。不要用重建后的文件冒充先前校验过的二进制哈希。
