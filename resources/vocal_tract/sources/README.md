# 原生资源重建

当前 R14 默认生成 257 纵向截面，顺序应用 R11/R12/R13/R14 补丁，新增原生舌叶站位与截面数量查询。运行 `tests/parity/test_m10_r14.py`、`scripts/probe_m10_r14.py`，使用 `scripts/register_m10_r12_resources.py r14` 更新来源和摘要。`scripts/benchmark_m10_r14.py` 对照独立 129 档与项目当前 257 档，先通过构建脚本的 `--sections 129 --output-dir output/validation/m10-r14/resolution129 --no-install` 生成基准。证据目录 `output/validation/m10-r14`，不覆盖 R13。

当前 R13 在 R12 后额外应用 `m10_r13_patch.py`。同一构建入口自动完成全部补丁。除下列回归外运行 `tests/parity/test_m10_r13.py`，并使用 `scripts/register_m10_r12_resources.py r13` 更新 R13 来源与资源摘要。R12 几何及 Qt 验证脚本支持独立输出目录，R13 证据存于 `output/validation/m10-r13`，不覆盖旧验收文件。

1. 解压本目录 `VTL2.4-API-source.zip`，得到 `Developer/Sources` 及 Visual Studio 工程。VocalTractLabApi.dll 可由原版工程的 x64 Release 配置重建；已有分发 DLL 的 SHA-256 见上级清单。
2. M10/3 桥接需要对原版 VocalTract.h/cpp 应用 `M10-geometric-readout.patch`、`m10_r11_patch.py`、`m10_r12_patch.py` 及 `refreshM10Geometry` 头文件声明，修改说明见 `M10-build.md`。在 v3 项目根运行 `.venv/m14/Scripts/python.exe scripts/build_m10_geometry.py`，脚本逐项检查并完成这些修改，使用已有 MSVC x64 工具链，不安装或修改全局环境。编译采用 C++14、`/LD /O2 /EHsc /DWIN32 /D_USE_MATH_DEFINES /MD`，include 设置为已应用补丁的源目录。
3. 同时编译该目录内 `Geometry Surface VocalTract Tube Splines XmlNode XmlHelper Dsp Signal IirFilter Constants TlModel Matrix2x2` 这些同名 `.cpp`，输出名 `geometry_p2.dll`。`build_geometry_prototype.py` 仅保留历史来源，不能直接重建当前 M10/3。
4. 将同一个原版 API DLL 分别复制为 `VocalTractLabApi.dll` 和 `VocalTractLabAnalysis.dll`。不要在运行时向 EXE 资源目录复制 DLL。
5. 当前改动先运行 `tests/parity/test_m10_r12.py`、`test_m10_r11.py`、`test_vocal_tract.py`、`test_vocal_process.py`、`test_vocal_recording.py`、`scripts/probe_m10_r12.py`、`scripts/verify_m10_r12_geometry.mjs` 和 `scripts/verify_m10_r12_qt.py`，再用 `scripts/register_m10_r12_resources.py` 更新限定资源清单并检查公共生成数据。冻结 EXE 需另行构建与验收。M10 使用应用拥有的私有进程，不新建 HTTP 服务。不要用重建后的文件冒充先前校验过的二进制哈希。
