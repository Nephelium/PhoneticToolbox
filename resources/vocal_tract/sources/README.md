# 原生资源重建

当前 R18 默认构建容量为 257、启动使用 129 纵向截面的 DLL，可在同一个实例中切换 65、129、257。构建按顺序应用 R11–R18 补丁，横向仍为 96 点。`--sections` 是编译容量选项，日常采样数量通过工作台选择，无须另编 DLL。修改含义见 `M10-build.md`。

历史 R13/R14 几何报告按需查项目 `docs/testing/2026-10-07-m10-r13-report.md`、`2026-10-07-m10-r14-report.md`，对应阶段的命令与数值边界不作为当前采样默认值。

1. 解压本目录 `VTL2.4-API-source.zip`，得到 `Developer/Sources` 及 Visual Studio 工程。VocalTractLabApi.dll 可由原版工程的 x64 Release 配置重建；已有分发 DLL 的 SHA-256 见上级清单。
2. M10/3 桥接需要对原版 VocalTract.h/cpp 应用几何读数补丁、`m10_r11_patch.py` 至 `m10_r18_patch.py` 及 `refreshM10Geometry` 头文件声明。在 v3 项目根运行 `.venv/m14/Scripts/python.exe scripts/build_m10_geometry.py --sections 257 --output-dir output/validation/m10-r18/native-build`，脚本逐项检查并完成这些修改，使用已有 MSVC x64 工具链，不安装或修改全局环境。编译采用 C++14、`/LD /O2 /EHsc /DWIN32 /D_USE_MATH_DEFINES /MD`，include 设置为已应用补丁的源目录。
3. 同时编译该目录内 `Geometry Surface VocalTract Tube Splines XmlNode XmlHelper Dsp Signal IirFilter Constants TlModel Matrix2x2` 这些同名 `.cpp`，输出名 `geometry_p2.dll`。`build_geometry_prototype.py` 仅保留历史来源，不能直接重建当前 M10/3。
4. 将同一个原版 API DLL 分别复制为 `VocalTractLabApi.dll` 和 `VocalTractLabAnalysis.dll`。不要在运行时向 EXE 资源目录复制 DLL。
5. 当前定向检查为 `tests/parity/test_vocal_r15.py`、`test_vocal_tract.py`、`test_vocal_process.py`、`test_vocal_recording.py`、`desktop/tests/test_m10_worker_startup.py` 与 `scripts/verify_m10_r17.py`、`scripts/verify_m10_r17_qt.py`，精确命令和范围见项目 `docs/testing/2026-10-12-m10-r17-report.md`。以 `scripts/register_m10_r12_resources.py r18` 更新来源/资源摘要，再生成并检查公共界面数据。冻结 EXE 需另行构建与验收。M10 使用应用拥有的私有进程，不新建 HTTP 服务。不要用重建后的文件冒充先前校验过的二进制哈希。

R18 的局部偏移与文件验证见 `tests/parity/test_vocal_r18.py`、`scripts/verify_m10_r18.py` 和 `scripts/verify_m10_r17_qt.py --r18`，当前证据见 `docs/testing/2026-10-12-m10-r18-report.md`。
