# 原生资源重建

当前几何方法为 `m10/r21`，数组布局为 `m10/3`。默认构建容量 257、界面启动使用 65 纵向截面，浏览器另支持 33，横向仍为 96 点。工程保留同一套 VTL 2.4 源码压缩包、R11–R21 几何补丁及可选声源微扰补丁，原 JD2/M01/W02 speaker 字节不改。

在工程根目录、既有项目 Python 和编译工具链中构建：

```powershell
.venv/m14/Scripts/python.exe scripts/build_m10_geometry.py --sections 257 --output-dir output/validation/m10-shared-20261018/geometry-r21
.venv/m14/Scripts/python.exe scripts/build_m10_source.py --output-dir output/validation/m10-shared-20261018/source-build
```

第一条用既有 MSVC 构建几何 DLL，第二条构建具有 jitter/shimmer 接口的合成库，并复制成独立分析库。运行时不得写入应用资源目录。替换本地资源前须关闭本任务持有的库，保留用户窗口。

浏览器使用 `scripts/build_m10_wasm.py --output-dir <候选目录> --sdk <既有Emscripten4.0.9目录> --write-lock resources/manifests/vocal-wasm-runtime.json`。运行包包含共用 Python 核心和 Wasm 库，另外生成对应源码 ZIP；更新 Python 后可加 `--refresh-runtime-only`。先更新锁定清单，再构建前端。

macOS 在 Mac 上使用 `scripts/build_m10_macos.py --output-dir <候选目录>`，当前实测 ARM64、最低 macOS 14.0。两份声学库分别使用独立 install_name 和签名，按 `native-pair.json` 核对共同源码与二进制身份，不能要求两个 Mach-O 文件字节一致。x86_64 及最终应用需单独构建验收。

完成平台验证后运行 `scripts/register_m10_resources.py` 更新资源身份，再运行前端 `npm run ui-data`。注册器保留历史来源关系，不覆盖上游署名。当前二进制摘要以 `../bundle-manifest.json` 与 Wasm 锁定清单为准，测试入口见项目 `docs/testing/m10-shared-controls.md`。历史 R11–R18 的测试和构建说明保留为对应阶段证据，不能替代当前默认值。

VTL、几何桥接及派生补丁采用 GPL-3.0-or-later。公开再分发时须保留对应源码、许可和修改声明；本地构建不等同于发布 EXE、Mac 应用或网站。

R21 Windows/Wasm 已重建，Mac构建入口使用相同补丁，但当前保留的Mac原生文件仍为R20，须重建后配合新核心运行。实际验收见 `docs/testing/m10-smoothing-review.md`。
