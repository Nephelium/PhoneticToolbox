# v3 的源码入口与修改方式

项目自有功能只维护工程源码一份，所有普通模块入口共用 `Start-Research-Workbench.ps1`。

## 启动

在工程根目录执行：

```powershell
.\scripts\Start-Research-Workbench.ps1 -CheckOnly
.\scripts\Start-Research-Workbench.ps1
.\scripts\Start-Research-Workbench.ps1 -Module M10
```

第一条只显示解释器、源码与计算环境路径，不打开窗口或数据库。第二条打开首页。第三条打开同一个应用并选中 M10。已有 `Start-Mxx-Workbench.ps1` 名称保留，全部转发公共入口；`-CheckOnly` 和 `-PrepareOnly` 同样可用。`-PrepareOnly` 会检查既有开发任务库，必要时建立原有规则允许的本地文件缓存，不运行数据库迁移。

需要开发 MFA 时可传 `-ComponentRoot <已存在的组件目录>`，该参数只绑定外部资源，不安装或打包 MFA。当前紧凑发行物排除 MFA 环境、模型和词典。

## 修改位置

| 修改内容 | 编辑位置 | 生效方式 |
| --- | --- | --- |
| 页面、按钮、布局 | `frontend/src` | 按前端流程生成 `frontend/dist` 后重启 |
| 桌面窗口、本地设备 | `desktop/src/ptb_desktop` | 重启源码工作台 |
| 任务与文件处理 | `backend/src` | 重启源码工作台及其计算任务 |
| 科学算法 | `packages/phonetic_core/src/phonetic_core` | 按科研验证要求修改，重启源码工作台 |
| 原生 DLL/WASM | 对应原生源码与构建脚本 | 重新构建对应资源并验证 |

已启动的进程不会自动重新导入修改后的 Python。打包成品也要重新构建才能包含新修改。环境中的旧项目包、构建快照和 EXE 内文件都不作为日常编辑目标。

## 为什么还有几个环境

主工作台选用现有 `.venv/m14`。它通过已有路径配置使用 m09-ui 的同一批 NumPy、SciPy、Pandas、OpenCV、Parselmouth 与 Qt 文件，并增加文档读写依赖。解释器还依赖 `.venv/runtimes`，整理时必须追踪 `.pth` 和解释器绑定，保留被借用的环境。EGG/LPC 的 MKL 计算环境和唇形的 MediaPipe/PyAV 环境继续独立，它们各自运行的项目代码来自当前工程。

`workbench_source.py` 在加载业务代码前绑定三组源码目录，覆盖子进程的项目代码搜索路径。已加载其他位置的项目模块时明确报错。公共 PowerShell 启动不改变父终端的环境变量，外部 `PYTHONHOME`、`PYTHONPATH` 不得覆盖启动时的绑定。

## 验证与发行

Windows 当前定向检查：

```powershell
.\.venv\m14\Scripts\python.exe -B -m pytest -c tests/pytest.ini tests/test_source_entry.py tests/test_source_snapshot.py tests/test_release_content_policy.py -q
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/verify_source_entry.py --output output/validation/<新的目录>
```

第二条运行公开合成素材的真实计算、文档回读和隐藏 Qt，创建独立测试输出，不使用用户工程或录音设备。测试输出目录必须尚不存在。

发行时，入口、依赖分析与真实文件形式的工作进程源码均取自同一冻结快照，记录逐文件 SHA。免安装版和安装版仍来自同一个应用，规则见 [发行工程](../../release/README.md)、[架构决定](../decisions/ADR-source-entry-and-snapshot.md)与[入口专项报告](../testing/2026-10-07-source-entry-and-package-audit.md)。

