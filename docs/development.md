# v3 开发入口 · P02

新工程入口为 frontend、backend、desktop、packages/phonetic_core。根目录 run.py / run.spec / pyproject.toml 保留为 v2 迁移来源，不用于安装或启动正式 v3。

## 本机独立环境

Windows x86_64，项目内独立 CPython 3.11.14；Node 24.13.0 / npm 11.6.2 为已存在工具，未安装新全局依赖。`.venv/v3-dev` 与 P01 探针、原 conda phonetic_311 分开。Python 传递依赖用 [带 hash 的 lock](../requirements-v3-dev.lock)，前端用 [npm lock](../frontend/package-lock.json)。此锁仅在 Windows 验证，不能当作其他平台已通过。

在项目根目录 PowerShell 执行：

```powershell
uv venv --python '.venv/runtimes/cpython-3.11.14-windows-x86_64-none/python.exe' '.venv/v3-dev'
uv pip install --python '.venv/v3-dev/Scripts/python.exe' --require-hashes -r requirements-v3-dev.lock
uv pip install --python '.venv/v3-dev/Scripts/python.exe' --no-build-isolation -e packages/phonetic_core -e backend -e desktop
npm --prefix frontend ci --ignore-scripts
```

已有环境不必重复建。新机器先安装同版项目内解释器（`uv python install 3.11.14 --install-dir .venv/runtimes --no-bin --no-registry`）；uv/Node 可由开发者选择现有可信安装，以上脚本不自动改系统。更新依赖须改 `.in` 和包声明，经试验后重新编译 lock、更新来源清单；不能自动升级 v2。

## 使用与验证

```powershell
# 只读服务器开发入口：绑定 127.0.0.1 随机端口，控制台输出实际 URL，Ctrl+C 退出。
& '.venv/v3-dev/Scripts/python.exe' -m ptb_api.cli --mode server
# 桌面本地服务诊断：启动、会话握手、退出，输出版本与退出结果。
& '.venv/v3-dev/Scripts/python.exe' -m ptb_desktop.main
# 前端开发入口，当前为 P04 工作台试用版。
npm --prefix frontend run dev
```

P02 server 模式仅为 loopback 开发入口，无账号/任务/语料接口，不能部署给研究者使用。local 模式由 desktop 通过标准输入提供一次性凭据，URL 不含 token，服务校验 Host/Origin/Authorization。桌面启动器不会 import backend；三个 wheel 由同一发行组合安装。后续 P06 负责接入完整窗口与工作进程生命周期。

```powershell
& '.venv/v3-dev/Scripts/python.exe' scripts/generate_contracts.py --check
npm --prefix frontend run contracts:check
& '.venv/v3-dev/Scripts/python.exe' scripts/check_architecture.py
& '.venv/v3-dev/Scripts/python.exe' scripts/validate_docs.py
& '.venv/v3-dev/Scripts/python.exe' scripts/verify_p02.py
```

完整验证会另建带时间戳的干净环境，真实构建/安装三个 wheel，并从非项目目录运行两种服务与定向测试；所有日志保存在忽略的 output/validation/p02 下。不会删除已有环境。

架构检查覆盖正式源码的 Python 导入、前端导入、旧路径与规定资源目录，不是任意动态 Python 的安全沙箱。历史文档快照和第三方摘录保留原字节：校验 hash，并另报其未解决的相对链接；现行文档缺失链接会使检查失败。

## P04 工作台试用

在项目根目录执行，使用现有隔离环境，无需再安装依赖：

```powershell
# 浏览器试用；终端保持运行，Ctrl+C 退出。
npm --prefix frontend run dev -- --port 5174 --strictPort
# Qt 原生窗口试用；先构建，同一静态前端直接由自定义 scheme 加载。
npm --prefix frontend run build
& '.venv/v3-dev/Scripts/python.exe' desktop/experiments/p04_host.py
```

浏览器地址为 http://127.0.0.1:5174/ 。Qt 入口不启动额外 HTTP 服务。两端目前仅提供共同 UI、本机 WAV 预览/选区/试听；15 模块的算法、实际分析任务、摄像头、目录批处理和正式发行仍待后续任务。仅支持单个不超过 64 MB 的 WAV。文件不会自动上传或跨启动保存。

试用 EGG：点击 EGG 信号分析，载入公开测试音频或选择自己的双声道 WAV。音频声道选择控制试听：1–4.wav 为右音频/左 EGG；牧歌.wav 为左音频/右 EGG。程序不按文件名猜测角色，需要显式选择，原文件不改写。可试切换主题、调整选区、缩放平移、参数草稿、标签切换与关闭提示。视觉验收和限制见 [P04 报告](testing/p04-workbench-report.md)。

## P05 已验证的隔离测试实例

[P05 报告](testing/p05-accounts-report.md) 区分测试替身、真实 PostgreSQL 与进程重启证据。井井已授权并执行 [专属空库迁移](testing/p05-migration-review.md)，数据库 ptb_p05_test_20260909 保存在 output/validation/p05/postgres-data，运行时位于 .venv/postgresql-17.11-3/pgsql；测试结束已停止。已有表不应再次建表；工具会拒绝非空库。后续数据库变更需按根规则独立授权。

后端仍可用 P02 命令启动只读骨架，未配置账号存储时 auth/projects 返回 503；本机模式返回 404。P05 原型服务使用 `python -m ptb_api.server --port 5175 --frontend frontend/dist`，从标准输入读取一行私密 JSON（dsn、signing_key），仅回环监听；不用 .env，不在命令行/URL中传凭据。该入口不作生产部署。

`tests/support/account_ui_host.py` 仅供自动验证浏览器流程，使用测试替身，重启不保存数据，不用于研究者试用或真实账号。项目中无默认生产管理员密码。正式账号创建使用 `scripts/p05_database.py create-user` 的隐藏提示；首次建表和测试数据创建都遵守迁移审阅授权。

## P06 已验证任务流程

见 [P06 报告](testing/p06-jobs-report.md)。任务 API 仅在显式配置已初始化存储后开放；后端入口不迁移数据库。服务器私密配置 enable_jobs=true 启用现有 PG 任务表，worker 通过 ptb_worker.cli 从 stdin 读取配置；桌面 LocalService(jobs_path=...) 使用相同接口和独立 worker。新表已按 [专项审阅](testing/p06-migration-review.md)建立，真实 PG/SQLite 与本机服务验收通过，勿重复执行初始化。默认未配置存储时仍返回 503。

复验入口：`python scripts/run_p06_validation.py --approved-p06-test-data`（复用已有表，不带 --apply-reviewed-schema），随后 `python scripts/verify_p06_local_service.py --approved-test-data`。使用现有 .venv/v3-dev 解释器。Codex 内置测试页关闭两次关联退出，暂行测试方式见 [恢复记录](testing/p06-recovery-and-codex-exit.md)。

## P07 存储准备（未执行数据库与删除验收）

见 [P07 报告](testing/p07-storage-report.md)与 [具体迁移审阅](testing/p07-migration-review.md)。后端私密配置可显式指定 storage_root；必须是已初始化且 instance_id 与 PG 匹配的私有目录，构造函数不建表。显式启用后的启动恢复会处理到期文件，因此在测试范围获确认前不得配置/运行该入口。未配置时文件 API 明确 503，桌面为 404。

原始文件分块接口每块最多 256 KiB。/uploads 创建保留幂等键，PUT /uploads/{id}/blocks 使用 offset，POST finalize 完成服务器 SHA-256 和尺寸核对。文件以二进制原样保存，不自动解析 ZIP、Pickle 或执行代码。下载按单个 Range 和逐块截止/会话校验；DELETE 仅从会话确定 owner。生成结果与归档仍待 P07/P06 联合门，不要将上传资源称为科学分析产物。
