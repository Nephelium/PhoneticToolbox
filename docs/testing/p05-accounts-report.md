# P05 账号与项目：Windows 定向验收

2026-09-09，状态 **verified（Windows 账号、会话与项目范围）**。接口、前端、审阅后的专属空库建表及真实 PostgreSQL 事务/并发/重启恢复已通过。任务、文件、下载、日志、取消和配额的联合隔离仍属 P06/P07 待验收项；不表示完整服务器或 v3 业务已完成。

## 实现

账号受控创建、Argon2id 校验；登录 CSRF 签名挑战；不透明会话与数据库摘要、8 小时固定截止、登录更换会话、退出撤销；HttpOnly/SameSite/默认 Secure Cookie、固定 Origin 和 Host、CSRF、敏感响应 no-store、16 KiB 账号请求体上限、校验错误不回显密码。账号和 IP 限流由 PostgreSQL 原子计数支持，账号预算经过真实多连接争用验证；测试替身结果单列，不充当数据库证据。

项目列表/创建/查看/改名全部显式传入会话 owner；请求体额外 owner 被拒绝，跨用户读取与改名返回 404。前端携带预期账号标识，后端只把它用于拒绝过期页面请求，不用于选择身份。项目元数据上限 100；P06/P07 尚无实际任务、文件、下载、配额和清理接口。

独立 /server/ 页面提供登录与项目管理，沿用 U2/K2 和浅/深/系统主题。服务器页不加载 P04 本机文件状态；退出、换账号清空项目与详情。Qt 和原本机预览仍无需登录。P04 的现有模块页面仍只是公共预览。

## 已执行验证

- `python -m pytest -c tests/pytest.ini backend/tests tests/security tests/contracts -q`：27 项通过（含已有契约检查）。新增用例覆盖 Cookie 标志/摘要、退出撤销、过期/禁用、凭据更换、旧 CSRF 拒绝、账号变更、Host/Origin 拒绝、请求过大与密码错误脱敏、两账号项目访问/改名拒绝。账号存储使用明确测试替身，未替代 PG 验收。
- 加入架构测试后的最终 Python 定向命令共 31 项通过；前端 typecheck、8 项测试、build 通过；contracts:check、ui-data:check、generate_contracts.py --check、check_architecture.py 无漂移/错误；validate_docs.py 检查 159 文件、293 来源、32 任务，无现行错误，历史快照失效链接保留另报。git diff --check 通过。
- `python desktop/experiments/p04_host.py --self-test output/validation/p04/p05-regression` 通过：新前端入口仍能在 Qt 显示 15 个模块与首页，无登录门。最初使用 p05 输出目录被该探针既有路径约束拒绝，改用允许的 p04 子目录后通过，未放宽路径检查。
- 真实浏览器连接本机 FastAPI 测试宿主（测试替身存储），完成错误密码、登录、中文项目创建、改名、刷新后恢复、退出及换账号空列表。Alice 的项目没有显示在 Bob 的页面中。未上传私人音频。
- 项目页在 1280×800 两主题与 390×844 窄窗口检查，未见横向溢出。截图位于本机忽略目录 output/validation/p05。首次自动输入因页面重新可见触发会话重建而失去焦点；已限制为已登录时才检查可见性恢复，随后输入与登录复测通过。
- 保留 Starlette/httpx、AnyIO 的已有弃用警告；未为了消除警告升级无关测试依赖。

## 依赖与来源

在现有 v3 隔离环境中新增 8 个锁定包：argon2-cffi 25.1.0、argon2-cffi-bindings 26.1.0、cffi 2.1.1、itsdangerous 2.2.0、psycopg/psycopg-binary 3.3.3、pycparser 3.0、tzdata 2026.3。requirements-v3-dev.lock 包含哈希。未安装全局依赖。

8 条依赖元数据、2 条 OWASP 参考和 1 条 PostgreSQL 测试运行时进入来源登记，共 294 条。新增项全部归软件组，学术组仍优先。PostgreSQL 17.11-3 由官网指向的 EDB 下载页取得，免安装运行时位于 v3 .venv；压缩包 341,325,378 字节，本地 SHA-256 为 4b8db0930c38f6ef845db919551dedda3b6b845aeb0927b3d79a6e8e9e4537cf。厂商独立哈希未提供，postgres.exe 未签名。版本、原生文件哈希和根许可清单见 [PostgreSQL 来源清单](../../third_party/p05-postgres-runtime.json)。原生包随发行物的传递许可证需后续审计；本轮安装不等于再分发验收。依据与设计边界见 ADR-017。

## 实际 PostgreSQL 验证

[实际数据库操作审阅](p05-migration-review.md) 在井井回复“允许”后执行。数据库 ptb_p05_test_20260909，PGDATA 为 output/validation/p05/postgres-data，监听 127.0.0.1:10255，SCRAM-SHA-256；随机凭据放入当前用户专用 ACL 的忽略目录，通过私有 stdin 提供。未启动 Docker Desktop、未连接已有数据库或注册系统服务。

执行命令（均使用 .venv/v3-dev/Scripts/python.exe，以下省略解释器路径）：

```powershell
python -X utf8 scripts/run_p05_isolated_postgres.py --runtime D:/PhoneticToolbox/PhoneticToolbox_v3/.venv/postgresql-17.11-3/pgsql --approved-empty-test-database
# 首轮已初始化实例后，从明确归属且已停止的实例恢复；没有重建或删除数据。
python -X utf8 scripts/run_p05_isolated_postgres.py --runtime D:/PhoneticToolbox/PhoneticToolbox_v3/.venv/postgresql-17.11-3/pgsql --approved-empty-test-database --resume-initialized-instance
# 上述父进程以私有管道调用：
python scripts/p05_database.py apply --approved-empty-test-database --dsn-stdin
python scripts/verify_p05_postgres.py --approved-test-data --dsn-stdin
```

首轮 PG 已正常启动，但 Windows 后代保留输出管道，导致 subprocess 等待超时。按 PGDATA 核对并停止自有进程后，将 pg_ctl 输出改为日志文件；恢复命令完成所有检查。这是启动探针问题，未改动数据库算法、DDL、容差或检查标准。完整迁移 SQL SHA-256 为 9c2349003e072a4b2e460c4841466da1aa0be6b593398be52e25c88e8909d586，与审阅稿相同。

- 两账号项目读取与改名隔离、独立 adapter/app 的会话/项目恢复、事务回滚、跨连接退出撤销、12 连接争用恰好 10 次获准：5 项真实数据库检查通过。
- 重复迁移被非空库检查拒绝；没有 DROP、DELETE 或覆盖已有表。
- 实际 HTTP 服务创建中文项目，停止并重新启动 API 与 PG 两个进程后，原 Cookie 和中文项目仍可恢复；退出后旧 Cookie 返回 401。
- 所属 API 与 PG 进程已停止，测试数据保留。此次是受控数据库重启，不声明断电、数据库崩溃恢复或公网 TLS 已验证。
- 证据：output/validation/p05/postgres-integration.json、postgres-*.json、postgres-control.log、postgres-server.log。31 项 Python 定向回归复跑通过，保留两条既有弃用警告。
- 新增来源后的前端 typecheck、8 项测试与 build 复跑通过；契约和架构检查通过。文档检查最初发现来源总数字段仍为 293，补齐为实际 294 后复验，161 文件、294 来源、32 任务，errors 为空；没有放宽校验规则。git diff --check 通过。

## 后续边界

下一项依赖是 P06 持久任务与 worker。任务/日志/取消/下载隔离与配额分别在 P06/P07 完成后补测，联合退出门仍保留。上线前还需反向代理/TLS、密码恢复、账户管理与限流维护策略验收；本轮不提供公网部署。

保留核对：实际数据库验证后再次捕获 v2 的 HEAD/index/status/环境路径/包元数据，五项与 P05 迁移前一致，427 个基线文件无变化；原 v3 锁定的 31 个依赖版本未升级，P05 仅新增 8 个 Python 包。证据 output/validation/p05/preservation-postgres.json，迁移前证据 preservation.json 继续保留。临时内存 UI 宿主和实际 PG/API 测试进程均已停止，不把测试服务作为研究者账号系统交付。
