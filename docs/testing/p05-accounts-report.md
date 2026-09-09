# P05 账号与项目：实现及迁移前验证

2026-09-09，状态 **in_progress**。本轮完成接口、前端、迁移草案与定向安全检查；没有执行数据库 schema 变更，不声明真实 PostgreSQL 事务/并发/持久化通过。

## 实现

账号受控创建、Argon2id 校验；登录 CSRF 签名挑战；不透明会话与数据库摘要、8 小时固定截止、登录更换会话、退出撤销；HttpOnly/SameSite/默认 Secure Cookie、固定 Origin 和 Host、CSRF、敏感响应 no-store、16 KiB 账号请求体上限、校验错误不回显密码。账号和 IP 限流由 PostgreSQL 原子计数实现草案支持，不能用本轮测试替身的结果证明数据库并发正确。

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

8 条依赖元数据与 2 条 OWASP 参考进入来源登记，共 293 条。全部归软件组，学术组仍优先。原生包随发行物的传递许可证需后续审计；本轮安装不等于再分发验收。依据与设计边界见 ADR-017。

## 未完成与下一步

[实际数据库操作审阅](p05-migration-review.md) 给出专属空库目标、完整 SQL、拒绝覆盖的执行工具和 PG 集成命令。当前 Docker 引擎未运行、PATH 未发现 PG 工具。未启动 Docker Desktop、未连接已有数据库、未运行 SQL。

获授权后，建立本工作区内的专属 PostgreSQL 测试实例，执行建表，再验证两账号、不同数据库连接/应用对象的会话恢复、事务回滚、退出撤销以及真实 12 连接登录限流。进程崩溃、任务/日志/取消/下载隔离与配额分别在 P06/P07 完成后补测。上线前还需反向代理/TLS、密码恢复、账户管理与限流维护策略验收；本轮不提供公网部署。

保留核对：v2 的 HEAD/index/status/环境路径/包元数据五项与 P04 收尾一致，427 个基线文件无变化；原 v3 锁定的 31 个依赖版本未升级，仅新增 8 包。证据 output/validation/p05/preservation.json。临时内存 UI 宿主已在本轮测试结束后停止，不把该服务作为研究者账号系统交付。
