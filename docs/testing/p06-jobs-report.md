# P06 任务执行：Windows 定向验收

2026-09-09，状态 **verified（限定 Windows 持久任务流程）**。井井在 P05 验收后授权继续 P06；19:31 在具体建表审阅之后回复“继续，刚刚不小心闪退了”，原任务随后完成了建表、真实 SQL 与网页任务验证。第二次退出前未同步阶段文档，本轮已从原始会话、实际数据库、截图和报告核对并完成复验。这里只开放 pipeline_check 流程探针，不代表语音算法、文件配额或完整桌面业务已完成。

## 已实现

- 同一 JobStore 政策配 PostgreSQL/SQLite 事务适配。不可变输入/配置及版本快照、owner 幂等键、同键异内容冲突、默认全局 2 槽/每人 1 槽、租约代数和截止、过期 interrupted、手动重试保留 retry_of；事件与状态、成功与有限 manifest 同事务保存。
- 每任务独立核心子进程。pipeline_check 对确定性字节计算摘要，明确不是语音算法；任务元数据有上限，未接入音频上传、输出文件、下载或不计配额的缓存。P07 文件提交门实现前不允许写文件任务。
- 任务创建/查询/事件/取消/重试从会话取 owner；本地令牌、精确 Origin、已有 Host 边界、CSRF、请求体上限及 no-store 同样覆盖任务接口。未知操作与额外 owner 字段拒绝。未配置任务库返回明确 503，能力清单不宣称可运行。
- LocalService 可传入已初始化任务状态文件，API 启动一个自有 worker；关闭/父进程 EOF 协调退出。SQLite 连接 mode=rw，API 和 worker 不自动建表。
- 网页项目详情增加任务区域，使用生成契约和公共 TaskPanel；显示真实可用性、取消、重试及有序事件，网络错误保留同次请求的幂等标识。账号/项目切换卸载组件并中止请求；携带预期账号防止旧页面误操作新账号。

## 首次建表前检查（历史阶段）

以下记录保留建表前的测试范围；其后的实际执行与本轮复验见下文。

解释器为现有 .venv/v3-dev/Scripts/python.exe，不安装全局包，不更改旧 conda 环境。

- `python -m pytest -c tests/pytest.ini backend/tests tests/security tests/contracts desktop/tests packages/phonetic_core/tests -q`：38 项通过。另 `tests/architecture` 4 项通过。新增原始字节摘要独立期望、错误边界、状态/取消/租约政策、本地权限边界及真实子进程协议/父管道 EOF 退出测试。存储事务未被这些测试替代。
- 原 P02/P05 测试期待任务路由不存在（404）；P06 注册了路由但未配置存储时必须返回明确 503。更新为精确状态、错误码和空能力检查，不允许伪成功。
- `npm --prefix frontend run typecheck`、`npm --prefix frontend run test`（8 项）、`npm --prefix frontend run build` 通过。生成 OpenAPI 与 TypeScript 已同步；P04 原始音频/选择区契约未改。
- `check_architecture.py`、架构测试通过；初次 `validate_docs.py` 检查 179 文件、296 来源、32 任务，无现行错误（最终新增报告后的计数以本机 final-checks.json 为准）。历史快照失效链接继续单列。
- 真实浏览器连接显式 P05 账号测试替身宿主，创建中文项目，检查任务尚未启用的禁用按钮和空态；浅/深主题实际检查，未见当前窗口横向溢出。修复了任务空态文字挤在一行的问题。此时没有启动任务数据库，不能把这个画面当作排队/取消实测。
- 本轮准备 `p06_database.py` 和 `verify_p06_jobs.py`；前者拒绝未授权初始化与覆盖，后者等待真实数据库验证。脚本语法检查通过，未执行其中 SQL。
- 保留 Starlette/httpx 和 AnyIO 两条已有弃用警告，没有为了消除提示升级依赖。

## 真实数据库、服务与网页验收

[P06 建表审阅](p06-migration-review.md) 列出唯一库名、本机状态文件、两份完整 SQL、拒绝覆盖及回退方式。具体审阅后的继续指令及实际执行已在原始会话核实；本轮复验只复用已存在的表，没有重复迁移。

- PostgreSQL 与 SQLite 的真实事务检查通过：并发幂等与异内容冲突、全局最多 2 槽且每账号 1 槽、租约过期/旧代数拒绝、取消/完成竞态、事件游标、回滚和重新连接恢复。PostgreSQL 另验证 10 个独立账号排队，以及任务/事件/取消/重试/项目归属的跨账号拒绝。SQLite 使用单个本地身份，不将其称为 10 账号测试。
- 两种适配都运行真实 worker 与独立核心子进程：摘要匹配独立字节期望、运行中取消、强制结束本轮 worker 后 interrupted、使用新 worker 显式重试通过。此处验证受控进程中断，不等于操作系统断电恢复测试。
- 初次测试沿用了 P05 的模拟客户端地址，正常登录改动了旧限流桶；原任务已给新测试分配独立地址并重新验证。本轮复验前已有的 83 条账号相关记录全部保持原值；不将此前那次限流计数变化隐去。
- 本机 LocalService 通过真实 loopback HTTP 验证提交幂等、结果、事件与服务重启后恢复；本机账号端点保持不可用，两次服务退出码均为 0。
- 原任务真实网页验证：完成、取消、重试完成三条记录，进度 0/19/41/62/84/100%，刷新后恢复，换账号为空；浅色、深色及 390 像素视口已保存截图。本轮重新查看这两张截图，未重开 Codex 内置浏览器；该图中的 EGG 是测试项目名称，不是实际 EGG 分析。

本机证据位于忽略目录 output/validation/p06：postgres-*.json、sqlite-*.json、database-validation.json、local-service.json、ui-validation.json、ui-shutdown.json、tasks-real-light.png、tasks-real-dark-narrow.png。网页宿主与 worker 退出码均为 0，PG 已停止。本轮复验的命令与文件摘要另见 recovery/final-checks.json。

## 本轮恢复后的复验命令

均在 v3 根目录使用现有 .venv/v3-dev/Scripts/python.exe，未安装全局依赖。

```powershell
& '.venv/v3-dev/Scripts/python.exe' -m pytest -c tests/pytest.ini backend/tests tests/security tests/contracts desktop/tests packages/phonetic_core/tests tests/architecture -q
& '.venv/v3-dev/Scripts/python.exe' scripts/run_p06_validation.py --approved-p06-test-data
& '.venv/v3-dev/Scripts/python.exe' scripts/verify_p06_local_service.py --approved-test-data
npm --prefix frontend run typecheck
npm --prefix frontend run test
npm --prefix frontend run build
npm --prefix frontend run contracts:check
npm --prefix frontend run ui-data:check
& '.venv/v3-dev/Scripts/python.exe' scripts/generate_contracts.py --check
& '.venv/v3-dev/Scripts/python.exe' scripts/check_architecture.py
& '.venv/v3-dev/Scripts/python.exe' scripts/validate_docs.py
```

Python 42 项通过（包含架构 4 项），前端 8 项通过，构建、生成契约和来源数据无漂移；保留 2 条已有弃用警告。实际数据库与本机服务复验均通过；文档最终计数以 recovery/final-checks.json 为准。只读保存性检查再次核对 v2 的 427 个基线文件、HEAD、索引、状态及旧环境包版本摘要，均与此前证据一致。

## Codex 退出与后续测试方式

两次退出都紧随内置浏览器测试页关闭；具体时间、证据边界和绕行措施见 [恢复与退出排查](p06-recovery-and-codex-exit.md)。本轮未复现该操作，未更改 Codex 设置或用户目录仓库；测试采用独立进程，原有服务已停止。避开触发路径不等于修复了 Codex 底层缺陷。

## 来源与边界

仅使用现有 CPython 所带 SQLite 3.50.4 和已安装的 PostgreSQL/psycopg；没有增加 Python 锁定包。新增 SQLite 与 PG 锁定文档来源，共 296 条，均在软件组，语言学/语音学组继续优先。原始 pipeline_check 无外部算法复制，不能用于科研指标。详见 ADR-018 与来源登记。

P07 配额/文件清理/下载、真实语音算法、完整 Qt 任务交互、原生设备和跨平台发行仍需各自验收。P05 通过范围保持账号/会话/项目，v2 的数据和源文件保持原边界。
