# P07 存储闭环准备与待执行验证

2026-09-09，**in_progress**。井井在 P06 验收后授权继续。本轮完成第一批可审阅代码及无数据库检查；尚未执行 P07 SQL 或合成文件的物理删除，不声称磁盘/数据库闭环已通过。

## 已实现的第一批代码

- PostgreSQL 配额账户与资源表草案；默认每人 5,000,000,000 字节。受控文件块最多 256 KiB，先持久预留再写入/fsync，随后原子转为已占用；文件写入后未结算的字节仍有预留覆盖。
- 资源幂等创建、明确 offset 的块重试、最终 SHA-256（服务端分块计算，可选客户端期望校验）、已知总长核对、未知总长逐块预留。文件以不透明二进制保存，不执行/解压/反序列化；格式解释留给对应模块。
- 所有磁盘操作使用同一根目录进程间文件锁与 PG advisory lock。目录需先显式初始化，标记与数据库 instance_id 对应；拒绝路径链接和非普通文件。用户文件名仅显示，不参与物理路径。
- 用户空间、列表、下载、直接删除接口。下载检查登录、归属、到期与单个 Range，每个后续块重新校验；网络中已发送的字节不能被撤回，不把这一机制描述成撤销用户已下载的副本。
- 资源最终化起算 7 天，未完成上传 24 小时；下载不续期。删除先标状态不可访问，实际 unlink 成功才释放用量；失败保留占用及有限重试元数据。启动恢复检查磁盘与账目，遇到未知文件冻结写入，不按超时直接清零。
- 清理线程由同一宿主持有，截止前只唤醒检查，截止后处理；服务退出发停止信号并等待。Windows 普通文件 fsync 不等于目录元数据的断电保证，真实断电持久性仍是待验证限制。
- 网页增加文件面板、空间/预留、上传、到期/大小排序、原生浏览器下载和公共删除确认框；不要求先下载再删。切账号/项目卸载中止请求。桌面不会启用服务器文件配额或 7 天清理。

## 本轮实际检查

- 新政策用例首次因 quota 模块不存在而失败；最小实现后通过。测试包括精确字节边界、逐块结算、Range、TTL/归档截止政策、显示文件名与额外 owner 拒绝。
- HTTP 使用明确测试替身验证身份、CSRF、Origin/Host、旧页面预期账号、JSON/二进制不同大小上限、桌面禁止服务器存储。额外直接调用 ASGI 响应验证第二块到期时不再发送；它不是实际 TCP 到期/慢下载实测。
- 收尾 Python 定向测试 51 项通过，保留 2 条已有弃用警告；其中包括新增的下载中到期测试。完整结果见 output/validation/p07/final-checks.json。前端 typecheck、8 项测试与 build 通过，契约已生成。
- 独立 Chrome + 现有 Playwright 1.62.1 实测：显式内存账号测试宿主登录、建项目、存储未配置时禁用上传、浅色/深色、390 像素无横向溢出、退出卸载文件面板，页面无 JS 错误。已查看实际截图；截图仅代表不可用态，不能替代真实上传/删除/配额验收。
- 迁移和验证三个脚本在未给授权参数时均以退出码 2 拒绝，且 P07 存储根未创建；语法检查通过。这不能替代实际 SQL 语义/事务/磁盘验收。
- 该独立浏览器与测试宿主已退出；未操作 Codex 内置浏览器关闭接口。本轮测试未出现相同退出现象，不将其称为 Codex 底层缺陷已修复。
- 来源新增标准库文件/锁与 Starlette 响应参考、既有测试工具登记，总计 299 条，均归软件组。没有新项目依赖或全局安装。

命令：`python -m pytest -c tests/pytest.ini backend/tests tests/security tests/contracts desktop/tests packages/phonetic_core/tests tests/architecture -q`；`npm --prefix frontend run typecheck` / `test` / `build`；`python -X utf8 scripts/generate_contracts.py --check`；`npm --prefix frontend run contracts:check` / `ui-data:check`；`python -X utf8 scripts/check_architecture.py` / `scripts/validate_docs.py`。全部 Python 使用现有 .venv/v3-dev/Scripts/python.exe。

浏览器复验脚本与截图保存在忽略的 output/playwright/p07；迁移前命令结果、语法/授权拒绝和来源校验写在 output/validation/p07。数据库脚本没有假结果文件。

## 下一步及剩余门

先确认 [存储表与合成文件清理审阅](p07-migration-review.md)，再执行真实 PostgreSQL 与文件闭环。其后继续全部 Q01–Q20：尤其账户最后配额的并发竞争、真实 TCP 下载中到期、ZIP 展开/归档预算、多个输出、活跃输入到期、P06 旧 worker 提交拒绝与结果清单原子提交。当前 pipeline_check 仍仅返回任务元数据，生成文件任务尚未开放。

未验证 Windows 硬断电、Mac/Linux 原生文件锁与发行、生产磁盘权限/容量、负载和完整桌面文件流程。P07 未完成前不推进全面业务迁移，不将单文件代码存在等同于整体 verified。
