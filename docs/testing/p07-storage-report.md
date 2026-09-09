# P07 单文件存储验收与剩余联合门

2026-09-09，P07 整体 **in_progress**，其中 **Windows 单文件存储范围 verified**。井井在具体 003 存储表与专属测试文件清理审阅后回复“好，继续”，已执行新表及真实 PG/磁盘验收。上传、下载、删除、配额与恢复证据不能替代尚未完成的 ZIP/结果/任务输入引用联合门。

## 已实现的第一批代码

- PostgreSQL 配额账户与资源表草案；默认每人 5,000,000,000 字节。受控文件块最多 256 KiB，先持久预留再写入/fsync，随后原子转为已占用；文件写入后未结算的字节仍有预留覆盖。
- 资源幂等创建、明确 offset 的块重试、最终 SHA-256（服务端分块计算，可选客户端期望校验）、已知总长核对、未知总长逐块预留。文件以不透明二进制保存，不执行/解压/反序列化；格式解释留给对应模块。
- 所有磁盘操作使用同一根目录进程间文件锁与 PG advisory lock。目录需先显式初始化，标记与数据库 instance_id 对应；拒绝路径链接和非普通文件。用户文件名仅显示，不参与物理路径。
- 用户空间、列表、下载、直接删除接口。下载检查登录、归属、到期与单个 Range，每个后续块重新校验；网络中已发送的字节不能被撤回，不把这一机制描述成撤销用户已下载的副本。
- 资源最终化起算 7 天，未完成上传 24 小时；下载不续期。删除先标状态不可访问，实际 unlink 成功才释放用量；失败保留占用及有限重试元数据。启动恢复检查磁盘与账目，遇到未知文件冻结写入，不按超时直接清零。
- 清理线程由同一宿主持有，截止前只唤醒检查，截止后处理；服务退出发停止信号并等待。Windows 普通文件 fsync 不等于目录元数据的断电保证，真实断电持久性仍是待验证限制。
- 网页增加文件面板、空间/预留、上传、到期/大小排序、原生浏览器下载和公共删除确认框；不要求先下载再删。切账号/项目卸载中止请求。桌面不会启用服务器文件配额或 7 天清理。

## 第一批：迁移前检查记录

- 新政策用例首次因 quota 模块不存在而失败；最小实现后通过。测试包括精确字节边界、逐块结算、Range、TTL/归档截止政策、显示文件名与额外 owner 拒绝。
- HTTP 使用明确测试替身验证身份、CSRF、Origin/Host、旧页面预期账号、JSON/二进制不同大小上限、桌面禁止服务器存储。额外直接调用 ASGI 响应验证第二块到期时不再发送；它不是实际 TCP 到期/慢下载实测。
- 收尾 Python 定向测试 51 项通过，保留 2 条已有弃用警告；其中包括新增的下载中到期测试。完整结果见 output/validation/p07/final-checks.json。前端 typecheck、8 项测试与 build 通过，契约已生成。
- 独立 Chrome + 现有 Playwright 1.62.1 实测：显式内存账号测试宿主登录、建项目、存储未配置时禁用上传、浅色/深色、390 像素无横向溢出、退出卸载文件面板，页面无 JS 错误。已查看实际截图；截图仅代表不可用态，不能替代真实上传/删除/配额验收。
- 迁移和验证三个脚本在未给授权参数时均以退出码 2 拒绝，且 P07 存储根未创建；语法检查通过。这不能替代实际 SQL 语义/事务/磁盘验收。
- 该独立浏览器与测试宿主已退出；未操作 Codex 内置浏览器关闭接口。本轮测试未出现相同退出现象，不将其称为 Codex 底层缺陷已修复。
- 来源新增标准库文件/锁与 Starlette 响应参考、既有测试工具登记，总计 299 条，均归软件组。没有新项目依赖或全局安装。

命令：`python -m pytest -c tests/pytest.ini backend/tests tests/security tests/contracts desktop/tests packages/phonetic_core/tests tests/architecture -q`；`npm --prefix frontend run typecheck` / `test` / `build`；`python -X utf8 scripts/generate_contracts.py --check`；`npm --prefix frontend run contracts:check` / `ui-data:check`；`python -X utf8 scripts/check_architecture.py` / `scripts/validate_docs.py`。全部 Python 使用现有 .venv/v3-dev/Scripts/python.exe。

浏览器复验脚本与截图保存在忽略的 output/playwright/p07；迁移前命令结果、语法/授权拒绝和来源校验写在 output/validation/p07。以上是当时的无数据库检查，以下为后续已获授权的真实执行结果。

## 第二批：已授权的真实数据库、磁盘和网页检查

首次执行 run_p07_validation.py --approved-p07-schema-and-test-files --apply-reviewed-schema 成功。003 SHA-256 为 f27fd45be2a4afabf8162c3f695787816da38d14a719b0ba098065617c8effc2。后续复验省略 --apply-reviewed-schema，未重复建表。当前验证脚本同时核对已有 P05/P06 行内容，最后一轮保留 500 条原有账号/项目/会话/限流/任务/事件/版本记录；只新增测试账号与验证状态。

| 场景 | 实际结果与证据边界 |
| --- | --- |
| Q01/Q11 原子预留、逐块计量和幂等 | 真实 PG 和磁盘锁；写入转账不重复计量。4 个并发请求争抢最后 1 字节，1 个成功、3 个 quota_exceeded；未写入 5 GB 实体文件 |
| Q02 未知长度 | 逐块追加并最终核对 14 字节；不依赖 Content-Length 先收全文件 |
| Q05/Q09 删除与重试 | 未下载即可删除；针对自有测试文件注入 unlink 失败，实有文件与用量保留；解除故障后删除才释放 |
| Q06 真实 TCP 中途到期 | 786,432 字节文件在首个 262,144 字节后暂停第二块，更新测试行截止后放行；客户端收到截断，后续整文件与 Range 都返回 410。属于受控暂停的真实网络/数据库验证，不是生产慢网长期压力测试 |
| Q08/Q10 故障恢复 | 在 fsync 后、账本结算前注入中断状态，恢复准确记入 17 字节并保留 33 字节预留；属于故障注入，不是真实硬断电 |
| Q12/Q19 隔离与期限 | 不同账号读取/删除他人资源 404，缺 CSRF 拒绝，合法 Range 字节精确匹配；下载不改变 expires_at |
| Q13 空间用满 | 真实 TCP 登录、下载原文件和删除成功，删除后恢复 1 字节额度 |
| Q15/Q16 恢复与磁盘水位 | 到期后恢复实际删除；注入磁盘可用量为零时拒绝新增且账目不变 |
| Q18 单文件清理竞态 | 清理与两次主动删除并发，空间只释放一次；生成任务的同类竞态仍待第三批 |
| 10 账号并发 | 各写入并读取 256 KiB，hash 与内容精确一致，最后一轮耗时 2.859 秒；属于有界功能并发，不是持续生产负载承诺 |
| 实际清理线程 | 截止前资源仍可读，截止后实际删除；最后一轮测得延迟约 0.0503 秒。异常/宕机不承诺这一延迟 |

独立 Chrome 使用真实 PG 与 Storage：上传 262,181 字节、通过页面下载链接的浏览器 HTTP 客户端核对全部字节及 attachment 响应，按占用排序并等待实际列表更新，取消删除保留文件，确认删除后不可下载。切换账号后，旧链接 expected_account 返回 409，去掉提示参数仍由资源归属校验返回 404。未测试系统“另存为”对话框。浅色、深色、390 像素无横向溢出、窄屏删除弹窗均留存并查看实际截图；无页面 JS 错误。

浏览器脚本初次因项目按钮包含日期、排序控件定位及删除列表异步刷新而失败；根据实际页面快照修正定位、等待真实更新后复验通过。没有改低业务标准或把替身结果改写为 SQL 成功。每次失败也执行了自有进程退出和测试资源清理。

可复跑的入口：scripts/run_p07_validation.py、scripts/verify_p07_storage.py、scripts/verify_p07_extended.py、tests/e2e/p07-storage.cjs。结果见 output/validation/p07/database-validation.json、extended-validation.json 与 output/playwright/p07/real-ui-validation.json。测试生成的 UUID.bin 已清理，每个新测试账户 used/reserved 均为零；专属根标记与元数据保留。PG、API、清理线程和独立 Chrome 均已退出。

本次采用已有独立 Chrome/Playwright，不关闭 Codex 内置页面；运行期间未出现同类退出，不声称已修复 Codex 底层原因。没有新增依赖、修改环境或 git push。

第二批收尾再次运行 Python 定向套件，51 项通过、2 条已有弃用提示；架构及文档检查无新增错误。v2 的 427 个源码基线文件均未变，HEAD、暂存区、工作树与 phonetic_311 包元数据同恢复基准一致；证据写入 output/validation/p07/context-after.json。Git 根为 D 盘 v3 的独立 worktree，共用相邻 v2 的 .git；C:/、C:/Users、C:/Users/13680 均无 .git，不存在将该用户目录当成本任务仓库根的现象。

## 下一步及剩余门

003 [存储表与合成文件清理审阅](p07-migration-review.md) 已执行，不重复申请。下一步为 [004 任务文件关联扩展审阅](p07-job-assets-review.md)和[联合设计](../plans/2026-09-09-p07-job-files.md)：ZIP 展开/归档预算、多个输出、活跃输入到期、P06 旧 worker 文件提交拒绝与结果清单原子提交。004 尚未执行，当前 pipeline_check 仍仅返回任务元数据，生成文件任务尚未开放。

未验证 Windows 硬断电、Mac/Linux 原生文件锁与发行、生产磁盘权限/容量、负载和完整桌面文件流程。P07 未完成前不推进全面业务迁移，不将单文件代码存在等同于整体 verified。
