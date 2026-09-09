# P07 任务文件与 ZIP 联合验收

2026-09-09。**verified，限定 Windows、本机 PostgreSQL、受控文件生成与下述 ZIP 支持范围。** 这不是完整 v3、科学算法迁移、生产负载、硬断电或跨平台发行验收。

井井对 [004 审阅](p07-job-assets-review.md) 回复“好，允许”，已在原专属测试库应用 004。没有新增数据库、移动语料或扩大生成文件清理目录。003/004 的实际 SQL 指纹、旧行保留和数据库正常停止见 output/validation/p07/database-validation.json；004 SHA-256 为 209d2271d143caaa1dc8538da829a48c9e6687a3a1a975fb23aa9c54e9a16497。

## 完成行为

- 任务提交与输入快照/关系同事务，数据库联合外键保证同账号同项目。认领前复查输入可用性；只配置旧元数据能力的 worker 不会认领文件任务。
- 受控 writer 复用既有 Storage 账本。每次创建、追加、读取与最终提交核对 worker/代数/租约/取消/输入期限；已经预留的在途块仍计量，不能因取消清零。输出在整批成功前不可下载，公开上传接口不能追加/最终化 worker 输出。
- 输出尺寸/hash、所有 ready 标志、未用预留释放、任务成功、事件和 manifest 在同一事务内完成；物理检查后再次核对期限，跨截止则全批回滚。取消或失败时，实有字节在真正删除前仍占用空间。
- storage_check 生成有明确标签的确定性工程测试文件，验证结果批次，不声称做了语音分析。archive_zip 打包所选资源，extract_zip 展开为同项目独立资源；不写系统临时 ZIP 或用户路径。所有归档头、数据、描述符与尾部也经过计量。
- 实际新生成结果从成功提交起算 7 天。归档与展开不超过最早输入期限；下载不续期。删除输入或暂存输出会请求停止活动任务，已完成的其他结果保留独立期限。
- 网页增加资源类型、文件选择、输出预算、打包/展开/存储流程检查，以及删除影响说明。任务记录区分操作名称和失败原因；账号切换仍卸载状态，所有资源访问重新验证身份。

## 实际验收

| 范围 | 验证方法与结果 |
| --- | --- |
| Q01/Q02/Q05/Q06/Q09/Q11/Q12/Q13/Q15/Q16/Q19 单文件门 | 原有真实 PG/磁盘、最后 1 字节竞争、未知长度、删除失败、满额登录下载、TCP 中途到期和启动清理复验通过，详见 [单文件报告](p07-storage-report.md) |
| Q04/Q20 多文件与结果 TTL | 两份 777 字节确定性输出逐字节/hash 匹配；输入改为剩余 1 小时，重新生成结果仍从任务成功起算 7 天；公开上传接口拒绝修改结果 |
| Q14/Q20 归档与展开 | ZIP 往返内容与 hash 一致，归档和展开资源截止严格等于原输入截止；180 字符最大显示文件名能够往返 |
| Q03 不可信 ZIP | 实际上传并处理路径逃逸、高展开比、伪造超大中央目录，任务失败且无成功 manifest；纯政策用例还覆盖链接、加密、反斜杠、驱动器、多级路径和不支持的方法 |
| Q04/Q14 额度边界 | 第二个输出超出预算/剩余额度时整批失败；ZIP 在可写入局部数据但容纳不下尾部时失败，原输入保留，临时输出物理删除后 used/reserved 回到原值；空间全满时提交新文件任务也立即拒绝 |
| Q12 真实外键 | 应用接口拒绝其他账号资源；绕过接口直接插入跨账号关联也触发 PostgreSQL ForeignKeyViolation |
| Q04 原生工具边界 | 任务契约拒绝 native_tool 和任意 output_path；没有强制总字节限制的原生目录输出没有开放，不能冒称已经实测该类工具配额 |
| Q07/Q17 活跃输入和提交 | 暂存文件不可下载；输入到期后读取与提交均被拒绝；显式取消后不能提交成功；旧代数写入与 manifest 提交都拒绝 |
| 提交过程中到期 | 在最终文件 stat 阶段注入短暂停顿，使输入跨过截止；最终检查拒绝并回滚所有 ready 标志与 manifest，不把事务早期的检查当成永久有效 |
| Q08/Q10/Q17 进程中断与重试 | 启动真实独立 worker，观察已写入的暂存块后终止该自有进程；按数据库时间等租约到期，启动恢复标 interrupted 并物理清理暂存；旧身份不能写/提交，显式新任务重试完成四份结果 |
| Q18 取消/删除竞态 | 后端完成与删除并发，序列化赢家决定是否成功；独立 Chrome 的两张标签页并发生成/删除也验证任务状态与 manifest 一致，不能失败却开放完整结果 |
| 实际网页 | 真实 PG、Storage、独立 worker 与 Chrome：上传、打包、展开内容 hash、两份生成结果、删除影响/取消删除、确认删除、跨账号旧链接、浅深色与 390 像素无横向溢出通过；已查看截图，无页面 JS 错误 |

程序使用的源版本始终为项目隔离 Python 3.11.14 与已有依赖。ZIP 来源新增 P07-PYTHON-ZIP，登记达到 300 条；复用标准库，未安装新依赖或复制声学算法。[Python ZIP 文档](https://docs.python.org/3.11/library/zipfile.html)用于核对流式输出与格式能力，实际支持边界由项目验证决定。

收尾回归：Python 定向套件 **60 项通过**，保留 2 条既有依赖弃用提示；前端 **8 项通过**，typecheck/build、契约和来源生成一致性通过。文档检查 207 个文件、300 条来源、32 项任务，无新增错误；未修改的历史快照缺链单独保留。v2 的 427 个基线文件、HEAD、暂存区、工作树、phonetic_311 包元数据保持一致。每轮核对保留此前已有的账号/会话/项目/限流/任务/事件/版本记录，具体数量见数据库 JSON；PG 已停止，测试根无残留 .bin。

## 支持限制与后续

ZIP 当前最多 16 个条目、128 KiB 中央目录、8 层逻辑名称、180 字符名、200 倍展开比；接受 stored/deflate，不接受 ZIP64、多卷、加密或链接。输出采用 stored 归档，超过当前标准库非 ZIP64 写入阈值会在生成前拒绝。短文件名加序号避免碰撞，最大长度名称保留原名；冲突名拒绝。逻辑子目录用“ · ”显示为平面资源，不建立真实目录树。

硬断电采用明确的 fsync/结算故障注入与自有 worker 终止来测试部分恢复边界，**没有拔电实测**。10 账号验证是有界功能并发，不代表持续生产性能。Mac/Linux 原生文件锁、生产磁盘 ACL/容量、原生工具受限卷与完整科学处理仍在后续阶段。用户本地原文件不应用网页 5 GB/7 天规则。

现有 P07 支持范围满足进入 P08/P09 具体模块规划的基础门；不自动授权全面算法迁移、生产部署或发布。后续科学模块仍需真实基线对照与模块级实施计划。

## 复验与交付证据

运行现有 .venv/v3-dev/Scripts/python.exe：

- `-X utf8 scripts/run_p07_validation.py --approved-p07-schema-and-test-files`：当前两次迁移均已存在，默认只验证，不重复 DDL。包含单文件、真实网络/浏览器、独立 worker 和联合文件脚本。
- `-X utf8 -m pytest -c tests/pytest.ini backend/tests tests/security tests/contracts desktop/tests packages/phonetic_core/tests tests/architecture -q`。
- `-X utf8 scripts/generate_contracts.py --check`、`scripts/check_architecture.py`、`scripts/validate_docs.py`；前端 typecheck/test/build、contracts:check、ui-data:check。

真实结果存于 output/validation/p07/jobs-validation.json、extended-validation.json、database-validation.json，浏览器结果为 output/playwright/p07/real-ui-validation.json，截图包括 joint-storage-light.png 和 joint-storage-dark-narrow.png。原始失败日志保留为历史证据，当前状态以有成功检查项的 JSON 和最后检查记录为准。

本轮没有调用 Codex 内置浏览器关闭/清理接口；独立浏览器、worker、API、清理线程、PG 均按所有权退出，专属根只保留标记/锁，生成测试字节清理后新账号用量归零。未触发同类 Codex 退出，不将绕行措施表述为底层缺陷已修复。没有 git push、环境/凭据修改或 v2 文件变更。
