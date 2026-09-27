# P07-POLICY 代码候选、验证与统筹交付

**2026-09-27 更新：** 后续 A 任务已完成公共接线、M08 复制期限保护、M14 明确分类、版本 manifest 和契约生成，见[收口报告](p07-policy-closeout-report.md)与[前置交接](2026-09-27-prerequisites-handoff.md)。目标专用库仍未迁移。以下“共享接线待完成”仅是本报告原轮次停止点。

2026-09-26。**本任务认领文件的代码与定向验证已完成。完整 P07-POLICY 尚待 M08 共享文件稳定后串行接线。迁移方案待确认，现存/服务运行库尚未生效。** 井井明确回复“仍在修改，先交接清单”，因此本轮没有修改 main.py、公共任务入口、job_models.py、files.py 或生成契约。

入口：[文件与精确接线清单](../plans/2026-09-26-p07-policy.md)、[006 迁移审阅及回滚边界](p07-policy-migration-review.md)、[用户帮助](../manual/storage-policy.md)。不更新总台账、根 AGENTS/README，不推进其他模块。

## 实现范围

| 文件 | 行为 |
| --- | --- |
| backend/src/ptb_api/storage_policy.py（新增） | 唯一运行时新政策源：1,000,000,000 字节、259,200 秒，24 小时暂存期限；显式命名历史读取常量、版本 1/2、科学重算操作集合 |
| backend/src/ptb_api/quota.py | 从政策源导入，精确整数边界，保留原子预留/结算/派生 expiry 算法 |
| backend/src/ptb_api/storage.py | 按数据库实际政策返回 usage；旧库阻止新写入但允许读删；上传 finalize 标政策 2 并三天到期；已有 ready 资产不续期 |
| backend/src/ptb_api/storage_models.py | UploadInput 新上限随单一源变化；AssetView 增量 policy_version，StorageUsage 增量 policy_version/retention_seconds/over_quota，兼容旧响应缺字段 |
| backend/src/ptb_api/acoustic_models.py | 移除独立 604800 常量与七天报错；旧 manifest 缺版本按 1 读取，版本 2 验三天及 1 GB，历史结果大小保持可读 |
| frontend/src/account/ProjectStorage.vue | 十进制 GB、精确字节/秒帮助；显示后端额度；旧库/超额禁新增、保留下载删除；输出预算用服务返回额度 |
| backend/migrations/006_storage_policy.sql（新增） | 原约束兼容升级提案、版本/切换时间、超额增长保护 trigger、新建资产校验；不改任何旧到期时间、manifest、snapshot、预留或资产字节 |
| scripts/p07_policy_preflight.py（新增） | 显式 READ ONLY 聚合检查，没有 apply 和数据清理功能 |
| 政策/契约/PG/UI 测试 | 新增 compat、fresh PostgreSQL 和浏览器测试；更新当前政策的旧单元预期和 M01 当前 manifest 测试的显式版本 |

`acoustic_boundary.py` 通过 acoustic_models 的兼容导出别名已取得当前 TTL，未改其源码。科学单任务预算、内存预算、历史测试报告及已执行 SQL 均未全仓替换。

## 实际验证结果

| 检查 | 实际结果与限制 |
| --- | --- |
| Windows Python 定向套件 | **108 passed**，2 条既有 FastAPI/Starlette 弃用警告；其中 **11 项真实 PostgreSQL 集成**，其余 97 项政策/存储 HTTP/归档/科研契约测试 |
| WSL 原生 Python | **97 passed**，1 条既有弃用警告；NInfer，Linux 6.6.87.2-microsoft-standard-WSL2，现有项目 Linux Python 3.11 环境；读取当前源码，不是 Windows Python。未在 Linux 创建 PG，未验证 Linux PG/文件锁链路 |
| 前端类型/测试/构建 | typecheck 通过，**137 tests passed**，build 通过；现有大 chunk 警告保留，未调整阈值 |
| 实际 Chrome UI | **3 状态通过**：旧政策待迁移、已迁移但超额、已迁移且可写；上传/生成/展开禁用正确，下载/删除可用；公共删除框可取消；1280 宽浅色和 390 宽深色无横向溢出，0 pageerror。接口为明确合成响应，不能替代真实 DB UI 联合验收 |
| 截图检查 | 已查看 over-dark-narrow、ready-light 两张实际截图；共输出六张状态/主题截图 |
| 架构检查 | scripts/check_architecture.py 返回 errors=[] |
| 生成契约只读检查 | **退出 1，预期未整合差异**：openapi.json、acousticfilemanifest.json、resultmanifestenvelope.json。按并行边界保留，不报告契约全绿 |
| 003 历史文件 | SHA-256 `f27fd45be2a4afabf8162c3f695787816da38d14a719b0ba098065617c8effc2` 与原 P07 执行报告相同 |
| 定向 git diff --check | 通过，只有仓库 CRLF 转换提示，无空白错误 |
| 文档检查 | 检查 743 份文件，退出 1，仅原有两处 M10-R5 EXE 缺链；本轮新增文档链接无错误。未为修复历史链接重建或删除旧包 |

## 新的真实 PG 覆盖

测试不接受现存 DSN。显式设置 PTB_POLICY_FRESH_PG=1 后，用项目已有 PostgreSQL 17.11 创建随机独立空集群、随机回环端口、每例独立数据库和资产根。未读取服务凭据。仅本轮合成资产允许测试删除，集群和日志保留于 output/validation/p07-policy/fresh-*，进程按所有权停止，最终无 postmaster.pid。

- 未迁移 schema 上读取、下载、删除旧文件成功，新写入返回 storage_policy_migration_required，usage 仍显示实际 5 GB/版本 1。
- 新迁移保留全部旧资产列、到期时间及在途 reservations；旧成功任务 snapshot 和七天 manifest 字符串逐字保持不变并可经 JobView 回读。
- 超额旧账号迁移成功，used/reserved 未清零，available=0。登录、usage、HTTP 下载、直接删除成功，新增上传和旧预留继续追加被拒绝。
- 四个实际并发 Storage 请求争最后一字节，恰好 1 个成功、3 个 quota_exceeded；直接数据库增长同样被 trigger 拒绝。
- 固定时钟验证新上传精确 259,200 秒，截止前可读，截止时禁读；下载不改变 expires_at。
- unlink 故障注入保留实物和 used/reserved，重试成功才释放；磁盘低水位注入拒绝写入，账本未变化。
- fsync 后/账本结算前故障注入，超额状态下恢复把已覆盖字节从 reserved 转 used，合计不增长，旧在途截止不变。
- 已全部写完的旧在途上传在超额状态可 finalize，释放预留并采用新三天及版本 2。
- 实际工程结果从成功提交起三天，ZIP 不晚于旧输入更早截止；既有真实 FilePipeline 产物可读取。这里未运行声学算法，不将 storage_check 视为声学模块验收。
- 输入到期/旧 worker 代数拒绝成功提交，不产生 manifest；清理请求取消在途任务，失败收尾只删除本轮临时输出。
- 迁移锁内账本再检查发现不一致时失败，ROLLBACK 后旧额度仍在、无新增 policy_version 列，验证事务原子边界。

额度大值通过持久预留测试，未为测试写入 1 GB/5 GB 实体文件。以上包含受控时钟、磁盘水位和删除故障，不冒充自然三天经过、硬断电、生产负载或公网下载。

## 实际命令与产物

工作目录为 `D:/PhoneticToolbox/PhoneticToolbox_v3`。Python 明确从当前 backend/core 源码导入，未安装 wheel 或更新项目/全局依赖。

```powershell
$env:PYTHONPATH='backend/src;packages/phonetic_core/src'
$env:PTB_POLICY_FRESH_PG='1'
& '.venv/v3-dev/Scripts/python.exe' -X utf8 -m pytest -c tests/pytest.ini backend/tests/test_storage_policy.py backend/tests/test_p07_policy_compat.py backend/tests/test_p07_policy_postgres.py backend/tests/test_storage_boundary.py backend/tests/test_archive_policy.py tests/contracts/test_m01_contract.py -q --junitxml=output/validation/p07-policy/windows-final.xml

npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
node tests/e2e/p07-policy.cjs
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/check_architecture.py
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/validate_docs.py
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/generate_contracts.py --check
```

WSL 命令（无安装、无数据库、无服务或系统设置变更）：

```powershell
wsl -d NInfer -- env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/d/PhoneticToolbox/PhoneticToolbox_v3/backend/src:/mnt/d/PhoneticToolbox/PhoneticToolbox_v3/packages/phonetic_core/src /home/ninfer/ptb-p11-20260926/venv/bin/python -m pytest -c /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/tests/pytest.ini /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/backend/tests/test_storage_policy.py /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/backend/tests/test_p07_policy_compat.py /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/backend/tests/test_storage_boundary.py /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/backend/tests/test_archive_policy.py /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/tests/contracts/test_m01_contract.py -q -p no:cacheprovider --junitxml=/mnt/d/PhoneticToolbox/PhoneticToolbox_v3/output/validation/p07-policy/wsl-final.xml
```

WSL 启动时提示现有 localhost 代理未镜像到 NAT 环境，测试没有联网，不改代理设置。Junit 结果位于 output/validation/p07-policy/windows-final.xml 和 wsl-final.xml；UI 的 checks.json 与六张 PNG 位于其 ui 子目录。合成 PG 日志及数据均保留于各 fresh-* 子目录，未迁移或清理旧 P05/P07 目录。

初轮问题已如实保留：测试 ROOT 计算多退一级导致 8 项跳过，修正后实际执行；Windows pg_ctl 输出管道被子进程继承而超时，停止本轮自有实例并改用日志文件后通过；M01 当前期限测试补显式 policy_version=2，另有旧版本回读测试；浏览器 Vite preview 与正式 /server/ 静态挂载不同，测试只映射同一构建资产后通过。没有放宽业务边界或跳过失败用例。

## 统筹摘要与剩余门

代码认领范围可审阅，Windows 108 项、WSL 97 项及前端/三状态 UI 已通过。**不可标整个 P07-POLICY verified 或政策已生效。**

1. 等 M08 稳定后，按接线清单处理 FileConfig 新请求上限、历史 ResultFile 命名常量、FilePipeline 科学结果独立期限与旧在途输出版本更新，再生成/复验契约。EGG/LPC/语谱转音频当前仍可能继承更早输入期限，此处没有虚报为完成后三天。
2. 迁移候选须确认旧到期时间保留、超额读删/禁增、在途处理及停写窗口。006 仅独立合成库通过，现存/服务库未生效。
3. Linux PostgreSQL/磁盘锁、目标服务器资源/清理延迟/生产压力和远程节点回传验收保留独立状态，本轮没有部署、push、EXE、系统配置、CI/CD 或凭据变更。
