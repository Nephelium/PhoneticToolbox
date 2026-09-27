# P07-POLICY 公共接线与迁移准备收口

2026-09-27。本轮任务 A。**代码、契约、Windows 独立合成 PG 和 WSL 非 PG 政策检查已完成；目标专用库未迁移，新政策未在该库生效。** 后续入口为[前置交接](2026-09-27-prerequisites-handoff.md)及[具体迁移审阅](p07-policy-migration-review.md)。本报告补充原 [P07 报告](p07-policy-report.md)，不扩大原模块验收。

## 行为与文件

- `backend/src/ptb_api/storage_policy.py` 为新政策唯一运行时来源：1,000,000,000 字节、259,200 秒。账户磁盘额度与 P11 的进程组内存预算各自独立。
- `job_models.py` 的新 FileConfig 上限引用 QUOTA_BYTES，历史 ResultFile 引用 LEGACY_QUOTA_BYTES 保留合法 5 GB 读取。旧失败文件任务若 snapshot 预算超新上限，retry 返回公开的 `output_budget_exceeded` / 413，旧 snapshot 不改，不隐式缩减预算或抛内部 ValidationError。
- `files.py` 发布以完整 snapshot 分类。科学结果从最后一次 fencing 成功、原子发布时刻起三天。ZIP、展开、纯切分取三天与所有输入记录/当前资产截止的最早值。读取、下载不更新时间，重试继续检查原输入截止。
- M08 `saved_copy` 为真，或存在 `source_ref` / `copy_result` 时，均按复制处理。真实保存任务的 input/job_assets 指向被复制的结果，期限从这些受管引用读取，不能从原音频身份字段推导新期限。保留原复制 worker、PCM 字节、流式接收和 generation fencing。
- M14 preview/export 都在 `m14_jobs.execute` 中重新 load → `PhonologyRules.analyze`，export 随后 configure/export。因此当前两种输出显式属于科研新生成，成功后三天。将来若新增缓存复制/保存接口，必须另行按复制语义接入，不能仅沿用 operation 白名单。
- FileManifest、AcousticTaskManifest、EggManifest、LpcManifest、Spec2WavManifest、M08Manifest、M14Manifest 均增加 `policy_version`，缺省 1 供历史回读。PG 和本地新发布器显式传 2；本地 `expires_at=None` 保持无服务器 TTL。D 阶段 AcousticFileManifest 已有版本兼容规则，正式 worker 发布的是 managed 清单。
- PG 最终资产更新在同一发布事务内写 expires_at 与 policy_version=2，覆盖迁移前创建、迁移后完成的输出。历史 ready 资产和已成功 manifest 不回写。
- `acoustic_boundary.py` 直接从政策源导入 TTL。现有模型兼容别名保留。新政策未混入科学默认参数、输入采样率、每任务计算/导出限制。
- `scripts/p07_policy_preflight.py` 只读检查 schema 版本、实例身份、冻结状态、账本、磁盘未知/缺失/尺寸不一致及元数据指纹。指纹不含可恢复数据，不是备份。
- 新增 `scripts/p07_policy_database.py`：默认 show；apply 需要精确授权标记和已审核 hash。只允许指定本地目标，拒绝已有运行集群、其他 DB 会话、running/cancel_requested 任务和预检异常；调用既有进程所有权 harness。提交后比对原资产、used/reserved、snapshot/manifest 指纹，失败保持停写，绝不自动恢复旧快照。目标库的 apply **未执行**。

## 本轮验证

证据根：`output/validation/p07-closeout-20260927/`。所有 PG 用例仅本轮随机新建、回环隔离集群，每例独立库/资产根，执行 001–006（含 005），保留集群数据/日志，结束后只停止自身实例。没有连接目标专用库，没有执行 007。

| 范围 | 实际结果 |
| --- | --- |
| Windows 政策/存储/发布/契约最终主套件 | **169 passed**，2 条既有框架弃用警告；`windows-policy-final.xml` |
| 新迁移执行器隔离验证 | **1 passed**，17 deselected；拒绝错误 hash/目标，真实合成库迁移并核对指纹；`migration-runner.xml` |
| M14 preview/export 明确分类补充 | **5 passed**，40 deselected，与主套件有重叠，不累加；`m14-publication.xml` |
| WSL 原生 Python 非 PG 政策/契约 | **141 passed**，1 条既有弃用警告；`wsl-policy.xml`，NInfer 既有 Python 3.11 环境，无安装/系统变更 |
| M08 本地真实原生链路 + 服务器 ZIP 门 | **20 passed**，1 条 JUnit 属性 warning；`m08-gates-fixed-path.xml` |
| M14 回归 | **14 passed**；`m14.xml` |
| M14 正式 Windows 本地 HTTP/持久任务 | **8 组通过**，五格式、完整三文件/hash、损坏输入、取消；`output/validation/m14/wiring/ef4d4bac77a44330a4084e2ddd3eb354/report.json` |
| 契约生成及双侧漂移检查 | Python generator 与 npm contracts/check 通过，保留六条 M08 和一条 M14 路由、JobView/结果 envelope 联合类型 |
| 前端初轮 | typecheck 通过，**140 tests passed**；没有 UI 改动，未重复构建或声称视觉复验 |
| 并行 M06 接线后的收尾 | 契约重新生成及两侧 check 通过，**124 项政策/契约回归通过**（`final-contracts.xml`）；全局 typecheck 出现 M06 的 3 项 TS2352，见下文，不能报告最终全局类型全绿 |
| 架构/版本同步 | `errors=[]`、`version_drift=[]` |
| 文档检查 | 845 文件，唯一错误为根 README 既有 M10-R5 EXE 缺链；本轮文档无新增缺链 |

主套件新增覆盖真实 PG 发布所有已接操作、源截止更短/三天上限、历史 5 GB 读取、超额读删/禁增、迁移前在途科学和复制输出的版本与期限、下载不续期、重试不延长输入、迁移原子失败、fencing、受控到期和清理。负载均为合成数据，跨操作 publication 测试写任意已知字节验证公共事务，**不冒充各模块科学计算或目标账号网页验收**。

最初 25 项新增无数据库回归为 18 failed / 7 passed，复现上限、版本字段与分类缺口。M08 初轮 4 failed / 16 passed 是本轮命令的相对 PYTHONPATH 在子进程工作目录失效，改绝对路径后原断言通过，未修改科学门。旧超预算 retry 的新增回归先复现内部 ValidationError，再修公开错误。

收尾时 D 并行加入 `M06Manifest`、speech_synthesis 联合类型及发布器分支。A 保留这些内容并重生成契约，不覆盖 D 的实现。最终 typecheck 错误位于 `frontend/src/modules/speech-synthesis/state.ts:4`（JSON 数组转固定元组）、`:6`（f0_range 元组）和 `frontend/src/platform/m06.ts:20`（Result 缺 metadata）；归 D 当前实施范围，A 未加 unknown cast 或修改其页面掩盖错误。交付是公共政策完成与明确的全局整合阻断，不能把初轮 typecheck 通过扩大到最新 M06 工作区。

## 可复跑命令

```powershell
$env:PYTHONPATH='D:/PhoneticToolbox/PhoneticToolbox_v3/backend/src;D:/PhoneticToolbox/PhoneticToolbox_v3/packages/phonetic_core/src;D:/PhoneticToolbox/PhoneticToolbox_v3/desktop/src'
$env:PTB_POLICY_FRESH_PG='1'
& '.venv/v3-dev/Scripts/python.exe' -X utf8 -m pytest -c tests/pytest.ini backend/tests/test_p07_publication_postgres.py backend/tests/test_p07_policy_postgres.py backend/tests/test_p07_policy_wiring.py backend/tests/test_p07_policy_compat.py backend/tests/test_storage_policy.py backend/tests/test_storage_boundary.py backend/tests/test_archive_policy.py tests/contracts -q
# 以上集合后追加 runner 和 M14 export 用例，重跑数量会增加；以各次 XML 为准。
& '.venv/m09-ui/Scripts/python.exe' -m pytest -c tests/pytest.ini backend/tests/test_m08_wiring.py backend/tests/test_p11_file_gate.py -q
& '.venv/m14/Scripts/python.exe' -m pytest -c tests/pytest.ini backend/tests/test_m14.py -q
& '.venv/m14/Scripts/python.exe' scripts/verify_m14_wiring.py
& '.venv/v3-dev/Scripts/python.exe' scripts/generate_contracts.py --check
npm --prefix frontend run contracts:check
npm --prefix frontend run typecheck
npm --prefix frontend test
& '.venv/v3-dev/Scripts/python.exe' scripts/check_architecture.py
& '.venv/v3-dev/Scripts/python.exe' scripts/sync_versions.py --check
& '.venv/v3-dev/Scripts/python.exe' scripts/p07_policy_database.py show
```

WSL 本轮命令使用 `wsl -d NInfer -- env PYTHONDONTWRITEBYTECODE=1`，PYTHONPATH 为本仓库 backend/src、packages/phonetic_core/src 的 `/mnt/d/` 绝对路径，解释器 `/home/ninfer/ptb-p11-20260926/venv/bin/python`。执行上述主集合去掉两个 postgres 文件，`-p no:cacheprovider`。WSL 启动的既有 localhost 代理提示不影响离线测试，未修改代理。

## 尚未完成

目标库在线版本/磁盘联合预检、006 实际迁移、停止/恢复该目标写者、M04/M08/M14 新政策双账号网页联验未执行。Linux PG/文件锁、新政策整合包重新生成真实 capability receipt、生产 TLS/站点及远程节点联验待后续授权/能力门。本次无 push、生产部署、EXE、额外整库备份、全局依赖或凭据变更。
