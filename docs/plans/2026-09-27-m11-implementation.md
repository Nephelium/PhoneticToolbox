# M11 独立 MFA 组件与正式任务迁移

**Goal:** 在统一工作台通过已有持久任务运行独立可选 MFA，保留既有对齐行为与用户数据。

**Architecture:** 主包只包含协议、组件管理、页面与外部进程边界。MFA 3.3.8 先以已有 Windows auto_alignment 环境建立基准，运行时与模型分别登记；每次运行使用独占 MFA_ROOT_DIR、临时目录与 SQLite，不连接账号 PostgreSQL。远程严格复用 remote/1，未具备能力的执行端保持关闭。

**Tech Stack:** 现有 Python 标准库/Pydantic/FastAPI、Windows Job Object/P11、Vue 公共组件。不新增全局依赖、不打包主 EXE。

状态：in_progress。用户已授权本设计实施。2026-09-27 起始分支 codex/v3-rebuild，大量既有并行差异保留。M05 已明确释放公共接线文件给本轮 M11，接线完成后需通知释放。

## A. 只读审计与独立基准

- 已读根/相关 AGENTS、架构、模块计划、8.1–8.3 说明书、四份旧源码、P07/P11/REMOTE 当前交付。
- 已定位相邻 V2 auto_alignment/env：Conda 记录 MFA 3.3.8，V3 仅有 bat。禁止调用旧 service 的 _ensure_runtime_app（会写入旧目录）。
- `scripts/verify_m11_runtime.py`、`tests/support/m11_baseline.py`：版本/依赖/原生程序/模型 hash、公开合成 WAV/LAB、旧 pipeline 双轮 TextGrid 回读及资源记录。源码由本轮读取/受控加载，不作产品动态依赖。
- 原始证据仅 `output/validation/m11/`，现有环境用 -B 禁止 pyc；不更改旧配置、语料、模型。

## B. 独占实现

- `packages/phonetic_core/src/phonetic_core/transcription/mfa_name_codec.py` 原样迁入名称编码，独立记录后续安全验证，禁止静默改编码。
- `backend/src/ptb_worker/mfa/`：组件可信清单、安全导入/版本目录与原子切换、已有环境检查、固定外部 child、任务独占缓存/进程树、资源及溯源。
- `backend/src/ptb_api/m11_models.py`：版本化请求与结果。无 DDL。
- `backend/tests/test_m11*.py`、`tests/parity/test_mfa.py`：先缺转写/编码/路径穿越/旧版本保护/取消等回归，再实现。
- 离线包不得自我认证：清单摘要必须由可信目录或用户从独立来源提供。未发布固定 URL 时线上入口显示待发布，不拼造下载地址。

## C. 正式宿主串行接线

- 独占 `backend/src/ptb_worker/m11_task.py`、`m11_executor.py`，`desktop/src/ptb_desktop/m11_bridge.py`。
- 串行最小改动 `backend/src/ptb_api/jobs.py`、`job_models.py`、`main.py`，`ptb_worker/executor.py`、`store.py`、`files.py`、`local_acoustic_files.py`，桌面 `host.py`/`task_bridge.py`。读取最新差异后补丁，不覆盖其他任务。
- 现有 owner/project/idempotency/输入 hash/租约/fencing/配额/TTL/发布事务。云端未知执行能力不可自动回退。
- M11 原生临时目录按实际受控预算预留，不将绕开 writer 的目录当作已具备服务器配额执行资格。未解决前 server execution 关闭。

## D. 公共界面

- `frontend/src/modules/mfa/{MfaAlignmentPage.vue,state.ts,port.ts}`、`frontend/src/platform/m11.ts`。
- 资源/组件/模型/词典/音频和已有转写/输出目录、Beam 10 / Retry 40 及联动、真实任务/取消/日志/结果、帮助来源、草稿与关闭保护。
- `AppShell.vue`、platform 接口与生成 contracts 仅串行接线，正式入口不注入测试 adapter。
- 无内部大标题/重复关闭，使用 ModuleFrame/Toolbar/Section/Status、公共字体/主题。

## E. 验证与退出门

拟执行（实际结果以报告为准）：

```powershell
$env:PYTHONPATH='D:/PhoneticToolbox/PhoneticToolbox_v3/backend/src;D:/PhoneticToolbox/PhoneticToolbox_v3/packages/phonetic_core/src;D:/PhoneticToolbox/PhoneticToolbox_v3/desktop/src'
& .venv/v3-dev/Scripts/python.exe -B -m pytest -c tests/pytest.ini backend/tests/test_m11.py tests/parity/test_mfa.py -q
& .venv/v3-dev/Scripts/python.exe -B scripts/verify_m11_runtime.py --help
& .venv/v3-dev/Scripts/python.exe -B scripts/verify_m11_wiring.py
& .venv/v3-dev/Scripts/python.exe scripts/generate_contracts.py --check
npm --prefix frontend run contracts:check
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
& .venv/v3-dev/Scripts/python.exe -B scripts/verify_m11_qt.py --help
& .venv/v3-dev/Scripts/python.exe -B scripts/verify_m11_qt_logs.py --help
```

Windows、WSL、实际服务器、实验室节点分列。现有 P06-REMOTE 仅独立协议、正式路由/文件发布未接通；trusted-worker 关闭；WSL NInfer 缺 systemd 可委托资源组。不能将新增 M11 单元测试说成真实远程验收。需要公共负责人完成既有接线后再验节点链路，不另建队列或认证。

最终交付 `docs/modules/evidence/M11-source-map.md`、`docs/manual/mfa.md`、`docs/testing/m11-report.md`、组件清单与测量；未完成项明确 in_progress/blocked，不扩大 verified。主 EXE 体积仅检查构建清单与代码增量，完整组件下载/解压量需实际候选包才报告。

## 当前交付

2026-09-27 Windows 本地 A–D 及 E 的限定范围已完成实际验证，见 `docs/testing/m11-report.md`。原 V2 基准实际失败，后续运行适配与拒绝规则在 ADR 单列，不写成旧数值等价。API/Qt/独立运行/离线组件和新建 PG 认证等待均有真实链路证据；完整模块保持 in_progress，Linux/节点/服务器/公开发行待门禁。最终浏览器交互以实际 Qt 的生产前端为证，不再使用初拟未创建的独立 m11.cjs 命令。公共接线已向 M05 释放。
