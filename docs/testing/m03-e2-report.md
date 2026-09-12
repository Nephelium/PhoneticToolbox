# M03-E2 网页、自然录音与切换竞争

2026-09-12。**E2 verified，限定 Windows 开发态 Qt 与本机托管 PostgreSQL/独立 Chrome。完整 E / M03 仍 in_progress。** 未部署生产服务、执行 DDL、修改原录音/v2/科学环境或更新旧 EXE。

## 实现与真实故障

`EggAnalysisPage.vue` 在开始读取新文件时关闭旧结果上下文，并为历史预览恢复增加结果窗口代次检查。此前只校验当前分析的输入代次，旧任务元数据回读迟到后仍可能恢复旧图并清空新文件选择。本轮先保持实际响应、延迟交付，再在等待中选择另一文件，稳定复现。修复后相同操作保留新选择，旧图不会出现。

保留失败证据：`output/validation/m03-ui/chrome-4d0c68647d064f01983dd36f6deed8d9/report.json`。修复后：`chrome-ed81caa24d7a43278078df26b91d4836/report.json`，含延迟旧预览与声道交换后图形/试听失效，两项通过、0 页面错误。验证命令为 `node tests/e2e/m03-races.cjs`，使用 Codex 已有项目运行时；修复前后均保留相同的响应延迟。

网页能力及 EGG 任务状态增加中文的空间不足、源文件失效/临近到期提示。没有改变配额、任务截止或科学算法。

## 真实网页验证

运行 `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m03_web.py`，进程内 `PYTHONPATH=backend/src;desktop/src` 使用绝对路径。复用已授权的专用测试 PostgreSQL 结构，新建本轮随机测试账号/项目与合成文件。所属服务、worker、浏览器通过已有入口启动和退出，不增加产品 HTTP 宿主。

最终证据：`output/validation/m03-e2/web-ba8633b522d44a04b5f3016ad37338f3/report.json`、`web-report.json`、`limits-report.json`，success=true、schema_applied=[]、postgres_stopped=true。

- 登录/上传、交互预览、单文件、IF和批次经过真实 API/数据库/独立 MKL 科学子进程。10 份实际下载文件校验 SHA-256，含批次 CSV、三 PNG、来源 JSON 和 IF 两 WAV；两 WAV 各 5292 帧、44100 Hz。
- 刷新浏览器保留任务。独立账号读取他人任务和结果均返回 404；同一浏览器退出后改用另一账号，文件、图形和任务列表为空。
- 为本轮账号预留剩余额度，真实 EGG worker 返回 quota_exceeded，无成功 manifest。取消测试预留后 reserved_bytes=0，失败任务无存活临时文件。
- 对本轮指定结果调整到期时间后，真实 HTTP 拒绝下载。临近到期的另一输入在建任务前以 input_unavailable 拒绝。随后显式调用现有 cleanup，过期结果文件实际消失。
- 到期时间由测试有意调整，清理由测试显式触发。此证据验证时间边界和同一清理实现，不冒充真实经过七天、后台定时唤醒或生产负载验收。正常调度历史证据仍见 P07。

追加限制阶段首轮因两处 alert 同时匹配而发生测试定位器错误，保留 `web-91f6e21bdae8468fb466b33ca3d5bc53/limits-browser.log`。定位器现按明确错误文本匹配，实际 HTTP 状态和任务失败断言保留。没有把测试误定位归因于科学计算。

## 自然录音与 Qt

运行 `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m03_qt.py --include-private`。仅使用 P03 已确认的 EGG-01/EGG-05，测试前后原文件哈希与冻结记录一致，声道方向分别按原确认值验证。测试输入副本、缓存、图和记录仅在忽略的本机验收目录，不加入源码/发行物。

最终证据：`output/validation/m03-ui/qt-4531f1c5bce94ddbaea7c434fba4a61b/report.json`，10 组通过。两文件既检查开头 0–0.5 秒，也检查按半秒平均绝对振幅选取的较响片段：EGG-01 为 6.5–7.0 秒，EGG-05 为 8.5–9.0 秒。实际四图、元数据、帧范围、角色与非零音频/Praat数据一致，分别有 45 / 50 个有效 Praat 点。相关截图已目视检查。

这不构成自然录音的生理有效性验证、人工声卡听感或全文件每个 ROI 的验收。原版数值等价继续以 B/C 的独立基准为准，本轮没有重复计算核心期望值充当新基准。读取自然 WAV 时一项未识别的非数据 chunk 提示保留，原文件和音频样本未改写。

初次自然录音脚本在模块重开后直接刷新，缺少重新授权输入目录，因而超时；修正测试操作顺序后通过。保留 `qt-9bd350d3511143afb9e4f046dbc8d5c7/report.json`，没有绕过原生文件授权。

## 回归与剩余项

`npm --prefix frontend test`：49 passed；`run typecheck`、`run build`、`run contracts:check` 通过。`scripts/check_architecture.py` errors=[]。`.venv/m09-ui/Scripts/python.exe -X utf8 scripts/validate_docs.py` 检查 540 份文件、328 条来源和 32 项任务，errors=[]；原封存历史快照的缺失链接仍单列保留。`git diff --check` 通过。本轮科研值与核心 wheel 未更改，未重复无关的整套科学构建。

A02 的本轮迟到恢复、A26 的受控配额/到期、A27 的双账号读隔离与切换已补充限定证据；A30 增加服务器三路径与自然录音页面，仍不含 F 冻结程序。当前剩余收口为 E3：长文件/5–5000 ms 微观窗差异、来源书目/页码与许可、尚未覆盖的设备/字体预检边界。M01 清理的历史一次 WinError32 仍未在本轮修改或宣称修复。M04 不接续。
