# M01-G 联合验收与说明书

2026-09-10，本轮联合验收批次 verified，完整 M01-G 仍 in_progress。井井在 F2 交付后回复“好，继续”，按既有 M01-G 计划执行；不需要新数据库表或环境依赖。实际结果和旧格式剩余项见[联合报告](../testing/m01-report.md)。

## 本轮可审阅范围

1. 检查 v2 §2.1 与结果加载源码。M01 导出需能被旧版 `load_excel` / `load_fastdb_window` 实际回读；任意历史结果导入与可信父结果来源是另外的证据，不能靠格式相似声称实现。保留 M01-A39 的未完成部分及 M02 的旧格式加载门。
2. 修复科学子进程到持久任务的错误原因丢失：固定白名单代码区分 WAV、TextGrid、唇形、采样预算、计算/导出预算、无参数帧及切分失败。不记录异常原文、路径或标签，不改变参数公式、范围或旧任务错误语义。
3. 独立 Chrome 实际选择并上传 WAV/TextGrid，完成分析和同步切分；下载 XLSX/SQLite/来源 JSON 并由独立读取器逐值核验。检查 390px、常用桌面与宽屏、主题、帮助、失败后的页面可操作性。桌面继续复用实际 Qt 与离线持久任务脚本。
4. 对已授权本机自然录音做只读定向比较，使用 P03 独立 v2 捕获；不上传、提交或改写语料。记录实际覆盖，不把少量样例说成完整语料验证。
5. 更新软件内操作帮助、`docs/manual/parameter-estimation.md`、`docs/testing/m01-report.md`、验收表和阶段账本。缺少的证据继续标明，不用一张截图或构建通过关闭整个 M01。

## 文件与验证

主要修改：`backend/src/ptb_worker/` 错误传递与持久白名单；`frontend/src/modules/parameter-estimation/BatchResults.vue`、共享帮助及项目文件提示；`tests/e2e/m01-tasks.cjs`、`scripts/verify_m01_web.py`、定向错误/自然录音验证脚本；上述说明书和证据文件。

命令：`pytest` 定向 M01/持久文件回归；`run_m01_validation.py --approved-m01-schema-and-synthetic-tests --verify-web`（不加 apply）；`verify_m01_local_tasks.py` / `verify_m01_task_window.py`；前端 typecheck/test/build；架构/契约/来源/文档检查。安装 wheel 后进行真实服务验证。SQLite 写入与保护哈希检查依次执行。

仅操作自行启动的 Chrome、Qt、API/worker 和已授权固定测试 PG。继续避开 Codex 内置浏览器关闭；没有证据可保证 Codex 本身永不退出。最后复核 v2 的 427 文件、HEAD/index/status 和环境，显式暂存本轮源码后本地提交，不 push。
