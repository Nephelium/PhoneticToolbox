# M03 实时更新修复

2026-09-29，井井要求对照 V2 说明书与源码仔细修复慢加载、完成后空白和实时更新。状态：verified，限定 [报告](../testing/2026-09-29-m03-realtime-report.md) 中的 Windows 源码/Chrome/Qt；完整 M03、跨平台与冻结发行不扩大。

## 已查明

- 原 V2 `Phonetic_Export/index.html` 3.1–3.4、`egg_widget.py` 的加载回调、ROI 与微观更新直接关联绘图。V3 手动提交与繁忙时丢弃更新偏离原交互。
- 当前本机两项任务均成功并有完整三文件结果，耗时约 39.75 / 36.67 秒。截图选区与两项任务的选区均不同。当前页面拒收旧选区结果后没有自动补算最新选区。
- 公共 Windows `collect_pipe` 每收到 4096 字节仍 sleep(0.005)，对完整归一化音频造成传输节流。相同实际输入直接计算约 2.29 秒（含 profiling/import）。

## 修复与退出条件

1. 公共管道只在无可读数据时等待，保留逐块取消、超时、大小、所有权及退出检查。用真实大结果、取消和越界检查验证。
2. EGG 加载后自动分析。参数和手势防抖，繁忙时合并最新请求，完成后自动补算。旧结果只能用原快照坐标显示并标明更新中，禁止将其用于当前选区导出。文件切换仍清空并隔离迟到响应。
3. 保留显式更新与取消、历史恢复、批处理/导出。科学核心、契约、数据库和 V2 不变。
4. 验收：`node tests/e2e/m03-realtime.cjs`，`python -m pytest backend/tests/test_pipe_drain.py`，既有 M03 定向后端检查，前端 test/typecheck/build，真实 77 秒输入前后输出逐字节比较及时间测量。平台证据单列，不宣称跨平台或冻结 EXE 已验。

涉及 `frontend/src/modules/egg-analysis/EggAnalysisPage.vue`、`backend/src/ptb_worker/native/reaper.py`、回归脚本、说明与报告。保留工作区已有其他改动，不 push、不改现存库、不发布。

追加定位：正式路径仍有 470 次写入，scratch 累计 11.56 秒。按适配器既有 1 MiB 限制合并本机 EGG 读写，完整执行由 16.85 秒降至 3.96 秒；保持 fsync、租约、配额与原子发布。科学事件按输出需求执行，普通预览不做无用的全段检测。详见 [ADR-M03-RT](../decisions/ADR-M03-realtime.md)。
