# M03-E3 微观范围与方法审阅

2026-09-12。本轮微观范围修复已验证，限定 Windows 开发态。**完整 E3 / E / M03 仍 in_progress**，长文件完整分析、部分方法原文/许可、设备/字体预检与冻结 EXE 尚未收口。

## 改动

- 微观数值输入、滚轮和键盘统一为 5–5000 ms，默认 50 ms。原手册10–200 ms与实际滚轮范围的差异已修正到当前说明书。
- 宽窗口先完整滤波和检测，再沿用 V2 二次幂抽点显示两条波形，快照记录 `micro_sample_stride`，图头显示步长。事件不抽稀，CSV/WAV 不受此显示规则影响。
- 事件列表上限从1000扩至10000，覆盖宽窗口下原1 ms最小峰距。300 Hz、5秒窗实际保留约1500个事件，未因旧上限拒绝或丢弃。
- 真实页面发现到达缩放边界仍重复提交任务，增加参数未变化时不提交的判断，原失败记录保留。
- 长录音在现有文件读取预算内可导航到EOF；超60秒或288万帧的完整科学计算仍明确拒绝。没有先切当前ROI再重新归一化/滤波，避免静默改变全文件预处理语义。

## 独立基准和回归

先运行新增端点用例，5/5000 ms均被旧API校验拒绝，随后实施。`scripts/capture_m03_ranges.py`在相邻v2原科学环境中离屏只读调用原EGGWidget，禁写pyc。使用公开合成输入重复8次生成6.4秒文件，两轮各捕获6种范围/中心×原始/滤波×两图。48个数组双轮精确一致，原记录不覆盖。

- 捕获：`.venv/m09-ui/Scripts/python.exe -X utf8 scripts/capture_m03_ranges.py`。
- 证据：`output/validation/m03-e3/v2-ranges-401b2b6f72d84168acb44d236d1bfb66`。
- 冻结数组：`tests/fixtures/m03/ranges.npz`，SHA-256 `7fb9f89d30addc2e33e62e98db9f0379e360252fc9698152fd42609422ba1402`；输入、原源码哈希、步长见同目录 `ranges-source.json`。
- 核心wheel：`scripts/Invoke-M03-Python.ps1 -PythonArguments @('-X','utf8','-m','build','--wheel','--no-isolation','--outdir','output/validation/m03-e3/wheels','packages/phonetic_core')`，仅通过 `uv pip install --python .venv/m03-compatible/python.exe --no-deps --reinstall output/validation/m03-e3/wheels/phonetic_core-3.0.0a1-py3-none-any.whl` 安装到M03兼容环境。wheel SHA-256 `4e4b1fdb61f6128646195b5d9a02d387147f233f0a8ac989356dd5b5aa6dd039`。
- `.venv/m03-compatible` 下，`backend/tests/test_m03_ranges.py`、`test_m03_preview.py`、`test_m03_exports.py`、`test_m03_export_names.py`定向pytest验证：59 passed，包含原版完整显示数组、8/48/96 kHz、宽窗事件、三PNG/双WAV/CSV及旧模式回归。
- `.venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini backend/tests/test_m03_contract.py -q`：14 passed，保留两条已有依赖弃用提示。
- `scripts/Invoke-M03-Python.ps1 -PythonArguments @('-X','utf8','scripts/verify_m03_core.py','--include-private')`：安装wheel后11样例31761项精确比较通过，`output/validation/m03-core/wheel-3fe1e2678e344717bed92729500cbdda/report.json`。原语料不改写，已有非数据WAV块警告保留。
- `npm --prefix frontend test`：50 passed；`run typecheck`、`run build`通过；`scripts/generate_contracts.py --check`、`run contracts:check`、`run ui-data:check`无漂移，`scripts/check_architecture.py` errors=[]。

首轮错误地将API契约测试一并放进纯科学环境，69项通过、2项因缺少psycopg/FastAPI失败；按既有环境分工重跑，不向科学环境安装这些依赖。科学用例和API用例的最终结果分别记录，不宣称该首轮全绿。

## 页面证据

`node tests/e2e/m03-ranges.cjs`，独立Chrome接真实本地任务与M03子进程，使用既有测试数据库的独立副本，schema_applied=[]：

- 最终 `output/validation/m03-ui/chrome-0528770179974651aa2519831eb2b342/report.json`，5组通过、0页面错误。5 ms和5000 ms从提交到绘图分别约5.86/7.51秒，仅为此机器和输入测量。
- 最大窗口两条波形均有界，显示步长32；边界重复操作不建新任务，缩回4500 ms成功。非法4 ms显示明确错误。
- 66.4秒文件总览平移到6.4秒起点，末刻度66.400秒；65秒起点分析被明确拒绝，无新增任务、无旧图、试听禁用。此项证明导航与拒绝行为，**不代表长文件分析通过**。
- `micro-5000.png` 已目视核对，保留四图、两行分组及V3公共视觉。宽窗口事件线密集是完整事件显示，缩小窗口可检查逐周期。
- 失败 `chrome-ae80c2af215e4e339825f463ad7ffef7` 实际出现上限处第3个重复任务，修复后同项通过。

Qt范围专项通过既有 `scripts/verify_m03_qt.py --ranges` 执行。初次范围断言因测试JavaScript字符串引号错误失败，记录 `qt-7e7ff96aa8674897a6b6f3cfa28811b2`，定位器修正后重跑；不把该测试脚本错误归为产品计算失败。最终结果在本报告末尾追加。

## 来源与剩余工作

[方法审阅](../references/m03-method-audit.md)核实北大馆藏书目、只读查看原手册截图，并读取Henrich等2004作者实验室原文。发现旧手册推荐不能直接代表论文结论，0.25与文献混合阈值不同，导数极值法不等于DECOM。新增两条reference-only来源并同步公共致谢，未复制论文或截图、未改变算法、未确立旧代码再分发许可。

下一项E3-B：保持全文件语义的长录音预算与实现方案，以及未完成字体预检/交互边界。已有60秒三PNG测量曾接近1.5GB进程预算，不能只提高时长常量。需分别实测长文件加载/全局归一化与去趋势/滤波、局部显示与完整导出；按原全文件时间轴对照，取消、输出预算与进程上限仍生效。分段重算数值如发生变化，须单独审阅。来源原文页码和许可缺口继续单列，不用数值等价替代。F候选范围仍待审阅，M04不推进。

本轮没有DDL、修改相邻v2/原语料/旧EXE、升级第三方依赖、push或发布。

最终Qt结果：`output/validation/m03-ui/qt-1ee78a20ceb9450588e5a7fe42977af9/report.json`，10组通过，含真实三路径保存、模块重开和5/5000ms范围。仅使用合成输入，原自然录音对照本轮另由安装wheel基准执行。

文档与边界复核：`scripts/validate_docs.py`检查544份文件、330条来源、32项任务，errors=[]，历史快照断链仍单列；`git diff --check`通过。相邻v2八份来源SHA-256与原迁移登记一致。

Qt前一轮 `qt-de28b951f88a4a3fbfc0a6259d174025` 的DOM断言通过，但截图捕获到合成器未更新的旧帧，不能作为视觉通过证据。捕获器增加事件循环等待绘制后重跑，最终5/5000ms深色截图已逐张目视确认，四图与步长说明正确显示。未改产品渲染流程。
