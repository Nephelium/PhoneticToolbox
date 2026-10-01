# M04-R1 LPC 输入范围与操作修复

2026-09-29。**verified：限定本轮 Windows 源码、Chrome、本机正式 Qt 宿主与原始自然文件。** 完整 M04、托管账号网页、Linux 本轮回归和 EXE 继续分列，不继承此状态。依据 [计划](../plans/2026-09-29-m04-repair.md) 与 [ADR](../decisions/ADR-M04-R1.md)。

## 原因与修复

截图对应 WAV 为 169728 帧、44100 Hz、3.848707482993197 秒。TextGrid 的两个层仍有空白尾段延伸到 9.085419501133787 秒。旧任务检查所有区间，因空白尾段越界而拒绝每次内部合法选区，错误码为 `lpc_textgrid_range`。新增合成回归在修复前两次失败，证明该检查可独立复现问题。

- 后端允许空字符串或纯空白字符区间超出 WAV，非空标签继续按原一个采样点容差检查。不会修改、裁切或重写源标注，JSON 继续记录原始 TextGrid 哈希。核心求解、默认值、标签拼接和半开样本映射保持不变。
- 界面提示空白越界，并在创建任务前阻止非空标签越界。错误提示给出正确关联或显式“不关联”的恢复方法。
- 波形和 Praat 语谱图均支持左键正反向拖选，Shift 仍可拖选。只移除 M04 的 Shift 必须按住选项，公共组件未改动。
- 点击跨音频末尾的空白标注只选交集，避免将 9 秒终点传入 3.849 秒音频。完全不相交的标注不创建选区。
- 时间范围、分析/取消、试听与参数/草稿进入原有可调宽度左栏，分析和试听优先排在参数前。图窗、结果保存和任务历史保留在主区，窄窗堆叠。
- 修复历史重试请求失败后，旧文件的迟到错误覆盖新文件提示的问题。取消、旧读取/保存归属、历史结果和字体预检通过定向复验。

## 实际验证

所有命令在项目根执行。科学测试 `PYTHONPATH` 指向 `backend/src;packages/phonetic_core/src`；Qt 增加 `desktop/src`。没有安装依赖或修改系统环境。

| 命令与范围 | 结果 |
| --- | --- |
| `scripts/Invoke-M03-Python.ps1 -PythonArguments @('-m','pytest','-o','addopts=','backend/tests/test_m04_exports.py','tests/parity/test_lpc_spectrum.py','tests/parity/test_lpc_boundaries.py','-q')` | 89 passed，使用既有 MKL 兼容环境；包含原数值冻结、PNG/WAV、空白越界及非空拒绝 |
| `.venv/m09-ui/Scripts/python.exe -m pytest -o addopts= backend/tests/test_m04_contract.py -q` | 11 passed，HTTP 契约使用既有宿主环境；2 条第三方弃用提示 |
| `npm --prefix frontend test` | 183 passed，含新增区间交集与空白/非空越界分类 |
| `npm --prefix frontend run typecheck`、`npm --prefix frontend run build` | 通过；保留既有大 chunk 提示 |
| `node tests/e2e/m04.cjs` | 32 组通过，真实本机文件桥接、HTTP 任务与 MKL 子进程，页面错误为空；覆盖双图选择、窄窗/主题、字体、历史、取消、重试、三文件下载与保存 |
| `scripts/Invoke-M03-Python.ps1 -PythonArguments @('-X','utf8','scripts/verify_m04_repair.py','--audio',<授权本机WAV>,'--roi','0.146859','0.572458','--roi','1.003544','1.173561')` | 截图两个范围成功，分别 18769 / 7498 样本，各 1024 点；关联/不关联 TextGrid 的谱值逐值相同且 WAV 字节一致 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m04_qt.py --audio <授权本机WAV>` | 实际 Qt 同名原 TextGrid、自然录音计算与 PNG/JSON/WAV 保存回读通过；PNG 2400×1350，WAV FLOAT64；原文件哈希未变 |
| `git diff --check -- <本轮文件>` | 通过 |
| `.venv/m09-ui/Scripts/python.exe scripts/validate_docs.py` | 扫描 1030 文件，仅主检查报 README 既有 M10-R5 EXE 缺链，未在本轮修补；历史快照缺链另列 |
| `.venv/m09-ui/Scripts/python.exe scripts/check_architecture.py` | **未通过**，报告既有其他任务修改的 `frontend/public/vocal-tract/{app.js,index.html,v3.css}` 三项资源哈希不匹配；本轮没有改这些文件或资源清单 |

首次 pytest 调用触发继承 V2 配置的 coverage 插件缺失，使用已沿用的 `-o addopts=` 清除旧 V2 覆盖率参数。随后将需要 FastAPI 的 HTTP 契约检查放回宿主环境，科学环境只跑科学/导出用例，没有跳过失败用例。首次 Chrome 启动暴露宿主安装核心包不包含后续 M14，测试明确从三个源码根加载后正常；设置已由其他任务改为标签页，测试同步关闭设置标签。首轮 Qt 实际计算已成功，但脚本误将输入 1.4 秒等同实际样本起点；改为根据既有 `int(seconds * fs)` 契约验证 61739–70560 / 8821 样本，未改计算结果、容差或算法。上述失败证据保留。

## 证据与范围

- 最终左栏顺序复验：`output/validation/m04-ui/bb47c20c0bf54bf689e54b224c7e9b69/report.json`，32 组通过，`r1-left-controls.png` 为最后布局。
- Chrome 首次完整通过：`output/validation/m04-ui/522397ea98cc4bd4a1eafe79a198e00b/report.json`，浅深主题、1440/1000/390px 与选区截图。
- 自然录音双选区：`output/validation/m04-r1/94e26ae81de0473d860c9c02b311cbc8/report.json` 和两个完整结果目录。原始路径、内容哈希仅存在该忽略目录。
- 实际 Qt：`output/validation/m04-e/qt-a7e573b57c054c799ce072f8a9e8dda4/report.json`、`R1-NATURAL-spectrum.png`、`R1-NATURAL-wave-spectrogram.png` 和 `saved/` 三文件。

本轮无第三方来源或许可变更。原 WAV/TextGrid、V2、旧 EXE 保持原状。当前源码入口为 `scripts/Start-M04-Workbench.ps1`，复用统一宿主和既有任务库，只给当前进程设置源码搜索路径，退出恢复，不更新安装包或数据库结构。既有失败任务保留为历史，可重试或重新分析。本轮未 push、DDL、服务器访问/部署、全局依赖更新或 EXE 打包。
