# M01-G 联合审阅报告

2026-09-10。本轮联合验收批次 **verified（下列 Windows 范围）**；完整 **M01-G / M01 / P08 仍为 in_progress**。本报告记录首轮联合批次；后续旧 PKL 图形转换及历史 XLSX/SQLite 显式导入已见[旧格式验收](m01-legacy-report.md)。[实施计划](../plans/2026-09-10-m01-joint-review.md)、[操作说明书](../manual/parameter-estimation.md)、[39项验收](../modules/evidence/M01-acceptance.csv)相互关联。

## 本轮修复

- 子进程原先把科学错误统一改成 `invalid_segment_input`，持久层再改为 `execution_failed`。现用固定白名单贯穿科学进程、任务和界面，区分损坏 WAV、无效 TextGrid/唇形、空参数帧、采样/输出/内存预算与切分错误。公共错误中不包含异常原文、标签或私有路径。
- 网页结果原先统一叫 `result.xlsx` 等，现下载时沿用原音频名并按音频分组；受控资源 ID 和校验和不变。桌面原先已有对应命名，本轮复验。
- 真实结果出现后右列进度条、长文件名曾撑出内容列。已限制进度条宽度并允许名称换行；用完成计算后的页面验证四个窗口宽度，不只检查空态。
- 软件帮助与项目上传提示移除“分析尚未接入”的过时表述，补充分析/切分范围、父参数来源、草稿与任务区别、失败恢复和长音频预算。科研公式、参数默认值、原始声音、schema、依赖版本均未修改。

## 实际验证与结果

开发环境 `.venv/v3-dev` 构建 API wheel；真实科学/桌面测试使用 `.venv/m01-ui` 中安装的 wheel。未操作 Codex 内置浏览器；Chrome、Qt、服务与原生进程均由当前验证持有。

| 实际命令或范围 | 证据与结果 |
| --- | --- |
| `run_m01_validation.py --approved-m01-schema-and-synthetic-tests --verify-web` | `output/validation/m01/persistent-96ec90edc3fb475ab9966106f86817d2/`：9组浏览器检查通过，pageerror为空，PG正常停止；没有 apply schema |
| 浏览器真实上传 | 在项目文件控件选择 WAV 与 TextGrid，实际上传块和 finalize，再进入工作台计算；不再由测试直接把成功资源放进数据库 |
| 下载与回读 | 10文件校验和逐个相同；原分析160帧×78列，两组片段各80帧×79列，XLSX/SQLite逐值与来源JSON一致。76声学列＋Time_s＋标注；选择80键但无唇形输入时四个Lip列不伪造 |
| 向旧读取器兼容 | 相邻v2原 `services/io/excel.py` 的 `load_excel`、`load_fastdb_columns`、`load_fastdb_window` 实际读取三组下载产物；列、时间窗口、非有限值与标签一致。原源码只读，仅在验证脚本加载，应用不导入v2 |
| TextGrid同步参数切分 | 两个实际WAV片段逐样本与原输入相同；参数保留原帧，分别验证Time_s、Source_Time_s和原值；IPA与`=1+1`是字面标签，未变为Excel公式 |
| 真实网页失败 | 上传损坏WAV和2000001采样值音频，得到invalid_audio/analysis_sample_limit；同批后续正常音频成功，失败项无结果发布，无残余temporary资源；明确显示可重试 |
| 响应式与主题 | 1920×1080、1280×720、1000×700、390×844的文档和三列均无横向溢出；实际浅/深截图、帮助打开及Escape关闭、换账号清空和越权404 |
| `verify_m01_failures.py` | `failures-a872bbe816dd41e8987b5f33a549ad27/report.json`：真实本地服务分别处理坏WAV、坏TextGrid、坏lip、空WAV、超采样预算、正常输入，五种错误准确保留；服务重启后仍一致、临时资源为零 |
| `verify_m01_natural.py` | `natural-45133546ecca4d5eafcd66139b3aaf04/report.json`：4份已授权本机自然录音、16/22.05/44.1kHz，与独立冻结P03 v2结果比较通过；时间/非有限mask精确，数值保持原容差。输入与标注hash前后不变，没有上传或提交语料 |
| `verify_m01_local_tasks.py --approved-m01-synthetic-tests` | `local-tasks-cbb1478869ae491788c6fadcf78c06a1/report.json`：真实科学计算、原v2 GUI-ALL冻结对照、同源参数切分、重名保护、后续输出失败回滚、重启保存及旧行保持，6组全部通过 |
| `verify_m01_task_window.py` | `task-window-fb4235e06556487ea74b4deb384656db/report.json`：7步实际Qt目录选择、全参数计算、WAV及参数同步切分、浅深/窄窗截图全部通过；窗口与所属服务正常退出 |
| Python回归 | `.venv/m01-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini backend/tests desktop/tests tests/contracts tests/security packages/phonetic_core/tests tests/parity -q`：**410 passed**，两个既有Starlette/anyio弃用提示；见`g-regression.log` |
| 前端检查 | `npm --prefix frontend run typecheck`、`npm --prefix frontend test`（**16 passed**）、`npm --prefix frontend run build`通过 |

Qt最终操作记录、源码/wheel哈希和保存性结果见[机器证据](../modules/evidence/M01-joint-review.json)。后续只补定向检查，不重复已通过的全部科学回归。

最终 API wheel 中7份受影响worker源码与工作区逐文件hash一致；架构、OpenAPI/TypeScript生成与323项来源生成检查通过。保存性7项均与F2相同：v2的HEAD/index/status、427份基线文件、包元数据及环境位置；本机活动任务为0，固定测试PG已正常停止。Git根仍为D盘v3项目，暂存范围仅本轮明确源码/说明书/验证脚本和摘要证据。

## 审阅中发现的问题

首次完整页面检查明确失败于1920px右列（client163、scroll198），据此修复后四种尺寸相等。首次回读脚本错误地预期没有唇形关联也有80个数值列；已改为同时断言选中80键、实际76列、缺失集合恰为四个Lip键，未改变应用或补假数据。首次Qt脚本在保存文件出现而目录仍刷新时过早点击，停在第5步；改为等待保存反馈与控件同时就绪，再执行后续切分。原失败证据保留。

## 未完成项与下一步

1. **M01-A08**：本报告首轮结束时缺少旧PKL图形转换和NumPy兼容；后续已在[旧格式批次](m01-legacy-report.md)实现并真实Qt验证。支持范围有明确结构和资源边界，不能扩大为任意Python对象读取。
2. **M01-A39 / M02-F01**：后续旧格式批次已完成历史XLSX/SQLite显式关联及同步切分。可信同源父结果路径仍单独校验；历史表标明来源未核实。M02其他功能仍planned，完整M01留待39项最终审阅。
3. 四份自然录音是定向回归，不是所有持续元音、气声、嘎裂类型的独立验证；没有本轮自然唇形同步采集证据。配额竞态/取消/租约复用F2及P07报告，本轮没有真实填满5GB或进行生产十人负载测试。
4. 科学计算仍限200万采样值、240秒、20万表格单元格/64MB结果；预览/切分的预算独立。桌面仍是开发入口；正式EXE、跨平台、硬断电和发行许可属于后续阶段。

没有新第三方代码、库或方法；来源仍使用统一登记的Praat/REAPER/IRAPT、既有参数方法、openpyxl和SQLite，许可未决项保持。没有push、数据库再次迁移、全局设置修改或相邻v2文件修改。避开内置浏览器关闭只是已采取的规避措施，不代表已修复Codex底层退出原因。
