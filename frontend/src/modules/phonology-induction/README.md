# 音系归纳 · M14

M14 将单音节调查字表解析为声母、韵母和尾部数字调值，经人工确认类别、顺序和归并后生成两份 DOCX 与一份二维 XLSX。页面没有音频输入、识别、声学测量或试听功能。完整操作正文的权威源为[手册章节](../../../../manual/chapters/m14.json)，本 README 说明工程入口、格式、来源和验证边界。

## 架构与入口

从统一工作台首页或侧栏进入音系归纳。开发态使用已有项目环境与已构建前端，运行[Start-M14-Workbench.ps1](../../../../scripts/Start-M14-Workbench.ps1)，不由启动器安装依赖、构建前端或执行数据库迁移。

| 层与文件 | 职责 |
| --- | --- |
| [PhonologyInductionPage.vue](PhonologyInductionPage.vue) | 四步导入、调类编辑、草稿、快照、任务和文件交付 |
| [SymbolEditor.vue](SymbolEditor.vue) | 声韵搜索、多选排序、页内归并、撤销和还原 |
| [ResultPreview.vue](ResultPreview.vue) | 三种预览、全表搜索、命中定位与逐行审阅 |
| [state.ts](state.ts)、[port.ts](port.ts) | 明确初始选项、设置与结果类型、分组和搜索规则 |
| [platform/m14.ts](../../platform/m14.ts) | 统一平台任务接口、取消、哈希回读和 ZIP 下载 |
| [m14_models.py](../../../../backend/src/ptb_api/m14_models.py) | 严格版本化请求、选列参数、设置和完整结果清单 |
| [m14_table.py](../../../../backend/src/ptb_worker/m14_table.py)、[m14_jobs.py](../../../../backend/src/ptb_worker/m14_jobs.py) | 有界格式读取、样例检查、正式分析与核心调用 |
| [m14_task.py](../../../../backend/src/ptb_worker/m14_task.py)、[m14_executor.py](../../../../backend/src/ptb_worker/m14_executor.py)、[m14_child.py](../../../../backend/src/ptb_worker/m14_child.py) | 复用 owner/project、资产和持久任务，Windows 有界子进程及完整三文件发布 |
| [m14_bridge.py](../../../../desktop/src/ptb_desktop/m14_bridge.py) | 本机导入及整组保存，非覆盖写入、校验和失败回滚 |
| [phonology 科学核心](../../../../packages/phonetic_core/src/phonetic_core/transcription/phonology/) | parser/rules/config：解析与分类；render/presentation/export：排序、字体和三文件生成 |

前端只通过平台能力提交文件、读取结果和保存草稿，不读取服务端路径，不在界面实现 Python 科学核心。重型文档解析与导出由受限 child 执行，原始调查文件不由分类或归并操作改写。正式任务当前有 Windows 能力门，Linux 核心可测试不代表 Linux 正式任务已开放。

## 功能、输入与输出

四步为导入、调类与调值、声韵排序归并、结果。导入先检查原始样例，确认成功后才替换当前数据；调类与声韵直接在页内保存和还原；结果页提供两个分组预览及矩阵、全表搜索和导入/归并审阅。

| 项目 | 格式与行为 |
| --- | --- |
| 输入 | XLSX、XLS、CSV、TSV、TXT、DOCX 表格；每条有效记录一个音节 |
| 表选择 | Excel 工作表、DOCX 表格、文本表格；新文件默认第一表 |
| 字段 | 必需字头和 IPA，可选备注；列号和开始行从 1 计数 |
| 默认选项 | 字头列 1、IPA 列 2、备注列 3、开始行 2、单辅音作韵母且声母为 Ø、编码/分隔符自动 |
| 文本编码 | 自动模式识别 UTF-16 BOM，否则 UTF-8；可显式 UTF-8、GB18030/GBK、UTF-16 |
| 分隔符 | 自动、Tab、逗号、分号、中文逗号、空格；空格需显式选择 |
| 新计算 | UI 显式 `computation_revision=m14/2`；请求与预览外层仍为 `m14/1` 协议 |
| 草稿 | `m14-draft/2` 本机账号/项目作用域保存，含源 ID、解析记录、审阅、设置和导入选项；不含生成三文件 |
| 输出 | `同音字表_韵母到声母.docx`、`同音字表_声母到韵母.docx`、`同音字表_二维表.xlsx` |
| 网页整组下载 | 正式接口可用时下载 `同音字表_完整结果.zip`，内含原三文件 |

单辅音、连音符、鼻化、成音节标记、近音、重音与上标调值的处理以[parser.py](../../../../packages/phonetic_core/src/phonetic_core/transcription/phonology/parser.py)为准。分类符号使用 Unicode NFD，源 IPA 原样保留。缺字头/IPA、只有调值或疑似多音节记录跳过并列明，含 `?` 或替换符号给人工核对警告。未知符号不自动转换。

字头中的首汉字成为条目；括号内容提取为备注，无括号的多汉字字段把整个汉字词放入备注。字头中的备注与备注列合并。重复的完整记录保留，原始 IPA 数按原字符串去重。

调类名称默认原调值，同名调类归组；调值拖动交换位置。声母/韵母可 Ctrl/⌘、Shift 多选及整组排序，归并要求单选源项并在页内确认目标及影响记录，支持取消、单条撤销和还原。归并只影响分类映射，不覆写字头、IPA、备注或原调值。

生成后预览绑定不可变数据和设置快照，后续编辑明确提示旧成果，未保存页内编辑会禁用生成。三文件以归并类别呈现字头和备注，不逐行输出原始 IPA、来源行或归并日志。科研记录应同时保存源表与人工设置决定。

## 快速使用

1. 选文件，核对所选工作表/表格、编码、分隔符和原始样例。
2. 映射字头/IPA/备注列与开始行，确认单辅音策略，点击确认导入并继续。
3. 在调类页核对原调值、记录数和例字，编辑名称、调整顺序并保存。
4. 在声韵页核对排序及归并，确认影响后保存设置进入结果。
5. 切换三种预览、搜索代表字项，展开审阅处理跳过行和异常符号。
6. 生成三文件，桌面选无同名结果的新目录整组保存，或逐项下载；再点顶部保存草稿。

切换步骤保留当前输入，页内保存只应用到工作数据，顶部保存草稿才持久写入本机。旧 `m14/1` 草稿可查看，重新导入后才使用当前解析生成。服务器源文件到期时，草稿中可见解析记录不保证源资产仍能执行生成。

## 文件预算与呈现

- 输入不超过 2,000,000 字节，读取记录不超过 10,001 条，有效记录不超过 10,000 条，整表不超过 64 列，每单元格不超过 512 字符。
- XLSX/DOCX ZIP 不超过 512 项、展开总量不超过 32,000,000 字节。含公式的 XLSX 拒绝，应另存纯值副本。
- 声韵矩阵不超过 20,000 格，单格不超过 32,767 字符，三文件总量不超过 16,000,000 字节；超预算拒绝，不截断。
- 子进程执行期限 60 秒，前端等待任务期限另为 180 秒；取消和失败保留已确认编辑及已有结果，未完成三件套不发布。
- 预览分组每页 100 条，矩阵窗口 12 行×10 列、每格每页 60 条，审阅每页 100 条。搜索覆盖全部有效记录，分页不改变导出内容。
- DOCX 含声母/韵母/声调统计与代表例字，正文保留重复条目和组内输入顺序。Excel 冻结 `B2`，列宽固定，行高按内容估计并受 409 pt 上限约束。
- IPA 固定 Doulos SIL，其他字体使用生成时公共导出字体快照；DOCX/XLSX 不内嵌字体。屏幕预览不模拟 Word 纸面分页。
- 桌面任一同名结果存在时整组拒绝，不自动改名或覆盖。取消目录选择保留结果；受控保存失败回滚本次新写入文件。逐项下载位置由宿主/浏览器控制。

## 方法、依赖和许可

[NOTICE](../../../../packages/phonetic_core/src/phonetic_core/transcription/phonology/NOTICE.md)和[来源映射](../../../../docs/modules/evidence/M14-source-map.md)记录历史实现来源。[统一来源登记](../../../../third_party/source-registry.json)中的 `PENDING-PHONOLOGY` 是保留的稳定 ID，现已登记作者确认的自有历史实现，部分历史项目有 AI 辅助；其旧名字不表示当前仍未确认所有权。第三方依赖与发行义务分别处理，项目总许可证由作者确定。

[requirements-m14.in](../../../../requirements/requirements-m14.in)、[增量依赖锁](../../../../requirements/requirements-m14-additions.lock)、[依赖核查](../../../../output/validation/m14/dependency-audit.json)记录 pandas 2.3.3、openpyxl 3.1.5、python-docx 1.2.0、xlrd 2.0.2、lxml 6.1.3 和 typing-extensions 4.16.0。核查分别登记 BSD-3-Clause、MIT、MIT、BSD、BSD-3-Clause、PSF-2.0，实际随包组件应按许可证原文复核。xlwt 1.3.0 只用于生成合成 XLS 测试材料，不是产品运行要求。Doulos SIL 字体及 OFL 按独立字体来源登记。

本模块没有将某一篇论文定义为自动音系分析依据。元音集合、修饰符、塞擦音和声母排序都是可检查的符号规则。研究者负责判断单辅音归属、调类合并与音位关系，矩阵空格仅表示当前字表未收录，记录数不等于独立词项或发音人数量。

## 开发验证与实际限制

以下为适用验证入口，从仓库根目录、已有项目环境运行，执行范围与结果写入对应报告。

```powershell
$env:PYTHONPATH="$PWD/packages/phonetic_core/src;$PWD/backend/src;$PWD/desktop/src;$PWD/scripts"
$env:PYTHONIOENCODING='utf-8'
& '.venv/m14/Scripts/python.exe' -B -m pytest -c tests/pytest.ini tests/parity/test_phonology_induction.py backend/tests/test_m14.py backend/tests/test_m14_r1.py desktop/tests/test_m14_save.py -q -p no:cacheprovider --tb=short
npm --prefix frontend run typecheck
node --test frontend/tests/m14.test.ts
& '.venv/m14/Scripts/python.exe' -B scripts/generate_contracts.py --check
npm --prefix frontend run contracts:check
node tests/e2e/m14-r1.cjs
& '.venv/m14/Scripts/python.exe' -B scripts/verify_m14_r1_qt.py
```

前端构建用 `npm --prefix frontend run build`。手册工程检查用 `python scripts/manual/validate.py --project manual --strict`，阅读资源由主线统一运行 `scripts/manual/build.py` 生成；模块作者不手工改 `frontend/public/manual`。

说明书截图入口为 [capture_m14_test_table.py](../../../../scripts/manual/capture_m14_test_table.py)，通过 `--source` 指定已授权字表，读取隔离副本并生成独立任务与报告。保留原字段和输入摘要；示例中的暂拟调类不当作实测音系结论。操作记录位于本机 `output/manual-work/m14-test-table/`，临时副本按根规则清理。

[R1 报告](../../../../docs/testing/2026-10-05-m14-r1-report.md)限定 Windows 源码、公开合成字表、实际 Chrome/Qt 及 WSL 纯核心。Windows 正式接口和三文件下载/保存已记录核验，WSL 环境缺 Pydantic 的 API 模型项单列。自然语料分类准确率、实体 IME/DPI、Linux 正式任务和 GUI、远程存储联合门未完成当前专项验收；R1 没有重新验 Office 实体程序的纸面显示。安装包包含情况及成品范围应另看对应包报告，不从源码测试推出全部平台可用。
