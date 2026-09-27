# M14 说明书与源码功能映射

2026-09-27。范围为 V2 `Phonetic_Export/index.html` 第 12.1、12.2 章与实际源码。所有基准为公开合成数据，来源哈希保存在 `tests/fixtures/m14/v2-baseline.json`。独立捕获脚本仅在验证阶段读取 V2，新运行入口不依赖 V2。

## 完整映射

下表中 widget 指 `phonetic_toolbox/gui/widgets/phonology_induction_widget.py`，service 指 `phonetic_toolbox/services/phonology_service.py`。验收结果以 [报告](../../testing/m14-report.md) 为准，表中入口存在不等同验收通过。

| 说明书操作 | V2 实际行为 | V3 正式入口及实现 | 验收证据入口 |
| --- | --- | --- | --- |
| 上传 Excel 或文本 | widget `_on_upload`、service `load_rows`；XLSX/XLS/CSV/TXT/TSV，首个工作表、前三列 | 音系归纳工具栏导入，`m14_import.load` | 20 组原 V2 基准、五格式正式任务 |
| 是否跳首行 | GUI 默认是，service 默认 False；常见表头仍自动过滤 | 导入设置默认勾选，提交显式布尔值 | 两种 skip 的原 V2 对照 |
| 单辅音处理 | `_ask_single_consonant_policy` 默认零声母，可选空韵 | 导入设置两选项，记录单辅音示例 | 两种策略及空韵归并 |
| 字头/备注 | `_parse_columns` 取首个汉字，整词/括号内容及第三列备注 | 原样 `rules.py`，导入审阅列 | 记录逐字段相等 |
| 音标分解 | 原 parser 的扫描/元音表；末尾数字为调值，无数字为 0 | 原样 parser，固定 Klatt 元音集合 | 中文、组合 IPA、全角调值、未知符号 |
| 调值排序 | `ToneMappingDialog` 的 `_swap_rows` 交换源行与目标行 | 第二步编辑，原生拖动交换 | 前端状态测试、真实拖动 |
| 调类与归并 | `collect_mapping` 去首尾空格、空名用调值；同名调类合并 | 名称编辑、确认/取消，调值→调类可审阅 | 三产物调类一致性 |
| 声母/韵母排序 | `SymbolOrderDialog` 扩展选择、Ctrl/Shift、组内拖动 | 第三步双列表、Ctrl/Shift、多项整组移动 | 状态测试和实际浏览器 |
| 声韵归并 | `_prepare_merge` 单选源，`_try_finish_merge` 选目标，链式 aliases | 源→目标，取消本次归并、取消整个弹窗、确认；映射审阅 | 源数据不可变、循环/未知项拒绝 |
| 生成同音字表 | `_on_generate`；`export_outputs` 生成两份 DOCX、一份 XLSX | 第四步持久后台任务，完整三文件才发布 | 正式 task→child→manifest→下载/保存 |
| 正序与逆序 | `_write_word_document` final_initial、initial_final；声韵调统计与备注下标 | `render.py` 保留结构；`presentation.py` 单列字体修正 | 原 V2 段落和单元格全文对照；Word 实际渲染 |
| 二维同音字表 | `_write_matrix_xlsx` 韵母为行、声母为列、调类组内原记录顺序 | 三者共享归并后记录；保留重复条目 | XLSX 字头/备注/调类、Excel 实际打开 |
| 保存目录/帮助 | 目录选择及模块帮助 | 公共目录授权与保存接口，帮助弹窗、方法来源、关闭保护 | 重名保护、失败恢复、正式宿主 |

## 说明书与源码差异及处置

1. 手册称取首汉字并把剩余汉字移入备注。实际多汉字字段把**整词**放入备注，V3 保留源码，例“合音”→字头“合”、备注“合音，2”。
2. GUI 跳首行默认 True，service 默认 False。V3 UI 默认 True，API 明示参数。标准表头即使不跳首行也被过滤，增加无表头合成用例区别二者。
3. Excel/CSV 经过 pandas 默认 NA 识别；TXT/TSV 字面 NA 保留。`_parse_columns` 会去掉空字符串再取列，文本的空 IPA 行可把备注当 IPA。合成表 Excel/CSV 有 16 条，文本有 18 条。原样保留、逐行展示，未静默纠正数据。
4. 手册多选/拖动全部保留。归并源码要求单个源项，V3 同样要求单选源，不擅自增加多源合并语义。
5. V2 空韵显示为空字符串，归并代码的真假判断使空韵不能作为源/目标。FIX02 明确修正，使两策略都能完整编辑，禁止自归并及循环。
6. V2 XLSX 富文本分支重新排序调类，忽略选择的调值顺序，与 Word 不一致。FIX01 修正为三文件使用同一顺序；原样 renderer 仍受独立基准保护。
7. V2 默认 TNR/宋体。V3 呈现层接生成时公共字体快照、IPA 固定 Doulos SIL。字音、分类与顺序不随字体改变。字体未嵌入 Office 文件，跨机需安装相应字体。
8. 手册称自适应单元格，V2 实际固定列宽/52 pt 行高。FIX04 按内容估计换行行高、保留行列冻结；Excel 行高最大 409 pt，大单元格可在编辑栏查看全文。超 32,767 字符明确拒绝，不截断。
9. FIX05：真实 Excel 16 拒绝含多调类的 V2 富文本 XLSX；空白分隔 run 缺 `xml:space="preserve"`。独立最小复现定位，添加标记后真实 Excel 可打开；openpyxl 能读并不能证明兼容。
10. V2 直接写目录可能覆盖/部分成功。FIX03：持久任务原子发布完整三件；本地同名拒绝，写失败回滚本次新文件；取消不改旧结果。
11. V2 缺 python-docx 有降级路径；V3 明确依赖真实 DOCX 库，不把伪文档当正常成功。非法格式、超预算、公式、非 UTF-8 文本明确拒绝，未知 IPA 保留原解析语义并提示人工审阅。

## 来源与基准隔离

原算法机械迁移已形成本地提交 `ad165a6`。后续一致性、字体、Office 兼容修正独立维护。两次 V2 捕获的输入记录、两策略分析、归并、Word 段落、Excel 单元格与六份源码/手册 SHA-256 全部一致。原 V2 环境没有 xlrd；XLS 基准在原 V2 解释器上临时使用 M14 项目依赖路径完成，V2 环境和文件未修改。许可/更早来源仍沿用 `PENDING-PHONOLOGY`，不新增已获第三方授权的声称。
