# M14 P17 逐控件规则

状态：in_progress。本规则依据实际 V2 手册 12.1–12.2、V2 源码与当前 V3 页面。历史通过不算本轮通过。

V2：`phonetic_toolbox/gui/widgets/phonology_induction_widget.py`，入口/事件 `_generate / tone / merge / export`。V3：`frontend/src/modules/phonology-induction/PhonologyInductionPage.vue`。V2 手册原文和源码只读，私有审阅摘录在 `output/validation/p17/source-review/`。

## 控件清单

每个动态 v-for 行还需逐一遍历所有选项/角色/参数。操作前记录当前值，按下表动作更改，检查可见值、dirty、数据归属，再恢复。数值控件另测上下界、空值、越界和键盘输入。按钮另测重复点击、禁用态与失败重试。文件选择器另测取消/无效格式/同名文件。下表逐行记录本轮实际范围。partial 表示所述路径通过，但整行通用边界或组合尚未全覆盖；not_run 未执行；blocked 有明确前置阻断；not_applicable 当前产品无该可用能力。不得把 partial 汇总为全交互通过。证据相对 output/validation/p17；报告相对本目录。

| ID | 手册 | V2 源 | V3 控件/事件 | 前置 | 操作与判据 | 证据 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M14-C001 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 导入调查字表 · `fileInput?.click()` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：真实文件选择与5格式导入、帮助对话框、跳过行诊断展开收起、取消实际生成后设置仍可用 | `M14/6b0850f9fbd5454f949ef78505b657fd/browser-report.json` | partial |
| M14-C002 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | fileInput · `importFile` | 空态/已载入/结果态按显示条件 | 填入有效值并触发提交；当前值准确更新，空/越界被拒或给出可理解反馈，原始文件不变<br>本轮：真实文件选择与5格式导入、帮助对话框、跳过行诊断展开收起、取消实际生成后设置仍可用 | `M14/6b0850f9fbd5454f949ef78505b657fd/browser-report.json` | partial |
| M14-C003 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 保存草稿 · `save` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：保存草稿、保护关闭并恢复。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C004 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 取消任务 · `controller?.abort()` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：真实文件选择与5格式导入、帮助对话框、跳过行诊断展开收起、取消实际生成后设置仍可用 | `M14/6b0850f9fbd5454f949ef78505b657fd/browser-report.json` | partial |
| M14-C005 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 帮助 · `help=true` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：真实文件选择与5格式导入、帮助对话框、跳过行诊断展开收起、取消实际生成后设置仍可用 | `M14/6b0850f9fbd5454f949ef78505b657fd/browser-report.json` | partial |
| M14-C006 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 方法与来源 · `emit('references')` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：实际AppShell方法来源对话框打开、Esc关闭；链接外站未执行 | `layout-c/1790855035363/report.json` | partial |
| M14-C007 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 0&&!preview)" :aria-current="step===i?'step':undefined" @click="step=i">{{i+1}} · {{name}} · `原生控件输入` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：导入/调类/声韵/结果步骤切换。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C008 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | options.skip_first_row · `options.skip_first_row` | 空态/已载入/结果态按显示条件 | 填入有效值并触发提交；当前值准确更新，空/越界被拒或给出可理解反馈，原始文件不变<br>本轮：已通过路径：skip false后重新生成。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C009 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 按零声母字处理按空韵字处理 · `options.consonant_only_as_zero_initial` | 空态/已载入/结果态按显示条件 | 依次选择每个选项，检查对应视图/模型/字段更新；禁用项不生效，切回保留合法状态<br>本轮：已通过路径：辅音声母规则开关后生成。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C010 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 编辑调值顺序与调类 · `editTones` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：打开调值对话框。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C011 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 编辑声韵顺序与归并 · `editSymbols` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：声母/韵母对话框。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C012 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 生成三份结果 · `generate` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：正式本地任务生成及输出。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C013 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | {{context.files.kind==='desktop'?'选择目录保存三份结果':'下载完整结果包'}} · `saveResults` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：保存三文件及重复同名冲突。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C014 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 下载 · `download(file.id)` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：实际下载三文件。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C015 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 跳过 {{preview.diagnostics.skipped.length}} 行 · `原生控件输入` | 空态/已载入/结果态按显示条件 | 展开再收起；内容可滚动且不撑高整页，焦点可达<br>本轮：真实文件选择与5格式导入、帮助对话框、跳过行诊断展开收起、取消实际生成后设置仍可用 | `M14/6b0850f9fbd5454f949ef78505b657fd/browser-report.json` | partial |
| M14-C016 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | '调类 '+tone · `toneDraft.tone_map[tone]` | 空态/已载入/结果态按显示条件 | 填入有效值并触发提交；当前值准确更新，空/越界被拒或给出可理解反馈，原始文件不变<br>本轮：已通过路径：调值调类映射修改。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C017 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 取消 · `toneDraft=undefined` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：调值取消。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C018 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 确认调类 · `confirmTones` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：调值确认。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C019 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 取消本次归并 · `pending=undefined` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：合并取消。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C020 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | {{kind==='initial'?'归并声母':'归并韵母'}} · `prepareMerge(kind)` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：多选合并和链式合并。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C021 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 取消 · `symbolDraft=undefined;pending=undefined;error=''` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：声韵取消。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-C022 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 确认声韵设置 · `confirmSymbols` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：声韵确认。其余通用边界/重复/取消未逐项覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-I01 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 调值拖动交换、同名调类合并、留空回退、取消回滚 | 对应功能就绪 | 逐项执行上述操作，对照V2与当前授权差异；记录原始样本/产物及失败<br>本轮：声韵Ctrl/Shift多选、拖动和链式合并及取消通过 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-I02 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 声韵Ctrl/Meta离散多选、Shift连续多选、整组拖动 | 对应功能就绪 | 逐项执行上述操作，对照V2与当前授权差异；记录原始样本/产物及失败<br>本轮：5格式输入与三文件生成保存下载已验 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-I03 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 声母/韵母归并链、禁止循环、自归并、取消本次归并 | 对应功能就绪 | 逐项执行上述操作，对照V2与当前授权差异；记录原始样本/产物及失败<br>本轮：异常输入、草稿关闭/恢复通过 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-I04 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | XLSX/XLS/CSV/TXT/TSV逐一导入、空文件/缺列/编码错误 | 对应功能就绪 | 逐项执行上述操作，对照V2与当前授权差异；记录原始样本/产物及失败<br>本轮：原交互组合未完整逐项覆盖，具体已验路径见本轮报告 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-I05 | 12.1–12.2 | `gui/widgets/phonology_induction_widget.py` | 两DOCX与一XLSX真实回读，完整条目和零声母/空韵/备注 | 对应功能就绪 | 逐项执行上述操作，对照V2与当前授权差异；记录原始样本/产物及失败<br>本轮：原交互组合未完整逐项覆盖，具体已验路径见本轮报告 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |
| M14-G-L01 | 12.1–12.2 | 同上 | 1920×1000 CSS及实际Qt最大化，记录DPR | 对应状态 | 按P17总规则执行并保存DOM几何/截图/原始计时<br>本轮：最新源码五页三栏、右栏收起增宽及重载记忆，四档CSS窗口/两缩放；最终production dist隐藏Qt已验，隐藏窗口inner1440×900，真实可见最大化在此前批次1707×996 | `layout-c/1790855035363/report.json；qt-c/3a44621f31624425bfdcad3fe4944e9b/report.json` | partial |
| M14-G-L02 | 12.1–12.2 | 同上 | 空态/有数据/结果/错误的scrollHeight，长列表只在独立容器滚动 | 对应状态 | 按P17总规则执行并保存DOM几何/截图/原始计时<br>本轮：见报告实际尺寸与截图；未覆盖的尺寸/状态不能外推 | `layout-c/1790852346905/report.json；qt-c/a2d31781aeee478e8969d0fe22e3dea6/report.json` | partial |
| M14-G-L03 | 12.1–12.2 | 同上 | 1366×768/1280×720及125/150%时末尾动作可达 | 对应状态 | 按P17总规则执行并保存DOM几何/截图/原始计时<br>本轮：见报告实际尺寸与截图；未覆盖的尺寸/状态不能外推 | `layout-c/1790852346905/report.json；qt-c/a2d31781aeee478e8969d0fe22e3dea6/report.json` | partial |
| M14-G-L04 | 12.1–12.2 | 同上 | 浅深色/IPA/侧栏拖宽，无裁切和重叠 | 对应状态 | 按P17总规则执行并保存DOM几何/截图/原始计时<br>本轮：见报告实际尺寸与截图；未覆盖的尺寸/状态不能外推 | `layout-c/1790852346905/report.json；qt-c/a2d31781aeee478e8969d0fe22e3dea6/report.json` | partial |
| M14-G-P01 | 12.1–12.2 | 同上 | 20次交互计时及冷/热加载分开 | 对应状态 | 按P17总规则执行并保存DOM几何/截图/原始计时<br>本轮：已有真实字表的导入参数开关各20次到两帧结束，min/median/P95/max=17/28/42/64ms；含自动化开销，不是科学任务耗时；未完成冷/热各20次加载 | `M14/51f809dcdb6e4dacb2c92351ff5df7d1/interaction-timing.json` | partial |
| M14-G-STATE | 12.1–12.2 | 同上 | 取消/失败/迟到/切页/关闭未保存保护 | 对应状态 | 按P17总规则执行并保存DOM几何/截图/原始计时<br>本轮：报告列明已验草稿/关闭/失败路径；取消/迟到矩阵未全覆盖 | `M14/d8a596fad5a24bfe93f4182781225a21/browser-report.json` | partial |

| M14-CNEW01 | P17追加 | V3公共布局 | 右辅助栏收起/恢复并记忆 | 默认三栏 | 点击收起，中间宽度增加；重新加载仍收起；恢复后全部原控件可达 | `layout-c/1790855035363/report.json` | verified |

## 当前授权覆盖旧行为

M12原始TextGrid优先、三图双击与整段剪贴按R5/R6现行约定，不退回旧手册。M11组件固定版本并独立管理，保存到新目录。所有模块不安装环境，不改现存库、不改V2/原录音。M15试音确认由自动化操作只证明流程门禁，真实听感/设备/物理时延仍需人工。
