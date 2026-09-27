# M14 · 音系归纳迁移计划

状态：in_progress。2026-09-27 已完成五功能组迁移和限定 Windows 正式功能验收，Linux 核心/导出有实际服务器证据；Linux 正式任务、原生浏览器及托管存储联合门未完成。见 [实施记录](M14-implementation.md)、[验收报告](../../testing/m14-report.md)。以下保留原规划，实际文件与命令以这两份记录为准。普通模块依 P03/P04/P06/P07；设备/原生模块另依 P01。共用 [架构](../../../ARCHITECTURE.md)、[测试规范](../../testing/verification-plan.md) 和 [UI 规范](../../design/UI_SPEC.md)。

## 现有代码与目标文件
现有路径均已确认存在；目录内逐函数对应由实施第一步记录，避免把旧类名机械套给新实现。

- [phonetic_toolbox/gui/widgets/phonology_induction_widget.py](../../../phonetic_toolbox/gui/widgets/phonology_induction_widget.py)
- [phonetic_toolbox/services/phonology_service.py](../../../phonetic_toolbox/services/phonology_service.py)
- [phonetic_toolbox/core/transcription/phonology_induction.py](../../../phonetic_toolbox/core/transcription/phonology_induction.py)

拟创建/修改路径（未来实现，不代表已经存在）：

- `frontend/src/modules/phonology-induction/PhonologyInductionPage.vue`
- `frontend/src/modules/phonology-induction/state.ts`
- `packages/phonetic_core/src/phonetic_core/transcription/`
- `backend/src/ptb_api/modules/phonology_induction.py`
- `tests/parity/test_phonology_induction.py`
- `frontend/tests/e2e/phonology-induction.spec.ts`

纯显示/客户端模块如果没有科学计算，不为凑层数创建空 core/API；仅创建真实需要的读取/转换边界。共用核心目录中的改动按文件独立提交，不能覆盖其他已迁移模块。

## 布局与全部原功能分组
页内四步导航：导入 → 调类与调值 → 声韵排序归并 → 结果；步骤可返回，确认后生成。

| 编号 | 原功能组 | 必须保留的功能 | 新位置 | 行为约束 |
| --- | --- | --- | --- | --- |
| M14-F01 | 导入与策略 | XLSX/XLS/CSV/TXT/TSV、是否跳过首行、单辅音按零声母字或空韵字处理 | 导入步骤 | 保留源码的表头选择与两种单辅音策略；取消不覆盖当前已载数据。 |
| M14-F02 | 调值与调类 | 调值顺序拖动、调类名称、调值归并映射、确定/取消 | 调类与调值步骤 | 展示映射和最终顺序；这是既有对话框功能，不能遗漏。 |
| M14-F03 | 声韵排序 | 声母/韵母列表、Ctrl/Shift 多选、拖动排序 | 声韵步骤 | 排序影响最终文档顺序，并保留多选操作。 |
| M14-F04 | 声韵归并 | 选中源音标、选择归并、点击目标音标、声母/韵母归并、确认/取消 | 声韵步骤工具栏 | 归并映射可审阅，不直接改写输入语料。 |
| M14-F05 | 结果生成 | 选择输出目录、生成正序/逆序两份 DOCX 与同音字矩阵 XLSX、帮助 | 结果步骤 | 成功列出三份文件，保留文件类型与原有内容语义。 |

## 状态、重用与双端差异
无有效行、缺字段、含单辅音、读取失败、步骤取消、导出失败分别反馈；返回上一步保留配置。

上轮概览中的“行范围选择”没有在本次核对的主流程中找到对应现成控件，不作为既有功能声称；如后续加入，单列增强项。

迁移重点：导入语义、调类/调值映射、两种单辅音策略、声韵排序归并都保留；使用纯配置对象代替依赖 Qt 的对话框状态。

平台边界：表格解析与文档生成由核心/导出服务；Web 使用原子结果集合提交，不能只保存其中一个却报全部成功。

## 实施步骤与每步证据
1. 只读列出上面每个功能组的旧控件、调用函数、默认值、输入输出格式、异常、现有测试。创建 `docs/modules/evidence/M14-source-map.md`，对新增/修复功能单独标识；不能把按钮文案当成功能已可用的证明。
2. 在 P03 固定环境与样例下捕获本模块 v2 行为；核对 actual backend、单位、输出时间轴与缺失值。存入本地被忽略的输出目录，并把可公开 fixture 与配置/hash 写入测试清单。随机算法固定种子。新增功能没有旧基准时用明确规则验收。
3. 把已有可用函数迁入核心；先保留数值步骤，只拆 UI/文件/设备/全局可变状态。目标函数显式收配置与输入，任务快照与结果记录方法版本。适配变动与算法优化分开提交。
4. 新建本模块契约并接应用服务；输入通过资源 ID/已授权本地文件接口，输出进入 manifest。进度、取消、失败和部分结果统一；Web 所有临时/最终写入经过配额 writer。只显示已有结果的页面不强造长任务。
5. 使用共享组件实现页面、主题、播放/选区、抽屉和页内子视图。把上述功能行一项项映射到组件与操作，保存/生成/试听等语义不能合并丢失。Web/desktop port 区分文件和设备能力，配置状态不跨账号/模块污染。
6. 运行下面的专项场景与数值对照；先修真实差异，再对浅/深色、窗口缩放、键盘/IPA 字体、错误空态截图审阅。不能只运行单位测试就把 UI 和设备标已通过。
7. 连接模块“方法与来源”入口，使用 PENDING-PHONOLOGY。更新说明书本模块操作、输入输出、参数单位、科学能力边界和来源；当前 unresolved 的条目不能伪装已核验。移植文件保留原版权/许可证。
8. 更新 `docs/plans/task-ledger.json` 与原功能矩阵的证据列，记录实际命令/报告/截图；只有每行有可定位证据、双端必测通过才标完成。评审通过前保留旧功能来源，不删除旧入口或数据。

## 专项验收
五种输入类型、跳首行、多选拖动、取消/返回、冲突归并、正序/逆序 DOCX 与矩阵 XLSX 恰好三份及内容对照。

每个分组至少一个正常路径和一个相关错误/边界路径；录制、原生时序和数值算法必须在真实目标环境验证。

拟建测试后的执行命令（当前不能当作已运行）：

```powershell
# 先激活 P02 建立并核验的 v3 环境，不改 phonetic_311。
python -m pytest tests/parity/test_phonology_induction.py -q
npm --prefix frontend run test:e2e -- phonology-induction.spec.ts
```

CLI 参数和 npm scripts 在 P02 明确定义后才能使用；如实现路径不同，先更新本计划与架构记录。共享测试另见 P03/P11。两条命令不能代替本模块原生/设备手工验收。

## 完成条件
- 本页全部 5 组功能以及原矩阵相关参数/设置有映射，没有把折叠项当作删除项。
- 数值/文件/时间轴差异均解释并审阅；已有功能不得静默改语义。
- 浅深色、未保存保护、错误恢复与真实操作可用；Web owner/配额检查适用的路径已覆盖。
- 软件/说明书的来源一致；尚缺权限/设备证据时状态仍为待处理，不能以隐藏控件绕过。
