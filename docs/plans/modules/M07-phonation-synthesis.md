# M07 · 发声类型合成迁移计划

状态：in_progress。2026-09-27 Windows开发态闭环已限定verified，Linux/远程仍待准入。现行文件、命令和验收以 [实施计划](../2026-09-27-m07-implementation.md) 与 [主报告](../../testing/m07-report.md) 为准。以下为原迁移设计，旧拟建命令不得当当前入口。普通模块依 P03/P04/P06/P07；设备/原生模块另依 P01。共用 [架构](../../../ARCHITECTURE.md)、[测试规范](../../testing/verification-plan.md) 和 [UI 规范](../../design/UI_SPEC.md)。

## 现有代码与目标文件
现有路径均已确认存在；目录内逐函数对应由实施第一步记录，避免把旧类名机械套给新实现。

- [phonetic_toolbox/gui/widgets/phonation_synthesis_widget.py](../../../phonetic_toolbox/gui/widgets/phonation_synthesis_widget.py)
- [phonetic_toolbox/core/manipulation/phonation_synthesis.py](../../../phonetic_toolbox/core/manipulation/phonation_synthesis.py)
- [phonetic_toolbox/services/phonation_synthesis_service.py](../../../phonetic_toolbox/services/phonation_synthesis_service.py)

拟创建/修改路径（未来实现，不代表已经存在）：

- `frontend/src/modules/phonation-synthesis/PhonationSynthesisPage.vue`
- `frontend/src/modules/phonation-synthesis/state.ts`
- `packages/phonetic_core/src/phonetic_core/manipulation/`
- `backend/src/ptb_api/modules/phonation_synthesis.py`
- `tests/parity/test_phonation_synthesis.py`
- `frontend/tests/e2e/phonation-synthesis.spec.ts`

纯显示/客户端模块如果没有科学计算，不为凑层数创建空 core/API；仅创建真实需要的读取/转换边界。共用核心目录中的改动按文件独立提交，不能覆盖其他已迁移模块。

## 布局与全部原功能分组
顶部源/目标/输出路径；中央波形与 F0 对比和可编辑表；右侧分析与连续统设置；底部生成任务。

| 编号 | 原功能组 | 必须保留的功能 | 新位置 | 行为约束 |
| --- | --- | --- | --- | --- |
| M07-F01 | 文件 | 源音频、目标音频、输出目录 | 顶部路径区 | 源与目标角色始终显示，不允许仅凭颜色判断。 |
| M07-F02 | F0 分析 | parselmouth/reaper、最小最大 F0、F0 帧隔、提取 F0 | 右侧分析组 | 切换分析配置标记旧结果失效；缺失 REAPER 时说明当前后端不可用。 |
| M07-F03 | 连续统与对齐 | 仅 F0、仅发声类型、二者同时；时长归一化或仅有声起点对齐；步数与编辑点数 | 右侧连续统组 | 三种连续统和两种对齐方式全部可选，保持现有语义。 |
| M07-F04 | 合成参数 | LPC 阶数、窗长/帧移（点）、窗函数、预加重、负峰阈值、脉冲内外范围、静音阈值/留白、有声边界、峰值限制 | 右侧可滚动高级设置 | 窗长/帧移按原模块使用采样点，不能因全局样式统一误标为毫秒。 |
| M07-F05 | 幅度策略 | 周期能量匹配、输出响度匹配源音频 | 合成设置尾部 | 两个独立开关分别保留，并记录到结果配置。 |
| M07-F06 | 编辑与输出 | 对齐时间/源 F0/目标 F0 表、应用编辑、保存 F0 CSV、生成当前/全部、取消任务、参数说明 | 中央表＋底部生成区 | 当前与全部的输出范围清楚；任务取消保持已完成结果并报告未完成部分。 |

## 状态、重用与双端差异
分析过期、表格非法、源/目标为空、取消中、部分完成分别显示；失败后保留输入和编辑。

这是参数最密集的模块之一。高级设置允许滚动/折叠，但全部字段必须能通过键盘定位，不做删减。

迁移重点：区分原论文与 Python 近似移植；保留 3 连续统×2 方向的全部输出、两种时间对齐和幅度开关；窗长/帧移按采样点。

平台边界：生成前计算组合规模并预留空间；写入过程受限，不能用估算通过就无限生成；原论文数据未获明确许可不随包带入。

## 实施步骤与每步证据
1. 只读列出上面每个功能组的旧控件、调用函数、默认值、输入输出格式、异常、现有测试。创建 `docs/modules/evidence/M07-source-map.md`，对新增/修复功能单独标识；不能把按钮文案当成功能已可用的证明。
2. 在 P03 固定环境与样例下捕获本模块 v2 行为；核对 actual backend、单位、输出时间轴与缺失值。存入本地被忽略的输出目录，并把可公开 fixture 与配置/hash 写入测试清单。随机算法固定种子。新增功能没有旧基准时用明确规则验收。
3. 把已有可用函数迁入核心；先保留数值步骤，只拆 UI/文件/设备/全局可变状态。目标函数显式收配置与输入，任务快照与结果记录方法版本。适配变动与算法优化分开提交。
4. 新建本模块契约并接应用服务；输入通过资源 ID/已授权本地文件接口，输出进入 manifest。进度、取消、失败和部分结果统一；Web 所有临时/最终写入经过配额 writer。只显示已有结果的页面不强造长任务。
5. 使用共享组件实现页面、主题、播放/选区、抽屉和页内子视图。把上述功能行一项项映射到组件与操作，保存/生成/试听等语义不能合并丢失。Web/desktop port 区分文件和设备能力，配置状态不跨账号/模块污染。
6. 运行下面的专项场景与数值对照；先修真实差异，再对浅/深色、窗口缩放、键盘/IPA 字体、错误空态截图审阅。不能只运行单位测试就把 UI 和设备标已通过。
7. 连接模块“方法与来源”入口，使用 SRC-ZAIWA, REF-ZAIWA, SRC-PRAAT, SRC-REAPER。更新说明书本模块操作、输入输出、参数单位、科学能力边界和来源；当前 unresolved 的条目不能伪装已核验。移植文件保留原版权/许可证。
8. 更新 `docs/plans/task-ledger.json` 与原功能矩阵的证据列，记录实际命令/报告/截图；只有每行有可定位证据、双端必测通过才标完成。评审通过前保留旧功能来源，不删除旧入口或数据。

## 专项验收
有声区缺失、源/目标长度不等、F0 CSV 往返、编辑非法、当前/全部六类输出数目、能量匹配、取消和配额耗尽时部分完成。

每个分组至少一个正常路径和一个相关错误/边界路径；录制、原生时序和数值算法必须在真实目标环境验证。

拟建测试后的执行命令（当前不能当作已运行）：

```powershell
# 先激活 P02 建立并核验的 v3 环境，不改 phonetic_311。
python -m pytest tests/parity/test_phonation_synthesis.py -q
npm --prefix frontend run test:e2e -- phonation-synthesis.spec.ts
```

CLI 参数和 npm scripts 在 P02 明确定义后才能使用；如实现路径不同，先更新本计划与架构记录。共享测试另见 P03/P11。两条命令不能代替本模块原生/设备手工验收。

## 完成条件
- 本页全部 6 组功能以及原矩阵相关参数/设置有映射，没有把折叠项当作删除项。
- 数值/文件/时间轴差异均解释并审阅；已有功能不得静默改语义。
- 浅深色、未保存保护、错误恢复与真实操作可用；Web owner/配额检查适用的路径已覆盖。
- 软件/说明书的来源一致；尚缺权限/设备证据时状态仍为待处理，不能以隐藏控件绕过。
