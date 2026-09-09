# M01 · 参数估计迁移计划

状态：in_progress；M01-A基准与M01-B科学核心已完成限定Windows验收，完整页面与原生预算/批任务尚未迁移。普通模块依 P03/P04/P06/P07；设备/原生模块另依 P01。共用 [架构](../../../ARCHITECTURE.md)、[测试规范](../../testing/verification-plan.md) 和 [UI 规范](../../design/UI_SPEC.md)。

2026-09-09 已完成迁移前源码核对和 [M01-A独立基准](../../testing/m01-baseline-report.md)，[M01-B科学核心](../../testing/m01-core-report.md)已通过149项测试，下一项为M01-C原生与格式预算适配。执行细节以 [文件级实施计划](../2026-09-09-m01-implementation.md) 为准；[源码对照](../../modules/evidence/M01-source-map.md)、[参数/设置清单](../../modules/evidence/M01-parameter-settings.json)、[30项验收表](../../modules/evidence/M01-acceptance.csv) 和 [本轮报告](../../testing/m01-planning-report.md) 已补齐。下列原功能规格继续有效；文件级计划修订了尚未存在的 E2E 命令和持久批次接入安排。

## 现有代码与目标文件
现有路径均已确认存在；目录内逐函数对应由实施第一步记录，避免把旧类名机械套给新实现。

- [phonetic_toolbox/gui/widgets/parameter_estimation_widget.py](../../../phonetic_toolbox/gui/widgets/parameter_estimation_widget.py)
- [phonetic_toolbox/core/acoustic](../../../phonetic_toolbox/core/acoustic)
- [phonetic_toolbox/services/settings_service.py](../../../phonetic_toolbox/services/settings_service.py)

拟创建/修改路径（未来实现，不代表已经存在）：

- `frontend/src/modules/parameter-estimation/ParameterEstimationPage.vue`
- `frontend/src/modules/parameter-estimation/state.ts`
- `packages/phonetic_core/src/phonetic_core/acoustic/`
- `backend/src/ptb_api/acoustic_models.py`、`backend/src/ptb_api/jobs.py`（统一任务入口）
- `tests/parity/test_parameter_estimation.py`
- `tests/e2e/m01-parameter-estimation.cjs`

纯显示/客户端模块如果没有科学计算，不为凑层数创建空 core/API；仅创建真实需要的读取/转换边界。共用核心目录中的改动按文件独立提交，不能覆盖其他已迁移模块。

## 布局与全部原功能分组
顶部目录与关联工具栏；左侧文件列表；中央波形与 TextGrid；右侧参数摘要；底部播放与批处理状态。

| 编号 | 原功能组 | 必须保留的功能 | 新位置 | 行为约束 |
| --- | --- | --- | --- | --- |
| M01-F01 | 目录与文件 | 输入目录、输出目录、浏览、xlsx 与 wav 同目录；刷新与文件多选 | 顶部目录条＋左侧文件栏 | 同目录勾选后输出路径同步输入并禁用独立编辑；取消勾选后可指定输出。 |
| M01-F02 | 关联与切分 | 读取 TextGrid、TextGrid 切分、切分层选择、保存切分音频、读取唇形数据 | 波形上方关联工具栏＋TextGrid 层级条 | 层级改变同步刷新分段；保存切分音频仍是独立动作，不与参数导出混淆。 |
| M01-F03 | 参数选择 | 80 项独立参数、全选、全不选、确定、取消、参数说明 | 右侧摘要＋完整参数抽屉 | 摘要分组不合并参数键；pF0/rF0 及校正变体可分别勾选。 |
| M01-F04 | 完整设置 | 10 项常用设置＋4 项 REAPER 设置、保存、关闭 | 设置抽屉的两个页签 | 保留参数默认值、范围与单位；REAPER 设置不当作 Praat 通用设置。 |
| M01-F05 | 浏览与试听 | 波形缩放、平移、播放选中音频、停止 | 中央图＋底部播放条 | 文件选择服务于试听；采样点与秒的转换使用同一时间模型。 |
| M01-F06 | 全列表处理 | 开始处理、进度、取消、失败详情 | 底部任务条 | 原有处理范围为整个文件列表；显式写“处理列表中的 N 个文件”，不得改成仅处理勾选项。 |

## 状态、重用与双端差异
没有目录时展示选择目录；TextGrid/唇形未关联分别提示但不伪造曲线；取消显示已完成数量，结果文件不报为全部完成。

选择参数只改变待处理配置；已经显示的结果保留原计算配置标识，防止把旧结果误当作新配置结果。

迁移重点：80 参数键、列名/顺序、14 设置默认值和单位逐项冻结；保持 p/r 结果独立；同目录导出不得覆盖输入 WAV。

平台边界：两端算法共用；桌面选择目录，Web 上传成明确文件集合；服务端路径不回显给用户。

## 实施步骤与每步证据
1. 只读列出上面每个功能组的旧控件、调用函数、默认值、输入输出格式、异常、现有测试。创建 `docs/modules/evidence/M01-source-map.md`，对新增/修复功能单独标识；不能把按钮文案当成功能已可用的证明。
2. 在 P03 固定环境与样例下捕获本模块 v2 行为；核对 actual backend、单位、输出时间轴与缺失值。存入本地被忽略的输出目录，并把可公开 fixture 与配置/hash 写入测试清单。随机算法固定种子。新增功能没有旧基准时用明确规则验收。
3. 把已有可用函数迁入核心；先保留数值步骤，只拆 UI/文件/设备/全局可变状态。目标函数显式收配置与输入，任务快照与结果记录方法版本。适配变动与算法优化分开提交。
4. 新建本模块契约并接应用服务；输入通过资源 ID/已授权本地文件接口，输出进入 manifest。进度、取消、失败和部分结果统一；Web 所有临时/最终写入经过配额 writer。只显示已有结果的页面不强造长任务。
5. 使用共享组件实现页面、主题、播放/选区、抽屉和页内子视图。把上述功能行一项项映射到组件与操作，保存/生成/试听等语义不能合并丢失。Web/desktop port 区分文件和设备能力，配置状态不跨账号/模块污染。
6. 运行下面的专项场景与数值对照；先修真实差异，再对浅/深色、窗口缩放、键盘/IPA 字体、错误空态截图审阅。不能只运行单位测试就把 UI 和设备标已通过。
7. 连接模块“方法与来源”入口，使用 SRC-PRAAT, SRC-REAPER, SRC-IRAPT, SRC-WMPC, SRC-VOICESAUCE, SRC-OPENSAUCE, REF-CPP, REF-HNR, REF-SHR, REF-ISELI, REF-HAWKS, REF-SOE。更新说明书本模块操作、输入输出、参数单位、科学能力边界和来源；当前 unresolved 的条目不能伪装已核验。移植文件保留原版权/许可证。
8. 更新 `docs/plans/task-ledger.json` 与原功能矩阵的证据列，记录实际命令/报告/截图；只有每行有可定位证据、双端必测通过才标完成。评审通过前保留旧功能来源，不删除旧入口或数据。

## 专项验收
持续元音/气声/嘎裂/静音、空目录、无 TextGrid、仅部分文件失败、Praat/REAPER 后端失效、取消时部分结果。

每个分组至少一个正常路径和一个相关错误/边界路径；录制、原生时序和数值算法必须在真实目标环境验证。

命令入口已在 [文件级计划第3节](../2026-09-09-m01-implementation.md) 按 A–G 子任务具体列出。原草案的 `npm run test:e2e` 不适用当前工程：现有前端测试脚本使用 Node test，真实浏览器脚本位于 `tests/e2e/`。未来 M01 E2E 由拟建 `scripts/run_m01_validation.py` 调度独立浏览器，不能将不存在的 npm script 写成已运行证据。

## 完成条件
- 本页全部 6 组功能以及原矩阵相关参数/设置有映射，没有把折叠项当作删除项。
- 数值/文件/时间轴差异均解释并审阅；已有功能不得静默改语义。
- 浅深色、未保存保护、错误恢复与真实操作可用；Web owner/配额检查适用的路径已覆盖。
- 软件/说明书的来源一致；尚缺权限/设备证据时状态仍为待处理，不能以隐藏控件绕过。
