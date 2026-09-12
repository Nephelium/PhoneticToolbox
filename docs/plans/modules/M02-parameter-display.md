# M02 · 参数显示迁移计划

状态：verified，限定Windows桌面与Windows托管Chrome的已列明范围。实际路径与命令由[本次实施计划](../2026-09-11-m01-m02-m09.md)、[源码映射](../../modules/evidence/M02-source-map.md)和[联合报告](../../testing/m02-m09-report.md)更新，下方保留迁移前规格；M09截图设备、多平台与发行未纳入完成声明。普通模块依 P03/P04/P06/P07；设备/原生模块另依 P01。共用 [架构](../../../ARCHITECTURE.md)、[测试规范](../../testing/verification-plan.md) 和 [UI 规范](../../design/UI_SPEC.md)。

## 现有代码与目标文件
现有路径均已确认存在；目录内逐函数对应由实施第一步记录，避免把旧类名机械套给新实现。

- [phonetic_toolbox/gui/widgets/parameter_display_widget.py](../../../phonetic_toolbox/gui/widgets/parameter_display_widget.py)
- [phonetic_toolbox/gui/widgets/acoustic_widget.py](../../../phonetic_toolbox/gui/widgets/acoustic_widget.py)

拟创建/修改路径（未来实现，不代表已经存在）：

- `frontend/src/modules/parameter-display/ParameterDisplayPage.vue`
- `frontend/src/modules/parameter-display/state.ts`
- `packages/phonetic_core/src/phonetic_core/results/`
- `backend/src/ptb_api/modules/parameter_display.py`
- `tests/parity/test_parameter_display.py`
- `frontend/tests/e2e/parameter-display.spec.ts`

纯显示/客户端模块如果没有科学计算，不为凑层数创建空 core/API；仅创建真实需要的读取/转换边界。共用核心目录中的改动按文件独立提交，不能覆盖其他已迁移模块。

## 布局与全部原功能分组
左侧 WAV/XLSX 目录与文件列表；中央波形与参数叠加图；右侧参数选择；上方视窗导航；底部选区播放。

| 编号 | 原功能组 | 必须保留的功能 | 新位置 | 行为约束 |
| --- | --- | --- | --- | --- |
| M02-F01 | 载入既有结果 | WAV 目录、XLSX 目录、浏览、音频筛选、刷新、选择文件 | 左侧资源栏 | 可以直接读取既有参数表，不要求重新做参数估计。 |
| M02-F02 | 参数与过滤 | 参数搜索、多选、reaper、correction 两个独立开关 | 右侧参数栏 | 筛选只改变候选项可见性，不静默修改已选参数或混合两个开关的含义。 |
| M02-F03 | 参数叠加显示 | 波形、多个参数曲线、语谱图、共享时间轴 | 上方波形、下方叠加图窗 | 同窗曲线共享绘图区和纵轴，旧量级规则可自动双轴，分窗显式分配；参数名/图例完整，时间同步。 |
| M02-F04 | 时间操作 | 可视范围、位置滑条、缩放、平移、选区 | 图顶导航＋底部读数 | 视窗时长、全文件时长、选区时长分开显示，0.8–1.2 秒必须对应真实坐标。 |
| M02-F05 | 输出与帮助 | 播放及选区播放、保存图片、参数说明、帮助、关闭 | 播放条＋页面标题栏 | 保存目标明确为当前图；关闭仅关闭模块页。 |

## 状态、重用与双端差异
缺少匹配 XLSX 时仍显示音频并列出缺失项；空筛选可一键清除；不把缺失值绘成零值。

配色在深浅主题中保持语义：Praat 与 REAPER 有稳定的线型/图例区分；不能只靠蓝橙颜色识别。

迁移重点：保留既有 XLSX 的真实列解析；显示视窗/选区/全长三个量；缺失值用断线而非补零。

平台边界：显示只需结果读取服务；大图用分辨率适配，不把完整巨型 XLSX 在每次拖动重传。

## 实施步骤与每步证据
1. 只读列出上面每个功能组的旧控件、调用函数、默认值、输入输出格式、异常、现有测试。创建 `docs/modules/evidence/M02-source-map.md`，对新增/修复功能单独标识；不能把按钮文案当成功能已可用的证明。
2. 在 P03 固定环境与样例下捕获本模块 v2 行为；核对 actual backend、单位、输出时间轴与缺失值。存入本地被忽略的输出目录，并把可公开 fixture 与配置/hash 写入测试清单。随机算法固定种子。新增功能没有旧基准时用明确规则验收。
3. 把已有可用函数迁入核心；先保留数值步骤，只拆 UI/文件/设备/全局可变状态。目标函数显式收配置与输入，任务快照与结果记录方法版本。适配变动与算法优化分开提交。
4. 新建本模块契约并接应用服务；输入通过资源 ID/已授权本地文件接口，输出进入 manifest。进度、取消、失败和部分结果统一；Web 所有临时/最终写入经过配额 writer。只显示已有结果的页面不强造长任务。
5. 使用共享组件实现页面、主题、播放/选区、抽屉和页内子视图。把上述功能行一项项映射到组件与操作，保存/生成/试听等语义不能合并丢失。Web/desktop port 区分文件和设备能力，配置状态不跨账号/模块污染。
6. 运行下面的专项场景与数值对照；先修真实差异，再对浅/深色、窗口缩放、键盘/IPA 字体、错误空态截图审阅。不能只运行单位测试就把 UI 和设备标已通过。
7. 连接模块“方法与来源”入口，使用 SRC-PRAAT, SRC-REAPER, SRC-VOICESAUCE。更新说明书本模块操作、输入输出、参数单位、科学能力边界和来源；当前 unresolved 的条目不能伪装已核验。移植文件保留原版权/许可证。
8. 更新 `docs/plans/task-ledger.json` 与原功能矩阵的证据列，记录实际命令/报告/截图；只有每行有可定位证据、双端必测通过才标完成。评审通过前保留旧功能来源，不删除旧入口或数据。

## 专项验收
WAV/XLSX 无匹配、不同帧率参数、长音频抽稀、F0 双算法、0.8–1.2 秒选区准确播放、独立 reaper/correction 开关。

每个分组至少一个正常路径和一个相关错误/边界路径；录制、原生时序和数值算法必须在真实目标环境验证。

拟建测试后的执行命令（当前不能当作已运行）：

```powershell
# 先激活 P02 建立并核验的 v3 环境，不改 phonetic_311。
python -m pytest tests/parity/test_parameter_display.py -q
npm --prefix frontend run test:e2e -- parameter-display.spec.ts
```

CLI 参数和 npm scripts 在 P02 明确定义后才能使用；如实现路径不同，先更新本计划与架构记录。共享测试另见 P03/P11。两条命令不能代替本模块原生/设备手工验收。

## 完成条件
- 本页全部 5 组功能以及原矩阵相关参数/设置有映射，没有把折叠项当作删除项。
- 数值/文件/时间轴差异均解释并审阅；已有功能不得静默改语义。
- 浅深色、未保存保护、错误恢复与真实操作可用；Web owner/配额检查适用的路径已覆盖。
- 软件/说明书的来源一致；尚缺权限/设备证据时状态仍为待处理，不能以隐藏控件绕过。
