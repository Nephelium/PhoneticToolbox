# M08 · 变速变调迁移计划

2026-10-05 M08-R2：按井井截图反馈修正历史图窗/空态、固定选框、小数中间态及本地导出位置提示，限定Windows源码verified；见[计划](../2026-10-05-m08-r2-input-layout-save.md)和[报告](../../testing/2026-10-05-m08-r2-report.md)。科学算法和下文跨平台未决门保留。

2026-10-04 M08-R1：井井明确要求整理生成 / 试听 / 保存。当前交互以[本轮计划](../2026-10-04-m08-r1-workflow.md)、[ADR](../../decisions/ADR-M08-R1.md)及[报告](../../testing/2026-10-04-m08-r1-report.md)为准。下表中的添加 / 清除拐点通过表内添加行 / 删除行完成，合成结果自动进入历史，外部保存与生成分离。旧功能映射作为迁移记录保留。

状态：in_progress。2026-09-27 已完成正式任务/存储 adapter、三格式解码、AppShell 注册及限定 Windows HTTP/Chrome/Qt 验收，见 [正式接线报告](../../testing/m08-wiring-report.md)。真实 PostgreSQL 网页因政策门受阻，Linux 正式受限链路与跨平台精确门仍未完成。普通模块依 P03/P04/P06/P07；设备/原生模块另依 P01。共用 [架构](../../../ARCHITECTURE.md)、[测试规范](../../testing/verification-plan.md) 和 [UI 规范](../../design/UI_SPEC.md)。


## 2026-09-26 实施检查点（历史，接线状态以新报告为准）

- 用户已明确实施授权，范围限定 M08 文件；并行 owner 的公共执行器、capability、入口、AppShell、tokens、公共组件和注册未改。总台账/全局 ADR 由统筹汇总。
- [六组映射](../../modules/evidence/M08-source-map.md)、[使用说明](../../manual/pitch-manipulation.md)、[精确接线需求](M08-wiring.md)、[实际验收](../../testing/m08-report.md)。
- 直接迁移三份科学文件与新增边界规则分离。V2 非零 Hz 偏移错误在 handler 显式修正为 Hertz，原错误仍有独立回归。
- Windows 源码/安装 wheel 定向 43 项，前端全量 137 项，真实 Chrome 15 组测试通过（限定独立组件及真实计算测试 adapter）。
- Linux WSL 原 Windows 冻结精确门 25 passed / 5 failed；64 诊断数组59完全相同。另 Linux 同环境原V2与V3 64/64精确相同。禁止把同平台对照替代跨平台精确门。
- 正式网页 owner/配额/下载到期、持久队列/硬取消、MP3/FLAC FileProvider、AppShell 注册与标签关闭保护均需公共接线后联合验证。服务器资源/远程节点/EXE未实施。
- 本轮不推进 M07，不改总台账、数据库、共享依赖清单、环境配置或旧发行包。

## 现有代码与目标文件
现有路径均已确认存在；目录内逐函数对应由实施第一步记录，避免把旧类名机械套给新实现。

- [phonetic_toolbox/gui/widgets/pitch_manipulation_widget.py](../../../phonetic_toolbox/gui/widgets/pitch_manipulation_widget.py)
- [phonetic_toolbox/core/manipulation/synthesis.py](../../../phonetic_toolbox/core/manipulation/synthesis.py)
- [phonetic_toolbox/core/manipulation/batch_utils.py](../../../phonetic_toolbox/core/manipulation/batch_utils.py)

拟创建/修改路径（未来实现，不代表已经存在）：

- `frontend/src/modules/pitch-manipulation/PitchManipulationPage.vue`
- `frontend/src/modules/pitch-manipulation/state.ts`
- `packages/phonetic_core/src/phonetic_core/manipulation/`
- `backend/src/ptb_api/modules/pitch_manipulation.py`
- `tests/parity/test_pitch_manipulation.py`
- `frontend/tests/e2e/pitch-manipulation.spec.ts`

纯显示/客户端模块如果没有科学计算，不为凑层数创建空 core/API；仅创建真实需要的读取/转换边界。共用核心目录中的改动按文件独立提交，不能覆盖其他已迁移模块。

## 布局与全部原功能分组
顶部文件与语速；中央波形/F0 编辑；右侧基频范围与参考线；底部试听与保存；批量工具为页内独立子页。

| 编号 | 原功能组 | 必须保留的功能 | 新位置 | 行为约束 |
| --- | --- | --- | --- | --- |
| M08-F01 | 单文件操作 | 加载音频、语速倍率、当前视野原音播放、当前视野合成、合成音播放、保存并编号 | 顶部控制＋底栏 | 当前视野与整段音频处理范围区分；保存编号沿用现有规则。 |
| M08-F02 | 曲线与参照 | F0 范围应用、频率参考线添加/清除、保存对比图、导入基频序列 | 中央编辑器＋右侧工具 | 导入不静默重置视野；基频单位和时间基准可见。 |
| M08-F03 | 本批次文件 | 删除本批次音频、批量重命名、历史对比 | 结果列表的文件管理菜单 | 删除列出明确文件和范围；不能扩大为语料目录全部文件。 |
| M08-F04 | 批量变速变调 | 输入文件夹、处理参数、开始、关闭 | 批量工具子页 | 逐文件报告输出与失败，不以一个成功样例代表全批成功。 |
| M08-F05 | 批量基频 | 起止时间、起止 F0 列表、拐点时间和 F0 列表、添加/清除/编辑拐点、直线插值、升降偏移、生成保存 | 批量基频子页 | 四种连接方式全连接/顺序/逆序/常量保留；起点终点的编辑约束沿用源码。 |
| M08-F06 | 拐点表与帮助 | 添加行、删除选中、保存更改、关闭、帮助 | 拐点表子视图＋标题栏 | 删除拐点不删除音频文件；两种删除动作使用不同文案。 |

## 状态、重用与双端差异
不合理倍率、时间区间越界、F0 列表长度/连接方式错误就地提示；对比结果保留生成配置。

文件管理中的删除仍是显式用户操作；样式优化不改变可删除文件的边界。

迁移重点：四种连接方式、当前视野的作用范围、编号/本批次列表、起点终点约束全部迁移；删除只能指向当前拥有的明确工件。

平台边界：Web 不能执行本地路径批量删除；所有文件操作走 owner 检查与资源 ID。

## 实施步骤与每步证据
1. 只读列出上面每个功能组的旧控件、调用函数、默认值、输入输出格式、异常、现有测试。创建 `docs/modules/evidence/M08-source-map.md`，对新增/修复功能单独标识；不能把按钮文案当成功能已可用的证明。
2. 在 P03 固定环境与样例下捕获本模块 v2 行为；核对 actual backend、单位、输出时间轴与缺失值。存入本地被忽略的输出目录，并把可公开 fixture 与配置/hash 写入测试清单。随机算法固定种子。新增功能没有旧基准时用明确规则验收。
3. 把已有可用函数迁入核心；先保留数值步骤，只拆 UI/文件/设备/全局可变状态。目标函数显式收配置与输入，任务快照与结果记录方法版本。适配变动与算法优化分开提交。
4. 新建本模块契约并接应用服务；输入通过资源 ID/已授权本地文件接口，输出进入 manifest。进度、取消、失败和部分结果统一；Web 所有临时/最终写入经过配额 writer。只显示已有结果的页面不强造长任务。
5. 使用共享组件实现页面、主题、播放/选区、抽屉和页内子视图。把上述功能行一项项映射到组件与操作，保存/生成/试听等语义不能合并丢失。Web/desktop port 区分文件和设备能力，配置状态不跨账号/模块污染。
6. 运行下面的专项场景与数值对照；先修真实差异，再对浅/深色、窗口缩放、键盘/IPA 字体、错误空态截图审阅。不能只运行单位测试就把 UI 和设备标已通过。
7. 连接模块“方法与来源”入口，使用 SRC-PRAAT。更新说明书本模块操作、输入输出、参数单位、科学能力边界和来源；当前 unresolved 的条目不能伪装已核验。移植文件保留原版权/许可证。
8. 更新 `docs/plans/task-ledger.json` 与原功能矩阵的证据列，记录实际命令/报告/截图；只有每行有可定位证据、双端必测通过才标完成。评审通过前保留旧功能来源，不删除旧入口或数据。

## 专项验收
全部批量基频组合、拐点增删/往返、超范围 F0、历史对比、批量重命名/重复名、满额但仍可下载删除。

每个分组至少一个正常路径和一个相关错误/边界路径；录制、原生时序和数值算法必须在真实目标环境验证。

现行测试入口（实际运行结果与环境见报告，不沿用已失效的 test:e2e）：

```powershell
# 先激活 P02 建立并核验的 v3 环境，不改 phonetic_311。
python -m pytest tests/parity/test_pitch_manipulation.py -q
node tests/e2e/m08.cjs
```

npm 现行 scripts 为 test/typecheck/build，没有 test:e2e。M08 按现有 Vite + Chrome + stdio 测试适配模式运行，不能代替公共 production adapter。共享测试另见 P03/P11。两条命令不能代替本模块原生/设备手工验收。

## 完成条件
- 本页全部 6 组功能以及原矩阵相关参数/设置有映射，没有把折叠项当作删除项。
- 数值/文件/时间轴差异均解释并审阅；已有功能不得静默改语义。
- 浅深色、未保存保护、错误恢复与真实操作可用；Web owner/配额检查适用的路径已覆盖。
- 软件/说明书的来源一致；尚缺权限/设备证据时状态仍为待处理，不能以隐藏控件绕过。
