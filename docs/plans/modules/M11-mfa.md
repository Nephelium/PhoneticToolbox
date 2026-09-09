# M11 · MFA 自动标注迁移计划

状态：planned，未开始实现。普通模块依 P03/P04/P06/P07；设备/原生模块另依 P01。共用 [架构](../../../ARCHITECTURE.md)、[测试规范](../../testing/verification-plan.md) 和 [UI 规范](../../design/UI_SPEC.md)。

## 现有代码与目标文件
现有路径均已确认存在；目录内逐函数对应由实施第一步记录，避免把旧类名机械套给新实现。

- [phonetic_toolbox/gui/dialogs/mfa_auto_alignment_dialog.py](../../../phonetic_toolbox/gui/dialogs/mfa_auto_alignment_dialog.py)
- [phonetic_toolbox/services/mfa_alignment_service.py](../../../phonetic_toolbox/services/mfa_alignment_service.py)
- [phonetic_toolbox/services/pipelines/mfa_alignment_pipeline.py](../../../phonetic_toolbox/services/pipelines/mfa_alignment_pipeline.py)
- [phonetic_toolbox/core/transcription/mfa_name_codec.py](../../../phonetic_toolbox/core/transcription/mfa_name_codec.py)

拟创建/修改路径（未来实现，不代表已经存在）：

- `frontend/src/modules/mfa/MfaAlignmentPage.vue`
- `frontend/src/modules/mfa/state.ts`
- `packages/phonetic_core/src/phonetic_core/transcription/`
- `backend/src/ptb_api/modules/mfa.py`
- `tests/parity/test_mfa.py`
- `frontend/tests/e2e/mfa.spec.ts`

纯显示/客户端模块如果没有科学计算，不为凑层数创建空 core/API；仅创建真实需要的读取/转换边界。共用核心目录中的改动按文件独立提交，不能覆盖其他已迁移模块。

## 布局与全部原功能分组
左侧模型/词典/输入输出表单；右侧任务日志；底部运行状态与开始对齐。

| 编号 | 原功能组 | 必须保留的功能 | 新位置 | 行为约束 |
| --- | --- | --- | --- | --- |
| M11-F01 | 输入资源 | 声学模型文件、字典文件、音频目录、输出目录及浏览 | 左侧表单 | 路径分别校验，文件与目录选择器类型保持正确。 |
| M11-F02 | 对齐参数 | Beam、Retry beam | 表单参数组 | 保留现有 Retry beam 随 Beam 的约束/同步行为；不能单纯改为两个无关联文本框。 |
| M11-F03 | 执行与诊断 | 开始对齐、运行进度、日志、完成/错误详情、帮助 | 底部状态＋右侧日志 | 缺失模型/环境明确指出；日志可回看，失败后保留输入。 |
| M11-F04 | 新增取消控制 | 新增：可取消的任务状态与后台进程停止协议 | 任务条 | 当前 v2 隐藏了取消按钮；实现前必须验证只终止本任务且不破坏已完成结果。 |

## 状态、重用与双端差异
资源未齐、运行环境缺失、运行中、失败、完成；只有后端真实支持取消时才启用取消按钮。

源码 290 行的“取消”构造文字并不代表现成功能，293 行 setCancelButton(None) 将它隐藏。本项在矩阵中标为新增，不计作原功能保留。

迁移重点：继承路径/名称编码、Beam 与 Retry beam 的约束；模型/字典须按各自许可证登记；当前隐藏取消不是已有通过项。

平台边界：本地外部环境由 launcher 管；服务端隔离执行及账号文件集合；模型共享只读，不能让上传模型以任意脚本执行。

## 实施步骤与每步证据
1. 只读列出上面每个功能组的旧控件、调用函数、默认值、输入输出格式、异常、现有测试。创建 `docs/modules/evidence/M11-source-map.md`，对新增/修复功能单独标识；不能把按钮文案当成功能已可用的证明。
2. 在 P03 固定环境与样例下捕获本模块 v2 行为；核对 actual backend、单位、输出时间轴与缺失值。存入本地被忽略的输出目录，并把可公开 fixture 与配置/hash 写入测试清单。随机算法固定种子。新增功能没有旧基准时用明确规则验收。
3. 把已有可用函数迁入核心；先保留数值步骤，只拆 UI/文件/设备/全局可变状态。目标函数显式收配置与输入，任务快照与结果记录方法版本。适配变动与算法优化分开提交。
4. 新建本模块契约并接应用服务；输入通过资源 ID/已授权本地文件接口，输出进入 manifest。进度、取消、失败和部分结果统一；Web 所有临时/最终写入经过配额 writer。只显示已有结果的页面不强造长任务。
5. 使用共享组件实现页面、主题、播放/选区、抽屉和页内子视图。把上述功能行一项项映射到组件与操作，保存/生成/试听等语义不能合并丢失。Web/desktop port 区分文件和设备能力，配置状态不跨账号/模块污染。
6. 运行下面的专项场景与数值对照；先修真实差异，再对浅/深色、窗口缩放、键盘/IPA 字体、错误空态截图审阅。不能只运行单位测试就把 UI 和设备标已通过。
7. 连接模块“方法与来源”入口，使用 SRC-MFA, REF-MFA。更新说明书本模块操作、输入输出、参数单位、科学能力边界和来源；当前 unresolved 的条目不能伪装已核验。移植文件保留原版权/许可证。
8. 更新 `docs/plans/task-ledger.json` 与原功能矩阵的证据列，记录实际命令/报告/截图；只有每行有可定位证据、双端必测通过才标完成。评审通过前保留旧功能来源，不删除旧入口或数据。

## 专项验收
中文/空格文件名、模型缺失、词典错误、超时、真实取消、仅停止自己的子进程、部分 TextGrid 输出和完整日志。

每个分组至少一个正常路径和一个相关错误/边界路径；录制、原生时序和数值算法必须在真实目标环境验证。

拟建测试后的执行命令（当前不能当作已运行）：

```powershell
# 先激活 P02 建立并核验的 v3 环境，不改 phonetic_311。
python -m pytest tests/parity/test_mfa.py -q
npm --prefix frontend run test:e2e -- mfa.spec.ts
```

CLI 参数和 npm scripts 在 P02 明确定义后才能使用；如实现路径不同，先更新本计划与架构记录。共享测试另见 P03/P11。两条命令不能代替本模块原生/设备手工验收。

## 完成条件
- 本页全部 4 组功能以及原矩阵相关参数/设置有映射，没有把折叠项当作删除项。
- 数值/文件/时间轴差异均解释并审阅；已有功能不得静默改语义。
- 浅深色、未保存保护、错误恢复与真实操作可用；Web owner/配额检查适用的路径已覆盖。
- 软件/说明书的来源一致；尚缺权限/设备证据时状态仍为待处理，不能以隐藏控件绕过。
