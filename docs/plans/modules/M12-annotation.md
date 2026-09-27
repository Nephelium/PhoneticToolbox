# M12 · 语音标注对齐迁移计划

2026-09-19 最新：R6 图窗优先及完整标注剪贴通过限定验证，生成用户授权的 R6 临时单文件 EXE，冻结基础链路 11 步通过。详见 [R6 报告](../../testing/m12-r6-report.md)，旧包保留。

2026-09-19：R5 交互、窗长、原文件优先及聚焦审查已完成限定 Windows 开发态验证，见 [R5 报告](../../testing/m12-r5-report.md)。旧 EXE 未更新。

2026-09-15 追加：R4 长 WAV 轻量读取与时间轴位置调整已获授权，按用户要求只修改和打包，不进行验证，见[R4 记录](../2026-09-15-m12-r4-long-audio.md)。下述 R3 verified 为历史限定证据，不代表 R4 已验证。

2026-09-14 最新：R3 本机试用修复与全局页面缩放已按用户追加范围 verified（限定 Windows/Chrome/Qt 及实际单文件 EXE），见[R3 报告](../../testing/m12-r3-report.md)。原 R1/R2 证据作为历史保留，当前操作以[使用说明](../../manual/annotation.md)为准。

状态：verified，限定 Windows 开发态功能；[2026-09-14实施记录](../2026-09-14-m12-implementation.md)和[验收报告](../../testing/m12-report.md)覆盖下述7组。[R1 追加修复与临时 EXE 验证](../../testing/m12-r1-report.md)覆盖空白/顺序标注、长录音匹配、框选与可配置微调。跨平台/生产未测。普通模块依 P03/P04/P06/P07；设备/原生模块另依 P01。共用 [架构](../../../ARCHITECTURE.md)、[测试规范](../../testing/verification-plan.md) 和 [UI 规范](../../design/UI_SPEC.md)。

## 现有代码与目标文件
现有路径均已确认存在；目录内逐函数对应由实施第一步记录，避免把旧类名机械套给新实现。

- [phonetic_toolbox/services/web_praat_server.py](../../../phonetic_toolbox/services/web_praat_server.py)
- [phonetic_toolbox/gui/resources/web_praat_editor](../../../phonetic_toolbox/gui/resources/web_praat_editor)
- [phonetic_toolbox/services/io/lip.py](../../../phonetic_toolbox/services/io/lip.py)

实际实现路径（替代原拟建 state.ts/API module/e2e 路径，使用现有统一宿主和文件能力）：

- `frontend/src/modules/annotation/AnnotationPage.vue`
- `frontend/src/modules/annotation/editor.mjs`、`format.ts`、`AnnotationTracks.vue`
- `packages/phonetic_core/src/phonetic_core/annotation/`
- `backend/src/ptb_worker/io/annotation.py`、`desktop/src/ptb_desktop/annotation.py`、`frontend/src/platform/annotation.ts`
- `frontend/tests/annotation-parity.test.ts`、`desktop/tests/test_m12_annotation.py`
- `tests/e2e/m12.cjs`、`m12-web.cjs`、`scripts/verify_m12_qt.py`

纯显示/客户端模块如果没有科学计算，不为凑层数创建空 core/API；仅创建真实需要的读取/转换边界。共用核心目录中的改动按文件独立提交，不能覆盖其他已迁移模块。

## 布局与全部原功能分组
左侧语料文件；中央波形/语谱图/唇形与词音素层；右侧层级/复用/保存设置；图下资源与搜索工具，图窗顶部保存。

| 编号 | 原功能组 | 必须保留的功能 | 新位置 | 行为约束 |
| --- | --- | --- | --- | --- |
| M12-F01 | 语料与层级 | 文件夹选择、扫描、文件筛选、词层名、音素层名 | 左侧资源＋右侧实际层名选择/显式创建 | R2 空白不自动建层，当前文件实际层优先；扫描不自动覆盖未保存标注。 |
| M12-F02 | 标注编辑 | 播放、波形/语谱图/层级编辑、时间导航、可视时长 | 中央三图＋顶部总览/音量＋空格选区试听 | R2 取消底部播放条，三图共用选区和边界、波形视窗振幅轴。 |
| M12-F03 | 文本与资源 | 音素自动填充、上传词典、上传词表、清除词表、词层搜索上/下一项、单个/全部替换 | 图下词典词表＋底部搜索条 | 词典与词表是两个资源；替换全部提示影响范围并按设计保留恢复机会。 |
| M12-F04 | 参考复用 | 选择/清除参考 TextGrid、区间之外/之内/起点之前/之后、起止时间、复用 | 右侧参考复用组 | 四个模式都保留，模式变化时说明对应操作区间。 |
| M12-F05 | 强度贴合 | 强度范围起止、边界内收或外扩毫秒数、强度贴合 | 图窗下方强度工具 | 负数外扩与正数内收语义保留，不将字段限制成仅非负。 |
| M12-F06 | 唇形对齐 | 唇开/唇宽独立可见性、共用时间偏移、左右微调、保存唇偏 | 右侧唇形组 | 只有一个共享 lip_manual_offset，写入原记录 metadata；保存唇偏与保存 TextGrid 独立。 |
| M12-F07 | 标注保存 | 保存 TextGrid、文件名后缀 | 图窗顶部保存＋右侧保存设置 | 后缀留空代表覆盖原文件，应明确显示目标与覆盖行为。 |

## 状态、重用与双端差异
未保存有标记；缺参考文件、找不到层、词典格式错误分别提示；页切换保持编辑与参考资源。

纠正上一轮生成图：不应出现唇开与唇宽两个独立偏移输入。图片中任何重复按钮均不进入实现规格。

迁移重点：提取既有编辑器状态/算法保留，不继承任意绝对路径接口；词/音素层独立，lip_manual_offset 只有一个，TextGrid 与唇偏分别保存。

平台边界：桌面原目录通过受控资源映射；Web 下载修改后的安全格式与 TextGrid；服务端旧 pickle 输入默认拒绝，转换须另审安全实现。

## 实施步骤与每步证据
1. 只读列出上面每个功能组的旧控件、调用函数、默认值、输入输出格式、异常、现有测试。创建 `docs/modules/evidence/M12-source-map.md`，对新增/修复功能单独标识；不能把按钮文案当成功能已可用的证明。
2. 在 P03 固定环境与样例下捕获本模块 v2 行为；核对 actual backend、单位、输出时间轴与缺失值。存入本地被忽略的输出目录，并把可公开 fixture 与配置/hash 写入测试清单。随机算法固定种子。新增功能没有旧基准时用明确规则验收。
3. 把已有可用函数迁入核心；先保留数值步骤，只拆 UI/文件/设备/全局可变状态。目标函数显式收配置与输入，任务快照与结果记录方法版本。适配变动与算法优化分开提交。
4. 新建本模块契约并接应用服务；输入通过资源 ID/已授权本地文件接口，输出进入 manifest。进度、取消、失败和部分结果统一；Web 所有临时/最终写入经过配额 writer。只显示已有结果的页面不强造长任务。
5. 使用共享组件实现页面、主题、播放/选区、抽屉和页内子视图。把上述功能行一项项映射到组件与操作，保存/生成/试听等语义不能合并丢失。Web/desktop port 区分文件和设备能力，配置状态不跨账号/模块污染。
6. 运行下面的专项场景与数值对照；先修真实差异，再对浅/深色、窗口缩放、键盘/IPA 字体、错误空态截图审阅。不能只运行单位测试就把 UI 和设备标已通过。
7. 连接模块“方法与来源”入口，使用 SRC-PRAAT, PENDING-DICTIONARY, ORIGIN-WEBEDITOR。更新说明书本模块操作、输入输出、参数单位、科学能力边界和来源；当前 unresolved 的条目不能伪装已核验。移植文件保留原版权/许可证。
8. 更新 `docs/plans/task-ledger.json` 与原功能矩阵的证据列，记录实际命令/报告/截图；只有每行有可定位证据、双端必测通过才标完成。评审通过前保留旧功能来源，不删除旧入口或数据。

## 专项验收
四种参考复用、强度内收/外扩负数、词典/词表、全部替换恢复、空后缀覆盖提示、切页保持编辑、并发保存冲突、偏移往返。

每个分组至少一个正常路径和一个相关错误/边界路径；录制、原生时序和数值算法必须在真实目标环境验证。

当前实际专项命令（环境与完整176项回归见验收报告）：

```powershell
# 先激活 P02 建立并核验的 v3 环境，不改 phonetic_311。
python -m pytest -c tests/pytest.ini desktop/tests/test_m12_annotation.py -q
node tests/e2e/m12.cjs
```

本轮已按 ADR-050 更新实际路径；同名网页文件保存为项目内新版本，本机源/目标双版本检查。共享测试另见 P03/P11。两条命令不能代替本模块原生/设备手工验收。

## 完成条件
- 本页全部 7 组功能以及原矩阵相关参数/设置有映射，没有把折叠项当作删除项。
- 数值/文件/时间轴差异均解释并审阅；已有功能不得静默改语义。
- 浅深色、未保存保护、错误恢复与真实操作可用；Web owner/配额检查适用的路径已覆盖。
- 软件/说明书的来源一致；尚缺权限/设备证据时状态仍为待处理，不能以隐藏控件绕过。
