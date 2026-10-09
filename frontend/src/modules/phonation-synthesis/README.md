# 发声类型合成 · M07

本目录提供源/目标音频的 LPC 残差连续统生成。名称与持久模块 ID 见[模块注册表](../../app/registry.ts)，完整操作内容位于[可编辑使用说明章节](../../../../manual/chapters/m07.json)，章节结构与写作约定见[说明书工程](../../../../manual/AUTHORING.md)。本文按 2026-10-05 的当前源码和模块验收报告整理。

## 功能概览

- 源音频、目标音频、F0 和合成音频四图对照。
- 显式提取 F0，编辑控制点并明确应用，生成三类连续统及双方向组。
- 单步/整组点击即试听，任务历史绑定生成快照；整组播放按实际 PCM 位置突出当前步骤。
- F0 显示纵轴独立保存，可导出当前范围与所选曲线的完整图例 PNG。

## 输入与输出

| 类型 | 内容 |
| --- | --- |
| 输入 | 源与目标 WAV、分析参数、F0 控制点与连续统设置 |
| 输出 | 11025 Hz 的各步 PCM16 单声道 WAV、`combined_steps.wav`、完整 1 ms 源/目标轨迹 `edited_f0.csv`、`m07.ptb.json` |
| 图片 | 当前 F0 图窗的 300 dpi PNG |
| 草稿 | 分析/生成参数及未应用控制点，不保存长期文件授权 |

## 快速流程

1. 打开音频目录，选择源与目标 WAV，等待真实波形出现，分别试听并核对角色。
2. 选择 Parselmouth / Praat 或 REAPER 及 F0 范围，点击提取 F0。文件选择不会自动提交分析任务。
3. 先选归一化有声时长或起点对齐及控制点数量，再编辑数值。修改后点击应用编辑。
4. 选择类型、方向、步数和幅度选项，生成当前组或全部六组。默认九步包含插值两端。
5. 在历史中选择成功生成任务，点击整组试听或 stepXX，观察对应生成控制 F0 与实时步骤高亮。
6. 保存完整组或下载 ZIP；需要图像时另行调整显示轴并导出 PNG。

## 格式与限制

- 每个输入同时限 10 秒、480000 帧、8000000 字节；目标采样率固定 11025 Hz，控制点 20–200、每组 2–50 步。
- 源/目标试听选区不裁剪分析输入。帧长和帧移以 11025 Hz 下的采样点计，F0 后端帧间隔以毫秒计。
- 合成 F0 是任务快照的生成控制轨迹，未从输出 WAV 重测；发声类型单独变化时曲线可重合。
- F0 显示范围不修改分析与生成设置。网页暂未接入合成曲线读取。
- 周期能量匹配实际为残差周期绝对峰值匹配，响度匹配实际为全段平均绝对振幅匹配。
- 控制表空白按 0 处理，正值间插值可填补内部空白。应用编辑只更新轨迹，复用原 LPC、残差和脉冲；更换输入或分析参数须重做分析。
- 六组按正向 F0/发声/同时、反向 F0/发声/同时依次独立发布。取消保留已成功组，重试沿用对应任务快照。
- CSV 保存完整 1 ms 轨迹，行数不等于控制点数。当前没有 CSV 或任意参数 JSON 导入入口。
- 单步按钮用于试听。完整组保存包含全部 WAV、CSV 与 JSON，PNG 使用独立导出入口。当前没有单个 WAV 另存按钮或统一四件套按钮。
- 历史组的试听、图面和导出读取自身快照。左栏当前输入不会自动恢复为历史组输入，须通过清单核对身份。
- 参数草稿重开后需重新选择输入并分析，仅文件哈希、分析参数、点数与对齐都匹配时才恢复控制点，恢复后仍需应用。
- 连续统未证明知觉等距、发声类别知觉效度或临床适用性。

## 源码结构

| 文件 | 职责 |
| --- | --- |
| [PhonationSynthesisPage.vue](PhonationSynthesisPage.vue)、[state.ts](state.ts) | 输入、控制点、任务与组选择 |
| [F0Comparison.vue](F0Comparison.vue)、[f0.ts](f0.ts) | 控制 F0、播放高亮与图片 |
| [history.ts](history.ts) | 历史任务描述与时间 |
| [SourceAcknowledgement.vue](SourceAcknowledgement.vue) | 常驻来源与许可说明 |
| [平台接口](../../platform/m07.ts) | 任务、受控读取、文件哈希核查与完整组下载 |
| [科学入口](../../../../packages/phonetic_core/src/phonetic_core/manipulation/m07_api.py) | M07 分析、对齐及生成编排 |
| [显示核心](../../../../packages/phonetic_core/src/phonetic_core/manipulation/m07_display.py) | 从生成快照恢复控制 F0，不执行新音高估计 |
| [任务接口模型](../../../../backend/src/ptb_api/m07_models.py) | 默认值、单位、有限范围及交叉参数约束 |
| [桌面桥接](../../../../desktop/src/ptb_desktop/m07_bridge.py) | 授权文件、组保存和只读 F0 显示 |

## 开发与定向验证

从仓库根目录运行[源码启动器](../../../../scripts/Start-M07-Workbench.ps1)。公共前端检查使用 `npm --prefix frontend run typecheck`、`npm --prefix frontend test` 和 `npm --prefix frontend run build`。

定向入口：[界面检查](../../../../tests/e2e/m07-r3.cjs)、[Qt 检查](../../../../scripts/verify_m07_r3_qt.py)。[R3 报告](../../../../docs/testing/2026-10-05-m07-r3-report.md)记录 Windows 源码、合成输入、实际 Chrome/Qt 和 PNG 回读。[结果组与 F0 归属 ADR](../../../../docs/decisions/ADR-M07-R1.md)及[分析快照与逐组发布 ADR](../../../../docs/decisions/ADR-M07-001.md)说明状态边界。实体听辨、自然嘎裂声准确率、Linux GUI、远程宿主和近期 EXE 的完整 M07 任务未验证。

章节编辑后，从仓库根目录运行 `python scripts/manual/validate.py --project manual --strict`。此检查核对说明书结构、引用与已登记媒体，不替代科学计算或图音制作。真实示例图音由全书制作流程集中维护，章节作者不复制媒体，也不把私有录音路径写入公开正文。

## 方法与来源

原连续统方法及 MATLAB 到 Python 改写许可见[来源映射](../../../../docs/modules/evidence/M07-source-map.md)、[许可摘要](../../../../third_party/evidence/SRC-ZAIWA/permission-summary.md)和[统一来源登记](../../../../third_party/source-registry.json)。登记 ID 为 `REF-ZAIWA`、`SRC-ZAIWA`、`SRC-PRAAT` 与 `SRC-REAPER`。

[Lu、Liang、Kong (2025) 论文](https://doi.org/10.1016/j.wocn.2025.101413)与[作者官方仓库](https://github.com/Luyao2025/Contribution-of-F0-and-phonation-to-tone-perception-in-the-Zaiwa-language)分别提供学术来源与原始实现入口。[Parselmouth 官方文档](https://parselmouth.readthedocs.io/en/stable/)及[REAPER 官方仓库](https://github.com/google/REAPER)列出 F0 组件。REAPER 是本工具新增的选项，适配效果不作为原论文方法效果的结论。代码改写、论文、录音与统计数据的许可分别处理。
