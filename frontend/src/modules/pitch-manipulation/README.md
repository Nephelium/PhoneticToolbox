# 变速变调 · M08

本目录提供单文件 F0 编辑、变速变调、组合生成与历史结果管理。名称与持久模块 ID 见[模块注册表](../../app/registry.ts)，完整操作见[可编辑说明书章节](../../../../manual/chapters/m08.json)，摘要见[模块说明](../../../../docs/manual/pitch-manipulation.md)。本文按 2026-10-05 的源码与已记录验证范围整理。

## 功能概览

- 原音与 F0 共用原始时间，支持手绘有声轨迹、恢复及基频序列导入。
- 分别合成当前视野或整段。拐点批量基频使用原始 F0，不使用手绘/导入实线及单文件语速倍率；文件夹变速变调处理整段。
- 每次成功合成都进入历史 F0 对照，可试听、重命名、明确删除和批量保存所选结果。
- 数值输入保留小数点/负号中间态，提交时校验；保存后显示实际目录和文件名。

## 输入与输出

| 类型 | 内容 |
| --- | --- |
| 输入 | WAV、MP3、FLAC；基频序列为每行一个值或时间/F0 两列 |
| 控制 | 语速、F0 曲线、拐点组合、音高倍率与 Hz 偏移 |
| 输出 | 实际合成 WAV、历史 F0 对比 PNG |
| 草稿 | 输入 hash、编辑轨迹与参数，音频需另存 |

## 快速流程

1. 选择音频，核对原波形与提取的 F0。
2. Shift+拖动修改有声帧，Ctrl+拖动恢复；或缩放到唯一连续有声段后导入 F0。
3. 设置语速后选择合成当前视野/整段，或在独立批量入口配置拐点组合。
4. 点击原音、合成音或历史版本直接试听。鼠标选区播放与直接播放保留独立用途。
5. 勾选需要的历史结果，保存合成音或批量保存，核对右栏实际目录与最终名称。

## 格式与限制

- 输入限 64 MB、800 万帧、8 声道，预计输出最多 3200 万帧；控制点 64、组合 256。
- 默认显示轴 50–350 Hz，仅影响图面。原音/历史使用 Sound.to_pitch() 默认提取，Manipulation 为 0.01 秒和 75–600 Hz。
- 历史 F0 来自实际输出 WAV 重新提取，按零起点对齐，不将目标编辑线当作输出测量值。
- 语速 0.8 为减速、1.2 为加速；批量流程先变速、再乘音高倍率、最后加 Hz。
- 桌面同名不同内容结果保留并追加数字。浏览器多结果 ZIP 另限 256 项/64 MB。
- 外部导出副本的管理关联限当前宿主会话，重启后不扫描外部目录。实际科研数值、幅值和时长可能因处理改变。

## 源码结构

| 文件 | 职责 |
| --- | --- |
| [PitchManipulationPage.vue](PitchManipulationPage.vue) | 单文件、批量、保存与历史 |
| [PitchCurve.vue](PitchCurve.vue)、[HistoryPlot.vue](HistoryPlot.vue) | 编辑 F0 与输出 F0 图 |
| [state.ts](state.ts)、[port.ts](port.ts) | 数值校验、组合、重命名及平台契约 |
| [科学核心](../../../../packages/phonetic_core/src/phonetic_core/manipulation/) | M08 Praat overlap-add 及批量规则 |

## 开发与定向验证

从仓库根目录运行[源码启动器](../../../../scripts/Start-M08-Workbench.ps1)。公共前端检查使用 `npm --prefix frontend run typecheck`、`npm --prefix frontend test` 和 `npm --prefix frontend run build`。

定向入口：[R2 界面检查](../../../../tests/e2e/m08-r2.cjs)、[Qt 检查](../../../../scripts/verify_m08_r2_qt.py)。[R2 报告](../../../../docs/testing/2026-10-05-m08-r2-report.md)与[R1 报告](../../../../docs/testing/2026-10-04-m08-r1-report.md)记录 Windows 源码、合成输入、Chrome/实际 Qt 和实际 WAV 回读。实体音频、Linux 正式任务/GUI、真实远程 PostgreSQL 与近期 EXE 未验证。

## 方法与来源

科学方法为 SRC-PRAAT，保留计算规则并明确 Hz 偏移修正。参见[Praat overlap-add 官方说明](https://www.fon.hum.uva.nl/praat/manual/overlap-add.html)与[Parselmouth API](https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Sound.to_pitch)。参见[来源映射](../../../../docs/modules/evidence/M08-source-map.md)、[保存 ADR](../../../../docs/decisions/ADR-M08-R1.md)和[统一来源登记](../../../../third_party/source-registry.json)。

## 草稿与参数保存

当前页面支持粘贴基频序列、保存编辑草稿、查看历史生成参数，未提供独立参数文件导入/导出或 F0 数值表导出按钮。草稿与 WAV、PNG 分别保存。M08 紧凑播放条隐藏音量与进度滑条；直接播放和波形选区播放保持独立。

当前视野批次仅包括同一来源且源区间与原音视野按两位小数匹配的结果。文件夹处理后选择各原音，恢复对应全长视野，再分别核对和保存。重命名后结果 ID 关联保留，删除与改名会影响当前会话已登记的外部副本。

说明书结构按统一校验入口检查，科学任务、冻结成品、跨平台 GUI 与长期稳定性按各自证据判断。
