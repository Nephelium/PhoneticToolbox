# 架构决策索引

本页只导航，新增决策使用单独专题文件。现行系统结构见[总架构](../../ARCHITECTURE.md)，任务状态见[台账](../plans/task-ledger.json)。旧决定可能被后续专题替代，按具体对象和证据判断，不将记录日期当作执行授权。

## 专题决策

- [相同依赖只保存一份，原路径恢复](ADR-cross-archive-dedup.md)
- [ADR-M01-M02-R1：列表、切分与长录音预览修复](ADR-M01-M02-R1.md)
- [ADR-M01-R4：30分钟有界分析与 EGG 联合参数](ADR-M01-R4.md)
- [ADR-M02-R2 参数图的选区与像素标注](ADR-M02-R2.md)
- [ADR-M03-R2：EGG 临时交互会话](ADR-M03-R2.md)
- [ADR-M03-R5：音频 F0 搜索范围与 REAPER](ADR-M03-R5.md)
- [M03-R7：有界长文件与逆滤波选区](ADR-M03-R7.md)
- [ADR-M03-RT：自动刷新与保持科学输出的传输优化](ADR-M03-realtime.md)
- [ADR-M04-R1 空白 TextGrid 尾段与直接拖选](ADR-M04-R1.md)
- [M05 方法身份与时间/缺失语义](ADR-M05-methods.md)
- [ADR-M05-R2：实时保存、本地输入与方法身份](ADR-M05-R2-local-recording.md)
- [ADR-M05-R3 录制选项与对齐工作流](ADR-M05-R3-recording-alignment.md)
- [ADR-M06-R1：编辑时域与旧合成音频的时间一致性](ADR-M06-R1.md)
- [ADR-M06-R2：修正声源标尺并保持 F0 编辑](ADR-M06-R2.md)
- [ADR-M06-R3：复用 F0 后端与共享播放栏](ADR-M06-R3.md)
- [ADR-M06-R4：保留原录音信息的可选重合成](ADR-M06-R4.md)
- [ADR-M07-001 分析快照与六组独立发布](ADR-M07-001.md)
- [ADR-M07-R1 结果组、直接试听与 F0 显示归属](ADR-M07-R1.md)
- [ADR-M08-R1：合成历史与外部保存分离](ADR-M08-R1.md)
- [ADR-M09-R1：基于原始相位的音频频谱绘制](ADR-M09-R1.md)
- [ADR-M10-R11：生理参数扩展与共享几何](ADR-M10-R11.md)
- [M10-R12 舌位自由度与刚性牙齿接触](ADR-M10-R12.md)
- [M10-R13 圆弧与组织连接](ADR-M10-R13.md)
- [ADR-M10-R14 舌体厚度、表面控制与 257 截面](ADR-M10-R14.md)
- [ADR-M10-R6 音标索引的本机构形库](ADR-M10-R6.md)
- [ADR-M11-001：可选运行时、旧运行适配故障与准入](ADR-M11-001.md)
- [ADR-M11-R1：显式转写来源与 CPU 运行开销](ADR-M11-R1.md)
- [ADR-M13-R1：右侧控制区和锚定选音](ADR-M13-R1.md)
- [ADR-M13-R2：显式草稿恢复与独立文字显示](ADR-M13-R2.md)
- [ADR-M14-R1：可配置导入、页内归并与结果快照](ADR-M14-R1.md)
- [ADR-M15-001 纯客户端感知实验](ADR-M15-client.md)
- [ADR-M16：可恢复本地录音工程](ADR-M16-local-recording.md)
- [ADR-M17：纯客户端符号表、Unicode 编辑与固定字体](ADR-M17-client.md)
- [ADR-M17-R3：静态内容发布与开发专用维护](ADR-M17-R3.md)
- [M18 独立论文分发与阅读](ADR-M18-paper-distribution.md)
- [ADR-P06-REMOTE-001：节点优先与有条件接管](ADR-P06-REMOTE-001.md)
- [ADR-053 补充：按任务快照确定发布期限](ADR-P07-publication-policy.md)
- [ADR-P10-R1：说明书目录跟随当前阅读小节](ADR-P10-R1.md)
- [ADR-P16：科学行为修正与平台边界](ADR-P16-review-repairs.md)
- [ADR-P17 统一显示与单页布局](ADR-P17-display.md)
- [ADR-P17-M08：即时原音显示与有界预览写入](ADR-P17-M08.md)
- [ADR-P19：配色方案与显示模式独立保存](ADR-P19-appearance.md)
- [首次准备、内容校验复用与退出清理](ADR-persistent-startup-cache.md)
- [Preview 1 单文件与必要运行时](ADR-Preview1-compact-packaging.md)
- [Preview 1 Windows 包装与独立科学环境](ADR-Preview1-packaging.md)
- [开发源码统一绑定与发行快照](ADR-source-entry-and-snapshot.md)

## 早期汇总

[早期决策原文](legacy-decisions.md)保留基础分层、初始方案及迁移时的取舍，仅在需要某个旧 ADR 时按 ID 检索。其旧配额、入口、候选方案和阶段状态不能覆盖当前规范。

## ADR-013 P02 开发主线、包版本与单一契约源

保留既有引用锚点，原决定见[早期 ADR-013](legacy-decisions.md#adr-013-p02-开发主线包版本与单一契约源)。现行环境与命令见[源码入口](../development/source-entry.md)。
