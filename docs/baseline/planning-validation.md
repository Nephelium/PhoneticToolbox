# D0.3 规划交付核验

2026-09-09。本记录只证明本轮规划、文件和来源登记检查，不证明 v3 程序已实现。

## 已执行的检查
- 在 conda phonetic_311 中运行本轮临时 validate_planning.py（Python 3.11.14，UTF-8）；完整本地输出在被忽略的 output/validation/planning-validation.json。
- 新编 Markdown 的 UTF-8/无 BOM 和相对文件链接；历史归档与第三方上游 README 不按当前工程路径解析。
- 15 模块、83 功能组、197 行原矩阵、20 行双端补充、80 唯一参数键与 14 设置条目的覆盖和编号。
- 17 项工程任务 + 15 个模块任务；全部模块来源 ID 能定位到 58 条注册记录。
- 根目录及 9 个职责目录的 AGENTS.md / ARCHITECTURE.md 存在并关联总计划。
- 来源许可证快照 hash、重点仓库当前观测 commit 与实际使用版本分开记录。
- 427 个 v2 基线文件 hash、v2 HEAD、暂存区 hash 和 git status 与工作前相同；v3 继承业务/资源文件与原目录逐字节相同。
- 含测试音频路径与本机状态的 local-evidence.json 被 Git 排除，未进入暂存区。
- Git 索引刷新消除换行判断带来的假修改：实际内容差异仅新规划、文档入口与忽略规则，没有业务算法差异。

本轮临时校验工具不属于 v3 运行代码；正式可移植文档/来源工具在 P02/P10 建立。原 v2 源码清单见 [source-manifest.json](source-manifest.json)。

## 未执行的验证
没有生成 v2 数值黄金结果，没有运行 v3 科研回归、前后端构建、多人服务器、Windows 包或 Mac/Linux 设备测试。没有新 UI 渲染结果；本轮 UI 文件是已确认设计图及规格。所有业务任务保持 planned。

来源核验仍有开放项，尤其原始 VoiceSauce 条款、载瓦语代码/数据授权、部分字典/IPA/EGG 方法链和实际原生构建包。不能因为清单与链接通过就标为发行许可已通过。

## 交付位置
[审阅入口](../REVIEW.md) · [总计划](../plans/2026-09-09-v3-master-plan.md) · [来源审计](../references/source-audit.md)。本轮只在本地保存，不推送或发布。
