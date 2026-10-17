# 工程文档导航

从当前任务需要的入口读起，不把历史计划和报告全部载入上下文。

| 要找的内容 | 维护位置 |
| --- | --- |
| 对外项目介绍与配图 | [项目主页](../README.md)、[配图来源](assets/readme/README.md)、[主页检查报告](testing/github-readme.md) |
| 工作边界与清理规则 | [根 AGENTS](../AGENTS.md)，子目录仅补充当地差异 |
| 当前结构与依赖 | [总架构](../ARCHITECTURE.md)，子目录架构定位代码 |
| 启动、修改、检查 | [开发说明](development.md)、[源码入口](development/source-entry.md) |
| 用户要求 | [需求](requirements.md) |
| 论文语谱图自动重建与播放 | [分阶段实施规划](plans/2026-10-16-paper-spectrogram-audio.md)、[设计取舍提议](decisions/ADR-paper-spectrogram-audio.md)；当前为规划，尚未实施 |
| 网页版上线 | [首期服务器计算计划与逐模块安排](plans/2026-10-16-web-launch.md)、[临时工作区与计算部署决策](decisions/ADR-web-ephemeral-client-first.md) |
| 模块入口及迁移基线 | [模块导航](modules/module-migration.md) |
| 当前任务状态与证据 | [当前实现与后续任务](project-status.md)；[任务台账](plans/task-ledger.json)按任务 ID 检索 |
| 每次修改与版本更新内容 | [更新记录](../CHANGELOG.md)累积未发布修改；正式版本说明位于 [release/notes](../release/notes/) |
| 界面、数据与远程协议 | [界面规范](design/UI_SPEC.md)、[账号/存储](specs/accounts-storage-jobs.md)、[remote/1](specs/remote-protocol-v1.md) |
| 验证要求与结果 | [验证策略](testing/verification-plan.md)；具体报告位于 testing/ |
| 发行及公开仓库边界 | [严格打包](../release/PACKAGING_RULES.md)、[仓库内容管理](development/repository-hygiene.md)、[产物生命周期](development/artifact-lifecycle.md)、[测试临时清理](testing/test-temporaries-cleanup.md) |
| 设计取舍及来源 | [ADR 索引](decisions/ADR.md)、[来源登记](../third_party/README.md) |

规则说明应该怎样做，架构说明代码怎样分工，台账记录目前到哪一步，报告记录当时实际验证了什么。每类信息只在自己的位置维护，其他文档使用链接。

`plans/` 的日期计划、`testing/` 的专项验收（含 p17 的 rules 清单）和 `baseline/` 是按需参考的证据。旧阶段的下一步、暂停、授权或未实现声明不能当作当前指令；改动时先查当前源码及适用报告。受保护的基线/许可不因年代久而删除。

临时日志、私有审计、截图工作副本和可重新生成的测试输入进入忽略的 output/。正式说明书正文及素材保留，自动恢复稿只保留最后一份，保存历史按[作者工具规则](../tools/manual-studio/AGENTS.md)清理。每次修改或清理同步更新台账对应条目。已有文档优先原位更新，新增 ADR 或验收报告必须确有新的设计或验证事实，避免同一交付在多个入口反复复述。
