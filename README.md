# PhoneticToolbox 3.0 · 重构规划工作区

**D0.3 · 2026-09-09 · 供井井审阅。当前尚不是可发布的 v3 应用。**

沿用 U2 紧凑工作台、浅深色主题和 K2 波形团子。一个自有仓库共用前端与科学核心，分别交付网页版、Windows 单文件直用版/安装版、macOS；Linux 桌面作为独立平台验收项保留。已有可用代码优先迁移，不重复实现同一算法。

| 建议阅读顺序 | 文档 |
| --- | --- |
| 1. 审阅入口 | [规划摘要](docs/REVIEW.md) |
| 2. 全部阶段与任务 | [详细总计划](docs/plans/2026-09-09-v3-master-plan.md) |
| 3. 总体边界 | [架构文档](ARCHITECTURE.md) |
| 4. 外观与操作 | [UI 规范](docs/design/UI_SPEC.md) |
| 5. 全部功能如何迁移 | [15 模块迁移规格](docs/modules/module-migration.md) |
| 6. 登录、5 GB、7 天 | [账号与存储规格](docs/specs/accounts-storage-jobs.md) |
| 7. 来源与论文 | [查验报告](docs/references/source-audit.md)、[引用与第三方清单](third_party/README.md) |
| 8. 测试与发布 | [验证策略](docs/testing/verification-plan.md)、[平台与发行](docs/deployment/platform-release.md) |
| 9. 开发约束 | [AGENTS.md](AGENTS.md)、[架构决策](docs/decisions/ADR.md) |

## 已完成和未完成
- 建立同一仓库的 codex/v3-rebuild 工作分支与独立本地 worktree。
- 从当前 v2 工作状态继承 427 个源码/资源/文档文件，原目录文件、HEAD、暂存区保持原样。
- 本地源码基线为 ccf4ff73c355e8d955c2a6b9605b32ac2c7255a6；这是迁移依据，不表示该源码已通过完整功能验收。
- D0.1 的功能与视觉资料已在本工作区归档；其架构部分以本轮文档为准。
- 新目录中的 AGENTS.md / ARCHITECTURE.md 定义将来的实现边界；并没有伪造空壳业务实现。
- 继承的 phonetic_toolbox、run.py、run.spec、pyproject.toml 仍是 v2 过渡代码。运行它们不会得到统一前端的 v3。完成迁移前不得对外称已经实现 v3。
- 尚未安装新依赖、启动服务器、生成新 EXE、创建数据库、发布或删除旧网页目录。
- 学术/代码来源已按证据登记；没有明确许可证的材料仍须在发行前解决。

## 工作区约定
frontend / backend / desktop / packages / contracts / resources / tests 是 v3 的目标边界。现有 phonetic_toolbox 是受控的迁移来源，两者不能无限期混用。详细路径、迁移顺序和每阶段退出条件见总计划。

用户数据、原始测试语料和本机绝对路径记录在忽略的本地证据文件中；不随公开源码或安装包分发。官方论文与手册使用来源链接；本地继承的历史 papers 目录不自动进入 v3 包。

所有对外部署、代码推送、发行物发布，待明确授权再执行。
