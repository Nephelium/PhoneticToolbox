# P19-R11：配色来源与许可登记报告

日期：2026-10-05。

状态：**verified，限定 Windows 当前源码的逐项登记、许可证据保存、生成数据及静态检查。**主题许可未全部闭合，不标为整体公开发行许可已完成。

## 完成内容

井井授权补充现有各主题的上游来源与许可。已对照当前 29 个 palette ID，新增 29 条统一来源记录，并由既有生成器同步到关于页的软件与代码来源。登记包含作者、公开上游、固定查验 commit、许可链接/本地原文哈希、使用位置、独立适配关系与具体未决项。

| 查验结果 | 数量 | 范围 |
| --- | ---: | --- |
| 公开上游完整 MIT 许可原文已核对 | 11 | Ayu、Catppuccin、Dracula、Everforest、GitHub、Night Owl、Nord、One、Rose Pine、Solarized、Tokyo Night |
| 上游许可声明已核对，完整通知待补 | 1 | Gruvbox：README 声明 MIT/X11，无独立完整许可文件 |
| 公开实现 MIT 已核对，原设计许可链待确认 | 1 | Monokai：Microsoft VS Code 公开实现与经典深色三项 seed 对应，原设计/历史来源分列 |
| 来源或许可待确认 | 16 | Absolutely、Codex、Linear、Lobster、Material、Matrix、Notion、OG、Oscurange、Proof、Raycast、Sentry、Temple、Vercel、VS Code Plus、Xcode |

完整 MIT 原文共 13 份：11 套主题中 One 的深浅上游各一份，加 Monokai 的 Microsoft 公开实现一份。Gruvbox 的 README 声明片段另存于证据目录，未伪造独立许可证或版权年份。

统一登记共 391 条，生成的有效应用致谢为 387 条，另 4 条沿用既有 retired 状态。最初引入版本/commit 仍 unknown；本次公开查验版本不能倒推为历史来源。现有 Codex 历史视觉参考条目保留并指向逐项登记，其他 361 条既有来源结构值与本轮前快照一致。

主要交付：

- [逐项审计表](../references/p19-theme-license-audit.md)
- [机器可读证据](../../third_party/evidence/p19-palettes/audit.json)
- [统一来源登记](../../third_party/source-registry.json)
- [上游许可原文](../../third_party/licenses/p19-palettes/)
- [设置说明](../manual/settings.md)与[既有外观来源记录](../references/p19-appearance-sources.md)

设置的关于配色说明增加来源查询位置和独立适配说明。未更改配色 ID、名称、色值、默认值或偏好。配色实现和生成器的 SHA-256 与本轮前快照一致。未新增运行依赖、导入上游主题代码或图标。

## 实际验证

| 检查 | 结果 |
| --- | --- |
| 逐项完整性与哈希核对，证据 `output/validation/p19-r11/integrity.json` | 29/29 一一覆盖；作者/版本/日期/使用位置/许可字段完整；固定上游 commit、13 份原文及声明哈希通过 |
| 既有来源及实现保留 | 361 条非目标原记录结构值不变；`themes.ts` 和生成器字节哈希不变 |
| `npm --prefix frontend run ui-data:check` | 80 参数、387 有效来源，生成一致 |
| `npm --prefix frontend run typecheck` | 通过 |
| `npm --prefix frontend run test` | 286 通过，0 失败/跳过 |
| `npm --prefix frontend run build` | 通过；保留既有大 chunk 提示，未调整阈值或依赖 |
| 当前目标文件 `git diff --check` | 通过 |
| 本轮 6 份文档的相对链接核对 | 52 条通过，0 缺失；证据 `output/validation/p19-r11/docs-links.json` |
| `python scripts/validate_docs.py` | 检查 1460 文件、391 来源；全库仍有 15 条既有 EXE 缺失链接，退出 1；没有本轮新增缺链 |

已读取 GitHub 公开仓库 API 与固定 commit 的静态色板/许可文本，未执行下载内容。部分颜色在当前上游通过 HEX 或 HSL 转换核对，仅说明配色家族对应；未宣称逐像素一致，也未把同名仓库许可套用到未决项。

## 未决项与边界

Material 原上游当前重定向至 Vira 后继项目，当前公开仓库无 tags、仅 live 分支，未取得可对应版本的官方许可原文。第三方 fork 中的历史 Apache-2.0 元数据不作为本项授权。Monokai Classic 与 Monokai Pro 分开，官方 Pro 社区移植条件未被套用为当前方案许可。

16 套待确认项目逐项记录具体缺口。品牌官网、品牌指南、扩展仓库或字体的许可均不外推为当前配色的再分发许可。来源登记和独立适配说明不替代必要授权，未决项继续保留为对应公开发行的核查条件。

原始 Codex 来源记录曾包含安装包元数据只读观察。本轮仅访问公开网页与仓库，没有继续读取或解包 Codex 安装包，也未认定那次历史获取方式符合当时适用条款。

本轮未重新打包或运行 EXE，未做实际 Chrome/Qt 窗口展示验收；生成数据与前端检查通过不能扩大为冻结成品、跨平台或完整法律授权验证。现有打包脚本会收集许可目录，此路径只做源码确认，未声称新成品已含这些文件。

无 push、公开发布、对外联络、现存数据库 DDL、环境安装、系统配置或用户文件清理。保留同期代码与来源改动。

## 文档核查记录

全库检查仍失败于 15 条既有 EXE 缺失链接，涉及旧 README 与 P17/P18/P20/P19 历史成品报告，未新增本轮错误，也未删除引用、降低检查标准或填入占位成品。旧源码快照缺链仍单独列示，不改写其内容。

本轮计划、主题审计表、报告、来源补充、设置手册与第三方 README 共 6 份文档的 52 条相对链接已核对通过。全库失败状态保留，不能写成全部文档检查通过。
