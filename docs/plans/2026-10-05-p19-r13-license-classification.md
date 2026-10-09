# 来源许可分类核查 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 核查当前 391 条来源，区分纯引用、现有许可、已获邮件授权及仍需查明权利的材料，并隐藏纯引用的界面许可说明。

**Architecture:** 保留来源登记与论文引用。以逐项核查表记录证据、实际使用方式、许可义务与是否需要联系权利人，界面显示策略单独登记，不把未查明的信息改写为已获授权。

**Tech Stack:** JSON 登记、Python 元数据与文件核查、Vue 公共致谢组件、官方许可资料。

---

当前会话已授权实施与核查，沿用当前工作区，保留同期改动，不提交或推送、不对外联络、不改变项目总许可证、不新建环境或删除材料。

追加信息：井井在执行中逐项确认五项历史项目属于自己的工作，部分有 AI 协助，并提供自有 IPA-lab 仓库。VoQS 引用改为王天恒的 Zenodo DOI；官方记录已核对为 CC BY 4.0，因此关闭译表额外邮件确认建议，56 个译名与稳定 ID 保留。

完成状态：四项任务已在本轮授权范围内完成。来源分类与界面 verified，完整原生发行包许可、对应源码及 SoE 具体复用许可为核查产出的明确未决对象，见[报告](../references/p19-license-classification-audit.md)。

### Task 1: 保存核查基线并核对直接使用

- 读取 `third_party/source-registry.json`、已有来源证据和许可原文。
- 核对本机环境中实际包版本、许可证文件和前端锁定元数据，区分历史、开发工具和运行依赖。
- 对代码移植、数据、字体、模型、词典、图表和颜色分别判断，不按学术分组一刀切。

### Task 2: 核查未决条目

- 查阅 VoiceSauce、Praat/Parselmouth、PyQt、模型、IPA/extIPA 等官方资料。
- 有开放许可的按条款列出义务，缺少打包或版本证据的保留具体未决项。
- 只有实际使用第三方受保护材料且缺少覆盖该用途的许可时，列为需邮件确认或替代；来源不明先查作者。

### Task 3: 更正致谢显示

- 在 `third_party/source-registry.json` 为纯引用登记显示策略。
- 更新 `frontend/scripts/generate-ui-data.mjs`、`frontend/src/components/MethodReferences.vue` 和生成数据。
- 保留 `SRC-ZAIWA` 邮件许可；不删除论文、作者、外链或引用功能。

### Task 4: 交付与验证

- 生成 `docs/references/p19-license-classification-audit.md` 和机器可读逐项清单。
- 校验 391 条覆盖、唯一 ID、邮件授权保留、许可说明只在指定条目隐藏。
- 执行生成一致性、类型、前端测试及构建，定向核查致谢渲染。
- 报告已核实与仍待确认部分。本轮不重新打包 EXE，后续以井井要求为准。
