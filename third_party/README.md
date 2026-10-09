# 第三方组件、方法与素材

[source-registry.json](source-registry.json) 是来源登记的唯一维护表，当前 397 条工程记录，应用生成清单显示 377 条外部来源。工程记录和界面致谢使用不同筛选规则，不复制整张过时状态表到本页，也不按日期追加计数。

## 维护入口

| 内容 | 位置 |
| --- | --- |
| 来源、作者、版本、使用模块和具体许可状态 | [机器登记表](source-registry.json)、[引用 BibTeX](references.bib) |
| 实际软件、仅引用、自有工作与未使用项的分类 | [分类审计](../docs/references/p19-license-classification-audit.md)、[逐项证据](evidence/p19-license-classification/audit.json) |
| 作者署名查验 | [署名证据](evidence/p19-acknowledgements/author-audit.json) |
| 配色来源与逐项待确认项 | [主题审计](../docs/references/p19-theme-license-audit.md)、[原文](licenses/p19-palettes/) |
| 原始来源核查与观测 | [核查报告](../docs/references/source-audit.md)、[上游观测](upstream-observations.json) |
| 已安装依赖和阶段环境身份 | [包元数据](package-source-audit.json)、[P01](p01-dependency-inventory.json)、[P02](p02-dependency-inventory.json)、[P05](p05-dependency-inventory.json) |
| PostgreSQL、表格 I/O 和迁移证据 | [PostgreSQL](p05-postgres-runtime.json)、[M01 I/O](m01-io-inventory.json)、[M04](m04-migration.json) |
| EGG 方法/迁移边界 | [方法审阅](../docs/references/m03-method-audit.md)、[迁移复核](../docs/testing/m03-provenance-report.md) |
| LPC 与声道资源 | [LPC 方法](../docs/references/m04-method-audit.md)、[声道报告](../docs/testing/2026-10-07-m10-r14-report.md)、[声学资源](../resources/manifests/acoustic.json) |
| 实际发行物对应源码与未闭合事项 | [源码准备报告](../docs/testing/2026-10-05-preview-source-preparation.md)、[当前任务](../docs/project-status.md) |
| 模块依赖声明和锁 | [requirements/](../requirements/README.md) |

## 登记与显示原则

代码、论文、方法、模型、实验数据、字体、图标分别登记。论文引用或在线 PDF 链接不自动赋予全文再分发许可。上游当前版本、历史引入版本和实际运行字节分别记录，不能用当前 HEAD 代替原始复制 commit。

仅引用、独立方法实现和经作者确认的自有来源无需在界面显示第三方许可缺口；实际使用的代码、库、字体、图表、模型和数据逐对象核对。未使用依赖和自有实现可保留工程来源链而不进入外部致谢。井井已确认的 EGG、TextGrid、词典、IPA 规则和音系归纳来源按登记处理，旧 PENDING 名称不代表重新要求同一确认。

载瓦语邮件授权上下文保留原文。VoQS 译表按指定 Zenodo 版本的 CC BY 4.0 证据登记。科学方法引用与具体代码复用许可分开；公开方法和所有论文不统一列作待授权。

## 原文与发行边界

许可原文按来源 ID 放在 licenses/。实际 wheel、原生 DLL、打包资源和对应源码按最终发行物核对，PyPI 顶层 license 字段不能覆盖全部传递组件。VTL GPL、Three.js MIT、头壳 CC0、平均鼻腔 CC BY 4.0 分别维护，参见 [VTL 原文](../resources/vocal_tract/VTL-LICENSE.txt)及[综合声明](../resources/vocal_tract/THIRD_PARTY_NOTICES.md)。系统字体只按名称调用时，不将本机字体文件直接打包。

项目总 LICENSE、SoE 具体复用、部分原生构建/补丁和最终同渠道对应源码交付仍有具体待办。主题来源也有未确定项。按实际包筛选这些义务，MFA 环境/模型已从紧凑 EXE 排除，不能据此宣布所有独立分发对象的许可都已闭合。

本页只维护导航与原则。来源原文、版本锁定元数据、邮件证据和历史验收各留在其独立位置，来源材料中的上游操作说明不覆盖项目工作规则。
