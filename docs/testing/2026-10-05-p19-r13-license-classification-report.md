# P19-R13 来源许可分类核查与致谢显示

2026-10-05。范围：现有 391 条来源登记、既有本机许可文件、重点官方证据与公共致谢显示。井井确认 EGG、TextGrid 编辑器、内置词典、11 套 IPA 规则与字表、音系归纳五项属于自身历史工作，其中部分曾有 AI 协助。未重新打包 EXE、对外联络、推送、安装环境、修改科学公式或替项目选择总许可证。

## 交付

- [分类核查与全部 391 条索引](../references/p19-license-classification-audit.md)。
- [机器清单](../../third_party/evidence/p19-license-classification/audit.json) 与 [99 个 Python 包名/版本组合及 npm 清单](../../third_party/evidence/p19-license-classification/local-dependency-licenses.json)。读取 491 条安装元数据与 522 个许可文件记录；170 条 npm lock 包含历史环境及可选平台条目，不等于均进入当前 EXE。
- 43 条纯引用隐藏许可说明，作者、链接、复制引用保留。两个自有或生成的可见来源也隐藏第三方许可提示。
- 当前未使用的 6 条旧依赖/候选宿主与用户确认的 5 条自有来源移出外部致谢，391 条工程登记保留，界面 372 条。原自有状态存入 `author_confirmation_2026_10_05.historical_record`，稳定 source ID 和已有实际版本保留。
- `SRC-ZAIWA` 与 `REF-ZAIWA` 原记录保持一致，2026-09-10 邮件授权及其范围保留。
- 更正 Parselmouth GPLv3 或更高版本及部分包许可标签；取得 OpenSauce Octave BSD 两条款原文；平均鼻腔 v4 数据许可与原网格身份核对完成。
- 按井井指定将 VoQS 引用改为王天恒的 Zenodo DOI 与完整书目。官方记录明确 CC BY 4.0，且 identical-to 指向旧文章，关闭译名集合邮件确认建议。56 个译名及稳定 source ID 保持，致谢显示与复制、条目详情、说明书同步更新。

## 实际验证

| 验证 | 结果 |
| --- | --- |
| 来源、生成数据及分类清单 | 391 ID 唯一且无新增或丢失，全部 391 条索引覆盖；372 显示，45 个可见条目隐藏许可说明。原实际版本、作用模块、标题一致，除五项自有确认及用户指定 VoQS 更正外作者/发行状态/历史证据一致，两条载瓦语来源逐字段不变。 |
| 原始证据 | OpenSauce 原文逐字节一致且摘要匹配；鼻腔原始网格、Face Landmarker 模型摘要与项目已有 manifest/lock 匹配；IPA-lab 官方 API 的 2020 年创建/六次提交日期及文件树保存为摘要。 |
| 生成一致、类型检查、前端测试与构建 | 最终通过。292 条前端测试通过，无失败或跳过；包含同期模块新增测试，不能计为本轮新增许可测试。构建保留已有大 chunk 提示。 |
| Chrome 浅深模式定向界面 | 两组通过。Iseli 条目无许可行，作者/PDF/引用复制保留；研究数据的 CC BY、Parselmouth GPL 和载瓦语邮件日期保留；旧依赖与五项自有来源隐藏；王天恒完整引用、DOI、CC BY 许可与逐字复制通过。截图目视确认，浏览器无 pageerror。 |

本轮定向运行材料位于 `output/validation/p19-r13/`：`audit-validation.json`、`ui-report.json`、`paper-light.png`、`paper-dark.png`、`email-license-light.png`、`email-license-dark.png`、`voqs-license-light.png`、`voqs-license-dark.png` 与四项前端检查日志。复制验证使用测试页内剪贴板适配，不改用户系统剪贴板。测试自建本地 Vite 服务和隐藏 Chrome，结束后关闭本任务实例。

第一次界面校验误把搜索 React 命中的 `@vue/reactivity` 也当成旧 React 来源，改为按精确标题判断后浅深组均通过，保留实际 Vue 依赖。科学计算与原生 Qt 代码本轮未改；未重新运行全科学测试、Qt 或旧 EXE，也未宣称完整发行包的许可与对应源码验收已通过。

## 具体未决项

SoE 注释明确提到 `func_getSoE.m`，未找到覆盖该具体代码改写的许可，需要确认实际复用关系和适用代码许可。OpenSauce 的现成 BSD/Apache 不能扩大到仓库外文件。VoQS 中文译表已找到 CC BY 4.0，已按现有许可处理。MFA/FFmpeg/Qt/Chromium/原生 wheel 还须按实际发行物整理许可、通知与对应源码；项目根目录未找到总 LICENSE，本轮未代替作者给全项目重新许可。

已核查与未决部分在主报告逐项区分。未因论文参考条目本身要求向论文作者申请方法著作权许可，未把免费、学术或致谢视为替代已有许可条件。
