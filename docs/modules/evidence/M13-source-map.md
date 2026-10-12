# M13 V2 说明书、源码与 V3 功能映射

2026-10-05 P19-R13 来源补充：项目作者已直接确认 11 套 IPA 规则与映射为自有历史工作，并提供其另一账号的 [IPA-lab](https://github.com/Nephelium-chryseum/IPA-lab)。官方 API 的 2020-03-14 至 15 日创建/提交记录已核对。`PENDING-IPA` 保留稳定 ID，之前未决字段留档，当前自有来源见[本轮核查](../../references/p19-license-classification-audit.md)。这项作者确认不替代各标准的书目核实，也不声称当前 V3 文件与 2020 仓库逐字一致。

2026-09-26 完成逐项对照。说明书来源为相邻 V2 `Phonetic_Export/index.html` 第 10.1–10.2 节，SHA-256 `a46c984929bcd8073ff1daf4e6b382a6685d4ca19e0fee0e29de3ecf0fb39ad5`。实际页面来源为 `phonetic_toolbox/gui/resources/ipa_trans/ipa_converter.html`，SHA-256 `11a5c8cc7315d04bc2beece64519247ab86f80e7e9fb37510704b8d1e4ec6451`；生成脚本 SHA-256 `676a03094f003a27575961c383ff59998bff284d02e01ed7fc9f584fc8b4dfba`。

11 个名称是旧数据列名或旧页面生成项。迁移时只验证本地来源、列和值的一致性。当前 `PENDING-IPA` 的整理归属已由作者确认，外部学术书目另行登记；自有整理归属与第三方文献、具体版次及逐字映射核验分别处理。

## 学术书目与转换列

现有十一项标准已关联九条标准学术参考，另有一条字表异常读音补核参考，书目和链接见[来源补充报告](../../testing/m13-academic-sources-report.md)与 `third_party/source-registry.json` 的 `REF-M13-*` 条目。界面和关于页由统一登记生成，引用采用 GB/T 7714-2015。两项 Beijing 列共用 Lee 与 Zee（2003），两项 UntPhesoca 列共用用户指认的2019年文章。

胡裕树、黄伯荣/廖序东、钱乃荣、赵元任与《汉语方音字汇》列保留原名，并分别列出已核实的参考版次。历史字表未记录其实际所据版次、页码，尚未逐字核对。吴宗济列只记作者名，1992年《现代汉语语音概要》作为补充参考，具体直接依据尚未确认。UntPhesoca首发2019年，当前两列依据作者网站2026-10-11访问正文重新生成，更新记录至2024-10-11。采用正文中派基准，自由变体取第一项。用户随后授权统一纠正已查证字音，当前修复10条异常记录及29条丢失的nan拼音，涉及39个拼音字段和4个声调字段。其他八列IPA各修复10个错误单元，其余单元保持，详见[当前验收](../../testing/m13-result-layout-report.md)。书目补齐本身不改变转换规则。

| 功能组 | 说明书承诺 | V2 实际源码与默认值 | V3 落点 | 验收证据 |
| --- | --- | --- | --- | --- |
| M13-F01 转换内容 | 输入汉字、实时显示、11 种标准、多音字选择；非汉字保留 | `hanziIndex`、`addToneMark`、`getDisplayText`、`convertToIPA`、`showVariants`、`selectVariant`；逐 Unicode 字符查表，默认取同字第一条，选择键为“字符＋文本位置”，没有词级上下文消歧；默认 `Standard Chinese (Beijing)严`；缺失单元显示 `?` | `state.ts` 的不可变索引、`convertText`、`variantsFor` 与逐位置选择；`ipa-data.json` 原样保留其余8列IPA，仅Unt两列按现行正文修订，汉语拼音继续用旧声调放置规则 | 21,572 行、20,771 个字符、681 个多音字符、最多 7 条读音；11 标准逐项、`银行` 上下文限制、空输入、标点/拉丁字符/换行见 `m13.test.ts` 与 Chrome 报告 |
| M13-F02 文字排版 | 汉字/音标独立字号、字音间距、行距、粗体、斜体、下划线 | `hanziFontSize=24`、`ipaFontSize=16`、`ipaHanziGap=0`、`hanziLineHeight=1.8`；仅音标且未手调时为 28 px；`getHanziStyle` 实际把粗斜体/下划线用于汉字。旧帮助气泡把对象写成音标，与说明书及代码不一致 | `MandarinIpaPage.vue` 的独立排版状态和有界恢复；IPA 固定公共 Doulos SIL；普通项和多音按钮统一盒模型，避免多音按钮公共 gap 抬高音标 | Chrome 测量 `银行花` 三个 IPA 顶坐标差及汉字顶坐标差均 < 0.5 px；浅深主题、组合符号、下划线与 Qt 截图见验收报告 |
| M13-F03 显示方式 | 仅音标/字音同显、横向/上下排布 | `showWithHanzi=true`、`verticalLayout=false`；切换只重排，不清空输入或多音选择 | 页面 display/layout 状态、公共模块容器查询；横向三栏和上下单栏均保留同一 token 列表 | 两种显示、两种排布、长文本 1,200 个已映射字符及 AppShell 关闭保存/重开恢复见 Chrome 报告 |
| M13-F04 输出与帮助 | 保存图片、使用说明 | `saveAsImage` 调用运行时 CDN `html2canvas`，3 倍白底 PNG；失败使用 `alert` | `export.ts` 直接使用浏览器 Canvas 和内置字体，无新增图片库、后端任务或文本上传；错误留在模块状态区，草稿与结果不清空 | Chrome 切到离线后真实下载 PNG，像素回读和 canvas 字体调用确认 Doulos；字体加载失败、PNG 编码失败均可恢复；Linux 只验证本地静态托管 |

## 2026-10-05 M13-R2 显示修订

本轮按井井授权：默认使用 Standard Chinese (Beijing)，重新打开空文本，旧草稿显式恢复。新增颜色、汉字字体和声调显示开关，IPA 固定 Doulos，原始映射与候选顺序不变。仅显示端将拼音键盘记法 v 规范为 ü，并修复修改文本后的选择位置与仅音标字号提示。导出复用已加载字体、自选样式和声调快照。此前表格记录的旧默认与自动恢复行为作为历史保留，当前行为以[决策](../../decisions/ADR-M13-R2.md)与[验证报告](../../testing/2026-10-05-m13-r2-report.md)为准。

## 映射资源

`generate-data.mjs` 只接受上述旧 HTML 的固定 SHA-256，提取内嵌 `ipaData`。旧 Python `json.dumps` 在 3 个缺失单元写出裸 `NaN`，生成器仅把这些非标准 JSON 值规范化为 `null`，显示路径仍为 `?`。首次原样迁移的 `ipa-data.json` 为3,103,992字节，SHA-256 `3a9b6b08c515a739e1b954919f99a580edc74c5884e16b33f21f6ce077cacd13`，此摘要仅表示历史基线。作者记音修订只替换Unt两列。后续按用户授权统一修复10条异常读音和29条nan拼音缺值，正常历史IPA保持原样；当前摘要和差异见[结果排版与字音修订报告](../../testing/m13-result-layout-report.md)。

Doulos SIL 7.000 复用 `frontend/src/assets/DoulosSIL-Regular.ttf`，887,828 字节，SHA-256 `cc89b87c047bcc8dc00246398218bb6343e0f2372b153cf560c29a77f27068ef`；OFL 文本在 `frontend/src/assets/Doulos-OFL.txt`，SHA-256 `940b145e55e07109fba236378a55eab74247189f4932cfd13b4de32c83ffddc3`。没有复制 V2 的 CDN 脚本，也没有引入外部转换服务。

## 明确限制

- 默认按单字转换，不实现变调、轻声词法判断、连读或词级多音消歧。Unt两列支持主动儿化组合、逐项或全部拆分；不判断词义。
- 多音字第一条仅是旧数据顺序，不能解释为上下文正确答案；页面必须保留人工选择入口。
- 十一项选项的学术参考已登记；历史字表具体版次、页码、其他标准的逐字对应未闭合；Unt修订版本及生成口径已锁定，功能迁移证据与书目存在均不代替映射核验。项目整理归属、文献引用和实际复用材料的许可分别处理。
- `SRC-HTML2CANVAS` 仅记录旧页面依赖。V3 运行时没有加载或发行 html2canvas。

## Unt现行正文与儿化

当前版本口径和可逆字对状态见[决策](../../decisions/ADR-M13-unt-erhua.md)，实测见[报告](../../testing/m13-unt-erhua-report.md)。规则资源保存作者正文文本摘要，自动组合及人工修订不会改原文字或其他标准。儿化操作区及小窗按钮仅在Unt两项下显示。21,572行与固定历史HTML逐行比较，单独列出已授权的字音修正，其余列逐单元保护。儿化按完整韵类替换，不通过追加r实现。
