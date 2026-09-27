# M13 V2 说明书、源码与 V3 功能映射

2026-09-26 完成逐项对照。说明书来源为相邻 V2 `Phonetic_Export/index.html` 第 10.1–10.2 节，SHA-256 `a46c984929bcd8073ff1daf4e6b382a6685d4ca19e0fee0e29de3ecf0fb39ad5`。实际页面来源为 `phonetic_toolbox/gui/resources/ipa_trans/ipa_converter.html`，SHA-256 `11a5c8cc7315d04bc2beece64519247ab86f80e7e9fb37510704b8d1e4ec6451`；生成脚本 SHA-256 `676a03094f003a27575961c383ff59998bff284d02e01ed7fc9f584fc8b4dfba`。

11 个名称是旧数据列名或旧页面生成项。本轮只验证本地来源、列和值的迁移一致性，没有把名称当作相应著作、体系或规范来源已经核验。`PENDING-IPA` 仍为来源与许可待处理项。

| 功能组 | 说明书承诺 | V2 实际源码与默认值 | V3 落点 | 验收证据 |
| --- | --- | --- | --- | --- |
| M13-F01 转换内容 | 输入汉字、实时显示、11 种标准、多音字选择；非汉字保留 | `hanziIndex`、`addToneMark`、`getDisplayText`、`convertToIPA`、`showVariants`、`selectVariant`；逐 Unicode 字符查表，默认取同字第一条，选择键为“字符＋文本位置”，没有词级上下文消歧；默认 `Standard Chinese (Beijing)严`；缺失单元显示 `?` | `state.ts` 的不可变索引、`convertText`、`variantsFor` 与逐位置选择；`ipa-data.json` 原样保留 10 列 IPA，汉语拼音继续用旧声调放置规则 | 21,572 行、20,771 个字符、681 个多音字符、最多 7 条读音；11 标准逐项、`银行` 上下文限制、空输入、标点/拉丁字符/换行见 `m13.test.ts` 与 Chrome 报告 |
| M13-F02 文字排版 | 汉字/音标独立字号、字音间距、行距、粗体、斜体、下划线 | `hanziFontSize=24`、`ipaFontSize=16`、`ipaHanziGap=0`、`hanziLineHeight=1.8`；仅音标且未手调时为 28 px；`getHanziStyle` 实际把粗斜体/下划线用于汉字。旧帮助气泡把对象写成音标，与说明书及代码不一致 | `MandarinIpaPage.vue` 的独立排版状态和有界恢复；IPA 固定公共 Doulos SIL；普通项和多音按钮统一盒模型，避免多音按钮公共 gap 抬高音标 | Chrome 测量 `银行花` 三个 IPA 顶坐标差及汉字顶坐标差均 < 0.5 px；浅深主题、组合符号、下划线与 Qt 截图见验收报告 |
| M13-F03 显示方式 | 仅音标/字音同显、横向/上下排布 | `showWithHanzi=true`、`verticalLayout=false`；切换只重排，不清空输入或多音选择 | 页面 display/layout 状态、公共模块容器查询；横向三栏和上下单栏均保留同一 token 列表 | 两种显示、两种排布、长文本 1,200 个已映射字符及 AppShell 关闭保存/重开恢复见 Chrome 报告 |
| M13-F04 输出与帮助 | 保存图片、使用说明 | `saveAsImage` 调用运行时 CDN `html2canvas`，3 倍白底 PNG；失败使用 `alert` | `export.ts` 直接使用浏览器 Canvas 和内置字体，无新增图片库、后端任务或文本上传；错误留在模块状态区，草稿与结果不清空 | Chrome 切到离线后真实下载 PNG，像素回读和 canvas 字体调用确认 Doulos；字体加载失败、PNG 编码失败均可恢复；Linux 只验证本地静态托管 |

## 映射资源

`generate-data.mjs` 只接受上述旧 HTML 的固定 SHA-256，提取内嵌 `ipaData`。旧 Python `json.dumps` 在 3 个缺失单元写出裸 `NaN`，生成器仅把这些非标准 JSON 值规范化为 `null`，显示路径仍为 `?`。生成的 `ipa-data.json` 为 3,103,992 字节，SHA-256 `3a9b6b08c515a739e1b954919f99a580edc74c5884e16b33f21f6ce077cacd13`。

Doulos SIL 7.000 复用 `frontend/src/assets/DoulosSIL-Regular.ttf`，887,828 字节，SHA-256 `cc89b87c047bcc8dc00246398218bb6343e0f2372b153cf560c29a77f27068ef`；OFL 文本在 `frontend/src/assets/Doulos-OFL.txt`，SHA-256 `940b145e55e07109fba236378a55eab74247189f4932cfd13b4de32c83ffddc3`。没有复制 V2 的 CDN 脚本，也没有引入外部转换服务。

## 明确限制

- 转换单位是单字，不实现变调、轻声词法判断、儿化、连读或词级多音消歧。
- 多音字第一条仅是旧数据顺序，不能解释为上下文正确答案；页面必须保留人工选择入口。
- 11 个标准的书目、映射来源和再分发授权仍按 `PENDING-IPA` 审查，功能迁移证据不能替代来源核验。
- `SRC-HTML2CANVAS` 仅记录旧页面依赖。V3 运行时没有加载或发行 html2canvas。
