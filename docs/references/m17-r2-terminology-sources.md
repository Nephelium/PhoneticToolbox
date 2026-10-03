# M17 R2 中文术语与解释来源核对

日期：2026-10-03。目录版本 `m17/1.2.0-20261003`。本报告仅覆盖目录内容、矩阵分类与出处，不替代真实窗口或打包验收。

## 本轮结果

保留625个输入入口（IPA351、extIPA209、VoQS65），所有原ID、输入序列、码位及插入模式不变。R2变换逐项断言输入序列不变。相对R1生成器，182项名称、77项含义说明、23项区别说明、426项来源定位更新。VoQS全部56个指定中文名称原样保留。目录内有示例与组合，625不能称为独立音标数。

IPA的基本辅音、非肺部辅音、补充组合合并为13行×14列的辅音矩阵。351个IPA入口在`extended/vowels/extras/diacritics/suprasegmentals/tones`中恰好各出现一次。矩阵只改变导航位置，entry.section保留旧值以保护持久ID和分类兼容。双重构音、同时构音、独立喷音号和连音线置于extras。所有扩展空格均不自动标阴影，不能将未收录误解为不可能构音。原来齿、龈、龈后共用的基本符号在龈栏作导航锚点，详情保留部位范围说明；腭化龈后`t̠ʲ/d̠ʲ/n̠ʲ`留在龈后栏，不直接等同于龈-腭音。

## 名称依据及冲突规则

井井提供的图4为《国际音标（修订至2005年）》、中文版©2007中国语言学会语音学分会，具有明确的中文术语优先权。井井于2026-10-03补充截图并明确校正click为啧音，本轮可见名称与正文统一采用啧音，早先将该字读为嗒的记录已纠正，不将早先读图结果当作原图确切用字。原R1吸气音检索别名保留；早先误读字样不写入产品别名，避免详情的检索别名字段再次显示。图中版式和原图没有复制到产品。中文表另有正式书目：Phonetic Association of China, *The chart of the International Phonetic Alphabet in Chinese (2007)*, JIPA 41(2), 2011，[DOI](https://doi.org/10.1017/S0025100311000156)。本次直接逐项读图；公开书目仅作为可追溯入口。IPA网站旧PDF链接返回404，不将其记为成功取回。

| 项目 | 图4采用名称 | 旧目录名称的处理 |
| --- | --- | --- |
| plosive、tap or flap | 爆发音、拍音或闪音 | 塞音、闪音／拍音保留为搜索别名 |
| alveolar、glottal | 龈、喉 | 旧齿龈／声门检索名保留；正文解剖名称仍可使用齿龈、声门 |
| click、voiced implosive、ejective | 啧音、浊内爆音、喷音 | 吸气音、挤喉音仅作旧名检索，不混淆肺部吸气言语 |
| vowel height | 闭、半闭、半开、开 | 高／低系列旧名保留为别名；中间位置用次闭／次开自主细分 |
| more/less rounded | 更圆、略展 | 旧较圆唇／较不圆唇别名保留 |
| advanced/retracted、raised/lowered | 偏前／偏后、偏高／偏低 | 解释区分元音舌位与辅音狭窄程度 |
| mid-centralized、non-syllabic、rhoticity | 中-央化、不成音节、r音色 | 不将r音色唯一解释为舌尖后卷 |
| breathy/creaky voiced | 气声性、嘎裂声性 | VoQS的气声／嘎裂声指定译名不变 |
| apical/laminal、tongue root | 舌尖性／舌叶性、舌根偏前／舌根偏后 | 保留作用对象的区别 |
| releases | 鼻除阻、边除阻、无闻除阻 | 无闻不等于声道从未解除闭塞 |
| suprasegmentals | 长、半长、超短、小(音步)组块、大(语调)组块、音节间隔、连接(间隔不出现) | 主重音、次重音保留 |
| tones | 声调与词重调；超高、高、中、低、超低；降阶、升阶、整体上升、整体下降 | 不用固定Hz定义相对调值 |

逐符号独立fixture在`frontend/tests/m17-r2-sources.test.ts`，实际映射位于`scripts/m17_catalog_r2.py`。IPA更名只作用于IPA条目，VoQS不参与替换。旧名称作为检索别名保留。

## 原资料、书目和页码

### IPA原理说明

国际语音学会编，江荻译：《国际语音学会手册：国际音标使用指南》，上海教育出版社，2008年8月第1版，ISBN 978-7-5444-1928-4。原著1999。实际渲染核对版权页、目录及正文pp.9–24，所给PDF页面为印刷页码加19。公开入口：[国际语音学会手册页面](https://www.internationalphoneticassociation.org/content/handbook-ipa)。图4名称与书中不同的地方以图4为准。

| 条目范围 | 参考页码 | PDF页面 |
| --- | --- | --- |
| 肺部气流辅音、共用格与附加符号细分 | 第2.4节，9–12 | 28–31 |
| 非肺部气流机制 | 第2.5节，12–13 | 31–32 |
| 元音空间、圆唇、连续性与参考点 | 第2.6节，13–17 | 32–36 |
| 超音段与声调 | 第2.7节，17–20 | 36–39 |
| 附加符号、辅助构音与除阻 | 第2.8节，20–23 | 39–42 |
| 其他符号与连音线 | 第2.9节，23–24 | 42–43 |

### VoQS解释

用户文件名VoQS.pdf的实际书目是 John H. Esling, Scott R. Moisik, Allison Benner & Lise Crevier-Buchman (2019), *Voice Quality: The Laryngeal Articulator Model*, Cambridge University Press，ISBN 978-1-108-49842-5，[DOI](https://doi.org/10.1017/9781108696555)。它是326页PDF书籍，不能按文件名登记为VoQS图表。印刷页码对应PDF页面加22。

| 说明 | 印刷页码 |
| --- | --- |
| 发声组合与构音制约 | 第1.4.1节，13–15 |
| 喉位 | 第1.4.2节，15–19 |
| 软腭、舌、颌、唇设置 | 第1.5.1–1.5.4节，19–28 |
| 常态浊声 | 第2.3.3节，44–47 |
| 耳语、气声、耳语声 | 第2.3.7–2.3.9节，53–60 |
| 假声、嘎裂声、糙声 | 第2.3.10–2.3.12节，60–71 |
| 室襞与杓会厌襞声源 | 第2.3.13–2.3.14节，71–75 |
| 紧／松设置的解释边界 | 第2.4节，78–79 |

本轮强化气声的较开放喉上管道与耳语声的杓会厌收缩区别，并分开听觉音质、实际振动来源、固定声学阈值和诊断结论。不是所有混合符号都表示各机制可任意相加。产品只含自主短解释和页码，未复制书中图像或连续段落。

56个中文名仍严格采用[UntPhesoca指定译表](https://zhuanlan.zhihu.com/p/203037479)。图表、组合规则和修改沿革依据 Ball, Esling & Dickson (2018), *Revisions to the VoQS system for the transcription of voice quality*, JIPA 48(2), 165–171，[DOI](https://doi.org/10.1017/S0025100317000159)。2016为修订表图版年，2017为在线发表年，2018为卷期年。来源定位从仅按分区改为对应pp.166、168–170及p.169 Figure 2；程度与范围定位到第2.4、3.4节。该论文是授权临床目录下的相关材料之一。

### extIPA历史与当前表

- 吕佳、江荻（2013），pp.665–668解读2002版。旧版外呼气流仍单列historical，不加入现行基本表。
- Ball, Howard & Miller (2018), JIPA 48(2), 155–164，[DOI](https://doi.org/10.1017/S0025100317000147)。第2.1节p.156明确齿唇为上唇接触下齿上缘，已修复原目录articulatory中三项解释的方向颠倒。原PDF第2页已渲染核对。
- Ball (2024), *Changes to certain extIPA diacritics*, 38(7), 692–695，[DOI](https://doi.org/10.1080/02699206.2024.2365205)。p.692列2024-05-14批准的partially denasal、竖波浪鼻漏气、上标腭咽擦音三项调整，p.693 Figure 2；本目录沿用已核对的2025表，并新增准确修订来源。
- 临床目录文件名含Ball2024的回复文，实际为Ball (2025), 39(9), 911–912，[DOI](https://doi.org/10.1080/02699206.2025.2489577)，2025-08-26在线发表。p.911讨论鼻擦音行使用旧号的图版错误，并写partially nasal。该用词与已核对2025表的partially denasal有差异，本轮如实登记，未据文件名或一段回复覆盖当前表含义，也不声称该词形差异已经解决。

## 复现与验证

```powershell
python scripts/verify_m17_catalog.py --write
python scripts/verify_m17_catalog.py
node frontend/scripts/generate-ui-data.mjs
node frontend/scripts/generate-ui-data.mjs --check
node --experimental-strip-types --test frontend/tests/m17-ipa-plus.test.ts frontend/tests/m17-r2-sources.test.ts
```

目录生成及核对通过，旧15项与新3项测试共18项通过。新增6项书目，统一来源记录347→353。产品来源只含公开链接、书目信息、页码与自主解释，不含用户本机绝对路径。原资料仅本机只读，SHA-256、渲染页和182项逐条更名审计存于本机忽略输出目录`output/validation/m17-r2-sources`，不打包、不上传原书。没有声称其扫描版获得可再分发许可。

真实Chrome、Qt及最终EXE结果由根报告另列。此处不宣称字体字形、实体DPI、设备或跨平台界面已验收。
