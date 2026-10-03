# M17 资料、术语与覆盖审计

日期：2026-10-02。实施依据见 [详细规划](../plans/2026-10-02-m17-ipa-plus.md)，验证见 [报告](../testing/m17-report.md)。

**2026-10-03增量：** 当前版本扩充为IPA351、extIPA209、VoQS65，共625入口。中文分组及2002/2025版本区别见 [来源差异](m17-expansion-diff.md)，验证见 [M17-R1报告](../testing/2026-10-03-m17-expansion-report.md)。下文500入口为2026-10-02历史基线，不作为当前总数。

**2026-10-03 R2增量：** 图4中文名优先，click依据井井2026-10-03补充截图校正为啧音；手册2008中文版与Esling等2019嗓音书籍的页码解释、extIPA版本差异和13×14辅音矩阵详见[术语与来源核对](m17-r2-terminology-sources.md)。625入口/字符不变，VoQS56指定名不变。下文原分区结构为历史基线。

## 原始资料与版本

| 资料 | 核对与用途 | 版本和分发边界 |
| --- | --- | --- |
| [IPA 官方表](https://www.internationalphoneticassociation.org/content/ipa-chart)及官方 IPA_Kiel.pdf | 实际下载并查看整张表，按分区逐格建立目录 | 2026 重印；内容修订 2015/2005，CC BY-SA 4.0。自绘交互衍生表注明归属并保留同许可，原 PDF 不随包 |
| 用户 extIPA PDF，[ICPLA 资源](https://www.icpla.org.uk/resources) | 一页图实际查看，提取文本仅辅助定位，不直接生成字符 | 2025 表；资源页标注 CC BY-SA 3.0，同时注明原表复制不作更改。自有中文说明和交互数据单独组织，原图不随包，不能据字体许可推导图版许可 |
| 用户 Ball 等论文，[DOI](https://doi.org/10.1017/S0025100317000159) | 读取 pp.165–171；实际看 p.169 Figure 2 与底部例句 | 2016 修订表、2017 在线发表、2018 卷期。原论文与图版版权保留，不打包 PDF |
| [UntPhesoca 中文文章](https://zhuanlan.zhihu.com/p/203037479) | 井井提供完整正文与双语图，另有知乎精确检索的前后两段记录；56 中文名与分类逐条采用 | 标题《VoQS：音质符号（2016 中英双语版）》；2020 unt 译；©2016 Ball、Esling、Dickson。文章图片及版式未复制到产品 |
| 用户 ipa.cin | 只读502映射/387不同输出/346字符；用于字符比对与检索别名 | 作者/许可未独立确认，原文件不分发、不执行、不修改。不是主清单或语义权威 |

三份用户原文件 SHA-256：

```text
ExtIPA_chart_(2025).pdf
971248df743bc6ca15ff8067238006f05010ddcae83cacd744dc5313d3a50528
Ball 等 - 2018 - Revisions to the VoQS system for the transcription.pdf
03ed513f97834f829be57cb88baf67b3b07f86c75a854984f7dbd67a01901e39
ipa.cin
291d5ab9de8f64d9d51c7c6d0d479493dcc253ac1541be75ff2d510995a7d983
```

本轮可复核图像在 `output/planning/2026-10-02-m16-m17/extipa-1.png`、`voqs-5.png` 和 `output/validation/m17/voqs-example.png`。知乎直接浏览最初失败，后续检索与井井贴文补齐，不能把前一次失败改记为成功直读。

## 覆盖定义

| 表 | 输入入口 | 分区 | 说明 |
| --- | ---: | ---: | --- |
| IPA | 244 | 7 | 肺部辅音、非肺部辅音、元音图、其他、附加、超音段、声调，独立符号和原表例示分开 |
| extIPA | 191 | 6 | 额外辅音、附加、发声、节奏、未知/圈围、其他声音 |
| VoQS | 65 | 8 | 原表56项 + 数字、括号、范围模板和底部例句9项 |
| 总计 | 500 | 21 | 原表出现项和输入操作计数，不冒充500个不同Unicode字符 |

[`m17-symbol-coverage.csv`](m17-symbol-coverage.csv)逐行记录来源、定位、表区、显示、ID、输入、码位、方式、介绍标题和点击测试ID。`catalog.json` 的矩阵 cells、元音 points 与分区 ids 完全一一对应。灰格/空格/分隔线是图表结构，未伪造为转写符号。重排仍保留辅音发音方式×部位、清浊配对、元音图位置和 VoQS 分类。

原表 56 个 VoQS 中文名另列 `frontend/src/modules/ipa-plus/data/voqs-zh-names.json`，与测试中独立抄录的指定译名逐项比较。无额外“嗓音”后缀。例如 V=常态浊声、V!=糙声、V͉=弛/松声、V͈=挤喉发声/紧声、V‼=室襞性发声。所有56项详情链接同时指向原论文和该中文文章。

## 编码和形态校核

- SIL 旧 PUA：F267→1DF06、F268→1DF04，按 [Doulos 官方记录](https://software.sil.org/doulos/history/)使用现代编码；不改 CIN 原文。
- 上标 AA 使用 U+10780，咽门化声上标带横 H 使用 U+A7F8；前者不把全尺寸 Ꜳ 直接接在 V 后。[Unicode VoQS 字形澄清](https://www.unicode.org/L2/L2020/20113-voqs-correction.pdf)。
- 部分清/浊化由下圈/上圈/浊化记号加对应组合双括号或单侧括号组成，display 中虚线圆不写入文本。随包字体 mark-to-mark 处理定位。
- 单字符圈围有13种实际圆圈连字。多字符圈围两项保留入口，以 `⟅…⟆`/`⟅n̥ã⟆`明确文本降级；无 PUA，无未分配提案码位，不冒称圆框跨度已由 Unicode 标准编码。[Unicode 圈围讨论](https://www.unicode.org/L2/L2024/24182r-cartouche.pdf)。
- 原图底部完整 VoQS 例句保留卷舌着色、重音、嵌套标签顺序。按钮显示紧凑预览，详情和实际插入保留全句。
- PDF 提取中的伪字符/多余数字未照搬。字形检查与码位检查分别保留，详见 [字体审计](m17-font-audit.md)。

中文说明由项目自行撰写，依据对应原表及论文解释用途，区分 whisper / whispery / breathy、creak / creaky、范围音质与单音段标记。extIPA 的 partially denasal 与 VoQS 的 denasalized 不混称；所有条目均有中文含义、用法、区别、码位、英文与来源定位。

## 布局参考与许可记录

实际读取 [ipachart.com](https://www.ipachart.com/) 的可交互 IPA 表结构，借鉴分区与可点单项；自行实现横向重排，未复制其代码、音频或图像。WestonRuter 备选页浏览器正文取回失败，保留失败边界，不声称已完整审阅。

新增来源独立交付为 [`m17-source-additions.json`](m17-source-additions.json)，由主代理合并统一来源面板。字体许可、论文引用、图表许可、中文译名来源分别登记。原论文、原 CIN、候选未采用字体均没有进入产品 assets。

再生成目录脚本在这台授权工作机只读原 CIN 以恢复相同检索别名；独立检出若没有原文件，需保留已经冻结的目录数据，不能把缺少别名的重生成结果当作等价。产品运行本身不读取 CIN 或本机资料路径。CSV 再现固定 LF，避免 Windows 换行转换造成伪差异。

2026-10-03起，目录核对脚本支持 `--cin <只读资料路径>`，可在WSL传入已挂载的Windows原文件路径，无需改HOME或复制资料。显式路径不存在时报错；不传参数继续使用本机Documents默认路径。
