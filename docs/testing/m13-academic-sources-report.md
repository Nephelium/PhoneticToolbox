# M13 学术来源与规范引用

核对日期：2026-10-10。任务 `M13-ACADEMIC-SOURCES`。当前状态 `verified`，范围为 Windows 当前源码的学术来源登记、书目和共用来源界面。历史映射对应、Qt、最终 EXE 和跨平台验收分别保留未验范围。

## 问题与修改范围

M13 的外部学术书目未逐项登记。自有字表整理条目 `PENDING-IPA` 按既有自有来源规则不显示，最终前端来源目录中 M13 只剩 Doulos SIL 字体，学术分组为空。

新增九条 `REF-M13-*` 学术来源，覆盖下拉框的十一项选项，同步 [统一登记](../../third_party/source-registry.json)、[BibTeX](../../third_party/references.bib)和[覆盖清单](../../third_party/acknowledgement-coverage.json)。前端生成器传递 `reference_note`，共用来源组件显示对应标准与引用版次，方法与引用、关于页仍用同一数据。现有转换名称、默认值、字表、字体和转换算法保持原值。

## 核实的书目与来源关系

引用格式为 GB/T 7714-2015。下表中的版本是已核实的参考书目；历史转换表没有保存的实际版本、页码和逐字对应另列，不把书目核实标成映射准确性验收。

| 转换选项 | 规范引用 | 核查来源与边界 |
| --- | --- | --- |
| Standard Chinese (Beijing)；Standard Chinese (Beijing)严 | LEE W S, ZEE E. Standard Chinese (Beijing)[J]. Journal of the International Phonetic Association, 2003, 33(1): 109-112. DOI: 10.1017/S0025100303001208. | [Cambridge 原文与书目](https://doi.org/10.1017/S0025100303001208)；核实作者、年份、卷期、页码、DOI。两列逐字对应未验。 |
| 胡裕树《现代汉语》 | 胡裕树. 现代汉语: 重订本[M]. 6版. 上海: 上海教育出版社, 1995. | [CiNii 馆藏书目](https://ci.nii.ac.jp/ncid/BA71639299)；ISBN 7532039900。历史列所据版次/页码未记录。 |
| 黄伯荣、廖序东《现代汉语》 | 黄伯荣, 廖序东. 现代汉语: 上册[M]. 增订5版. 北京: 高等教育出版社, 2012. | [出版社书目与目录](https://xuanshu.hep.com.cn/front/book/findBookDetails?bookId=59cd328fba9eb884cf819448)；ISBN 9787040316216，出版日期2012-01-13，第二章为语音。历史列所据版次/页码未记录。 |
| 钱乃荣《现代汉语》 | 钱乃荣. 现代汉语[M]. 北京: 高等教育出版社, 1990. | [编写者左思民的华东师大著作目录](https://zhwx.ecnu.edu.cn/ae/57/c14647a175703/page.htm)同时列1990年高教版与2001年江苏教育修订本。当前引用前者，历史列所据版次/页码未记录。 |
| 吴宗济 | 吴宗济. 现代汉语语音概要[M]. 北京: 华语教学出版社, 1992. | [CiNii 馆藏书目](https://ci.nii.ac.jp/ncid/BA3058811X)；ISBN 7800521370，吴宗济主编，赵金铭、朱竹、刘骥编写。作为补充语音学参考，作者名列究竟据哪一著作/页码尚未确认。 |
| 赵元任《汉语口语语法》 | 赵元任. 汉语口语语法[M]. 吕叔湘, 译. 北京: 商务印书馆, 1979. | [出版社产品页](https://www.cp.com.cn/book/9a9ee7ef-1.html)核实著作、翻译关系；[出版社历史](https://www.cp.com.cn/Content/2017/02-16/1622474939.html)明列1979年出版。产品页另列2005出版/2015重印，未将重印日期当初版。历史列所据版次/页码未记录。 |
| 《汉语方音字汇》 | 北京大学中国语言文学系语言学教研室. 汉语方音字汇[M]. 2版重排本. 北京: 语文出版社, 2003. | [日本国立国会图书馆书目](https://ndlsearch.ndl.go.jp/books/R100000002-Ia1000027590)；ISBN 7801840348，出版2003-06。历史列所据版次及北京音条目页码未记录。 |
| UntPhesoca宽；UntPhesoca严 | UntPhesoca. 新老派普通话的宽严式记音（含儿化韵）[EB/OL]. (2019-04-22)[2026-10-10]. https://zhuanlan.zhihu.com/p/38258415. | 用户直接提供[知乎原文](https://zhuanlan.zhihu.com/p/38258415)作者、标题、链接与对应关系。原页抓取失败，采用[作者网站](https://phesoca.com/aws/268/)核实2019-04-22首发知乎及现存2024-10-11更新记录。本地字表所据修订版未锁定。文章包含儿化韵，模块按单字查表，仍不自动组合儿化韵。 |
| 汉语拼音 | 中国文字改革委员会. 汉语拼音方案[S/OL]. (1958-02-11)[2026-10-10]. https://www.moe.gov.cn/jyb_sjzl/ziliao/A19/195802/t19580201_186000.html. | [教育部方案入口](https://www.moe.gov.cn/jyb_sjzl/ziliao/A19/195802/t19580201_186000.html)及[GB/T 16159—2012](https://www.moe.gov.cn/ewebeditor/uploadfile/2015/01/13/20150113091717604.pdf)第2、3.2节核实批准日期1958-02-11。门户页面标注1958-02-01，未用作正式批准日期。当前逐字拼音不声明支持按词分写正词法。 |

登记与界面仅包含书目、链接及自主编写的短说明，不下载或随包添加著作全文、原图和扫描件。文献引用、字表整理来源和实际复用材料的许可分别核查。

## 验证

实际环境为 Windows、Node 24.13.0、既有 `.venv/m14` Python 与项目独立 headless Chrome。没有调用 Codex 内置浏览器或修改用户浏览器配置。

| 实际命令／检查 | 结果 |
| --- | --- |
| `npm --prefix frontend run ui-data`；`npm --prefix frontend run ui-data:check` | 生成并漂移检查通过，本轮全局可见来源388条，其中 M13 学术9条。数字为本轮源码快照，不作为后续固定门槛。 |
| `npm --prefix frontend run contracts:check` | 通过，生成契约匹配 OpenAPI。 |
| `npm --prefix frontend run typecheck` | 通过。 |
| `node --test frontend/tests/source-coverage.test.ts frontend/tests/m13.test.ts` | 13项通过。十一项选项与学术来源绑定完整，引用与 BibTeX 登记、作者/版本说明、知乎链接、首发与引用日期覆盖；原转换边界及字表摘要检查仍通过。 |
| `npm --prefix frontend run build` | 通过，当前前端产物已生成；保留既有大 chunk 提示，没有调整警告门槛。 |
| `.venv/m14/Scripts/python.exe -B -X utf8 scripts/validate_docs.py` | 通过，`errors=[]`；未修改历史快照中的缺链由脚本另列，无新增活动文档缺链。 |
| `.venv/m14/Scripts/python.exe -B -X utf8 scripts/verify_m13_sources.py` | 四组通过：浅/深主题 × 1440×900、900×700；M13 与关于页九条来源一致，逐条引用与说明可见，搜索Unt/吴宗济、原文/作者网站链接与复制引用正确，软件组及M10范围未混入。无页面错误、外部请求或横向溢出。 |

浏览器证据保存在本机 `output/validation/m13-sources/chrome-e4ebe6c65f204510bd5defb402ee4f4e/`，含 `report.json`、`browser.log`、四组完整来源窗及四组 Unt 条目截图。实际查看浅色完整窗口与深色900px条目截图，标题、引用、说明与链接可读。独立浏览器配置和 Vite 缓存进入 scratch；自有进程停止后 `cleanup.json` 为 `deleted`，报告和必要截图保留。

转换资源基线 SHA-256：`3a9b6b08c515a739e1b954919f99a580edc74c5884e16b33f21f6ce077cacd13`。完成后实际复核一致。

## 未验范围

历史字表的实际版次、页码、宽严两列逐字对应及 UntPhesoca 所据文章版本未验证，吴宗济具体直接著作待确认。本次没有执行算法修订、自动儿化、自然语料准确率验收、Qt 专项、最终 EXE 打包或跨平台 GUI 验收。当前安装/免安装 EXE 尚未纳入本次来源补充。
