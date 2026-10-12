# 全局来源引用规范化验收

任务：`SOURCE-CITATION-NORMALIZATION`。核查日期：2026-10-11。状态：`verified`，范围为Windows当前源码、来源显示/复制、说明书阅读资源。

## 修改与来源核查

- 根据用户授权，将私人 `ipa.cin` 从公开致谢和模块致谢覆盖清单中隐藏。原登记、输入别名和来源证据保留。
- 软件来源整理题名、署名与可见版本。Praat和Parselmouth分别署名，SciPy使用官方团队署名，IPAChart.com与Weston Ruter的IPA Unicode Keyboard分别引用。软件包的许可名称与内部审阅说明分开，内部许可审阅措辞及分发状态保留。
- 对48条当前可见学术来源，按用户指定的三个 `gpt-6.1-sol/high` 子代理分为每组16条，以出版方、作者原文、DOI登记、官方图表/数据元信息和馆藏资料核查。48条核心书目信息核实。尹基德论文的年份、月份与博士学位层级由用户随后提供的原论文封面补核。全部补入GB/T 7714—2015引用，原来32条缺少完整citation的卡片已补齐。
- 注册表与BibTeX同步，现有BibTeX共62条，主键唯一。多份资料保留独立引用与真实换行。界面不显示内部kind代码和阶段性版本审阅句。
- 说明书现有29条引用索引、M17参考资料中的11条学术引用、入门章引用操作说明以及M17条目来源同步。中文手册链接指向中文译本馆藏，吕佳/江荻论文链接指向已核实的期刊条目。

关键纠正：ASSET-NASAL v4作者第4位为Hans Lamecker、第5位为Nora Schmidt，题名改为正式数据集题名；VTL手册2.4与2.3分别使用封面日期2025-12-05和2020-08-09；CPP/HNR保留当时刊名；VoQS/extIPA的修订图年份、论文在线年与卷期年分别登记；中文版IPA图按2011年出版条目引用，保留2007年图版本；中文手册直接引用江荻译、上海教育出版社2008年译本；Mills 2020补全作者和期号。

## 逐项书目核查

下表状态只针对核心书目字段。历史字表对应、未核实DOI与日期差异单列于后文。完整字段与各字段的支持来源保存在 `source-registry.json` 的 `citation_verification` 中。

| 来源ID | 正式题名 | 核查依据 | 核心书目 |
| --- | --- | --- | --- |
| REF-CPP | Acoustic Correlates of Breathy Vocal Quality | [依据1](https://api.crossref.org/works/10.1044/jshr.3704.769)、[依据2](https://pubs.asha.org/doi/abs/10.1044/jshr.3704.769)、[依据3](https://pubmed.ncbi.nlm.nih.gov/7967562/) | 已核实 |
| REF-HNR | A Cepstrum-Based Technique for Determining a Harmonics-to-Noise Ratio in Speech Signals | [依据1](https://api.crossref.org/works/10.1044/jshr.3602.254)、[依据2](https://pubs.asha.org/doi/10.1044/jshr.3602.254)、[依据3](https://pubmed.ncbi.nlm.nih.gov/8487518/) | 已核实 |
| REF-SHR | Pitch determination and voice quality analysis using Subharmonic-to-Harmonic Ratio | [依据1](https://api.crossref.org/works/10.1109/ICASSP.2002.5743722)、[依据2](https://ieeexplore.ieee.org/document/5743722/) | 已核实 |
| REF-ISELI | An improved correction formula for the estimation of harmonic magnitudes and its application to open quotient estimation | [依据1](https://api.crossref.org/works/10.1109/ICASSP.2004.1326074)、[依据2](https://www.seas.ucla.edu/spapl/paper/iseli04.pdf)、[依据3](https://www.seas.ucla.edu/spapl/publications.html) | 已核实 |
| REF-HAWKS | A formant bandwidth estimation procedure for vowel synthesis | [依据1](https://api.crossref.org/works/10.1121/1.412986)、[依据2](https://profiles.wustl.edu/en/publications/a-formant-bandwidth-estimation-procedure-for-vowel-synthesis-4372) | 已核实 |
| REF-SOE | Epoch Extraction From Speech Signals | [依据1](https://api.crossref.org/works/10.1109/TASL.2008.2004526)、[依据2](https://repository.ias.ac.in/view/fellows/Yegnanarayana%3D3ABayya%3D3A%3D3A.html) | 已核实 |
| REF-KLATT | Software for a cascade/parallel formant synthesizer | [依据1](https://api.crossref.org/works/10.1121/1.383940)、[依据2](https://sail.usc.edu/~lgoldste/Ling582/Week%2012/klatt1980.pdf) | 已核实 |
| REF-ZAIWA | Contribution of F0 and phonation to tone perception in the Zaiwa language | [依据1](https://api.crossref.org/works/10.1016/j.wocn.2025.101413)、[依据2](https://www.sciencedirect.com/science/article/pii/S0095447025000245) | 已核实 |
| REF-GRIFFINLIM | Signal estimation from modified short-time Fourier transform | [依据1](https://api.crossref.org/works/10.1109/TASSP.1984.1164317)、[依据2](https://dub.ucsd.edu/CATbox/Reader/GriffinLimMSTFT.pdf) | 已核实 |
| DOC-VTL24 | VocalTractLab 2.4 User Manual | [依据1](https://www.vocaltractlab.de/download-vocaltractlab/VTL2.4-manual.pdf)、[依据2](https://www.vocaltractlab.de/index.php?page=vocaltractlab-download) | 已核实 |
| DOC-VTL23 | VocalTractLab 2.3 User Manual | [依据1](https://www.vocaltractlab.de/download-vocaltractlab/VTL2.3-manual.pdf)、[依据2](https://www.vocaltractlab.de/index.php?page=vocaltractlab-download) | 已核实 |
| REF-VTL2006 | Construction and Control of a Three-Dimensional Vocal Tract Model | [依据1](https://api.crossref.org/works/10.1109/ICASSP.2006.1660160)、[依据2](https://www.vocaltractlab.de/publications/birkholz-2006-icassp.pdf) | 已核实 |
| ASSET-NASAL | Data for "Characterization of the Airflow within an Average Geometry of the Healthy Human Nasal Cavity" | [依据1](https://api.figshare.com/v2/articles/9585410/versions/4) | 已核实 |
| REF-MFA | Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi | [依据1](https://api.crossref.org/works/10.21437/Interspeech.2017-1386)、[依据2](https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html) | 已核实 |
| REF-YIN-EGG-THESIS | 汉语韵律的嗓音发声研究 | [依据1](https://lib.chinese.pku.edu.cn/xueweilunwen?combine=&degree=All&degree_str=&field_xwlw_callno_value=&major=All&page=314&sort_by=field_xwlw_year_int_value&sort_order=DESC&year=&year_int_max=&year_int_min=) | 已核实（原论文封面补核：2010年12月、博士学位论文） |
| REF-HENRICH-2004-DEGG | On the use of the derivative of electroglottographic signals for characterization of nonpathological phonation | [依据1](https://api.crossref.org/works/10.1121/1.1646401)、[依据2](https://www.lam.jussieu.fr/Membres/Castellengo/publications/2004b-Voice%20DEGG.pdf) | 已核实 |
| REF-MAKHOUL-1975-LPC | Linear prediction: A tutorial review | [依据1](https://api.crossref.org/works/10.1109/PROC.1975.9792)、[依据2](https://www.commsp.ee.ic.ac.uk/~xl404/papers/Linear%20prediction%20A%20tutorial%20review.pdf) | 已核实 |
| M17-IPA-CHART | The International Phonetic Alphabet | [依据1](https://www.internationalphoneticassociation.org/content/ipa-chart)、[依据2](https://www.internationalphoneticassociation.org/IPAcharts/common_files/pdfs/pdfs_IPA_charts_E/IPA_Kiel.pdf) | 已核实 |
| M17-EXTIPA-2025 | ExtIPA symbols for disordered speech | [依据1](https://www.icpla.org.uk/resources)、[依据2](https://drive.google.com/file/d/1innt3AxeLuAj0p8_7_PpHV8aOY8LbpK3/view) | 已核实 |
| M17-VOQS-BALL-2018 | Revisions to the VoQS system for the transcription of voice quality | [依据1](https://api.crossref.org/works/10.1017/S0025100317000159)、[依据2](https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/abs/revisions-to-the-voqs-system-for-the-transcription-of-voice-quality/67662EA00A0B1D77136AF8514B6F230B)、[依据3](https://www.icpla.org.uk/resources) | 已核实 |
| M17-VOQS-ZH-UNTPHESOCA | VoQS: 音质符号 (2016中英双语版) | [依据1](https://zenodo.org/records/10206204)、[依据2](https://zenodo.org/api/records/10206204)、[依据3](https://phesoca.com/wang/) | 已核实 |
| M17-UNICODE-PHONETIC-ENCODING | Unicode 音标与附加符号编码资料 | [依据1](https://www.unicode.org/versions/Unicode16.0.0/core-spec/chapter-7/)、[依据2](https://www.unicode.org/versions/Unicode16.0.0/)、[依据3](https://www.unicode.org/L2/L2020/20113-voqs-correction.pdf)、[依据4](https://www.unicode.org/L2/L2024/24182r-cartouche.pdf)、[依据5](https://software.sil.org/fonts/bridging-diacritics/) | 已核实 |
| M17-EXTIPA-LV-JIANG-2013 | 国际音标扩展表的分类、命名与功能 | [依据1](https://jasptl.haoyicn.cn/summary/19224?eng=0&type=history)、[依据2](https://www.nopss.gov.cn/n/2015/0721/c358205-27339164.html) | 已核实 |
| M17-IPA-CHART-ZH-2007 | The chart of the International Phonetic Alphabet in Chinese (2007) | [依据1](https://api.crossref.org/works/10.1017/S0025100311000156)、[依据2](https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/abs/chart-of-the-international-phonetic-alphabet-in-chinese-2007/372C851EDD231F03709B56339F4DB400)、[依据3](https://www.internationalphoneticassociation.org/IPAcharts/IPA_charts_T/IPA_charts_T.html) | 已核实 |
| M17-IPA-HANDBOOK-JIANG-2008 | 国际语音学会手册：国际音标使用指南 | [依据1](https://ci.nii.ac.jp/ncid/BB01991047) | 已核实 |
| M17-VOICE-QUALITY-ESLING-2019 | Voice quality: The laryngeal articulator model | [依据1](https://api.crossref.org/works/10.1017/9781108696555)、[依据2](https://www.cambridge.org/core/books/voice-quality/62327F676CC6B8823389A76449731F85)、[依据3](https://ci.nii.ac.jp/ncid/BB28502458) | 已核实 |
| M17-EXTIPA-BALL-2018 | Revisions to the extIPA chart | [依据1](https://api.crossref.org/works/10.1017/S0025100317000147)、[依据2](https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/abs/revisions-to-the-extipa-chart/06C01EA81DA2AECA2AC52AAF21556B33)、[依据3](https://www.internationalphoneticassociation.org/sites/default/files/extIPA_2016.pdf) | 已核实 |
| M17-EXTIPA-BALL-2024 | Changes to certain extIPA diacritics | [依据1](https://api.crossref.org/works/10.1080/02699206.2024.2365205)、[依据2](https://www.tandfonline.com/doi/full/10.1080/02699206.2024.2365205) | 已核实 |
| M17-EXTIPA-BALL-2025-REPLY | Reply to letter concerning the revisions to some diacritics of the extensions to the IPA (extIPA), Ball (2024) | [依据1](https://api.crossref.org/works/10.1080/02699206.2025.2489577)、[依据2](https://www.tandfonline.com/doi/pdf/10.1080/02699206.2025.2489577) | 已核实 |
| REF-DRUGMAN-GLOTTAL-COMPARISON | A Comparative Study of Glottal Source Estimation Techniques | [依据1](https://arxiv.org/abs/2001.00840v1) | 已核实 |
| REF-VOCODER-COMPARISON-2018 | Sound quality comparison among high-quality vocoders by using re-synthesized speech | [依据1](https://www.jstage.jst.go.jp/article/ast/39/3/39_E1779/_article/-char/en)、[依据2](https://api.crossref.org/works/10.1250/ast.39.263) | 已核实 |
| REF-GLOTTDNN-2016 | GlottDNN — A Full-Band Glottal Vocoder for Statistical Parametric Speech Synthesis | [依据1](https://www.isca-archive.org/interspeech_2016/airaksinen16b_interspeech.html)、[依据2](https://api.crossref.org/works/10.21437/Interspeech.2016-342) | 已核实 |
| REF-M10-JORDAN-2017-MRI | Examining Velopharyngeal Closure Patterns Based on Anatomic Variables | [依据1](https://api.crossref.org/works/10.1097/SCS.0000000000003284)、[依据2](https://pubmed.ncbi.nlm.nih.gov/27941550/) | 已核实 |
| REF-M10-OLIVEIRA-2012-MRI | An MRI study of the oral articulation of European Portuguese nasal vowels | [依据1](https://api.crossref.org/works/10.21437/Interspeech.2012-668)、[依据2](https://www.isca-archive.org/interspeech_2012/oliveira12b_interspeech.html) | 已核实 |
| SRC-M18-ARTICULATORY-TTS-2610 | Articulatory Source-Filter TTS: Physically Grounded Control through Vocal Tract Kinematics | [依据1](https://arxiv.org/abs/2610.00735v1) | 已核实 |
| SRC-M18-COARTICULATION-2408 | Towards a Quantitative Analysis of Coarticulation with a Phoneme-to-Articulatory Model | [依据1](https://arxiv.org/abs/2408.05641v1) | 已核实 |
| REF-M13-LEE-ZEE-2003 | Standard Chinese (Beijing) | [依据1](https://api.crossref.org/works/10.1017/S0025100303001208)、[依据2](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S0025100303001208) | 已核实 |
| REF-M13-HU-1995 | 现代汉语: 重订本 | [依据1](https://ci.nii.ac.jp/ncid/BA71639299) | 已核实 |
| REF-M13-HUANG-LIAO-2012 | 现代汉语: 上册 | [依据1](https://xuanshu.hep.com.cn/front/book/findBookDetails?bookId=59cd328fba9eb884cf819448) | 已核实 |
| REF-M13-QIAN-1990 | 现代汉语 | [依据1](https://zhwx.ecnu.edu.cn/ae/57/c14647a175703/page.htm)、[依据2](https://dflt.qdu.edu.cn/__local/2/B3/E7/A8EF8B094C86F3225A79033416E_CBC44C31_18310E.pdf) | 已核实 |
| REF-M13-WU-1992 | 现代汉语语音概要 | [依据1](https://ci.nii.ac.jp/ncid/BA3058811X) | 已核实 |
| REF-M13-CHAO-1979 | 汉语口语语法 | [依据1](https://www.cp.com.cn/Content/2017/02-16/1622474939.html)、[依据2](https://search.worldcat.org/zh-tw/title/10579397)、[依据3](https://www.cp.com.cn/book/9a9ee7ef-1.html) | 已核实 |
| REF-M13-FANGYIN-2003 | 汉语方音字汇（第二版重排本） | [依据1](https://ndlsearch.ndl.go.jp/books/R100000002-Ia1000027590)、[依据2](https://lib.chinese.pku.edu.cn/node/69735) | 已核实 |
| REF-M13-UNTPHESOCA-2019 | 新老派普通话的宽严式记音（含儿化韵） | [依据1](https://phesoca.com/aws/268/) | 已核实 |
| REF-M13-PINYIN-1958 | 汉语拼音方案 | [依据1](https://www.moe.gov.cn/ewebeditor/uploadfile/2015/01/13/20150113091717604.pdf)、[依据2](https://www.moe.gov.cn/s78/A18/A18_ztzl/jnhypyfa/201805/t20180517_336337.html) | 已核实 |
| REF-M10-MILLS-2019 | What is a tongue tie? Defining the anatomy of the in-situ lingual frenulum | [依据1](https://api.crossref.org/works/10.1002/ca.23343)、[依据2](https://pubmed.ncbi.nlm.nih.gov/30701608/) | 已核实 |
| REF-M10-MILLS-2020 | Understanding the Lingual Frenulum: Histological Structure, Tissue Composition, and Implications for Tongue Tie Surgery | [依据1](https://api.crossref.org/works/10.1155/2020/1820978)、[依据2](https://pubmed.ncbi.nlm.nih.gov/32774383/)、[依据3](https://onlinelibrary.wiley.com/doi/10.1155/2020/1820978) | 已核实 |
| REF-M13-READING-REPAIRS-2021 | 重編國語辭典修訂本（2021年版）及異體字字典 | [依据1](https://dict.revised.moe.edu.tw/page.jsp?ID=40)、[依据2](https://dict.variants.moe.edu.tw/page.jsp?ID=12)、[依据3](https://dict.variants.moe.edu.tw/page.jsp?ID=15) | 已核实 |

## 保留的具体未决项

- `REF-YIN-EGG-THESIS`：年份和博士学位层级已由原论文封面确认；全文总页数与旧手册引述对应页码未核对。
- `M17-EXTIPA-LV-JIANG-2013`：文献核心字段已核实；原有DOI未在本轮可访问的主办页面显示，未写入规范citation或BibTeX，保留于urls供继续核查。Crossref未收录不构成判假的依据。
- `REF-M13-LEE-ZEE-2003`：Beijing宽严两列与论文的逐条对应尚未在本次书目核查中验证。
- `REF-M13-HU-1995`：历史字表未记录所据版次和页码；所引1995年第6版书目不能证明实际采用版本。
- `REF-M13-HUANG-LIAO-2012`：历史字表未记录所据版次和页码；引用2012年增订五版上册，不替换为后版。
- `REF-M13-QIAN-1990`：历史字表未记录所据版次和页码；1990高教版与2001江苏教育修订本独立。
- `REF-M13-WU-1992`：吴宗济列历史只记录作者名，本书为补充参考；其具体依据是否本书及对应页码仍未确认。
- `REF-M13-CHAO-1979`：历史字表所据版次和页码尚未核对；本条引用1979年初版。
- `REF-M13-FANGYIN-2003`：历史字表所据版次及北京音条目页码尚未核对；本条引用2003年第二版重排本。
- `REF-M10-MILLS-2020`：出版方和Crossref首发日期2020-07-21，PubMed记录2020-06-28；具体月份差异留档，引用仅用一致确认的2020卷年。

Unt依据作者正文更新记录2024-10-11，并保留2026-10-11访问日期。网页更新时间2024-10-20不替代文章正文修订日期。本次未重新解释或更换科研映射。Praat的公开版本行注明文档所列版本，未把文档书目信息作为本机二进制实测。

## 实际验证

用户补充尹基德论文封面后，将引用中的推定年份 `[2010]` 改为 `2010`，BibTeX改为博士学位论文，记录封面日期2010-12。随后重新运行3项来源测试、生成数据检查、类型检查和前端构建。此前浏览器验收针对引用显示与复制机制，本次仅更新来源内容，未重跑浏览器。

- `node --test tests/source-coverage.test.ts tests/manual-reader-render.test.ts`：4项通过，覆盖48条学术引用、私人CIN隐藏、全局/模块/补充来源一致性、M13标准引用和说明书共享渲染。
- `npm run typecheck`、`npm run ui-data:check`、`npm run build`通过。构建保留原有较大分块提示，未以此改动架构。
- `python -X utf8 scripts/verify_source_citations.py`：[最终浏览器报告](../../output/validation/source-citation-normalization/chrome-a41e884ad35f41f4916b046f38aabc0e/report.json)通过。独立Chrome，浅/深两种主题×1440/900两种宽度。每组核对48条完整引用及复制文本，共192次；检查私人CIN隐藏、应用引用、换行、全局/模块来源一致，无页面异常、外部请求或横向溢出。剪贴板写入使用浏览器端捕获，未声称系统剪贴板实测。
- 同一浏览器读取真实说明书，核对M17参考资料的11条完整学术引用及参考资料目录导航，横向溢出为0。[阅读截图](../../output/validation/source-citation-normalization/chrome-a41e884ad35f41f4916b046f38aabc0e/manual-citations.png)仅作本轮界面验证，不替换正式说明书中的150%DPI工作台截图。
- `python -X utf8 scripts/manual/validate.py --project manual --strict`通过，20章、1044素材、0警告/错误。说明书software阅读资源已重新生成，1066文件的索引及摘要核对无漂移。
- 说明书第一次生成因正在被读取的既有音频替换失败（WinError 5）。生成器改为内容逐字节相同时直接保留既有文件，内容变化仍原子写入。[实际Windows锁检查](../../output/validation/source-citation-normalization/manual-lock-d11bf9baaa6542d9bea69c133f603823/report.json)通过：相同内容保留、变更且锁定时拒绝、解除锁定后正确发布、临时文件清理。未关闭用户进程。
- `python -X utf8 scripts/check_architecture.py`：说明书相关索引/摘要检查通过；全项目检查未通过，仅报告同期任务资源 `frontend/src/assets/feedback-avatar.jpg` 未登记。该资源未由本任务修改，未绕过架构检查。
- 候选文件 `git diff --check`通过。源码文本UTF-8，无BOM。

M13转换数据摘要仍为 `0213de35105d0251a4776f60461fee73b52a415620b711be1a69c1ac9343773f`，与本次规范化之前一致。没有重建最终EXE、重新核准全部下游软件许可或执行跨平台验收。

## 本机证据与产物

`output/validation/source-citation-normalization/proposals-a.json`、`proposals-b.json`、`proposals-c.json`保存48条提案及支持来源。该目录另有Crossref/figshare元数据、整合脚本、完整性记录和浏览器截图。临时浏览器配置、Vite缓存、锁检查输入按各运行目录的cleanup回执清理。正式说明书素材保持原样。
