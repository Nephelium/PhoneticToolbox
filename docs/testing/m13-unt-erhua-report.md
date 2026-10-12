# M13 Unt现行正文与儿化组合验收

本页保留Unt记音与儿化首轮验收快照，状态`verified`。后续用户授权的统一字音修正、结果排版、引用精简、当前Qt配图和说明书摘要以[最新报告](m13-result-layout-report.md)为准。下文旧数据摘要、Chrome截图及验证数量不代表当前资源。最终EXE及跨平台GUI未验。

## 来源与取值

- 作者网站：<https://phesoca.com/aws/268/>，2026-10-11访问。更新记录末项2024-10-11，原文首发知乎2019-04-22，<https://zhuanlan.zhihu.com/p/38258415>保留为原文链接。
- 正文按BeautifulSoup提取`.entry-content`文本并归一空白，SHA-256 `075c54091ce145180f5dee974f4d8013578436514881b74d98d4dfce307b0eb6`。本机核查材料在`output/validation/m13-unt-source/`，正文全文不进入分发资源。
- 两列采用正文中派基准，自由变体取第一项。宽式上声214，严式中派上声213。轻声缺上下文时不填调符；的/么等明确的轻声虚语素用ə。新派q接i的送气及新派四声已核对，未混入中派。
- 正文与更新摘要的ɝ等记法存在不一致，本文取正文，不全局替换。fu正文仍列ʋ̩及ʋ̩ˠ，后记又讨论其问题，本次保留正文。儿化正文韵尾写ɻ˕，前言说明称下移◌̞，也保留正文原符号。
- hm/hng仅出现在更新记录，现行正文未列音值；不追加未经证实的汉字读音。io、ê及成音节辅音没有正文儿化对应，禁用组合。a/ia/ua儿化按正文并入anr/ianr/uanr。

引用由统一登记生成，采用GB/T 7714-2015，删除原先版本未锁定的提示，标明访问快照、中派基准和儿化操作。按用户要求删除引用说明中的独立的儿需人工核对一句，生成引用数据同步，ui-data:check和两项source-coverage检查通过。异常读音核对参考单独列为官方字典补充来源，不当作新转换标准。

## 映射审计

固定历史HTML SHA-256 `11a5c8cc7315d04bc2beece64519247ab86f80e7e9fb37510704b8d1e4ec6451`不改。生成器仅重建Unt两列，并由`--check`检查可复现输出。

21,572行逐行比较：汉字、声调、拼音和其余8列IPA全部零差异，读音顺序、字符库存和缺值均保留。Unt宽式6,858个单元变化，严式9,613个单元变化。当前JSON为3,095,305字节，SHA-256 `407b0c6577532070d985b4df7dbdc2ade3ea2671ce74ccffa7b6b76702745a07`。详见`output/validation/m13-unt-source/mapping-audit.json`。

除已讨论的塞擦音宽式连结弧、er宽式、ŋ附加符号顺序和轻声外，还核对了：

- zh/ch/sh/r声母及ï₂，与z/c/s及ï₁分开，修复春等历史混用。
- j/q/x后的u实际对应ü，修复去等严式值。
- 零声母yi/wu/yu，以及ui/un在软腭、舌冠条件下的韵类。
- er按ér/ěr、èr区别严式，fu、iu及其他自由变体按声明的第一项输出。
- 历史空拼音nan行仅在Unt内部补齐；原字段保留。diu1、bei0、ashu及葛/be仅在Unt内部按已核实读音补核。

异常键依据教育部官方词条：丟<https://dict.revised.moe.edu.tw/dictView.jsp?ID=2222&la=0&powerMode=0>、銩<https://dict.revised.moe.edu.tw/dictView.jsp?ID=2223&la=0&powerMode=0>、唄<https://dict.revised.moe.edu.tw/dictView.jsp?ID=200&la=0&powerMode=0>、葛<https://dict.revised.moe.edu.tw/dictView.jsp?ID=3973&la=0&powerMode=0>、数<https://dict.variants.moe.edu.tw/dictView.jsp?ID=18983>。萡/be没有可靠读音依据，继续显示?。未对21,572行的汉字读音本身执行完整字典重新审订。

## 可逆儿化

原文字和每字读音选择保留，组合只记录字对位置。自动组合按钮一次选择可转写的相邻字＋儿/兒，点击结果可手动组合/拆分，全部拆分恢复逐字结果。保留前字声调，不追加儿的独立声调，不跨空格、标点或换行。组合按完整韵类处理合流、鼻化和韵尾，不追加r。词义由用户复核，例如女儿。

仅选择Unt宽式或严式时显示左侧儿化操作区和读音小窗中的组合、拆分按钮。切换其余9个选项隐藏这些控件，继续输出其原逐字值，切回Unt保留组合决定。旧v1草稿默认无组合；新草稿、显式恢复及PNG携带决定。文本编辑只保留未变公共前后片段中的完整字对。

## 实际验证

环境：Windows 11 10.0.26300，Node 24.13.0，Chrome 154.0.8037.98。未安装新的全局依赖。

| 检查 | 结果 |
| --- | --- |
| `node frontend/src/modules/mandarin-ipa/generate-data.mjs --check` | 当前资源可复现 |
| `node --experimental-strip-types --test frontend/tests/m13.test.ts frontend/tests/m13-unt.test.ts frontend/tests/source-coverage.test.ts` | 18项通过，包含逐单元保护、正文声韵条件、20类儿化示例、位置状态与旧草稿 |
| `npm --prefix frontend run typecheck` | 通过 |
| `npm --prefix frontend run ui-data:check`、`contracts:check` | 通过 |
| `npm --prefix frontend run build` | 通过，既有大chunk提示保留 |
| `python -X utf8 scripts/verify_m13_unt.py` | 浅/深×1440/900px四组通过，自动组合、单独拆分、直接手动组合、文本插入迁移；每组遍历其余9个选项，儿化区域和多音小窗的儿化按钮均不存在，回切Unt保留决定；草稿重开恢复通过 |
| 离线实际PNG | 84,791字节、1638×666，绘制记录包含组合花儿与拆分女/儿，下载文件为PNG |
| `python -X utf8 scripts/verify_m13_sources.py` | 十条学术参考，模块/关于一致，Unt原文及网站链接、复制引用、搜索、浅深/两种宽度通过 |
| 说明书严格校验及生成 | errors=[]、warnings=[]；当前M13正文、章节索引、新截图及生成副本摘要一致 |
| `python -X utf8 scripts/verify_m13_unt.py --manual-only` | 实际帮助入口载入儿化小节，2560px新截图完整加载。初次精确标题匹配未包含自动编号而超时，改按标题内容定位后通过 |
| 全仓`validate_docs.py` | 扫描发现README中8个缺失的docs/assets/readme链接；M13本次文件无对应错误，未修改其他任务的README及素材 |

当前交互证据：`output/validation/m13-unt/chrome-4227bd567091411cb7a0fc68650d5ada/report.json`。来源界面：`output/validation/m13-sources/chrome-d88c6e9d0c694f58802ad11173c9c2fe/report.json`。两个独立浏览器均无页面错误或外部请求，scratch清理回执为deleted。实际截图检查了默认Beijing下隐藏的儿化区域、Unt拆分小窗与字组，说明书4张浅色2560×1600截图均已更新，未裁切/未缩放。

当前说明书打开与图像加载证据：`output/validation/m13-unt/chrome-b9bc543a0500425b95c415af6e65f5c0/report.json`，scratch清理deleted。正式工程的M13搜索索引同步更新；生成正文与构建副本的SHA-256均为`b380d04cce562526ec98356dd0a13bbdba817c673e376aa35622d6326b659ba0`。

验证定位诊断保留：`chrome-e91463fba8b24f8da4dca581b8ffdd71`首次未匹配标题自动编号，`chrome-b3cdd13071404671ae4086b60d6c8edd`首次选择单读音“女”而没有多音小窗。随后采用标题内容和确有多读音的“行”，保留原验收要求并重新通过，scratch均清理deleted。

仍有边界：其余标准的历史书目版次和逐字准确性未扩大核验；自动组合不作词义判断；文章未给出的音值不造值；实际语流、实体输入法/DPI、Qt及EXE另验。
