# 汉字转国际音标自定义字表验收

2026-10-21。任务 `M13-CUSTOM-TABLES`，状态 `verified`，范围为当前共享源码、Windows Chrome生产页面、Windows Qt及后续实际HTTPS追加验收。实现方案见[计划](../plans/2026-10-21-m13-custom-tables.md)。macOS/Safari及最终EXE/Mac包未纳入运行验收。

## 已实现

- 两列汉字/IPA或普通话拼音音节/IPA，支持XLSX、UTF-8 CSV/TSV/制表符TXT，可带表头。XLSX复用已有锁定SheetJS CE 0.20.3并按需装载，未增加依赖或复制库；支持多工作表和共享/内联字符串，拒绝公式、空/缺列/多余非空列、错误汉字、非法拼音与超预算输入。
- 先选择类型、读取文件和预览，再命名、保存并选用。同名拒绝覆盖，每表50,000条/文件8 MB，最多32套/总存储8 MB。字表中重复汉字保留多读音，相同键值对去重。普通话拼音支持ma、ma1、mā及ü/v/u:，0/5轻声；带调优先，无调回退，第二列IPA原样使用。未覆盖汉字显示问号，标点、空白、Unicode字符和换行保持。
- 库保存到现有LocalAssets values，以CAS避免多窗口覆盖。网页按账号隔离、使用固定字表分组，临时服务器项目变化不丢失；桌面跟随原宿主持久浏览器配置。保存回执成功后才修改当前表和选择。自定义表及其ID不进入账号设置，研究文本不上传。
- 重开自动读取字表库并选中最近使用的自定义表，原文本仍显式恢复草稿。原11项内置标准、声调开关、Unt儿化、排版与PNG规则保留，PNG用自定义名称标注标准。使用说明新增导入/命名/自动恢复章节、示例与搜索索引，Windows/macOS/Web共用。

## 实际检查

命令在Windows、工程现有Node与 `.venv/m14` 下执行：

```powershell
npm --prefix frontend run typecheck
node --test frontend/tests/m13.test.ts frontend/tests/m13-unt.test.ts frontend/tests/m13-custom-tables.test.ts frontend/tests/source-coverage.test.ts
npm --prefix frontend run contracts:check
npm --prefix frontend run ui-data:check
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/manual/validate.py --project manual --strict
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/manual/build.py --project manual --output frontend/public/manual --distribution software
npm --prefix frontend run build:workbench
.\scripts\Start-Research-Workbench.ps1 -CheckOnly
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/verify_m13_custom_tables.py
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/verify_m13_custom_tables_qt.py
```

- 26项定向与来源测试通过。覆盖内置转换/Unt、Unicode多音、旧草稿兼容、精确IPA、拼音规范化、XLSX真实工作簿和公式拒绝、库重新打开、账号隔离/跨项目、CAS冲突与损坏输入。类型、协议、UI来源检查通过，保留活动窗口旧资源的生产构建通过。统一源码入口确认绑定本工程现有解释器和frontend/dist。
- 说明书严格检查21章、1095素材，零错误/警告；software阅读资源成功生成。此数量反映本轮共享工程状态，不代表全书逐条人工重验。
- [独立Chrome证据](../../output/validation/m13-custom-tables/chrome-1cfb4b02a24240378bc0859f1912f359/report.json)：生产AppShell六组通过，含BOM/引号/表头、非法行拒绝、命名/同名拒绝、多读音和补充汉字、TSV带调/无调/轻声、XLSX两种ZIP方式和两种字符串、多表切换、完整关闭重开浏览器后的持久库/选择、独立草稿恢复、PNG文件与绘制值、模块帮助及生成正文、1440×900浅色/900×700深色和窄窗弹窗。存储权限拒绝时显示重试，内置标准仍可切换并正常转换。页面错误和POST/PUT/PATCH上传请求均为空。
- [Windows Qt证据](../../output/validation/m13-custom-tables/qt-0078a754a10442489ba249624e168155/report.json)：实际宿主隔离配置四组通过，使用原生QTest触发文件选择、字表保存和多音选择；Qt Chromium读取压缩XLSX/共享与内联字符串/多工作表，刷新恢复字表和草稿，通过原生保存PNG回读签名并核对绘制的表名及音标，帮助含新章节。浅深结果与导入截图已查看。
- 原有排版浏览器回归也通过[八组场景](../../output/validation/m13-result-layout/chrome-d397b2e0b8cd4e80881757712c30ea68/report.json)，覆盖浅深主题、两种窗口、100%/150%页面缩放、等宽/行距、儿化与实际PNG；回归scratch清理为deleted。文档检查2375文件、448来源、172任务零错误，架构检查零错误，旧历史快照断链单列。
- 自有测试文件、浏览器配置、Qt临时配置与导出均放scratch，完成后关闭各自进程，再由共享收尾器清理。两次成功运行的cleanup.json均为 `deleted`；报告、日志和截图保留。

## 失败与修正

首轮浏览器脚本在异步模块尚未出现时读取disabled，已增加元素存在条件。Qt worker首轮受解释器路径配置影响未找到验证辅助模块，已显式绑定脚本目录；第二轮在刷新仍显示旧DOM时过早点击导航，现等待实际loadFinished后进入工具。失败报告/截图/日志保留，各轮scratch均按回执清理。

首次来源覆盖检查发现新增M13的SheetJS使用尚未进入覆盖清单，已同步登记及生成数据，原断言保持。全项目类型检查曾在同期TextGrid文件生成过程中缺少IpaKeyboard.vue，待该任务文件就绪后原命令通过；本任务未修改TextGrid代码。未降低校验、扩大容差或跳过失败项。

## 边界

Windows、macOS及网页版消费同一MandarinIpaPage与转换/导入/持久库实现，平台仅提供原宿主与本机存储。Windows Chrome证明独立浏览器完整重启持久化，Qt测试使用离线记录配置，仅验证当前宿主刷新恢复，未据此宣称其他操作系统或最终安装包重启已验。

未运行macOS/Safari/其他实体设备，未重建最终Windows EXE或macOS包，同日已公开部署并复验，未push或变更数据库schema。字表为当前设备资料，更换浏览器/设备需保留原文件后重新导入。历史全字表逐字准确率、书目版次/页码及自然语流判断不在本次自定义导入范围。


## 2026-10-21 网页上线与源码工作台

自定义汉字/拼音音标字表已与TextGrid一起发布`m12-m13-20261021-77b3ae2db3cc`，入口 <https://www.phonetictoolbox.com/app/>，说明书M13正文、目录和搜索条目同步。

候选和实际公网独立Chrome各五组通过，覆盖CSV/TSV/XLSX多工作表、命名和多读音、拼音匹配、PNG、完整浏览器重启恢复、独立草稿、帮助及浅深/窄窗。研究上传和页面异常均为零，截图已检查。整批候选14组/公网15组、Linux九类、发布/回退与清理记录见[同批部署验收](m12-three-tiers-keyboard.md#2026-10-21-网页上线与源码工作台)。证据`output/validation/m12-m13-web-20261021/`，两个模块共6个QA身份已停用，11项远端临时文件与自有scratch已清。

当前源码工作台通过统一入口打开到M12，窗口响应并在前台，字表功能可从侧栏进入。未push或重建最终EXE/macOS包，Mac实机/Safari等未验项保持原边界。

## 2026-10-21 字表替换与删除追加

本轮状态verified，范围为三端共用源码、本地生产浏览器与实际Windows Qt。此前导入功能已上线，本轮替换/删除随后按授权公开部署，未重建Windows EXE/macOS包，macOS/Safari未实测。

- 选中自定义字表后显示替换字表、删除字表。替换从同一导入窗读取新文件，预览后可沿用或修改名称，保留表ID且不增加数量；内容修订号递增。删除核对表名后确认，CAS成功才从本机库移除，切回内置标准，保留当前文本与其他表。取消、非法文件、同名冲突和保存失败保留原表。
- 可选revision/tableRevision字段兼容原v1记录，旧表与旧草稿按第1版读取，无数据库schema变更。恢复已替换表的旧草稿时保留文本和排版，清除失效多音序号并提示重新核对；表已删除时仍恢复文本与排版，选用当前可用标准，删除不会把草稿文本一起丢失。
- 28项M13/Unt/来源测试、类型/协议/UI数据检查、build:workbench通过。新测试覆盖替换ID/版本/数量、旧草稿多音失效、删除后的文本恢复、其他表保留、模拟写失败和CAS防止旧窗口复活已删记录。说明书严格校验21章/1112素材、software生成零错误/警告，源工程/阅读目录/生产目录均含新增操作说明。
- [Chrome追加证据](../../output/validation/m13-custom-tables/chrome-ccf7ecda60ba4d18a4d9f8337569c3e0/report.json)八组通过：包括替换取消、错误表/同名拒绝、改名保持ID、完整浏览器重启后的新内容和草稿校对提示、删除确认/取消、删除后切回内置标准且文本保留、其他表不受影响、再次完整重启不复活和已删表草稿文本恢复。原导入/多音/XLSX/PNG/帮助、浅深/窄窗与存储拒绝回退也保持，页面错误及上传请求为空。
- [Windows Qt追加证据](../../output/validation/m13-custom-tables/qt-ff6ba64537c94082bb828fbd16c1b5ad/report.json)五组通过：原生QTest导入、替换与删除确认，刷新后移除持久化且旧草稿恢复文本，原生PNG/帮助等原检查继续通过。nativeDiscardConfirmed为true，测试在有未保存编辑时通过实际QMessageBox的Yes按钮明确放弃当前编辑，随后检查先前草稿。浅深界面、替换与删除截图已查看。
- 追加Qt初轮到达刷新时，被原有未保存草稿确认框拦住。已记录诊断并只停止核实归属的自有worker；保留产品确认逻辑，在测试中补上QTest确认后复验通过。初轮与两次最终运行的scratch回执均为deleted，报告、日志和截图保留。

本轮收尾：文档2380文件/448来源/172任务零错误，历史快照断链单列；架构与受影响文件diff检查零错误。

## 2026-10-21 选列、起始行与声调映射

本轮共享源码及已授权网页更新已完成，实际本地Chrome、Windows Qt、候选及公网分别验证。

- 从多列源表选择汉字/音节列、音标列与1起算的起始行，忽略其他列。原始预览显示起始行附近最多8行及真实行号，转换预览显示前8条有效记录。XLSX保留A1起的空行和列，真实文件列坐标不压缩；表头建议可手动覆盖。
- 声调默认原样。末尾五度调值显式逐位转˩˨˧˦˥，普通/上标/全角数字等价。调类可原样显示，或逐类指定调值，无预设普通话对应。原始音标随字表保留，可在声调映射中改回原值；已有声调符号不重复转换，开关/多读音/PNG共用结果。tones可选字段兼容原v1库，不改schema，修改映射以CAS回执和revision为界。
- 33项M13/Unt/来源测试，typecheck、contracts:check、ui-data:check、build:workbench通过。说明书严格21章/1112素材及software阅读构建零错误/警告，M13章节/目录/搜索、模块README与操作页同步。
- [本地Chrome10组](../../output/validation/m13-custom-tables/chrome-83090aff76e94805a34a60bb2afa39dd/report.json)通过。含错误列/相同列/起始行、额外备注、原表预览、三种数字形式、五度符号、调类保留/非法映射拒绝/取消/任意调值、完整浏览器重启与恢复原调类。原导入/替换/删除/多音/PNG/帮助/浅深窄窗/存储拒绝继续通过，无页面异常或上传，截图已查看。
- [实际Windows Qt7组](../../output/validation/m13-custom-tables/qt-d23368fa74eb40eaa5ece6958fa97750/report.json)通过，原生QTest导入与保存、原列和行、声调转换/映射修改、刷新恢复、原数值回显、PNG和帮助均实测。未保存确认仍由实际QMessageBox确认，未绕过产品保护；浅深截图已查看。初轮脚本分别因预览双格相同文字及CSS空格转义定位失败，改成明确结果格/属性查找后复验，产品未降低校验。失败和成功scratch均为deleted。

macOS/Safari和最终Windows/macOS包本轮未运行，三端共用源码路径已同步。字表仍在各设备本机保存。

网页版已发布 `m13-import-tones-354e493fce3344888cb8aa7fdb4ecb5c`，入口 https://www.phonetictoolbox.com/app/ 。以M18 DOI当前线上冻结源为基线，54个可达控制构建分块与线上规范化内容一致，仅叠加9个M13源码文件与M13说明章节/目录/搜索。保留其他模块、论文资料和说明书正文。候选与实际HTTPS各9组通过，零研究上传/页面异常，浅深与窄窗截图已查看。

[发布与公网核对](../../output/validation/m13-import-tones-web-20261021/public-readback.json)：17项变化HTTPS逐项字节/摘要一致，2298发布文件、774运行文件及1863文件回退核对通过；后端、科学回执、官网首页、论文目录、Nginx配置和服务PID保持，无服务重启。两个任务QA身份已停用，活动数0。首次最终只读检查早于公网QA身份停用回执，身份活动断言未通过，等待QA完整退出后原检查加上进程/配置保护再次通过。自有scratch清理为deleted，冻结源码、发布增量、报告/日志/截图与服务器回退保留。

此前替换/删除已按授权发布 `m13-table-manage-f59a0326cb7f44648054f569f2d5b4f2`，候选与实际HTTPS各7组通过，2个任务QA身份已停用，其自有scratch本次收尾为deleted。此前通过统一入口重开的M13源码窗口进程、标题、非零句柄及Responding已确认。本次构建已更新，用户随后明确要求停止桌面操控以便观看视频，未继续刷新或重启正在使用的源码窗口。当前新界面由上述隔离Qt运行验收覆盖。

收尾文档2386文件/448来源/173任务与架构检查零错误，受影响文件diff检查通过。

源码工作台后续收尾：按用户新要求，仅通过统一启动脚本以-Module home重新打开源码工作台，未打开汉字转国际音标；实际源码进程59384、窗口句柄1512276、PhoneticToolbox 3.0及Responding=True已由只读进程信息确认，未使用桌面操控。启动前旧源码进程已退出。证据为`output/validation/m13-import-tones-web-20261021/source-home.json`；本次只核对启动参数与进程/窗口状态，未读取页面或操作窗口。
