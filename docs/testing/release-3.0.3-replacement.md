# 3.0.3 同版本成品替换

2026-10-17，状态`verified`。井井明确要求将优化后的桌面程序重新打包，替换服务器与GitHub上的3.0.3，随后指定直接替换EXE、不写新文案。本轮保留现有Release正文、notes、版本号、标签与源码分支，仅替换同版本应用及必要的更新包/大小/SHA/来源清单。自动更新ZIP必须与新EXE同字节，不把旧ZIP继续当作新应用。

## 构建与验证范围

采用当前唯一源码与独立前端输入，版本仍为3.0.3。已核对上一轮桌面预览缓存、请求合并、资源后台读取和当前已验证共用适配代码。MFA环境/模型/词典不入包，沿用已验科学依赖与Qt缓存，三件分发均不得超过500,000,000字节。服务器/GitHub原3.0.3在新包验收完成前保持在线，替换前保存旧成品与关联清单的完整SHA和服务器回退副本。正式封装入口按既有规则已清理旧本机dist/build/installer三目录，不把这一步称为新包运行验收通过，报告及发布前完整旧摘要保留。

构建`PhoneticToolbox-303-20261017-163326-25f561`，编排337.370秒；说明书21章/1059素材，365项派生媒体均命中校验缓存，457项前端、类型及UI数据/独立构建通过。源码快照SHA为`3481adc4932159c8796e771dbc38e238698d1c21a9519fcd9b8db8f565614085`，相对旧冻结60份应用输入变化，六份关键桌面/前端文件身份明确核对，详见忽略目录`output/validation/release-303-replacement-20261017-r1/frozen-closure.json`。公开发行的Git基线与标签仍为b596c3da，未整体推送同期源码。

| 新成品 | 字节 | SHA-256 |
| --- | ---: | --- |
| 免安装EXE | 495186511 | `897d059381325d4aa4b84d305ee6410636d69a25e9dc35d1b84f0b214f3e3051` |
| 正式安装EXE | 497102829 | `055bb0d1de1fa12eafc37f506f7544817f0f0d9441fafc8d3fac0420146623d4` |
| 自动更新ZIP | 495187077 | `f365e253a225d3455d186b3d805679ef7c3726e0bf781c42787316fa225edd6c` |

三个分发均低于500 MB，ZIP的EXE与免安装版同SHA，两EXE的Windows文件/产品版本去除填充后均为3.0.3。打包身份核验、静态审计和封装检查通过；科学压缩包仍为235146080字节、SHA`14354853dabf5a6b81dffd3236e5ef531ce8b6a9699ac11f6d4869077cd821e9`。实际运行验收`output/validation/local-package-0f4ab890/`全部15阶段通过，服务器/GitHub同版本替换及公网核对已完成。源码公共启动器本轮实际可见首页/字体就绪、指定自有窗口取图及正常退出0，证据`output/validation/release-303-replacement/source-preview-44449910ed484e75b03d149727bdd3dd/`。

## 最终字节运行验收

实际执行`.venv/m14/Scripts/python.exe -B -X utf8 scripts/verify_local_package.py --package dist/PhoneticToolbox-303-20261017-163326-25f561`。独立目录/Windows-only PATH下11项科学任务、18模块帮助及分发探针通过；OwnedQA真实中文深目录安装、缓存预热、活动租约阻止卸载及退出后卸载通过，恢复文件最长350字符，研究资料/设置摘要保持。损坏/中断/两进程并发缓存、设置清理与重开、免安装及安装版实际更新交接/重启通过。

默认GPU可见普通/最大化窗口的浅深主题、三字体及字号/滚动共168帧通过，逐帧字形像素检查通过；发声类型页楷体15号临界宽度288次采样通过。最终原生窗口截图`native-owned.png`已查看。同一说明书章节切换无子树重建、媒体保持；生成PDF三页初始绘制后无重复渲染。仅本任务拥有的运行副本、隔离缓存/联接、安装包和生成输入已按清单清理，保留报告及必要截图，清理回执`cleanup-apply.log`为`deleted`。

补充对最终冻结应用的本机语谱接口验证在`output/validation/release-303-replacement/final-preview-2552a50886c54053922dddb13f4fdc39/`通过：相同完整响应SHA与源码Qt证据一致，重复请求约4.30–4.82ms，换通道/区间正确失效，正常退出0及scratch清理通过。这是本机接口耗时，不能外推为首次启动或端到端GUI提速。初次验证配方误把外层窗口启动器当作stdio工作进程而超时，失败与脚本摘要保留；修正为生产使用的已校验内层冻结入口，未改成品以适配测试。

## 体积复核

旧免安装EXE494149024字节，新包495186511字节，增加1037487字节。先前裁剪仍完整保留，科学/宿主压缩包与Qt原生文件身份均相同，详见`runtime-identity.json`。主要新增资源为网页登录说明字体1514968字节，通用前端会包含该未被桌面使用的分块。井井随后指定沿用现有成品；短暂的桌面构建排除试改已撤回，未重新构建或混用另一包验收，体积差异记录`frontend-size-diff.json`。此前约26 MB为展开文件省量，不能当作EXE下载减幅。

| 严格打包规则第0节 | 本轮适用与验证 |
| --- | --- |
| 可见默认GPU/字体与滚动 | 基础项，最终EXE正常可见窗口验证音标页、侧栏、浅深/字体及普通最大化、实际滚动与像素 |
| 材质/侧栏/声道 | 公共绘制基础回归，未新增材质设计；核对焦点、层数、声道引擎，不以关闭GPU代替 |
| 发声类型尺寸反馈 | 楷体15号临界宽度与滚动条状态采样，保留原断言 |
| WinError448卸载 | 同字节OwnedQA真实安装/预热/托管结果/活动租约/卸载及资料设置摘要 |
| 中文深目录/长路径 | OwnedQA with-workspace/long-paths，实际恢复路径超过260字符 |
| helper与换版 | 两种真实换版及新界面、设置/草稿/工程/组件与媒体恢复 |
| 冻结身份 | 唯一构建名、独立前端与源码快照，全部证据绑定最终SHA |
| 测试依据与原生取图 | 沿用自有PID/标题取图和像素边界求交，保留失败并修根因 |
| 同版本替换与关联清单 | 逐项列旧/新摘要，正文/notes/标签保持，必要ZIP/清单同步，HTTPS及GitHub附件核验 |
| QA与联接清理 | 仅清理本轮拥有的已结束scratch/运行副本，报告与必要图像保留，审批拒绝项不重试 |

未改变打包方式、科学依赖选择、共享映射或启动恢复机制，三轮新旧冷启动比较按实际冻结差异判断适用性；最终包原生身份、共享路径、缓存异常与复制边界继续执行。运行与上传命令、最终SHA及完整回执已在下节与忽略的本轮证据目录记录。未验证实体设备、第二台干净Windows或其他平台，不作扩展声明。

## 公开替换与收尾

服务器9个正式路径的完整SHA/大小与新成品一致，包含两个EXE、同字节更新ZIP、来源/构建/校验清单及stable/preview频道。旧文件在替换前逐一核对旧SHA并备份到`/var/backups/phonetictoolbox-releases/3.0.3-897d059381325d4a`，正式文件以临时文件原子替换，频道清单最后切换；未改网站正文、配置或服务。

GitHub原v3.0.3 Release的7个附件均已替换，实际API回读大小/digest/state全部匹配。Release ID、名称、原正文、标签、Git基线和发布时间保持，`release/notes/3.0.3.md`字节保持SHA`85f400df8d6a4ace387a2ac35c74edb425d3d74dbf953ab9648db0aa635631cf`，未推送源码。首次附件切换在3件分发已完成后遇到本机网络连接中断，回读当前状态后按预期旧/新SHA续做，原失败回执保留于`github-replace-attempt-1.json`；未重复删除已完成附件。

实际HTTPS检查3个分发HEAD长度及Range 0–1023的内容/长度/总量，三份更新清单公网完整字节SHA、冻结应用更新解析器及GitHub对应清单通过，共6项公网检查。未从公网重复下载整包；服务器完整SHA和GitHub附件digest已核对。公网链接为[官方免安装版](https://www.phonetictoolbox.com/releases/windows-x64/3.0.3/PhoneticToolbox-windows-x64.exe)与[原GitHub发行页](https://github.com/Nephelium/PhoneticToolbox/releases/tag/v3.0.3)。

本轮忽略目录`output/validation/release-303-replacement-20261017-r1/`保存`replacement-plan.json`逐项旧/新SHA、`server-published.json`/`server-final.json`、`github-published.json`/`github-final.json`、`https-final.json`和`completion.json`。操作顺序为`release_ops.py prepare → stage-server/stage-github → publish-server/publish-github`，随后`verify-published.py`与`cleanup-upload-parts.py`。仅7个不同内容的本轮服务器`.part`传输文件经完整SHA及正式/回退文件核对后删除，旧成品回退保留，`upload-part-cleanup.json`成功；GitHub传输硬链接scratch亦已清理，原本地成品保留。

收尾执行`git diff --check`定向检查通过；`scripts/validate_docs.py`检查2227文件、445来源、150任务，错误为0，未改历史快照的既有断链单列。原更新文案SHA、成品回执及任务verified状态再次核对一致。
