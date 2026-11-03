# 应用三语验证

2026-11-03：本轮源码已纳入3.1.3的Windows/Web发行并同步GitHub，最终字节与完整发布/复验边界见[3.1.3发行验收](release-3.1.3.md)。Mac仅源码同步，安装包暂保留3.1.1。

任务：APP-I18N-20261030。以下按实施阶段记录检查范围，早期未验收、未发布的描述仅对应当时状态，当前发行结果以上方3.1.3验收为准。

## 共用语言和账号

- 前端采用三语命名键和命名参数，默认中文。英文、日文按需加载，同一语言并发请求合并，重复选择复用已加载资源。语言选择不重挂载工作台。
- 语言偏好捕获具体账号的设置会话。新账号没有语言偏好时使用中文，退出恢复匿名本机选择；切换账号使旧的异步选择失效，不能写入下一账号。
- 状态和已知错误保存可重新翻译的信息描述。外部未知诊断保持原文。兼容旧的错误文字只在错误边界匹配，不查找替换 DOM，也不处理研究材料。
- WAV 解析和声道 Worker 传递信息描述，主线程按当前语言显示。Worker 只带自身使用的中文诊断回退，不导入 Vue 或整套语言包。原 RIFF、PCM、浮点、非有限值及通道顺序检查保留。
- 图片处理和声道资料冲突改用错误类型、稳定错误码，不依赖翻译后的前缀或片段。显示字段的原对象与 Vue 代理共享同一描述，避免重复注册。

2026-10-30 在当前 Windows 源码执行：

```powershell
npm run typecheck --prefix frontend
npm test --prefix frontend
```

类型检查通过。完整前端测试最近一轮为 837 项通过、0 项失败，日志为 `output/validation/app-i18n-root/frontend-node-tests-latest.log`。实际 SFC 生命周期测试接入真实语言模块，原有录音/试听、异步关闭、保存失败、可见性轮询和取消断言保留。这些检查使用受控设备或服务边界，不代表实体录音设备、最终 EXE 或完整网页服务验收。

## 模块和语言内容

十九模块已完成首轮界面接入，根公共界面、账户、设置、平台错误及音标资料已加入三语词典。当前仍需进行全文展示扫描和实际三语操作；词典键一致不代表全部界面完成。

真实 Qt 的英文语言地图截图发现的整句拼接及方言层级同名问题已修复，两语各 8 张实际截图已重拍。设置两语各 4 张已登记，合计 24/680 张应用截图，未将草稿图计入。对应证据位于 `output/validation/m19-manual/` 和 `output/validation/settings-localized/`。开发截图不能代替最后成品的运行验收。

Qt 原生词表由同一三语源生成 Python 和前端资源，当前 496 条。新增 383 条产品错误已逐条审阅，并核对 41 个 Python 文件中的实际中文 AST 常量与审计全集一致；另补专用缓存目录和启动失败恢复提示，原始诊断仍写日志。错误边界取消长度门槛，已登记短错误也可重译，普通研究文字和文件名不处理。14 项前端语言测试、8 项原生语言测试通过。Win32 实际启动卡片此前完成三语各 14 阶段、5 DPI 共 210 组检查，重复窗口后的 GDI/USER 数量保持，证据为 `output/validation/app-i18n-startup/`。后续原生错误和启动失败变更仍需在最终 EXE 验证，不能由词表数量扩大结论。

独立真实 Chrome 开发预览最近 16 项检查通过，含中文默认、主页/设置同步、重复英文只加载一次、820/1280/1920 三语布局、日文实际字体、未应用的字体草稿保留，以及已调整宽度的现有分隔条随语言更新无重建、宽度保持。见 `output/validation/app-i18n-browser/1793317062145/report.json`。这不等同于登录网页候选验收。

## 可选离线说明书

- 中文为默认随包内容。英文、日文为独立可安装数据包，绑定当前中文内容摘要和完整文件校验，损坏包不替换旧包。
- 原生安装器 14 项定向检查通过，覆盖实际本机 HTTP 下载、重复安装免下载、损坏文件重装修复、离线重开、包及文件摘要、路径归属、坏包保留旧包、分块导入顺序、两个管理器并发提交、失败原子提交和退出取消。退出只清本实例未完成的导入，既有手册与其他窗口传输保留；活动网络传输在读取返回后协作停止。见 `desktop/tests/test_manual_packs.py`。
- 小型合成包的真实 Chrome 流程已覆盖下载、IndexedDB 保存、切换、禁用下载后离线重开和读取次数，证据为 `output/validation/manual-language-packs/`。这仅证明安装/读取链路，未代表完整手册翻译或全部截图通过。
- 当前说明书引用的 340 张应用截图需重拍英文、日文，另 3 张带自动中文标题/轴标的科学图需同数据重新导出；其余 5 张科学图已逐张核对复用条件。源图摘要变化会使外语覆盖失效。
- 22 章英文、日文全文最近机械覆盖均为 6207/6207 单元，真实结构化阅读回读及搜索通过，见 `output/validation/manual-language-packs/complete-readers-01bfbafacca341f0ba397d29bd250215/report.json`。M10 当前功能事实正在按实现修订，完成后须重新校验全文源摘要。作者预览可用于真实截图，明确不是候选包，最终外语包仍须满足全部实拍覆盖。

## 缓存隔离与退出

验证窗口的缓存与手册安装目录显式绑定本次自有临时根，不读取生产缓存偏好或清理用户缓存；生产窗口沿用原位置。缓存候选保持稳定身份，仅展示标签/原因随语言切换，验证窗口明确禁止退出清除全部/图形缓存。7 项上下文与关闭定向测试、缓存行实际三语反应测试和上述真实 Qt 设置采图通过。

公共语言注册表同时生成 API 语言代码，15 项校验检查通过。独立新建 PostgreSQL 的真实账户 API 流程通过：中英日依次保存、其他会话读取、错误输入和 CSRF 拒绝、科研设置与另一账户保持隔离。该测试使用新建临时库，未访问现存数据库，收尾已停止自有集群并清理 scratch，见 `output/validation/web-launch/postgres-eeacbde5e9f243829fc498d8cafe6d49/cleanup.json`。ASGI API 检查与后续实际登录浏览器验收分开记录。

原生桥失败在错误边界保留可重译描述，未知诊断保持原文；说明书、论文与反馈通道未以译文改变协议。28 项语言/缓存/论文/反馈定向前端测试、类型检查通过，独立于前述较早完整 819 项测试。首次新增断言误将日文原文中的“独立”预期为“隔離”，核对现有词表后修正测试依据，产品文案未为测试改写。

## 对应语言实拍的进展

M13 已新增英文、日文各 17 张当前 Qt 全屏实拍并逐张目视登记，均为原生 DPR1.5、页面 100%、2560×1600 和浅色，源在采集期间未变。实际文字输入、多音选择、儿化、等宽/上下排版、草稿恢复、未保存关闭与缺字体提示通过；实际 PNG 导出回读记录尺寸/摘要并确认不导出选择三角。报告分别为 `output/validation/m13-result-layout/en-16e2fc08f03e49b0a562a0f6e7eb547f/report.json` 和 `ja-0352a348a71743fb84854c333fc4fb3d/report.json`。源中文图片、实际研究文字与数值未改，只有对应外语覆盖新增。

M17 旧采图配方已修为当前禁用播放状态，保留真实编辑、原生指针、复制/UTF-8导出和帮助返回的行为断言。适配中保留失败诊断：无障碍名称不能只按按钮文字查找，隐藏最大化窗口需先按实际屏幕初始化尺寸；阅读器上下文跳转按现有规则只标记三级位置，不能将目录章按钮的 `aria-current=page` 当作必然结果。按实际章节 article 身份、阅读语言和面包屑核对帮助目标，不降低截图尺寸/字体门槛。

M17 两语各 15/16 张已逐张查看后登记，包含当前禁用播放、真实复制、UTF-8 文件回读、字体许可和由实际 DOM 区域派生的编辑区状态图。英文最后一轮为 `output/validation/m17-manual/en-794f5c9fc10f4001988afe9b5cfb0dc2/`，日文主批为 `ja-6e4efcc9005a4d02b10ea1c5214aab4f/`，连接符无悬浮卡的新父图为 `ja-ec5ae626da384113ab07773dccad745a/`；派生报告位于 `output/validation/manual-localized-details/`。两张帮助图待最终目录标题后补拍。英文字符数改为中性标签 Characters，避免单字符仍显示复数。

后续两张帮助图已按稳定全书摘要 `a304a9365b7c81525c4170b19b7b802e954fced683ac51ae869b1dade09f17a7` 补拍、目视并登记，M17 现两语各 16/16。报告为 `output/validation/m17-manual/en-e8db7aaddb85406fa1f21fb9c09a1e10/` 和 `ja-5d019f8b468b41308efe0720fd0a0051/`。M10 事实及工具标题同步后，英文/日文全文均 6230/6230、22 章结构与每语 2425 搜索项回读通过，见 `output/validation/manual-language-packs/complete-readers-0c595087955d4a6d8b68f97787ac6580/report.json`。当前UI登记快照已增至108/680。

M13 两张同输入科学图重导出时发现英日标题与标准名称沿用固定间距而重叠。现按实际文字宽度布局，空间不足时改为两行，保留原转写、参数与输出文件名。独立构建的两语实际 Qt PNG 均为 3600×906，新增真实 Canvas 标题边界断言并逐张目视后严格登记。输入和完整字体/字号/排版/颜色参数与 Git HEAD 原配方一致，行距滑块 22 对应实际倍率 2.2，源中文图未改。报告为 `output/validation/m13-result-layout/en-e5b2d937be5a4c8987e1dfa30b95e065/report.json` 和 `ja-2a133abb925a4c9b8d8ca5802add2ae8/report.json`，科学图覆盖目前 2/6。较早重叠图未登记，首次仅导出适配漏执行弹窗打开动作的测试错误也保留诊断。

2026-10-30 本轮登记快照为 UI 英文 60/340、日文 46/340，共 106/680，另有上述两张科学图。该计数来自当前引用 ID 与外语覆盖注册表交集，不含未审阅草稿。

## 实际登录网页集成

新增 `backend/tests/test_app_i18n_web.py` 与 `tests/e2e/app-i18n-web.cjs`，使用全新本地 PostgreSQL、实际回环 HTTP、独立 Chrome/Edge 配置及当前生产构建。七组行为通过：新账号默认中文、真实 WAV 与参数在五次切语后保留、三语九种宽度、Edge 恢复账户语言与参数且不取得另一浏览器本地 WAV、刷新保留、退出恢复匿名语言、复用配置登录另一账号仍隔离。语言切换期间无新上传或科学任务 POST，浏览器错误为零。公开公告用固定合成响应覆盖并通过实际按钮确认，正文保留原文，不依赖线上内容。

实际命令为 `PTB_POLICY_FRESH_PG=1` 下执行 `.venv/m14/Scripts/python.exe -B -X utf8 -m pytest -o addopts= backend/tests/test_app_i18n_web.py -q`，1 项通过，2 条既有 FastAPI/Starlette 弃用提示。报告 `output/validation/app-i18n-web/account-browser-4d2b122cbd664ef384e83b0662b272f5/report.json`，自有浏览器、HTTP 和 PG 已退出，scratch 回执为 deleted。适配期修正语言 helper 变量、三下拉框误合计、公告异步弹出和数值序列化的测试依据，未降低业务断言。此轮为源码集成，最终完整 Web 候选另验。

同一实际流程补入 Worker 发送次数及 CDP 性能检查后再次通过。最新报告为 `output/validation/app-i18n-web/account-browser-0757221a0ee443e5b31f0767126f2dad/`，八组行为、零浏览器错误。15 次已预热切语后，DOM 节点803→803、事件监听210→210、Worker发送保持1次，EN/JA语言包各只加载1次；主动GC后堆14,804,136→15,167,620字节。`performance.json`中的时长包含偏好保存及其去抖和HTTP往返，不能当作纯绘制耗时或最终EXE启动指标。原生语言/桥/手册/缓存相关测试合跑30项与15子检查通过。

## 实际MFA、原生错误分支与阅读器窗口回归

M11两语各13张完整Qt截图逐张目视后严格登记，资源局部从实际model/dictionary DOM框裁切并加定位红框，各14/14齐备。报告为`output/validation/m11-manual/en-f22918e8e7e44cc0a5d8f4aeadfed44c/report.json`与`ja-18f755128fa1421baab5bb03678f2f0e/report.json`。实际MFA3.3.8运行资源自检、LAB/TextGrid格式重载、4个a1词对齐回读、两结果原子保存及OOV失败日志通过；环境注册表、模型、词典和合成输入原始SHA保持。独立工作区和scratch退出清理完成。中文旧两图文件名与当前实现不一致，已按同配方更新当前源并保留旧图证据，两语登记绑定新中文SHA。

原生错误审计扩展到条件表达式和字典返回值，补MFA name/format、频谱六类失败及fallback、参数读取和旧PKL转换等。当前525条三语同源消息，生成与review pending=0。`test_native_bridge_errors.py`执行真实TaskBridge分派的两种频谱操作×八种HTTP错误，保持原始错误、配置对象、输入文件名和单次请求，附已登记display描述。与native_i18n合跑25项/2子检查通过，前端16项检查显示已有失败随切语更新而不重试计算。

M18全屏退出丢失最大化已由独立真实pointer→Chromium→Qt闭环复现：进入前保存状态正确，退出后引擎后续showNormal覆盖下一事件轮的showMaximized。`host.py`在该次退出generation内有限确认目标状态，两次稳定后结束。英文`output/validation/reader-fullscreen/en-66ce0fc014894b2dbbe432eb30467e7d/report.json`、日文`ja-5b86c48126b2498ba700293bcbb08ae1/report.json`各最大化/普通原尺寸/最大化三轮通过，退出后持续1.5秒状态稳定。独立probe未替代完整论文流程，完整M18对应阶段也已通过，后续流程继续；Mac未新增运行验收。

## 参数估计全流程、来源说明与候选身份核对

M01两语各26/26实际界面及局部已逐张目视严格登记。英文`output/validation/m01-localized/en-d9725d7e250e491f80fbbe141d85234b/report.json`、日文`ja-ea01af5293e74a4da35b0397811b18e8/report.json`覆盖选中/全列表真实REAPER计算、XLSX/SQLite回读、TextGrid区间语谱图、缺关联拒绝、同源22段切分和Source_Time_s、EGG低通非法实际失败、原始周期/对齐两种输出及gF0列、重开宿主后重新授权目录保存既有结果且不重提计算。原始四份授权输入和TextGrid摘要保持，26.6MB测试EGG与所有可重建输出均在已退出的自有scratch中清理，回执deleted。首次缺REAPER显式参数、首次恢复窗口尺寸与辅助配方漏选音频等采集错误保留失败诊断，不计通过。

两张干净REAPER对话框与无遮挡角色父图分别为`output/validation/m01-native-details/en-844dc4ef7aa346c39a1f59cd1f82e29f/report.json`、`ja-cec95566a8df40b1895db03527124952/report.json`。F0非法范围、REAPER、EGG角色局部由当前语言实际DOM框派生，包含红框定位，未替换控件或数值。正式图和派生分别经过来源摘要门禁登记。

来源公开面板和M07归属说明接入71个项目说明字段三语词典，涵盖许可摘要、适配、引用附注，当前值必须与原来源注册表一致。原始作者、论文标题、引文、版本与URL原样保留；未识别或已变化字段回退原文并由生成检查拦截待审。新增`source-notes-i18n.test.ts`实际检查71字段、英日译文、未知/变更回退和注册表字节保持，1项通过，类型检查通过。

实际Web集成与科学流程的测试入口增加`PTB_VERIFY_FRONTEND_DIST`显式候选目录，测试前后逐文件大小/SHA和树摘要写入`frontend-input*.json`并要求不变。不提供时在自有scratch重新构建，不复用历史web-launch目录。新增帮助代码及两个入口语法检查通过；最终冻结候选的实际浏览器/科学运行仍待执行，不能将本轮语法检查记作候选验收。

公共错误尾审修复`SpectrogramViewport`将已格式化字典值存入messageRef、`ProjectStorage`将LocalStorageError.message存入notice的问题，保留原descriptor；三个本地打包验证也直接传descriptor。更新通道补64条原生产品提示与地区标签，UpdatesError保留code并继承可重译错误，返回的来源HTTP状态附带display，正文/版本/哈希和原始失败信息保持。真实MemoryAssetDatabase版本冲突单次失败、更新通道单次失败与各语即时显示共8项Node通过，原生更新70项通过，类型检查通过。后续模块错误边界仍由各负责代理收尾。

重复预览源码检查`output/validation/desktop-preview-performance/app-i18n-final-source-cce49a97c57d4ba0823d81cda94dcceb/report.json`：同输入连续四次语谱图SHA一致，科学进程交换0；同一录音预览只扫描1次、频谱1次，四次输出摘要一致。这是Windows源码服务检查，不能替代最终EXE性能。最终可见成品专用脚本`verify_app_i18n_package.py/.mjs`已备，计划57次模块打开、已有校验错误切语、同转换输入保留、完整两语ZIP实际导入及重开逐章离线读取；当前只通过语法检查，未计成品通过。

冻结只读审查发现`copy_runtime_frontend`遗漏新手册下载清单。现验证清单schema并按原字节纳入冻结，仍只复制正文引用的中文资源，不附外语ZIP；`tests/test_frontend_freeze_manual_languages.py`实际冻结两个检查通过。实际语谱图SFC回归执行其完整setup和run，六种已知/原始错误只调用一次loader，已显示错误三语切换后即时更新，1项通过。手册/原生语言/桥/缓存合跑47项、17子检查通过。

预冻结全量前端854项中851通过。新缓存测试使用Node24不支持的构造参数属性、M08计数断言旧文案已分别由C/A修正，未改Node选项或业务规则；地图一文件进程失败无堆栈，单独重跑三项通过。此时仍有同期写入，最终冻结前须全量重跑，不把独立复跑扩写为全量通过。文档检查本轮2766文件/462来源/195任务errors=[]，历史原样快照的旧链接单列。

## 1210冻结阶段记录（后续结果见文末）

三语界面和647条原生提示完成尾审，英日各22章6230单元、680张界面图和6张科学导出已登记，两完整语言包摘要核对通过。默认中文开发树按架构保留1128资产，发行冻结仅保留383引用资产，sourceDigest为2c2f8e7a0015d24d60c5309144fc8f611a9d98d6ad6bfe3c62432d484f9097a4。实际Chrome全断网读取两语全书、40项共享媒体及例音解码、卸载重开帮助、新译文更新和坏缓存补存通过。完整前端869项和类型检查通过。正在构建1210冻结Windows候选并以同份冻结前端执行真实Web验收，最终成品与性能验收尚未结束。

- 完整候选包：`output/manual-packs/3.1.2-manual-20261030-2c2f8e7a/candidate-manifest.json`。EN 120528578字节，SHA `18c802cdc8df6ee0e684d673d2111f31a5cdd29815d1ea1c801c48dce698fa79`；JA 114968290字节，SHA `dd8fea99a61f1aefa1bd549087212168335fda688df96484c72b28315a0c9942`。每语343覆盖映射到338独立PNG，逐文件SHA由构建回读；Root另核整包SHA、22章与映射数。
- 全书当前读回：`output/validation/manual-language-packs/complete-readers-262044e1ef034161aedd9f59a2b6bf18/report.json`。桌面按既有规则隐藏Web入门章，真实EXE需读21章，Web需读完整22章，不能将桌面过滤误判为缺章。
- M10正式资源表79项生成及check、native647/pending0通过。真实VTL Web原功能与新拒绝信封合跑14项通过，保留2条既有httpx/AnyIO弃用提示；音频/worker路径23项通过。首次直接pytest缺tests/support或统一科学解释器绑定只属于环境未按入口配置，已经统一workbench_source.configure重跑，不计产品失败或未修bug。
- 最终EXE脚本已按实际组件修正全屏按钮、同值语言保存等待、桌面章过滤与CDP表达式；媒体期望从ZIP正文导出稳定assetId，不允许异步0张图片跳过。将实际滑动/解码、音视频metadata、已有错误切语、PDF往返、真实ZIP导入与重开。脚本准备不等于成品通过。
- 仅autoReceive:false仍有论文启动元数据请求。已只读确认公共urllib继承环境代理、Qt无覆盖、本机API专门直连回环；新增本次子进程专属拒绝代理，真实HTTPS CONNECT与真实UpdateService均在本机502拒绝且不转发，2项通过。最终Windows只可据实际记录声称所审HTTP(S)出站被拒绝条件下的本地阅读，不能称操作系统或任意socket全部断网。
- 中文重复图实际2组、额外584794源字节。冻结时同输入/编码配置只编码并存储一次，保留各assetId/caption/原始SHA和无损像素，audit按唯一文件统计实际字节；3项真实冻结检查通过。原PNG不改。

## 完整手册离线与缓存尾验

最终源码真实Chrome报告为`output/validation/manual-language-packs/complete-web-offline-363b4aa5746c4fc7ad4f69c521c07a02/report.json`。两个完整包各22章，343覆盖图与40共享项逐Blob摘要回读，图片实际解码，34例音实际AudioContext解码；整个浏览器context离线后关闭和重开帮助可恢复精确匹配的索引。来源9文件始末SHA保持，778个对象URL均回收。只声明已经打开的网站离线阅读，不声明网站离线导航启动。新译文版本及坏MP3定点修复实际UI报告为`output/validation/manual-language-packs/browser-318fff2d6f9d4deebd4f2259555a5ab4/report.json`，27项定向Node通过。单帮助页安装锁顺序安装EN/JA共享资源仅取一次，不声明多窗口并发全局合并。

开发中文完整树1128资产按架构契约重建，发行冻结仍只选383引用资产。`scripts/manual/register_reader_resources.py --check`同时核对作者源、生成字节和原登记摘要，未改变任何来源pin；真实失配拒绝回归1项通过。当前架构和文档检查无错误。首次1206冻结因最后英文音频元信息修订而主动停止，未交付，1210已用新前端重新冻结。

## 1210阶段待验项目（已由后续验收覆盖）

1210候选正在构建。待最终Windows免安装/安装/更新ZIP及默认GPU可见详细试用、独立实际HTTP/PG/科学Web候选、启动和资源生命周期检查。当前不标verified或published；所有失败保留诊断并查根因，最终字节变化后按规则复验。Mac只同步源码，不声称构建或实机通过。

## 1210冻结候选首轮运行与测试环境修正

Windows应用已生成，EXE为498960888字节、SHA `2a2c159b49521c67c1a1e0b90462877f85cc394ffde964bffd57cc3995c5b99c`。实际安装包498877021字节，更新ZIP498961454字节，均小于500000000，内容/嵌入归档/ZIPCRC审计通过。首轮最终16阶段正在执行，已通过脱离开发环境的distribution阶段，不能扩写为全部通过。

最终冻结前端实际Chrome/Edge账户语言报告为`output/validation/app-i18n-web/account-browser-c76baa7643854312ac0e284eb3619ca1/report.json`，19模块×3语实际打开、M10原iframe保持、账户跨浏览器语言/参数恢复及隔离通过。15次预热切语DOM803→803、监听210→210、Worker发送仍1，英日语言包各加载1次，主动GC后堆15043076→15422032字节。前端逐文件及树摘要前后保持，见同目录`frontend-input*.json`，没有以历史构建替代候选。

首轮新增模块通用断言误把iframe父section的innerText作为M10正文，已改为真实frame body/ready/lang/Frame身份。首个server fallback提示由实际确认按钮关闭。独立科学Web测试缺既有011账号偏好迁移，补齐后又暴露未确认首次公告的测试流程遗漏，现固定合成公告并经真实按钮确认。仅调整自有新数据库夹具和验收脚本，未修改业务源或冻结成品，原失败日志保留；科学Web最终复跑仍待结果。

## 1210最终冻结Web完整说明书

`backend/tests/test_manual_candidate_web.py`明确要求同份冻结前端目录，通过真实Chrome账号登录、主页帮助和本地filechooser导入两个实际完整ZIP，再以context.setOffline(true)关闭和重开帮助，逐22章核对ZIP正文导出的媒体ID/种类/重复次数。1项通过，报告`output/validation/manual-candidate-web/authenticated-offline-e194782320ea461a95d549032bccb1b1/report.json`。两语各343覆盖+40shared全部payload摘要匹配独立ZIP/实际压缩host清单，图片解码、34例音真实解码、视频尺寸/时长metadata通过，浏览器错误0，关闭帮助后的观察BlobURL不能再读取。没有注入产品store/state，没有使用源码harness。

冻结前端613文件，树SHA `17c7769ffcafaaa6b8e2d6867597b2172fdb086554450c3f2df65f731e4bd3d0`，前后保持。该试验只证明已经打开的网站离线阅读，不包含离线地址导航启动网站。保留2条既有httpx/AnyIO弃用提示。实际截图已目视，英日正文与界面无缺字。

## 1210中间Windows验收与科学Web发现

1210实际EXE完整16阶段通过，报告`output/validation/local-package-710854e7/report.json`。覆盖11科学普通启动、长路径安装卸载、缓存锁/取消/重建、便携及安装更新交接和重开、默认GPU可见168字体滚动帧、重复工作、阅读器与实际像素。此包后因Web科学验收发现业务问题，将被新冻结候选替代，不能将这份中间字节的16阶段结果冒充新字节验收。

实际EGG批量科学任务完成后，`batchSubmit`把含`font/1`导出字体快照的frozen对象存入batchConfig。useWebSettings随后同步`batchConfig.font`，后端现有白名单拒绝422，页面刷新重放pending时无法恢复账号。实际请求/响应见`output/validation/web-launch/compute-chrome-4d68354ed94d4470917959cdff0541e9/failure.json`。修复限定保留任务字体快照、从参数持久化及旧待保存记录中剔除这一瞬时字段，不放宽服务器字段校验，不改变算法或数据。修复与旧残留恢复回归尚在执行。

另将英文在线人数改为中性`Users online: {v0}`，避免单人数时语法错误，account review pending0。最终须按修复后冻结新成品重新验收。

## 1255修复后重新冻结

M03仅把导出font留在任务frozen，持久化参数使用原parameters；旧本机batch配置载入和旧账户session/local pending恢复仅剔除`M03.batchConfig.font`，所有科学参数、其它模块与未知校验错误保持。新增4项实际SFC函数边界和账号pending回放回归，与原7项合计11通过。修复后完整Node873项、typecheck、i18n11346/pending0及架构检查通过。业务源冻结后以新的1255名称重新构建，并用其snapshot/frontend/dist重跑实际Web账户/完整手册及科学流程。

1210附加EXE脚本已通过57模块打开、15次切语保留输入和三语既有F0校验提示，后因测试漏加载native虚拟namespace停在更新提示预期取词。补读canonical native JSON，并将关闭后的CDP清理等待改为1.5秒有界，语法通过。原失败报告保留`output/validation/app-i18n-package/final-exe-bd974691296c4b4e8c6c892337fea44d/initial-report.json`，owned node按准确PID/路径结束，EXE及scratch由wrapper收尾，cleanup=deleted。该轮不计完整附加验收通过。

## 1255实际Web通过与原生手册图片发现

修复后最终Web合计5项实际运行通过：账户与完整手册2项见`output/validation/app-i18n-root/web-1255-account-manual.log`，Chrome/Edge原生科学及受管API3项见`output/validation/app-i18n-modules-b/final-web-science-1255.log`。实际科学报告为`output/validation/web-launch/compute-chrome-baf8fce1dbb947e5bc77a81a236f9aa0/checks.json`、`compute-edge-702404ff2d4e474e8e5ff61be05918c9/checks.json`，三计算/原生REAPER/下载/取消幂等/离线重连/刷新/退出全过，Python科学结果回读与退出数据库、配额清理通过。613文件树SHA `8440d3c1fc1fb9892eeaeb84374d90da35482114e7f2cf176a4b4854bf248455`，前后保持。对应账户报告`output/validation/app-i18n-web/account-browser-b2d8919cd836482989057dcdfaef8852/report.json`，完整手册`output/validation/manual-candidate-web/authenticated-offline-a91c10f6fb094e69b20eeeb720250e3d/report.json`。

1255新EXE498974057字节，SHA `00d295284a9760dbb1f271c5d929d2e6924d298349d97cd5440c7019159f343e`，内容审计通过。实际附加试用57模块三语、15次切语无新科学调用、已有F0校验/更新失败三语和PDF6次全屏退出保持均通过，但完整外语手册首图失败，不能计完整通过。报告`output/validation/app-i18n-package/final-exe-1667e1886dfc4eb39d073b3e9fa642fa/initial-report.json`及同目录截图。

最小原因已从源码定位：桌面origin为ptbapp://app，URL.createObjectURL产生blob:ptbapp://app/...，safeMediaSource只允许blob:HTTP(S)而将其过滤成null。修复将精确限定本应用origin与UUID，不扩大至任意协议/host，并做实际Qt与最终EXE回归。1255仍为中间候选，最终交付需再次冻结该修复，不能将Web通过替代桌面验收。

## 1310桌面Blob修复后冻结

最小产品修复仅content.ts允许精确的`blob:ptbapp://app/<8-4-4-4-12十六进制UUID>`，其它host、port、credentials、query/hash、路径、blob:null、file或裸应用协议仍拒绝，HTTP规则保持。新增正反例后16定向与typecheck通过。真实native ZIP读取同一PNG394906字节，原始Blob可直接解码2560×1600但旧生产过滤返回null，修前报告`output/validation/manual-language-packs/desktop-blob-142d9e4e45a3477cab2f972655cca188/report.json`；修后经当前生产过滤实际Qt解码成功见`desktop-blob-cb4e908fe1b047e2be4e0ead00e6e5f0/report.json`。这些为独立Qt证据，未冒称最终EXE通过。

content.ts于13:05:11冻结，SHA `01162f423f6b7ff42cbdc95202694fc5f2424e98e8b02995e743b72c431be3f2`，随后全量前端874项/类型/i18n11346及架构检查通过。1310使用新前端重新官方构建，两组实际Web测试正在绑定新的冻结目录；最终EXE三语完整手册、16阶段、地图背景与启动对比待运行。附加EXE脚本补收CDP Network真实语言JS请求，若Qt资源计时缺项可保留事实并用请求事件验证一次加载，仍要求EN/JA各一次，不把缺项视为重载。

## 1310冻结Web完整实际通过

新冻结前端613文件，树SHA `c643cc0c1e87d5eaf02285add08d245380390ce1b62a131e1ba6f4e5c72da864`。账户及完整手册两项通过（106.30s，2条既有弃用提示），报告分别为`output/validation/app-i18n-web/account-browser-de5d416822ab420a801cdab57b80a8ad/report.json`、`output/validation/manual-candidate-web/authenticated-offline-8ba42e2f3b004f2dab2b68cf28fe0582/report.json`。同份最终ZIP、各22章全部媒体SHA、图片/34例音解码、视频元信息与断网关闭重开通过，758个实际观察BlobURL均在关闭后不可读，两个fresh PG和浏览器均已清理。

Chrome/Edge科学及受管入口三项通过（176.63s），`output/validation/app-i18n-modules-b/final-web-science-1310.log`。对应`output/validation/web-launch/compute-chrome-3c33a56170b54121a82060208d5b1160/checks.json`、`compute-edge-ec1a4fc7ecdc462797740ce1c57a43f9/checks.json`及`preview-entry-b97cccbc0e484f00ab5f1444101e7b0d/report.json`，三类计算、原生REAPER、下载、取消幂等、断网重连、刷新、退出、Python数值回读、数据库和配额清理全部通过。各前端前后613文件摘要保持，三份scratch均deleted，测试源和产品源未变。

1310 Windows实际EXE498976085字节/SHA `fbe87f91b478ebf3d326a4e45462829c7cb6fec3cf974dbb5b254593a6b2dd0e`，更新ZIP498976651字节/SHA `51a76616c2f94444f53ff4da55c804c79290d19b90b007bcf9a2006f857a5071`，安装包498842744字节/SHA `e7678864b5b78b4003a0faa18bf9c0f1525a0df76be2642100a00bc6084d013b`。归档内容、Qt/科学身份和三文件上限审计通过。官方资料打包仅复制16份现有运行数据90540566字节到同目录地图库，原始书籍/扫描/审计没有复制，没有公开上传。最终字节Windows验收尚在执行。

## 1340双包索引修复与当前最终验收

实际1310保留现场证明JA已正常安装，索引85966字节（每语343覆盖），旧读取硬限65536使安装后的load/list失败。该失败与CDP文件输入无关，未执行的fileChooser替代方案已撤回。修复只改原生manager及其测试，索引读写共用`(MAX_MANIFEST_BYTES+4096)*len(LOCALES)`，当前8008192字节；在原子提交前检查实际UTF-8字节，维持现有格式、SHA/路径校验和已有包兼容。17定向通过，包含两语大索引安装与同/新Manager重开、真实超限读前拒绝、提交超限保留旧index及无tmp，冻结13:33:22。失败现场原记录保留，原先临时文件锁造成inventory失败，待所有自有进程退出后用原verification_artifacts.finish重新盘点并正常清理，cleanup=deleted。

1340新冻结EXE498969662字节/SHA `5e64600e6934f95ba4c83b89cd0ca3a0ad3211b2dff14137980301ecacbd2abc`，源码快照SHA `45ad3dc1c6c4717b3d0f89f5f81a0c4f39b40453f8b186b2eab86c8d8726ade2`。安装包498815811字节/SHA `1afb96b3515e6837ba5eeb3931ac14c29b11d90b435762f4951cbdb972b4c05d`，更新ZIP498970228字节/SHA `07ef4cbbf72920086055920b12b3535de9e65108419f7690c40fa50656f1c853`，上限与归档审计通过。

绑定1340的实际Web五项再次全部通过，同一613文件/c643cc0c树摘要保持，源码和测试未改，所有自有浏览器/PG清理。账户报告`output/validation/app-i18n-web/account-browser-c87a27e9ebfd478eb53a1467d1422469/report.json`，完整书`output/validation/manual-candidate-web/authenticated-offline-58b3bc4bda2045769c0419f13ef15f6f/report.json`，Chrome/Edge科学分别`output/validation/web-launch/compute-chrome-6d2aba8736274d68a93167ab2b2002b3/checks.json`、`compute-edge-b04e8f48395041349d33377d9b0c61bc/checks.json`。受管入口`preview-entry-ca691c902d6e451d978b5d3717876b06/report.json`也通过。

1340实际Windows首轮57模块、15切语保持、F0/真实更新失败三语、6轮PDF全屏、两语各21章全图片/音视频metadata全部通过，报告`output/validation/app-i18n-package/final-exe-e353ccbefafb4e45aec7e02ab33c08fc/initial-report.json`。Qt resourceTiming为空，真实CDP Network记录en/ja JS各一次，渲染器为默认Intel ANGLE/D3D11。该轮正常退出等待超时，尚未计整个附加验收通过。验收driver对照beforeunload离开确认补入明确的关闭处理，再验证正常退出/重新启动；已装JA启动可能早于调试连接，重开阶段以Debugger.scriptParsed补全已存在脚本并核对后续Network无重复，保留计时缺项事实。未修改产品源或降低业务断言。最终16阶段仍运行，后续地图和启动性能待执行。

## 1340最终Windows完整结果

`output/validation/local-package-4cf43dfd/report.json`完整16阶段成功，实际最终EXE SHA为5e64600e…，脱离开发环境普通启动与11科学任务、长路径安装卸载、缓存取消/锁/重建、便携及安装更新交接和重开、可见默认GPU168字体滚动帧、重复工作、阅读器与像素均通过。单独只读Computer Use也观察同一私有最终窗口实际IPA密集页，未输入或改动其它用户窗口。源码/隐藏Qt/其它电脑证据没有替代此成品。

三语与全书最终报告`output/validation/app-i18n-package/final-exe-92b20cbfcfc6445c814912a5d25274d1/report.json`成功。initial/reopen两次实际普通启动、初始中文与保存日语恢复、57模块打开、15切语输入/转换保持、F0已有校验及真实网络拒绝更新结果三语、6轮PDF全屏保持最大化、完整EN/JA ZIP真实导入，各语21桌面章全部图片实际显示及音视频metadata、退出后再次全章读取通过。Qt ResourceTiming缺项，初始CDP Network en/ja各一次；重开期Debugger回补保存JA的已存在脚本，并核查后续请求无重复。默认渲染器Intel ANGLE/D3D11，未禁用GPU。

前两轮正常退出driver漏处理Qt原生beforeunload确认，非全书失败。最终按真实标签关闭按钮逐个关闭20个自有测试标签，地图测试草稿经界面的日文丢弃按钮关闭，已保存M13草稿和语言包保留，再普通窗口退出。两进程均退出0并重开，未清空用户数据或绕过退出保护。该轮scratch已deleted，早期失败诊断保留且自有scratch也均已清理。

最终地图绘制报告`output/validation/geolinguistics-background/exe-41d1b8a20dae4dae8a74654cb91419ce/background-report.json`及`exit.json`成功：12种主题/背景/单双四图组合、48手势、24PNG的透明叠层/底图像素和比例尺对比度检查通过，实际EXE正常退出0，scratch deleted。只使用已授权的16份现有运行数据，未复制书籍扫描或公开上传。

启动对比正在独占运行，使用旧3.1.2与1340实际EXE、交替各3次新缓存，随后3次暖启动和跨构建复用。明确隐藏窗口/软件渲染只衡量就绪、未清操作系统缓存，不代替上述可见GPU验收。当前尚未计启动通过或交付完成。

## 最终启动性能对照

`output/validation/package-trim/startup-5a9a655778f142dca00b22c93039d807/report.json`性能门槛通过。旧/新各3轮按交替顺序使用新应用缓存，旧中位41840.9059ms，新41765.6021ms，差值-75.3038ms；允许增量2092.0453ms。该差异不作显著加速结论。新包三次暖启动8638.8633/8650.5241/8304.1292ms，科学、宿主和应用组件全部复用；跨构建19677.6993ms，只准备变更应用内容、复用science/host。每次内容校验保持，没有凭版本号/mtime豁免文件散列。

计时采用隐藏窗口/软件渲染的就绪信号，操作系统缓存未清，期间没有并行大规模构建/测试；不能当作可见窗口绘制时长或其它机器保证。默认GPU可见实际帧另已通过。性能运行已完成逐文件盘点，14:27:44清理自有临时缓存并退出0，随后才执行旧构建清理，未破坏比较输入或活动进程。

## 最终本地交付与清理

本次本地交付范围已验证，当前有效构建为`PhoneticToolbox-312-i18n-20261030-1340`。最终回读`output/validation/app-i18n-root/final-delivery-readback.json`为success：三件成品的大小/SHA与package-report一致，ZIP CRC和内含EXE/应用元数据相同，当前运行源码与冻结快照一致，613文件前端树摘要为`c643cc0c1e87d5eaf02285add08d245380390ce1b62a131e1ba6f4e5c72da864`，两个完整说明书ZIP摘要与候选清单一致。正式包名仍为本地3.1.2候选，未选择新公开版本。

| 交付物 | 实际位置 | 字节 |
| --- | --- | ---: |
| Windows免安装EXE | `dist/PhoneticToolbox-312-i18n-20261030-1340/PhoneticToolbox-windows-x64.exe` | 498969662 |
| Windows安装EXE | `output/release-staging/i18n-20261030-1340/artifacts/PhoneticToolbox-3.1.2-windows-x64-setup.exe` | 498815811 |
| Windows更新ZIP | `output/release-staging/i18n-20261030-1340/artifacts/PhoneticToolbox-3.1.2-windows-x64-portable.zip` | 498970228 |
| Web编译候选 | `output/build-PhoneticToolbox-312-i18n-20261030-1340/snapshot/frontend/dist/` | 613文件 |
| EN/JA完整说明书 | `output/manual-packs/3.1.2-manual-20261030-2c2f8e7a/` | EN 120528578 / JA 114968290 |

最终Windows16阶段、附加三语与全书、默认GPU地图、启动对照，以及绑定同一冻结前端的5项实际Web验证全部通过，前文报告给出实际命令、输出和边界。重复切语未引入新科学请求，预热后DOM/监听器803/210保持，英日资源各一次；完整Web手册关闭后的758个已观察Blob URL均释放。完整前端874项、类型、原生双包17项、翻译/资源/架构检查通过。最终文档校验2777文件/462来源/195任务errors=[]，`git diff --check`通过；原样历史快照36条旧断链单列。结果见`output/validation/app-i18n-root/docs-check-delivery.log`和`diff-check-delivery.log`，未把历史断链当作当前文档有效链接。

所有本次实际运行的独立浏览器、隔离数据库、Qt进程及运行scratch均已结束并按各报告清理。按既有`release/cleanup_old_builds.ps1 -KeepBuild <1340目录> -KeepStage <1340暂存目录>`清理12处已识别旧成品、构建和安装暂存，保留当前成品/输入及必要诊断。`dist/PhoneticToolbox-312-i18n-20261030-1340/cleanup-report.json`另列10处不完整历史构建，依安全保护原样保留，未扩大删除。

3处旧前端输入目录`output/app-i18n-final-frontend-20261030-{1206,1210,1255}`共约1.498GB逻辑字节已完成路径、联接、占用及内容清单审查，但直接删除被自动审批以`blocked by policy`拒绝，没有执行。现有安全清理脚本不覆盖这些位置，未修改其范围或改用其它手段绕过。保留计划和`output/validation/app-i18n-root/obsolete-frontend-cleanup.json`回执，当前有效原始输入`...-1310`也保留。此清理例外不影响最终成品或运行性能。

未公开发布、部署、push或向外部发送内容。外语手册下载机制已实现，生产清单保持`published:false,packs:[]`，当前以实际完整ZIP本地导入交付。Mac仅共享源码，未构建或实机验收；未新增实体音频设备、独立干净系统、Safari及移动端验收。Web离线证据覆盖已打开页面的阅读器关闭/重开，不包含离线地址导航启动整个网站；桌面网络限制为本次子进程HTTP(S)拒绝，不宣称OS或任意socket断网。公开发布仍须另行确认对应版本及完整更新文案。
