# 发声类型合成窗口尺寸跳变

核对日期：2026-10-13。范围为 Windows 当前源码、独立 Chrome、实际 Qt WebEngine与最终R5免安装/安装成品，两件EXE已在服务器和GitHub同版本替换。

## 截图与定位

用户提供的两张截图均为 2559×1527。RGB 差分边界为 `(760,140)–(2533,1366)`，最大通道差超过 16 的像素有 261673 个，占 6.7%。差异集中于中间四图和右侧任务栏，左侧参数栏未移动。预览提示在单行和换行之间切换，图窗高度随之调整。用户确认问题随窗口大小出现，当前使用安装版、楷体 15 号。

只读核对相关界面偏好后，测试采用导航栏 184、模块左栏 294、右栏 357 CSS px，中文楷体 15 px、拉丁 Times New Roman、图表宋体 12 px。不读取研究语料，不修改用户偏好。

默认及细密尺寸扫描未稳定复现截图中的持续往返抖动。已确认一个对应的宽度变化机制：工作区出现垂直滚动条时，工作台宽度缩减 9 CSS px，右栏横移 9 CSS px，四图随之变窄。图窗内部滚动条也会改变图宽。这与截图中的位移及预览提示换行一致，但尚不能据此断言已完整复现安装版的持续反馈过程。

## 修改

`frontend/src/modules/phonation-synthesis/PhonationSynthesisPage.vue` 仅为本模块的 `.m07-workspace` 与 `.workbench-center` 设置 `scrollbar-gutter:stable`。垂直滚动条出现或消失时保留相同内容宽度，消除滚动切换参与的横向尺寸反馈。公共布局、音频处理、合成参数、算法与其他模块保持原实现。

回归脚本为 `scripts/verify_m07_resize.cjs` 和 `scripts/verify_m07_resize_qt.py`。浏览器脚本显式撤销 Playwright 默认的 `--hide-scrollbars`，以实际可见滚动条验证；测试以可控内容溢出穿越滚动临界点，再连续逐帧测量固定窗口的几何。Qt 脚本以临时合成 WAV 完成实际 F0 分析、九步同时改变 F0/发声类型的生成及 step09 试听，使用独立任务库、缓存与隐藏窗口，不改用户文件或运行窗口。

## 验证

- 修复前的可见滚动条测试失败：工作台宽度 1493→1484，右栏 x 为 1330→1321。该失败报告保存于 `output/validation/m07-resize-20261013/before-gutter.json`。
- 修复后工作区滚动切换时宽度及栏位相同，图窗内部滚动切换时图宽相同，差值均为 0。
- `node scripts/verify_m07_resize.cjs --kaiti`：42 组布局通过，七种窗口尺寸 × 浅深主题 × 70/100/150% 页面比例。每次固定窗口采样 36 帧，后 24 帧均只有一个几何状态，无页面异常。报告为 `output/validation/m07-resize-20261013/browser-report.json`。
- `npm --prefix frontend run typecheck`、`npm --prefix frontend run build` 通过。构建仍有既有 chunk 大小提示。
- 在 `frontend` 运行 `node --test tests/m07.test.ts tests/m07-r1.test.ts tests/m07-r3.test.ts tests/panel-widths.test.ts`：12 项通过。
- 实际 Qt 完成 41 个相邻窗口尺寸及同一滚动切换断言，最终 DPR 与清理状态见下列本机报告。

最终 Qt 证据：`output/validation/m07/resize-qt-041070e6f20e41b09a869b00eb3dbbd9/report.json`。截图差分及完整 PNG 位于 `output/validation/m07-resize-20261013/`。测试 WAV、隔离任务库及缓存按 scratch 归属清理，保留报告、参数、摘要与必要截图。首次测试的 DOM 对象返回值判断已修正为布尔判断，失败证据保留；自动收尾一度因活动子进程保留 scratch，退出后用既有清理入口复核并清理。

## 学术致谢与字体扩展

用户追加要求三项学术来源与移除可见 VoiceVista。`third_party/source-registry.json` 新增 Shue 2010 博士论文、VoiceSauce 2011 会议论文、凌锋等 2019《方言》论文，前两项与 UCLA 官方资料核对，中文引文由用户原刊 PDF 首页核对。用户名为2009的文件实际是42页ICPhS 2011讲稿，采用原文年份并保留官方论文和讲稿链接。论文原PDF不进入应用或Git。VoiceVista设置 `show_in_acknowledgements:false`，保留内部来源审计。BibTeX、致谢覆盖清单和生成UI数据同步。

浏览器三字体为宋体、微软雅黑、楷体，各检查10/12/14/15/18/20/24 px，三种窗口与浅深主题共126组；全部稳定，三条学术来源实际各出现一次，VoiceVista在软件组不出现。实际Qt同一字体/字号、三窗口63组稳定，实际DPR1.5。报告分别为 `output/validation/m07-resize-20261013/font-matrix-report.json` 与 `output/validation/m07/resize-qt-49da0b5387964162975b1fbf29f07220/report.json`。完整前端392项通过，首次两项来源测试因新增三项及移除一项后的预期/覆盖清单未同步而失败，已按实际登记同步，保留断言。

全项目架构检查发现既有M10资源摘要登记落后及未登记default-presets.json，七条错误保留，不修改本任务外M10。对照已发布R2冻结前端，七文件均逐字节相同。普通文档检查2006文件/427来源通过。R3的Python业务冻结仅web_assets.py格式差异，AST一致；后续R5另外纳入用户明确授权的卸载修复。

## 未验范围

原安装版的持续抖动未完整复现，已确认的滚动切换宽度变化已在源码和R5成品消除，实际成品字体/字号验证通过。用户原窗口与录音的再次复核、另一台电脑及跨平台仍未验证。

## 同版本应用替换与卸载修复

用户追加授权在完成字体与致谢修改后重新打包3.0.1，替换服务器和GitHub的两件EXE，其他内容保持；对应下载校验字段随两文件同步，现有ZIP、Release正文/标签和官网不改。R3最终EXE的工程外Windows-only PATH分发、63组真实Qt成品字体布局与致谢检查通过。隔离OwnedQA安装/缓存准备通过，卸载失败：Inno 7.1.0的RedirectionGuard继承至子进程，清理入口仍初始化科学运行时联接，`native_runtime._directory_link`报WinError448。用户明确选择先修复卸载再重打包，不豁免该失败。

卸载修复保持Windows安全检查和普通启动的科学运行时核验。`v3_local_preview_entry.py`为清理主入口及其专用本地清理服务跳过不需要的科学初始化；`cache_cleanup.py`经`LocalService(cache_cleanup=True)`使用该内部入口，现有结果缓存清理协议和正常任务服务保持。R3失败与候选身份留证，不公开上传，后续候选须重新冻结并完成实际卸载。

专用清理服务同时停止启动计算worker，回环接口仅允许GET健康与POST受管结果清空，Host/Origin/token校验保持。37项清理入口、服务、租约和结果保留测试通过。`verify_cached_install.py --with-workspace`增加可重建的已完成结果夹具；首次夹具遗漏清单条目触发`output_not_committed`，原失败留证，补齐已提交清单后原断言不变，重跑成功。

最终候选为R5，应用495831923字节、SHA `5dcde8fc895bf776826b12482f57e00850bfa56b8e46140465b8719d0c051174`；安装497748262字节、SHA `303da3562fe768be52a0a2dfc389794720a7ab8be2e818b635faf5f1440d9553`。源码快照SHA `61cf46f40d5c6eed2a1271a0dc6eb3ace6bf5f4c790435876323a1c857826202`。同一应用生成两件EXE和本地验收ZIP，线上既有ZIP按范围保留。

R5成品校验：20个主Qt DLL、共享/科学压缩包与MFA排除、500MB约束通过；工程外Windows-only PATH的科学探针、11任务、18帮助、真实GUI字体63组与三论文/VoiceVista可见性通过。真实OwnedQA安装/预热与卸载成功，19字节已完成受管结果确实清除，研究/设置摘要保持，活动租约拒绝卸载，安装目录及小型启动临时目录无残留。免安装/安装两个真实helper与下一进程、草稿/主题/文件标记、九音频互斥播放及图片放大通过；较旧版本号仅是测试元数据，目标包实际为3.0.1。

成品报告与日志保存于 `output/validation/m07-resize-20261013/qa-evidence/` 对应的R3/R5测试目录，两个换版回执为同级 `portable-handoff/relaunch-report.json`、`installer-handoff/relaunch-report.json`。测试用OwnedQA身份关闭正式卸载登记和快捷方式，不更新井井当前安装或数据。公开替换开始前保存服务器/GitHub两应用、校验文件与ZIP身份，替换清单为同级 `replace-plan.json`，远端原文件保存在私有回滚目录。

服务器六个文件的原子替换与完整摘要回读成功，GitHub四附件替换成功。`replace-server-receipt.json`、`replace-github-after.json`、`replace-final.json`记录最终身份；`public-https.json`直接流式读取公网两EXE的完整字节及四个校验文件，大小/SHA全部匹配。固定版本与两频道清单SHA为 `da1073901600beff2b7a3311a1baf2a84d44aa003f605a7e7259b03a453460aa`。首轮上传为三个相同内容的JSON共用临时文件，第一份发布后临时文件消耗导致后续缺失；改为内容摘要加目标路径的独立临时名，跳过已到位字节后续传成功，首轮日志保留。没有改服务器系统或服务配置。

现有线上ZIP仍为原R2字节，两个EXE为R5，这是用户限定仅替换两应用的范围。Release正文、标签目标、名称和发布时间逐字段保持；本轮源码未提交或推送。原安装版持续抖动未完整复现的边界保持。QA副本、合成WAV、隔离缓存与QA安装器按逐项路径/进程/联接检查后清理，必要报告、截图及摘要保留。共183376个文件逐项留存摘要，含101616个独立文件身份，硬链接只减少重复读取；原PowerShell摘要扫描精确停止后在Python继续，不跳过最终联接/进程复核。2026-10-13 03:13清理回执为 `deleted`，scratch确实不存在，见同级 `qa-cleanup/cleanup.json`。当前R5成品、构建缓存、失败R4诊断及原始用户文件保留。
