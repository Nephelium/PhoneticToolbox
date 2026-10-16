# PhoneticToolbox 官网

静态产品介绍、正式下载和试听页，源文件为 `index.html`、`site.css`、`site.js`。不承载应用登录、计算或文件上传。

- 本地预览：`python -m http.server 8765 --bind 127.0.0.1 --directory temporary-site`
- 品牌与装饰构建：`python scripts/build_website_assets.py`。首次取字体需联网，仅从 Google Fonts 官方仓库读取并保留 OFL。
- 当前应用截图：`node scripts/capture_website.cjs`。使用当前 Vue 源码、独立 Chrome 配置、公开示例文字，不读取用户任务或素材。
- 页面验证：`node scripts/verify_website.cjs`。
- 三张连续长图：`node scripts/export_website_long_images.cjs`。从公网首页截图，在区块留白处分界，报告与PNG保存到 `output/validation/website/long-images-*`。
- 资源来源、许可、主题及截图身份见 `assets/provenance.json` 和 `assets/licenses/`。

顶端版本及两个下载按钮统一为3.0.2，链接经服务器完整摘要核对的免安装版与安装版，提示文件大小同步。assets/downloads.json保留实际字节数、摘要与来源，构建不会退回旧链接。网页版预计于 2026 年 10 月 31 日发布，目前本站提供产品介绍、声音试听与支持内容。静态发布不重新上传安装包。

滚动使用原生页面滚动、IntersectionObserver 和 requestAnimationFrame，没有滚轮劫持。减少动态效果时取消漂浮、视差及入场变形；无 JavaScript 时正文、截图与导航保持可见。点击截图可查看完整原图。

IPA 装饰参考 Doulos SIL 的字形结构，重新绘制圆润的单线条、浮雕层与阴影，保留辨音所需钩形；它们是图形资产，并非研究结果。语谱背景为确定性装饰图形。新增截图使用实际界面，不用设计图伪装软件结果。

首页标题六字使用自绘 SVG 笔画，保留可访问的真实标题文本；其余标题使用较简洁的正文同族粗体。更多功能提示使用 ZCOOL KuaiLe 轮廓编排，作者文字泡使用其改字形并重命名的 OFL 修改版 Phonetic Whisper。版权与完整许可保留，详见 `docs/testing/website-font-license-audit.md`。外侧音标为独立设计的粗细笔画与浮雕图形。

展示共十六组、二十六张截图，依次为 Everforest 浅色首页、Catppuccin 深色 IPA、Rose Pine 浅色诗歌转写、Gruvbox 深色发声类型合成、Solarized 浅色生理参数合成、Nord 深色唇形、GitHub 白底 EGG、Ayu 深色语谱重建、One 浅色变速变调、Monokai 深色 MFA、Material 浅色 TextGrid、Night Owl 深色 LPC、Gruvbox 浅色音系归纳、Dracula 浅色图标、Temple 深色说明书，以及两张自定义背景图。自定义背景位于说明书展示后、小实验前，DeepSeek设置截图居前，2026-10-13用户更新的首页截图居后，延续错落叠放。旧深浅配色展示整块已移除。两张用户指定截图逐字节复制，保留2559×1527原始尺寸与来源摘要。

音系归纳用 `scripts/capture_website_phonology.py` 导入用户指定的测试表，完整保留5453条记录。调值1/2/5/6/7/8依次以阴平/阳平/上声/去声/阴入/阳入演示，明确标注“声调仅为举例，并非真实调类”。三图为调类排序归并、声母到韵母、二维表格，二维表格在最前。原XLSX摘要不变，只发布截图，不发布原表或生成的文档。

原生视图使用 `scripts/capture_website_modules.py`，两张定向修订使用 `scripts/capture_website_examples.py`，说明书用 `node scripts/capture_website.cjs --manual-only`。原生截图使用此前获准展示的语句与 EGG 示例，临时副本归属各次 scratch，除用户明确指定的试听片段外，截图原始录音不加入网站。唇形截图由用户提供，`scripts/prepare_website_lip.cjs <源截图路径>` 将视频矩形永久替换为全不透明保密贴纸，并验证区外像素完全相同。仅派生图片进入网站。页脚不展示素材与字体链接，来源与许可文件保留。

变速变调展示使用 `scripts/capture_website_pitch.py`：用户指定目录中的单音节临时副本，原 F0 增加 15 Hz，实际合成后截图。原文件不改动。

音标展示包含 IPA/extIPA/VoQS 三图，转写包含Standard Chinese (Beijing) 与 UntPhesoca严两版《春江花月夜》全文等宽排版，每七字及标点换行。EGG 增加逆滤波，图片重建增加实际涂鸦合成与写入 PTB 的音频藏信息，变速变调增加起点/拐点/终点各三基频全连接的 27 项生成。新截图为 2560×1440，应用缩放 150%，用户唇形截图维持原始尺寸。

补充截图用 `capture_website_stacks.py`、`capture_website_spectrum.py` 及 `capture_website.cjs --stacks-only`，作者寄语构建用 `build_website_author_note.py`。只有头像链接知乎主页，语谱图转音频文章按钮指向用户指定的知乎回答，原图提示仅出现一次。首页左侧分层轻微鼠标跟随，工具之间加入十五个错落波浪/星点分隔。减少动态效果时不跟随鼠标。

获明确部署授权后使用 `deployment/temporary-site/publish.py --ssh-helper <仓库外既有连接脚本> --apply`。只发布页面引用的静态资产与字体许可，版本化目录先写入，最后原子切换首页。既有论文目录和反馈服务保持不变，旧首页和站点配置在服务器私有备份目录保留。安装包不在发布清单。


新增内容由 `scripts/build_website_expansion.py` 生成。八项导航顺序见下方顶栏说明。工具巡游后直接进入使用教程；原更多功能与省略号提示已移除。背景区采用四张用户截图错落叠放，从前到后为第2、1、3、4张，分别是首页、外观设置、国际音标表Plus、感知实验，每张都可点击查看原图。`assets/release.json` 保留 GitHub v3.0.0 原文。更新小记默认全文展开，版本按钮切换对应正文。当前收录v3.0.2、v3.0.1与v3.0.0，按版本号从新到旧排序，默认显示最新。v3.0.2与v3.0.1分别逐字导入release/notes/3.0.2.md与release/notes/3.0.1.md，来源摘要保留，不自动同步GitHub；后续将版本快照加入assets/releases目录，构建时生成版本列表。反馈直接调用同源/api/v1/feedback/，首次验证邮箱，可发送文字/图片、查看回复、切换邮箱与控制邮件提醒。会话凭据只存当前标签页sessionStorage，邮件通过既有反馈服务处理，不跳转GitHub。

试听使用用户确认的M07完整导出：`build_website_experiments.py --phonation-export <导出目录>`核对m07.ptb.json输入与输出摘要，按step01–step09原样复制，不重合成或调幅。原声为dl_m02_0558_ma.wav与dl_m05_0558_ma.wav，均属SCSC，逐字节复制。九步同时改变基频和发声类型，导出分析为Praat/Parselmouth、40–300Hz、1ms，页面基频以相同设置复测原声和最终WAV。处理后中位F0约127/122/118/112/106/102/100/97/94Hz。小实验原有22段音频保持；下一步展示六组对比，新增六段GL对照和三段处理后音频，全页共31个WAV，来源/参数/摘要见assets/audio/provenance.json。已确认的手工导出会阻止普通自动生成命令覆盖，后续用新的明确导出更新。

`capture_website_expansion.py --mfa-only` 使用用户指定的原拼音 LAB、mandarin_pinyin_tab.dict 和相容 mandarin.zip，在独立副本内实际运行。TextGrid 截图用用户手工标注副本。原录音、转写、标注、词典和模型均核对哈希不变。`--lpc-only` 检查150%缩放下完整坐标轴。`capture_website.cjs --poem-only` 通过页面读音选择修正诗歌的处、扁、卷、还，并核对两种记音标准全部252字，使用现代普通话单字本调。

页尾统计读取同源 `/site-stats.json`。`deployment/temporary-site/install_stats.py` 安装匿名 Nginx 日志与一分钟汇总服务，`collect_stats.py` 增量汇总、支持轮转与压缩、不记录访客身份。统计从启用时间开始，页面请求计浏览、整文件或首段请求计下载发起，续传中间分块不计入。重试首段仍是新请求，不能解释成安装人数或独立用户。服务测试：`python deployment/temporary-site/test_stats.py`。统计失败显示不可用，不伪造零。

网页反馈定向验证：`node scripts/verify_website_feedback.cjs`。独立回环服务复用真实FeedbackService，使用隔离SQLite和邮件捕获器，检查验证码、图片、回复、会话恢复与丢失回执后的幂等重试；不向公网发送测试邮件或消息。联系入口为两张紧凑便签。统计起点显示北京时间年月日中的月日和时分秒，正式服务从2026-10-12 18:23:30累计，跨日不清零。

语谱图试听用 `scripts/build_website_spectrogram_audio.py`：用户提供的Smith图8.10、Chappell图7.4/7.3，作为说明图片近似重建的有限图像引用，作者、书名和链接位于例子下方，不声明开放再分发许可。三段实际M09输出，保留参数与哈希。`--phonation-only`仅更新两个原声和七步，保留其他音频。右侧阅读目录带进度环、当前板块提示、Escape关闭和返回顶部，尊重减少动态效果设置；底部慢慢逛提示已移除。

顶栏八项站内导航依次为使用教程、小实验、方法与引用、常见问题、更新小记、下一步…、问题反馈、联系我们，最后为GitHub。桌面18px，窄屏15–16px。下一步介绍保留快乐体、Aoki论文及四步流程，展示Smith、levantan、observo、Lu modal/pressed和知乎例图六组前后试听。全部处理前统一10ms Hann/60dB随机相位GL，256次迭代，不用脉冲初始化。处理后前三段保留stream-v4结果，新增三段逐字节取stream-v5对照页的new-base80-repaired。每对按RMS衰减播放音量，无JS可手动调节。Lu给出载瓦语论文完整引用和DOI。共31WAV/34播放器，逐对预测模型/图源/音频SHA保留于comparison.json与provenance；本机路径不公开。导入入口：build_website_spectrogram_audio.py --six-defaults-study <六图10ms默认GL的generated目录> --additional-comparisons <three-predicted-parameters-v5目录>，随后重建HTML并检查幂等。WEBSITE_NEXT_ONLY=1验证12段完整播放/来源/默认设置/导航/五档布局与无JS，WEBSITE_DIRECTORY指定冻结候选，WEBSITE_ORIGIN指定最终公网。当前最终命名与引用版rbe930860a22563bb已上线，90项公网大小/SHA和本地/最终公网12段完整播放及名称/引用/布局检查通过，回执见docs/testing/website-redesign.md。

当前品牌采用用户确认的黄底、雾紫声波、深蓝末端W1，图标页保留31项。`capture_website.cjs --branding-only`更新首页、外观、图标和说明书四图。语谱图试听设置按各条processing读取，模型预估值与固定Hann/迭代设置分开显示。导航提示按内容宽度排版并允许换行，首尾保持在视口内。

语谱图转音频展示下方用按钮跳转井井指定的知乎回答，按钮不显示原始网址。当前连续统A/B均为SCSC录音，按用户最后指定的两个输入更新。两段源音频保持原字节。

语谱图三段试听当前使用固定stream-v4预测参数重新生成的80带修复/BigVGAN输出，逐图21.73/66.80、4.65/55.79、5.32/59.59（ms/dB），处理前独立采用未知参数默认随机相位GL，处理后文件保持。先使用研究项目生成入口，再用 `build_website_spectrogram_audio.py --from-pulse-study <明确实验目录>` 导入，重建HTML后核对来源和音频摘要。默认旧GL生成入口阻止覆盖已确认导出。`WEBSITE_SPECTRUM_ONLY=1`检查小实验三段，`WEBSITE_NEXT_ONLY=1`检查下一步12段/参数/导航/宽窄/无JS；`WEBSITE_DIRECTORY`用于冻结候选，`WEBSITE_ORIGIN`用于最终公网。服务器更新范围和回执见docs/testing/website-redesign.md。

使用教程位于小实验前，左侧小电视按点击加载 B 站官方介绍播放器，支持播放器及独立全屏按钮、前往 B 站观看。右侧介绍为黄色可用按钮，其余18个模块灰色禁用，按钮使用不规则边线、轻微倾斜和随文字变化的宽度；正文标明正在施工中，按钮上方不再显示选一节提示。教程至作者寄语的九个大板块搭配三种分隔符，层级高于工具巡游的细分隔线。手绘声波用于教程/小实验，撕纸边用于方法与引用/常见问题，胶带接缝用于更新小记/下一步/问题反馈/联系我们/作者寄语。模块名称与本工程README对应。无JavaScript时可用B站链接。`WEBSITE_TUTORIAL_ONLY=1 node scripts/verify_website.cjs`检查导航、五档布局、禁用状态、键盘、全屏及无JS入口，外部媒体加载结果单独记录。

分隔符本地设计对照：`divider-preview.html`提供手绘声波、撕纸边、胶带接缝三种样式，作者已确认全部采用。手绘声波用于教程/小实验，撕纸边用于方法与引用/常见问题，胶带接缝用于更新小记/下一步/问题反馈/联系我们/作者寄语。主页面与对照页的SVG统一由 `scripts/build_website_expansion.py` 维护和生成。对照页未接入正式导航，不进入生产发布清单。`WEBSITE_CHAPTERS_ONLY=1 node scripts/verify_website.cjs`验证九处分配、桌面/窄屏、目录和无JS；`WEBSITE_DIVIDERS_ONLY=1`检查独立对照页。

正文中的使用说明与源码小节已移除，顶栏GitHub入口保留，第二入口位于页脚；阅读目录随之移除该节。首页音标采用更明显的漂浮、抬升轻转与明暗悬停反馈，正文按组错落渐入并保留适量卡片悬停。离屏/后台暂停首页装饰动画，减少动态效果模式静态显示，无JS正文可见。`WEBSITE_MOTION_ONLY=1`检查实际位移、悬停恢复、离屏暂停、宽窄屏及减少动态效果。上线CSP保留其他指令，只追加官方B站播放器的frame-src许可。

本轮教程、分隔、页脚和动效已发布为 `r4698051b31a01f4e`。89项公网大小/SHA及最终公网教程起播/全屏、导航/布局、动效/减少动态效果/页脚检查通过；具体范围见 `docs/testing/website-redesign.md`。
