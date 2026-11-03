# PhoneticToolbox 官网

静态产品介绍、正式下载和试听页，不承载应用登录、计算或文件上传。

当前中文维护源为 `content/home.zh-CN.html`，来自2026-10-30服务器实际首页。`locales/en.json`和`locales/ja.json`集中维护英文、日文文案，`site.css`、`site.js`、`ime-downloads.css/js`和`site-languages.css/js`分别维护样式与行为。`index.html`、`en/`和`ja/`是生成结果，正文不要直接改在生成文件上。

- 构建中英日三版：`python -B -X utf8 scripts/build_website_expansion.py`（工程根目录）。
- 检查生成结果与翻译覆盖：`python -B -X utf8 scripts/build_website_i18n.py --check`。
- 查看待翻译文案：`python -B -X utf8 scripts/build_website_i18n.py --inventory --language ja`（英文使用`--language en`）。中文文案变化会成为待翻译项，版本号或数值变化可复用同一句型，参数占位符必须完整保留。
- 中文模板和每个语言包各维护一份。语言包中的完整文案优先，`fragments`只用于模块名称、截图说明和版本等重复术语。引文正文、用户反馈、下载地址、科研数值与音视频沿用原文或原身份。新增语言只需增加同结构的语言包及语言信息，生成器、语言菜单和归档器自动发现。
- 本地预览 `/`、`/en/`和`/ja/`，顶栏切换语言后从页面顶部开始，不携带旧章节锚点或自动定位阅读板块。根地址始终默认简体中文，不按浏览器语言或已存语言偏好自动跳转；英文、日文独立地址按明确选择显示。英文、日文页均说明应用和视频教程当前使用中文。
- 顶栏语言入口在各版固定显示`Language`，下拉选项保留各语言自己的名称与当前语言勾选。顶栏按品牌、八项导航和操作区的实际宽度判断是否换行，按钮保留内容宽度；800px及以下将导航排成四列两行。日文页统一采用日文字体族，避免中文手写字体只覆盖部分汉字造成混排；代码和IPA字体独立保留。
- 定向验证：`python -B -X utf8 -m unittest discover -s tests -p test_website_i18n.py -v`与`node scripts/verify_website_languages.cjs`。
- 公网只读验证：`WEBSITE_ORIGIN`指定实际HTTPS根地址后运行同一浏览器脚本，只做导航、布局、语言切换和媒体读取，不发送反馈。`WEBSITE_DIRECTORY`配合`WEBSITE_READONLY=1`检查冻结归档候选。
- 需要重新采用公网基准时，先运行`python -B -X utf8 scripts/sync_website_from_public.py`比较，明确需要后加`--apply`。它只写本机，保留被替换文字与公开摘要。已有模板时更新模板，随后补齐翻译并构建。线上复用的旧资产目录不等于当前下载元数据，脚本按首页链接与稳定频道核对下载身份。
- 发布器在准备归档前检查各版生成结果。归档同时准备`/`、`/en/`、`/ja/`和共用媒体；实际服务器切换仍须获得本次部署授权。

- 本地预览：`python -m http.server 8765 --bind 127.0.0.1 --directory temporary-site`
- 首页网页版入口以6秒一周期、最多4像素轻微上下漂浮；离屏/切到后台暂停，减少动态效果时静态显示。漂浮动效保留本地预览范围。当前公开入口保持禁用，文案为灰测中，即将开放，已随re51d6c9a7c14facb上线。
- 品牌与装饰构建：`python scripts/build_website_assets.py`。首次取字体需联网，仅从 Google Fonts 官方仓库读取并保留 OFL。
- 当前应用截图：`node scripts/capture_website.cjs`。使用当前 Vue 源码、独立 Chrome 配置、公开示例文字，不读取用户任务或素材。
- 页面验证：`node scripts/verify_website.cjs`。
- 三张连续长图：`node scripts/export_website_long_images.cjs`。从公网首页截图，在区块留白处分界，报告与PNG保存到 `output/validation/website/long-images-*`。
- 资源来源、许可、主题及截图身份见 `assets/provenance.json` 和 `assets/licenses/`。

当前下载身份以`assets/downloads.json`及已核实的服务器链接为准，2026-10-30基准为Windows 3.1.2、macOS 3.1.1和国际音标输入法0.1.7。服务器版本化资源目录可能复用较早的元数据，不能按目录中的旧`downloads.json`覆盖当前版本。各语言页使用相同下载地址；发布归档不重新上传应用安装包。

滚动使用原生页面滚动、IntersectionObserver 和 requestAnimationFrame，没有滚轮劫持。减少动态效果时取消漂浮、视差及入场变形；无 JavaScript 时正文、截图与导航保持可见。点击截图可查看完整原图。

IPA 装饰参考 Doulos SIL 的字形结构，重新绘制圆润的单线条、浮雕层与阴影，保留辨音所需钩形；它们是图形资产，并非研究结果。语谱背景为确定性装饰图形。新增截图使用实际界面，不用设计图伪装软件结果。

首页标题六字使用自绘 SVG 笔画，保留可访问的真实标题文本；其余标题使用较简洁的正文同族粗体。更多功能提示使用 ZCOOL KuaiLe 轮廓编排，作者文字泡使用其改字形并重命名的 OFL 修改版 Phonetic Whisper。版权与完整许可保留，详见 `docs/testing/website-font-license-audit.md`。外侧音标为独立设计的粗细笔画与浮雕图形。

展示共十八组、三十四张截图，依次为 Everforest 浅色首页、Catppuccin 深色 IPA、Rose Pine 浅色诗歌转写、Gruvbox 深色发声类型合成、Solarized 浅色生理参数合成、Nord 深色唇形、GitHub 白底 EGG、Ayu 深色语谱重建、One 浅色变速变调、Monokai 深色 MFA、Material 浅色 TextGrid、Night Owl 深色 LPC、Gruvbox 浅色音系归纳、Lobster 深色论文精读、Notion 浅色语言地图、Dracula 浅色图标、Temple 深色说明书，以及四张自定义背景图。自定义背景位于说明书展示后、小实验前，DeepSeek设置截图居前，2026-10-13用户更新的首页截图居后，延续错落叠放。旧深浅配色展示整块已移除。两张用户指定截图逐字节复制，保留2559×1527原始尺寸与来源摘要。

音系归纳用 `scripts/capture_website_phonology.py` 导入用户指定的测试表，完整保留5453条记录。调值1/2/5/6/7/8依次以阴平/阳平/上声/去声/阴入/阳入演示，明确标注“声调仅为举例，并非真实调类”。三图为调类排序归并、声母到韵母、二维表格，二维表格在最前。原XLSX摘要不变，只发布截图，不发布原表或生成的文档。

原生视图使用 `scripts/capture_website_modules.py`，两张定向修订使用 `scripts/capture_website_examples.py`，说明书用 `node scripts/capture_website.cjs --manual-only`。原生截图使用此前获准展示的语句与 EGG 示例，临时副本归属各次 scratch，除用户明确指定的试听片段外，截图原始录音不加入网站。唇形截图由用户提供，`scripts/prepare_website_lip.cjs <源截图路径>` 将视频矩形永久替换为全不透明保密贴纸，并验证区外像素完全相同。仅派生图片进入网站。页脚不展示素材与字体链接，来源与许可文件保留。

变速变调展示使用 `scripts/capture_website_pitch.py`：用户指定目录中的单音节临时副本，原 F0 增加 15 Hz，实际合成后截图。原文件不改动。

音标展示包含 IPA/extIPA/VoQS 三图，转写包含Standard Chinese (Beijing) 与 UntPhesoca严两版《春江花月夜》全文等宽排版，每七字及标点换行。EGG 增加逆滤波，图片重建增加实际涂鸦合成与写入 PTB 的音频藏信息，变速变调增加起点/拐点/终点各三基频全连接的 27 项生成。新截图为 2560×1440，应用缩放 150%，用户唇形截图维持原始尺寸。

补充截图用 `capture_website_stacks.py`、`capture_website_spectrum.py` 及 `capture_website.cjs --stacks-only`，作者寄语构建用 `build_website_author_note.py`。只有头像链接知乎主页，语谱图转音频文章按钮指向用户指定的知乎回答，原图提示仅出现一次。首页左侧分层轻微鼠标跟随，工具之间加入十五个错落波浪/星点分隔。减少动态效果时不跟随鼠标。

获明确部署授权后使用 `deployment/temporary-site/publish.py --ssh-helper <仓库外既有连接脚本> --expected-state <已审阅服务器快照JSON> --apply`。快照包含`index_sha256`和`localized`下各语言旧页摘要或null，发布前及暂存后核对，防止覆盖同期修改。发布页面引用的静态资产、既有音频来源清单登记的WAV及字体许可，版本化目录先写入，再写入各语言静态首页，最后原子切换中文首页。既有站点配置、论文目录和反馈服务保持，旧首页和站点配置在服务器私有备份目录保留；仅切换静态文件，不重载服务。安装包不在发布清单。


新增内容由 `scripts/build_website_expansion.py` 生成。八项导航顺序见下方顶栏说明。工具巡游后直接进入使用教程；原更多功能与省略号提示已移除。背景区采用四张用户截图错落叠放，从前到后为第2、1、3、4张，分别是首页、外观设置、国际音标表Plus、感知实验，每张都可点击查看原图。`assets/release.json` 保留 GitHub v3.0.0 原文。更新小记默认全文展开，版本按钮切换对应正文。当前收录v3.0.4、v3.0.3、v3.0.2、v3.0.1与v3.0.0，按版本号从新到旧排序，默认显示最新。v3.0.3、v3.0.2与v3.0.1分别逐字导入release/notes/3.0.3.md、release/notes/3.0.2.md与release/notes/3.0.1.md，来源摘要保留，不自动同步GitHub；后续将版本快照加入assets/releases目录，构建时生成版本列表。反馈直接调用同源/api/v1/feedback/，首次验证邮箱，可发送文字/图片、查看回复、切换邮箱与控制邮件提醒。会话凭据只存当前标签页sessionStorage，邮件通过既有反馈服务处理，不跳转GitHub。

小实验第3张声道卡使用井井提供的《春江花月夜》视频，替换原三段元音试听。视频与桌面原文件逐字节相同，复用已有MP4/JPEG，16:9原生播放/暂停/进度/内联，默认不预载或自动播放；与其他音频互斥。旧三WAV保留。`WEBSITE_VOCAL_VIDEO_ONLY=1 node scripts/verify_website.cjs`检查四宽度、来源/HTTP摘要、播放解码/结束、互斥与无JS。该卡已随r192fbd47a2437145上线，更新小记内的同视频位置仍仅本地。

试听使用用户确认的M07完整导出：`build_website_experiments.py --phonation-export <导出目录>`核对m07.ptb.json输入与输出摘要，按step01–step09原样复制，不重合成或调幅。原声为dl_m02_0558_ma.wav与dl_m05_0558_ma.wav，均属SCSC，逐字节复制。九步同时改变基频和发声类型，导出分析为Praat/Parselmouth、40–300Hz、1ms，页面基频以相同设置复测原声和最终WAV。处理后中位F0约127/122/118/112/106/102/100/97/94Hz。小实验原有22段音频保持；下一步展示六组五路对比，GL均32次；前三路和原73段保持原字节，新增12段青木初始化及其修复/声码器音频，资源保留85个WAV，来源/参数/摘要见assets/audio/provenance.json。已确认的手工导出会阻止普通自动生成命令覆盖，后续用新的明确导出更新。

`capture_website_expansion.py --mfa-only` 使用用户指定的原拼音 LAB、mandarin_pinyin_tab.dict 和相容 mandarin.zip，在独立副本内实际运行。TextGrid 截图用用户手工标注副本。原录音、转写、标注、词典和模型均核对哈希不变。`--lpc-only` 检查150%缩放下完整坐标轴。`capture_website.cjs --poem-only` 通过页面读音选择修正诗歌的处、扁、卷、还，并核对两种记音标准全部252字，使用现代普通话单字本调。

2026-10-30起，下载合并官网、应用更新与两个公开GitHub仓库中的应用包，范围为2026-10-12起的工具箱v3.0.0及以上Windows/macOS和国际音标输入法四端。官网记录每分钟刷新，GitHub每5分钟采样，按附件ID去重并保留已观测累计，排除JSON/校验/源码/修补附件。普通日志可确认的Windows更新ZIP按真实时间补入，缺少时间的GitHub历史累计在首次观测时并入曲线，页面说明补录，末端与顶部三类一致。总数和曲线核对同批采样后一起更新，交叉时最多重取一次，失败保留上一组。定向检查：`python -B -X utf8 deployment/temporary-site/test_stats.py`和`node scripts/verify_website_downloads.cjs`，公网检查在PowerShell先设`$env:WEBSITE_ORIGIN='https://www.phonetictoolbox.com/'`；本地夹具可加`WEBSITE_STATS_RACE_CHECK=1`检查刷新交叉和失败保留。仅更新采集脚本时使用`deployment/temporary-site/publish_stats_update.py`并核对现存脚本SHA，不重装Nginx/systemd配置。

页尾统计读取同源 `/site-stats.json`。`deployment/temporary-site/install_stats.py` 安装匿名 Nginx 日志与一分钟汇总服务，`collect_stats.py` 增量汇总、支持轮转与压缩、不记录访客身份。统计从启用时间开始，页面请求计浏览、整文件或首段请求计下载发起，续传中间分块不计入。重试首段仍是新请求，不能解释成安装人数或独立用户。服务测试：`python deployment/temporary-site/test_stats.py`。统计失败显示不可用，不伪造零。

网页反馈定向验证：`node scripts/verify_website_feedback.cjs`。独立回环服务复用真实FeedbackService，使用隔离SQLite和邮件捕获器，检查验证码、图片、回复、会话恢复与丢失回执后的幂等重试；不向公网发送测试邮件或消息。联系入口为两张紧凑便签。统计起点显示北京时间年月日中的月日和时分秒，正式服务从2026-10-12 18:23:30累计，跨日不清零。

语谱图试听用 `scripts/build_website_spectrogram_audio.py`：用户提供的Smith图8.10、Chappell图7.4/7.3，作为说明图片近似重建的有限图像引用，作者、书名和链接位于例子下方，不声明开放再分发许可。三段实际M09输出，保留参数与哈希。`--phonation-only`仅更新两个原声和七步，保留其他音频。右侧阅读目录带进度环、当前板块提示、Escape关闭和返回顶部，尊重减少动态效果设置；底部慢慢逛提示已移除。

顶栏八项站内导航依次为使用教程、小实验、方法与引用、常见问题、更新小记、下一步…、问题反馈、联系我们，最后为GitHub。下一步保留快乐体、92%行内代码及正常基线。六图每例五按钮，无波形与按钮下方小字：原始GL、动量GL、预测参数动量GL、预测参数青木动量GL、同一第四路PCM接80带NAFNet/BigVGAN。前三路沿用原字节，前两路10ms Hann/60dB，后续动量0.99；后三路用stream-v5预测值、窗宽整数ms四舍五入，GL均32次、FFT512/16kHz/hop24。前三路随机相位，后两路青木脉冲初始化；修复仍为第6000步权重。五路说明默认展开并高亮重点。悬停/聚焦每个按钮显示五路实际输出的语谱图浮窗，当前图加粗/边框/当前标记，支持移入、Esc及窄屏避让。作图统一10ms Hann、1.5ms帧移、60dB，同例共享频率/幅度参考且按等RMS播放电平绘制。five-way-baseline.json锁定前三路及原73段、预测/激励/模型与同PCM，新增12WAV及30图。显式scripts/build_website_five_comparison.py --generate只生成一次；普通构建/导入/发布只验证。WEBSITE_NEXT_ONLY=1检查30段及浮窗/键盘/宽窄/无JS；WEBSITE_CODE_ONLY=1检查行内代码。WEBSITE_DIRECTORY用于冻结候选，WEBSITE_ORIGIN用于公网。

当前品牌采用用户确认的黄底、雾紫声波、深蓝末端W1，图标页保留31项。`capture_website.cjs --branding-only`更新首页、外观、图标和说明书四图。语谱图试听设置按各条processing读取，模型预估值与固定Hann/迭代设置分开显示。导航提示按内容宽度排版并允许换行，首尾保持在视口内。

语谱图转音频展示下方用按钮跳转井井指定的知乎回答，按钮不显示原始网址。当前连续统A/B均为SCSC录音，按用户最后指定的两个输入更新。两段源音频保持原字节。

语谱图三段试听当前使用固定stream-v4预测参数重新生成的80带修复/BigVGAN输出，逐图21.73/66.80、4.65/55.79、5.32/59.59（ms/dB），处理前独立采用未知参数默认随机相位GL，处理后文件保持。先使用研究项目生成入口，再用 `build_website_spectrogram_audio.py --from-pulse-study <明确实验目录>` 导入，重建HTML后核对来源和音频摘要。默认旧GL生成入口阻止覆盖已确认导出。`WEBSITE_SPECTRUM_ONLY=1`检查小实验三段，`WEBSITE_NEXT_ONLY=1`检查下一步30段/预测参数与青木初始化/同PCM/五图浮窗/导航/宽窄/无JS；`WEBSITE_DIRECTORY`用于冻结候选，`WEBSITE_ORIGIN`用于最终公网。服务器更新范围和回执见docs/testing/website-redesign.md。

使用教程位于小实验前，左侧小电视按点击加载B站官方播放器，支持独立全屏按钮和前往B站观看。右侧介绍、录音、唇形提取、参数估计、参数显示、EGG信号分析、LPC谱图、语谱图转音频、国际音标表Plus、TextGrid标注、音系归纳、感知实验与语音学论文精读13个频道可用，其余6个模块禁用。语谱图转音频默认播放新11期 `BV17mnc6QEYY`，分集仅保留专题与本期教程两个按钮，专题 `BV1RuJm6gEWE` 在前，本期教程在后。切换频道替换唯一播放器，同步标题、父频道/分集选中状态及外链；重复选择保留播放进度。按钮沿用现有样式、原字号和自然宽度，正文保留正在施工中；无JavaScript提供14个视频外链。`WEBSITE_TUTORIAL_ONLY=1 node scripts/verify_website.cjs`检查五档布局、12期教程映射、专题与教程顺序/切换/选中、第二集按钮移除、键盘、重复选择、全屏和无JS；`WEBSITE_PLAYER_ONLY=1`配合`WEBSITE_ORIGIN`检查实际HTTPS/CSP下电脑及Android UA/触摸模拟的14段加载与播放。发布范围见[官网验收](../docs/testing/website-redesign.md#教程06与11至16接入2026-10-20)。

方法与引用保留四张常用方法卡片，下方分为学术引用、软件与代码库引用两个独立列表，各自默认折叠，分别搜索和筛选，互不影响。生成器读取桌面共用的 `frontend/src/generated/sources.json` 公开投影和模块注册表，保留完整引用、来源链接、版本、许可、署名与改写说明。展开后可搜索、按模块筛选及复制引用；无JavaScript仍可展开完整列表。`WEBSITE_METHODS_ONLY=1 node scripts/verify_website.cjs`逐项核对与桌面来源一致，并检查折叠、筛选、复制、宽窄布局及无JS。本轮完整引用列表和提示句删除已于2026-10-16发布，公网逐条匹配及交互检查通过。

分隔符本地设计对照：`divider-preview.html`提供手绘声波、撕纸边、胶带接缝三种样式，作者已确认全部采用。手绘声波用于教程/小实验，撕纸边用于方法与引用/常见问题，胶带接缝用于更新小记/下一步/问题反馈/联系我们/作者寄语。主页面与对照页的SVG统一由 `scripts/build_website_expansion.py` 维护和生成。对照页未接入正式导航，不进入生产发布清单。`WEBSITE_CHAPTERS_ONLY=1 node scripts/verify_website.cjs`验证九处分配、桌面/窄屏、目录和无JS；`WEBSITE_DIVIDERS_ONLY=1`检查独立对照页。

正文中的使用说明与源码小节已移除，顶栏GitHub入口保留，第二入口位于页脚；阅读目录随之移除该节。首页音标采用更明显的漂浮、抬升轻转与明暗悬停反馈，正文按组错落渐入并保留适量卡片悬停。离屏/后台暂停首页装饰动画，减少动态效果模式静态显示，无JS正文可见。`WEBSITE_MOTION_ONLY=1`检查实际位移、悬停恢复、离屏暂停、宽窄屏及减少动态效果。上线CSP保留其他指令，只追加官方B站播放器的frame-src许可。

本轮教程、分隔、页脚和动效已发布为 `r4698051b31a01f4e`。89项公网大小/SHA及最终公网教程起播/全屏、导航/布局、动效/减少动态效果/页脚检查通过；具体范围见 `docs/testing/website-redesign.md`。

公开统计：四项总数及四张历史图从同源site-stats.json/site-stats-history.json读取。collect_stats.py兼容现有计数状态并回填保留的匿名日志，Star固定官方API每5分钟采样；webUses仅累计成功新建临时工作区。历史缺失保留null，失败保留上一采样及时间。下载图分别展示Windows与macOS，顶部下载总数为两平台合计，Windows内部沿用两种安装包合计。历史第六列macosDownloads为新增Mac计数，原五列值保持，第二列统计字段明确为windowsDownloads；未记录Mac的历史为null。旧普通日志仅回填可确认的GET 200，未保存Range头的206不补计；回填只执行一次，不保存IP或UA原文。底部不显示免安装版/安装版次数。2026-10-16采集器、统计map与四图首次随r196722d9534d7cb4发布，后续页面版本见官网验收。网页版大号入口放在系统下载区上方，当前已公开改为禁用并显示灰测中，即将开放，桌面顶栏1024px起单行；公开计数排除验证器合成测试，不混入私有内测数据。89资源摘要、公开数据刷新及七宽度浏览器验证通过，原反馈代码/站点路由保持。

下一步行内代码数字按正常文本基线对齐，移除该处不需要的overflow:auto，按井井要求字号由正文82%增至92%，保持底色/等宽/nowrap。WEBSITE_CODE_ONLY=1在1280/390/320三宽度用零尺寸基线探针检查实际对齐，复现旧滚动样式约5.16px上浮，并保存DPR3截图。


统计区将最新数字与真实趋势合并为四张紧凑卡片，下载卡直接显示Windows/macOS次数，起止日期统一显示。曲线纵向从卡片底线铺到大数字所在高度，文字置于前景，历史读数显示在下方。四图保留紫/珊瑚橙/鼠尾草绿/暖黄，曲线和所选点同色；曲线上覆盖不可见的原生时间控件，支持悬停、触摸点按及键盘方向键/Home/End，四图共用选中时间，End或Escape可回到最新；底部不另显示查看状态或返回按钮。最新总量与历史读数分开，刷新保留历史或跟随最新。缺失不插值、不虚构，单点禁用控件。`WEBSITE_STATS_CARDS_ONLY=1 node scripts/verify_website.cjs`检查卡片和共享交互，`WEBSITE_STATS_AXIS_ONLY=1`检查时间轴/缺失/刷新/键盘/空单点，`WEBSITE_STATS_SAMPLING_ONLY=1`检查长历史，`WEBSITE_MACOS_STATS_ONLY=1`检查双平台。`WEBSITE_STATS_DIRECTORY`可提供公共统计快照，夹具不上传。已随r29eed14bdf8c6e67上线，实际证据见官网验收。

2026-10-17已同步3.0.3下载版本、两入口/体积及井井确认的更新小记，旧版小记保留。downloads.json使用已核验服务器成品的大小/SHA。WEBSITE_RELEASE_ONLY公网验证会额外对照稳定更新频道的版本、安装器身份与同版更新正文，防止官网独立元数据留在旧版。此轮只发布3份变化文件并复用134项线上原资源，本机未上线CSS/JS保持，证据见官网验收。

下载平台布局定向检查：`WEBSITE_PLATFORM_ONLY=1 node scripts/verify_website.cjs`，覆盖七宽度、Windows/Mac下载路由及安装说明、网页版禁用、平台图例、聚合统计/键盘与无JS。本轮已随re51d6c9a7c14facb上线，冻结及公网七宽度/两下载HEAD/平台布局/真实统计/无JS通过，137项资源大小与SHA回读一致；仅三文件上传，134资产原字节复用。

macOS统计定向验收：`WEBSITE_MACOS_STATS_ONLY=1 node scripts/verify_website.cjs`。公开真实数据另设`WEBSITE_ORIGIN=https://www.phonetictoolbox.com/`与`WEBSITE_MACOS_STATS_LIVE=1`。采集检查：`python -B deployment/temporary-site/test_stats.py`；Linux实际Nginx隔离检查：`python -B deployment/temporary-site/test_nginx_stats.py`。升级脚本`upgrade_macos_stats.py`默认仅预演，`--apply`只替换采集器及统计map，保存私有回退并保留现有计数与服务配置。

2026-10-19随Windows 3.0.4同步下载版本、两入口/体积与作者确认的更新小记，全文来自`release/notes/3.0.4.md`。仅原子替换版本相关3文件，复用原线上资源，配置/反馈进程保持；其他工作区官网改动保留。验收见[Windows发行报告](../docs/testing/release-3.0.4-publish.md)。

更新小记 v3.0.4 的生理参数合成说明末尾嵌入《春江花月夜》短视频。原 MP4 与封面位于 `assets/videos/`，来源和 SHA 见该目录 `provenance.json`。视频点击播放、支持原生全屏、默认不预载，切换版本暂停；发行正文快照保持。发布器收录视频、封面与来源清单并重写 src/href/poster 前缀。`WEBSITE_RELEASE_ONLY=1 node scripts/verify_website.cjs` 核查来源、位置、四宽度、实际音视频解码/进度/播放结束/版本切换暂停与无 JS 入口。当前仅本地，公网尚未更新。


2026-10-25新增论文精读两图与语言地图四图，截图均为当前Windows源码Qt的完整视图。论文精读使用Lobster深色，展示栏目说明《请看这里~》及全屏荧光标注/本页批注，指定的普通论文阅读图已移除；示例为Fan等（2024）的CC BY 4.0论文，页面保留原文和许可入口。语言地图使用Notion浅色，以方言相似性为首图，另有韵尾特征、双图及苏州/广州/西安/哈尔滨/南京五地全部记录对照。浅色/深色标注与其他图一致，新增两套配色均未重复。仅首页旧图重新拍摄，现含十九工具，其他旧图原字节保持。

采集入口为 `scripts/capture_website_learning.py --papers <已核实论文发布目录> --maps <已核实地图资料目录>`，`--home-only`可只补拍首页。使用隔离任务库和阅读批注，最终截图不裁剪、不改色。`python -B -X utf8 scripts/build_website_expansion.py --learning-only`只更新两组展示与首页图片尺寸，`--index-file`可指定已核实公网基线副本。`WEBSITE_LEARNING_ONLY=1 node scripts/verify_website.cjs`核对七图、首页摘要、八种宽度、正常/减少动态效果、鼠标/键盘原图查看和无JS显示。发布按实时首页构造最小候选并复用旧资产，详见[官网验收](../docs/testing/website-redesign.md#论文精读与语言地图展示2026-10-25)。
