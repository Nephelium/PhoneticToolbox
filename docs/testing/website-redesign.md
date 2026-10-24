发布结果：输入法浮窗资源 `ipa-popover-011-2d49e4dc9211` 已上线，线上十一组、四包完整下载、GitHub公开链接与双更新清单一致通过。Windows链接使用r2。

## 2026-10-24 · 国际音标输入法下载区

官网在原工具箱下载组下增加独立输入法四平台下载入口，安装包与 GitHub 对齐。首版十组线上响应式/无 JS 通过，四包全量 HTTP 下载 SHA-256 与本地一致。随后按井井反馈改为默认隐藏的浮窗，悬停预览、点击固定、移开/再次点击/空白/Esc 关闭，触屏点击，无 JS 点击。浮窗本地及线上各五宽度开关 JS 共十组加触屏一组通过，不改变 hero 高度。仅发布线上首页的下载区与独立样式/脚本，不带入其他未发布内容。

## 教程06与11至16接入（2026-10-20）

状态：`verified`，七期接入首版为 `r940c0b4de70df6e4`，同日分集调整已上线 `rfd524881c628735c`。查找新发布的七期教程并接入对应按钮。B站官方视频接口核对全部作者为井韶子（mid 15511376）、标题对应、公开状态为0，时长与成片一致。

| 模块 | BVID | 时长 |
| --- | --- | --- |
| LPC谱图 | `BV17mnc6QExx` | 228秒 |
| 语谱图转音频 | `BV17mnc6QEYY` | 420秒 |
| 国际音标表Plus | `BV17mnc6QEYL` | 182秒 |
| TextGrid标注 | `BV17mnc6QEJN` | 398秒 |
| 音系归纳 | `BV17mnc6QEJf` | 273秒 |
| 感知实验 | `BV1Bmnc6QEjn` | 273秒 |
| 语音学论文精读 | `BV1Bmnc6QELa` | 209秒 |

新11期为语谱图转音频按钮的默认视频，原 `BV1RuJm6gEWE` 保留为专题。按井井后续截图要求，分集仅保留专题、本期教程并按此排序，第二集按钮已移除。共13个可用频道、6个未发布模块按钮及14条无JS直达链接。切换专题时父频道保持选中，分集选中状态与唯一播放器、标题和外链一致，重复选择不重载。

生成幂等、JS/Python语法和来源核对通过。以活动 `r192fbd47a2437145` 冻结候选 `r940c0b4de70df6e4`，只替换教程HTML和播放器事件块。候选及公网Chrome155五宽度的按钮/BVID/标题/外链、键盘/鼠标、分集选中/重复选择、唯一播放器、实际全屏及14条无JS链接通过；桌面和Android UA/触摸模拟最终覆盖14视频共28组readyState4和播放推进。页面错误及失败HTTP请求为空，桌面/320手机尺寸截图已核对。

初次Android的国际音标表就绪等待超时，定向复查通过后又在音系归纳出现同类超时，抓取到的媒体已readyState4、networkState1、时长有效且无媒体或网络错误。手机点击下方频道后iframe位于视口外，默认rAF轮询可能暂停。验证脚本改用100ms定时轮询，继续要求readyState至少3、有效时长及播放推进，45秒超时门槛保持；增加选择待复查频道和UA、去查询参数的网络错误记录与超时媒体状态。两个失败报告保留，补齐定向实测后合并的28组证据在 `public-players-combined.json`，没有将超时轮次标为通过。 补充把手机播放器滚入视口后检查57帧实际视频解码并留存可见播放截图，报告 `output/validation/website/verify-1792487083592-133620/report.json` 通过；原先整块长截图的画面层未绘出与视口外合成状态相符，播放器在可见视口正常显示。

仅首页/JS两文件88958压缩字节上传，182项原资源复用，184项服务器大小/SHA及两个变化资源/根首页HTTPS完整字节核对一致。其他网页内容、CSS、配置、论文目录及反馈PID保持。原子切换前两次活动首页摘要防并发核对，私有回退首页保留，自有浏览器与候选scratch均已清。

来源及操作配方：`output/validation/website/tutorials-06-16-20261020/`。候选、发布、HTTPS和保护对象回读：`output/validation/website/tutorials-06-16-public-f36361ca270d4ef990d779bff2e0b108/`。实体手机、其他浏览器及完整视频时长未验，未推送Git或重建应用。

同日分集后续调整已发布 `rfd524881c628735c`。生成源、播放器标题和无JS专题文案同步更新，按钮精确为专题、本期教程两个可用入口，默认仍为本期教程。候选及公网独立Chrome155五宽度、顺序/数量、键盘与鼠标切换、父频道/分集选中、标题/BVID/外链、重复选择、全屏及14条无JS链接通过。两视频在电脑和Android UA/触摸模拟共4组加载、播放推进与实际视频解码通过，已目视320宽度分集截图。

文档检查2356文件、448来源、170任务零错误，历史快照断链单列。本次只改变首页/JS两文件，上传88902压缩字节，182资源原字节复用，184项服务器摘要及变化资源/根首页HTTPS逐字节回读一致，保护对象保持，回退首页保留，scratch清理回执齐全。配方和文档检查：`output/validation/website/tutorial-episode-order-20261020/`；发布/公网/HTTPS证据：`output/validation/website/tutorial-episode-order-public-4314ee627e3c40358738d78a5b89b2e2/`；播放器报告：`output/validation/website/verify-1792488836946-70748/report.json`。实体设备、其他浏览器及完整时长未验。

## 声道演示视频替换元音试听（2026-10-19）

状态：`verified`，已按井井指定的截图位置上线 `r192fbd47a2437145`。第3张绿色小实验卡片改为声道动起来，原三元音按钮/波形和160Hz、三构形、JD2、每段1.20秒的说明移除，放入《春江花月夜》片段及声道动态说明。播放器保持16:9、原生controls/playsinline、preload=none、无autoplay，带封面与原视频fallback链接。原音频文件保留，网页旧音频和该视频互斥播放，原音频按钮状态、后台暂停及本地更新小记的版本切换暂停继续适用。

井井指定的桌面原MP4与工程已有副本均662593字节、SHA `c0f19521fdde5e2c34162588280e57b906975bb640744e405f0f4d827a7766ae`，FFprobe确认H.264/AAC、1280×720、2.36秒，无转码、重采样或重合成。复用既有0.4秒JPEG封面并更新来源用途，未再次导出。维护范围及流程见[计划](../plans/2026-10-19-website-vocal-video.md)。

- `WEBSITE_VOCAL_VIDEO_ONLY=1 node scripts/verify_website.cjs`：本地 `verify-1792423132500-30168`、最终候选 `verify-1792423310248-89576`、公网 `verify-1792423322303-38072` 均通过1600/800/390/320四宽度、三元音播放元素移除、无旧科学参数文案、视频/封面本地与HTTP SHA、原三WAV摘要保持、原生键盘播放/暂停、1秒跳转、播放结束、实际音视频解码、音频→视频/视频→音频及重建试听互斥、无JS原生控制与fallback。错误和失败请求均为空，公网解码83帧、45349音频解码字节、无媒体错误。已目视公网手机尺寸及播放完成画面。
- 从当前活动 `r29eed14bdf8c6e67` 冻结候选，只替换第3卡、媒体事件块并追加3条视频样式。既有更新小记HTML和其他公开内容经比较保持，本地更新小记播放器位置未随本次上线。新增MP4/JPEG/视频来源3资源，仅首页/CSS/JS3主文件改变，178资源保持，全部旧音频字节保持。
- 六文件837056压缩字节上传并原子切换，184项服务器大小/SHA、六个变化资源和根首页HTTPS完整字节核对通过。配置、论文目录、反馈PID保持，无服务重载或统计变更。回退首页私有保留，全部自有浏览器及冻结副本scratch的cleanup回执为deleted。完整证据 `output/validation/website/vocal-lab-public-8fd5190e0c724ef5b5b9515f283c582f/`，配方及幂等记录 `output/validation/website/vocal-lab-video-20261019/`。
- 生成幂等、基础GL6项锁与漂移拒绝、JS语法及定向差异检查通过。原重建试听互斥断言改为与新视频互斥；默认音频按钮验证的退役元音目标改用现存变调目标，未声称旧默认整站流程全部通过。文档检查2356文件/448来源/170任务零错误，原历史快照断链单列，日志 `vocal-lab-video-20261019/docs-validation.log`。

本轮未推送Git、修改科研算法或重建应用。实体手机、其他浏览器和主观听辨未验。解码与播放状态的验证不扩展为声音质量结论。

## 数字与趋势合一卡片（2026-10-19）

状态：`verified`，最新已公开上线 `r29eed14bdf8c6e67`，首版为 `r9a001debb7201074`。按井井确认的方案，把四项最新数字放进对应四色趋势卡片，移除重复总量栏与常驻粗滑杆，下载卡直接显示 Windows/macOS 次数，起止日期共用一行。沿用真实时间轴、阶梯路径、缺失留空与全部记录绘图降采样。悬停、触摸点按和原生 range 键盘共用一个选中时间，四图同步更新游标/读数，最新总量保持显示；底部独立查看状态及回到最新按钮已按井井截图删除，刷新保留历史或跟随最新。

维护源为 `scripts/build_website_expansion.py` 与 `temporary-site/site.css/site.js`，首页由生成器写入。实施范围见[计划](../plans/2026-10-19-website-stats-cards.md)。验证均在本机独立 Windows Chrome155，使用自有回环服务器与独立配置，未操作用户浏览器或生产计数。

同日按井井最新截图移除底部整行正在查看时间和回到最新按钮，删除其HTML、样式与事件绑定，曲线内历史读数及四图同步、End/Escape返回最新保持。已上线 `r29eed14bdf8c6e67`，候选四专项及公网卡片七宽度/历史选择后整行不存在/鼠标/键盘/触摸模拟/刷新/失败/无JS、时间轴检查通过，页面错误与失败请求为空。仅三文件112737压缩字节改变，178资源及其他线上内容保持，181项服务器大小/SHA及变化资源与根首页HTTPS通过，配置/目录/反馈PID保持，自有scratch已清。公网报告 `verify-1792421394318-41896` 与 `verify-1792421394313-125812`，完整证据 `output/validation/website/stats-cards-public-2cd82c31fea04b28ad06510bbf10a24c/`，生成幂等见 `stats-cards-20261019/controls-removed-generation.json`。

同日按井井截图追加：曲线纵向从底线延伸到大数字所在高度，取消独立110/100px底部绘图区。图层覆盖卡片数字行至底部，窗口变化时按实际数字位置对齐，数值/图例局部底色保护文字，历史读数移到下方。SVG原始时间/计数坐标及阶梯路径保持，视口与原生选时区域同步扩展，端点按实际宽高保持圆形。旧版先在 `verify-1792420330033-63092` 复现绘图区/最大值未到数字行的失败，未放宽检查。

纵向扩展最终候选的卡片/时间轴/双平台/长历史四专项分别为 `verify-1792420812049-126924`、`verify-1792420812049-87740`、`verify-1792420817793-140312`、`verify-1792420823010-107880`，全部通过。公网卡片 `verify-1792420829326-27532` 七宽度检查绘图区及最大值达到数字高度、数字实际命中未被遮挡、控件几何、鼠标/键盘/触摸模拟、刷新/失败/无JS，公网时间轴 `verify-1792420829328-116732` 通过原有数值及三宽度边界。页面错误/失败请求为空，公网1440/390真实统计截图已目视核对。仅三文件112903压缩字节改变，178资源保持，181项服务器大小/SHA及变化资源和根首页HTTPS一致，配置/目录/反馈PID保持，回退保留，自有scratch已清。完整证据 `output/validation/website/stats-cards-public-4231d37a21c24fa2bd02f372c0423d03/`，生成幂等见 `stats-cards-20261019/vertical-generation.json`。

- `WEBSITE_STATS_CARDS_ONLY=1 node scripts/verify_website.cjs`：最终 `output/validation/website/verify-1792418785599-40420/report.json` 通过七宽度1920/1440/1280/1024/800/390/320、数字与平台数入卡、双列/单列、默认隐藏粗滑杆、共同日期、四图同步悬停/键盘/触摸、回到最新、刷新跟随、失败保留旧值/首次失败与无JS。报告错误及失败请求为空。补充只读公开匿名快照2831条记录，在本地捕获并目视桌面1440和手机390截图，22:05快照总量为1881/389/52/37，不作为公网新布局上线证据。
- `WEBSITE_STATS_AXIS_ONLY=1 node scripts/verify_website.cjs`：`verify-1792418635615-38448` 通过真实秒数、不规则间隔、缺失前缀、243平台、邻近记录键盘操作、刷新保留/跟随、四时段、单点/空历史、三宽度，以及不可见原生控件与曲线/点的物理像素对齐。
- `WEBSITE_STATS_SAMPLING_ONLY=1 node scripts/verify_website.cjs`：`verify-1792418661142-47308` 通过160001条不规则记录、全部历史510个绘图点、端点/孤立峰/真实缺口保留与原始记录查询。最近1/7/30天的240/1680/7200点仍全部绘制，全部记录切换约75ms。
- `WEBSITE_MACOS_STATS_ONLY=1 node scripts/verify_website.cjs`：`verify-1792419075247-86184` 通过双平台总数/独立曲线颜色、七宽度、聚焦显示读数、未知与0区分、稀疏平台事件、旧五列格式、无效总数保留旧值和无JS。
- 两次重复生成逐字节幂等，`output/validation/website/stats-cards-20261019/generation.json` 保留首页摘要。基础GL音频锁6项及漂移拒绝、JS/Python语法和定向差异检查通过。所有本轮浏览器运行的 `cleanup.json` 均为 `deleted`。

失败与修复：旧页面先在 `verify-1792418458891-104576` 正确复现重复总量栏。`verify-1792418635610-129768` 发现触摸抬手触发pointerleave后读数消失，改为仅鼠标离开收起并重验通过。`verify-1792419006871-79228` 是旧平台专项测量默认可见output，与按需显示的新交互冲突，增加实际聚焦后测量，没有删除边界检查。一次变量误写为 `WEBSITE_MAC_STATS_ONLY` 导致 `verify-1792418785612-55792` 误入默认整站流程，在已有 `.github-button` 选择器处超时，未改无关页面或绕过默认断言。

按后续明确授权，从活动公网 `rdbdad5513312c950` 冻结 `r9a001debb7201074`，仅替换统计HTML、统计JS模块和77条统计CSS规则（含原媒体条件）。其他线上HTML、CSS和JS经范围投影比较保持，181项资源中178项原字节复用，未混入本地标题、更新小记视频或其他同期修改。仅上传3文件112651压缩字节，原首页保留私有回退，发布前及原子切换前检查活动首页摘要。181项服务器文件大小/SHA、三个变化资源及根首页HTTPS完整字节均核对通过；nginx配置、论文目录和反馈PID保持，无配置写入或服务重载，未修改统计采集器、计数和历史。

最终冻结候选四专项分别为卡片 `verify-1792419513921-45308`、时间轴 `verify-1792419513921-38732`、双平台 `verify-1792419520773-139520`、长历史 `verify-1792419526624-71000`，全部通过。公网卡片 `verify-1792419533395-88696` 通过七宽度、实际统计读取与鼠标/键盘/触摸模拟、刷新/失败/无JS，公网时间轴 `verify-1792419533395-105208` 通过三宽度及相关边界，页面错误和失败请求均为空。公网真实统计截图1440/390已目视核对。发布配方位于 `output/validation/website/stats-cards-20261019/publish.py`，完整范围、部署、HTTPS回读和完成证据位于 `output/validation/website/stats-cards-public-dad9d5721c524e498d6eb0e01acd5c63/`。所有六轮浏览器及冻结候选scratch均已按归属清理，回执为deleted。

`python -B -X utf8 scripts/validate_docs.py` 检查2355文件、448来源、170任务零错误，未修改的历史快照断链另列。未推送Git或重建EXE。其他浏览器、实体设备、Qt和最终成品未验。

## 教程01至05接入（2026-10-19）

状态：`verified`，按井井本轮明确授权已公开上线 `rdbdad5513312c950`。对应关系为录音 `BV1xsnJ6nES2`、唇形提取 `BV1NsnJ6nE2Z`、参数估计 `BV1NcJd6UEZv`、参数显示 `BV11cJd6UESc`、EGG 信号分析 `BV1KFJd6JEvg`。介绍及语谱图转音频第一集沿用，七个入口可用，其余12个模块继续禁用。新增五段按键盘或点击切换唯一iframe，同步标题、外链和选中状态，重复选择保留播放器；无JS时提供全部七个视频直达链接。

维护生成器和播放器映射，扩展已有教程/真实播放器验证。生成两次首页SHA一致，JS语法通过。从当前活动公网 `r42de50b4c3c837fa` 冻结候选 `rdbdad5513312c950`，仅 `index.html` 和 `site.js` 变化，179项资源复用。首页的活动字节与归档目录旧首页不同，初次冻结在摘要检查处停止且未部署，随后明确按活动首页自身SHA核验；不放松其他资产摘要或发布并发保护。本地更新小记短视频及其余未发布内容未混入候选。

`WEBSITE_TUTORIAL_ONLY=1 node scripts/verify_website.cjs` 在冻结候选与实际公网分别通过，五宽度1600/1280/800/390/320检查18模块名称、七启用/12禁用、导航、布局无溢出、每段准确BVID/iframe标题/外链、键盘和鼠标切换、唯一播放器、重复选择保留、语谱图分集显示/返回、实际全屏与七条无JS链接。候选报告 `output/validation/website/verify-1792411163649-41152/report.json`，公网报告 `output/validation/website/verify-1792411196577-24088/report.json`。

`WEBSITE_PLAYER_ONLY=1` 配合实际官网HTTPS完成独立Chrome155电脑UA与Android UA/触摸模拟，每端七段共14组均readyState4、video无错误、实际播放进度约2.1秒且未暂停、切换后单播放器和外链正确，CSP违规/页面错误/失败请求均为空。新五段手机播放器时长分别485.157732/430.497959/646.907937/164.977778/472.502857秒，与已剪辑成片对应；电脑播放器按整秒呈现。报告 `output/validation/website/verify-1792411247861-101244/report.json`，公网桌面/窄屏及EGG播放截图已目视核对。实体手机、其他浏览器及完整视频时长未验。

仅上传首页和JS共87658压缩字节，179项原字节资源在服务器复用，181项活动文件大小/SHA全部一致，两个变化资源及根首页HTTPS完整字节回读一致。发布前再次核对活动首页，原首页保留私有回退，nginx配置/论文目录/反馈PID保持，不重启服务。Python AST、JS语法、定向diff通过；文档全检2354文件/448来源/170任务，零错误，既有历史快照断链单列，回执保留在入口的 `docs-validation.log`。失败冻结、最终候选与三轮独立浏览器scratch均按归属清理并留下 `deleted` 回执。

脚本和证据入口为 `output/validation/website/tutorials-01-05-20261019/`，完整冻结/部署/HTTPS回读与完成记录位于 `output/validation/website/tutorials-01-05-public-047616462a68466eb35d16657a3c21b9/`。初次失败冻结为 `tutorials-01-05-public-adc07fa5e9c64add9091248284dbaa62/`。本轮未推送Git、发布应用版本或重建EXE。

## 手机端教程播放器被屏蔽（2026-10-18）

用户截图中的错误已用Chrome155/Android UA及触摸模拟复现。B站官方`player.bilibili.com/player.html`按手机UA跳转到`https://www.bilibili.com/blackboard/webplayer/mbplayer.html`，官网原CSP只允许前一个域名，控制台明确记录`frame-src`违规。桌面播放器正常加载，问题定位在手机跳转后的来源限制。

`deployment/temporary-site/publish.py`只增加官方手机播放器的具体路径，保留已有来源及其余全部指令。四种策略输入覆盖已有frame-src、none、缺省及已含手机路径，幂等和其他指令保持通过，Python远程程序与JS语法通过。在`verify_website.cjs`加入`WEBSITE_PLAYER_ONLY=1`专项，必须使用实际HTTPS响应头和真实手机UA，不能只缩小视窗；两端分别检查介绍与语谱图转音频第一集的官方地址、video就绪、播放时间推进、单播放器/外链切换及零CSP违规。

Windows Chrome155候选验证在PowerShell设置`$env:WEBSITE_ORIGIN='https://www.phonetictoolbox.com/'`、`$env:WEBSITE_PLAYER_ONLY='1'`及`$env:WEBSITE_PLAYER_POLICY_PREVIEW='1'`后运行`node scripts/verify_website.cjs`，仅在测试浏览器替换首页响应头，四组加载/实际播放均通过，报告`output/validation/website/verify-1792294666907-49748/report.json`。井井授权修复后，忽略目录中的`output/validation/website/mobile-player-20261018/repair.py`复用维护源的策略变换，以已审阅配置/活动文件摘要作并发保护，仅原子替换服务器CSP，执行`nginx -t`与reload并保留配置备份及失败回滚。回执`deployment.json`记录操作`mobile-player-f4394d645c224d40a9d56963b0d6ae40`，活动release仍为`rc57cf2fda18a63c6`，182项活动文件/首页/论文目录摘要和反馈PID保持，未上传页面或其他同期修改。

公网复验关闭响应头预览，运行同一专项，电脑/Android模拟两段均为readyState4、无video错误，播放时间分别推进至1.83/2.15/2.13/2.13秒且未暂停，零CSP违规。报告`output/validation/website/verify-1792294728491-92012/report.json`；手机介绍与桌面第一集截图已查看。裸域名及www域名HTTPS响应均含新策略、首页SHA保持，证据`output/validation/website/mobile-player-20261018/https-readback.json`。两轮scratch回执均为deleted。`python -B -X utf8 scripts/validate_docs.py`检查2296文件/447来源/162任务零错误，历史快照断链另列；定向`git diff --check`通过。实体手机、Safari和完整时长播放未验。

## 下载平台区分与灰测提示（2026-10-17）

首页下载区保持与网页版入口同宽，左侧两个Windows按钮，短竖线分隔后为灰色macOS按钮，原生disabled且没有href。Windows版本/体积和两个3.0.3链接保持，按钮文字精简为免安装版、安装版。网页版仍禁用，文案改为灰测中，即将开放。下载发起图显示Windows曲线与Windows所选次数，灰色macOS图例注明尚未上线，不伪造macOS历史；底部安装包分类次数已移除，统计采集和原数据格式保持。

`WEBSITE_PLATFORM_ONLY=1 node scripts/verify_website.cjs` 在独立Windows Chrome155通过1920/1440/1280/1024/800/390/320七宽度、按钮文字/图标边界、分隔顺序、Windows附件名与路由、macOS/网页版禁用、统计聚合120+149=269、键盘选择及无JS。报告与截图位于 `output/validation/website/verify-1792219048052-79908/`，截图已查看。另运行 `WEBSITE_STATS_AXIS_ONLY=1 node scripts/verify_website.cjs`，不等间隔采样、邻近记录键盘移动、平台前缀下的平段曲线、刷新保留/跟随与单点/空历史通过，报告 `output/validation/website/verify-1792219065481-98728/`。两轮失败报告保留：首轮测量包含绝对定位tooltip，改为单独测文字与图标；第二轮macOS窄屏确有拥挤，精简可见文字后修复。各轮scratch清理回执为deleted。

图例随后改为绝对定位在图内上方留白，保持四图坐标高度；截图验证前清焦点/悬停，避免安装版提示遮住网页版入口。最终复验通过，报告与已查看截图位于 `output/validation/website/verify-1792219142516-49184/`，页面错误/失败请求为空，scratch回执deleted。生成器连续两次逐字节幂等，JS语法、定向diff通过；文档2206文件/445来源/142任务零错误，历史快照断链单列，日志 `output/validation/website/platform-downloads-docs-20261017.log`。本阶段先使用明确的回环隔离夹具，不修改公网计数；后续按本轮上线授权发布并检查实际数据，见下段。


已按井井本轮上线授权发布re51d6c9a7c14facb，仅上传首页/CSS/JS三文件103594压缩字节，134项资产复用原字节。冻结候选及公网Chrome155七宽度、两Windows下载HEAD/附件路由、macOS与网页版禁用、平台图例/真实统计、滑块及无JS通过，页面错误/失败请求为空。137项公网资源大小/SHA与首页SHA回读一致，论文目录、Nginx配置及反馈服务PID保持，未重启服务。最终桌面/手机截图已查看，scratch回执deleted。未改正式/app/、统计采集器或安装包，未推送；实体手机及其他浏览器未验。 发布前保存活动首页、CSS/JS与全量资产清单，在当前线上HTML上仅替换网页版入口、下载区、下载图和底部分类文案。CSS仅追加本轮平台样式，JS仅增加Windows所选数值前缀并移除旧分类输出，保留同期其余线上/本地差异；发布前及原子切换前均校验活动首页SHA。Windows两个成品服务器全量SHA验证通过。回退首页私有保留，未改站点配置。

发布与保护核对回执：`output/validation/website/platform-publication-b9d2ccf926084b21950e6b6cb409c693/completion.json`、`server-before.json`、`server-after.json`和`deployment.json`。冻结候选浏览器报告 `output/validation/website/verify-1792219559089-57040/`，公网报告与实际统计/七宽度截图 `output/validation/website/verify-1792219578779-55164/`，真实数据采样为2026-10-17 14:45，Windows累计269次，页面浏览1261次；数字随服务后续实际请求更新。

## 官网同步3.0.3发行信息（2026-10-17）

用户截图指出官网仍显示3.0.2。实查首页及两个下载链接确为3.0.2，服务器固定成品与stable/preview频道已为3.0.3；上一轮未同步官网独立的downloads.json和版本小记。读取服务器两EXE完整大小/SHA，与最终包报告一致后更新downloads.json，新增assets/releases/v3.0.3.json，全文逐字使用已确认的release/notes/3.0.3.md，SHA 85f400df…，旧三版正文保留，展示日期使用北京时间2026-10-17。

通过标准生成器生成并验证幂等。发布前识别本地index与公网不同，CSS/JS存在同期未发布修改，故候选仅从生成结果提取两个下载锚点、版本/体积和更新小记，其他线上HTML及活动CSS/JS/图片/字体/音频保持原字节，本机同期修改保留。上线前及切换前核对原首页SHA，防止覆盖并行更新，私有回退首页保留。rf077ec7d8e36d930已上线，首页SHA 99c9e5513ff80d221d7ee51c17e3e7e5ce05752de8be82f8cb2100e01b08d53a；只上传3文件/71,108压缩字节，134项不变资产校验后复用，未重复上传。论文目录保持、反馈服务在运行，本任务未改nginx配置、正式网页版或EXE。

实际独立Chrome155的冻结候选及公网通过1600/800/390/320四宽度、两按钮下载附件名及实际HEAD长度、体积提示、默认3.0.3/四版本鼠标及键盘切换、全文/Markdown与摘要、无JS正文。页面错误/失败请求为空，公网下载区截图已目视确认为3.0.3。修正原验证入口无JS检查固定写3.0.2，改读当前元数据；新增公网与稳定频道的版本、安装器大小/SHA及更新正文交叉核对，最终verify-1792179336029-49184四组通过。验证未完整下载500MB成品、未重新运行科学任务，成品全量SHA已在服务器核对。

首轮夹具缺少音频锁校验所需的原输入图，补齐仅验证使用的硬链接，不扩大公开文件表；第二轮自组装HTML与标准构建清单摘要不符，改为直接使用标准构建器归档中的HTML，没有删断言或改摘要。两轮均在上线前中止，失败证据及日志保留，scratch均已清。成功冻结证据output/validation/website/release-303-3240ef8902284ca59521a3614c92db50，scratch已deleted；实际浏览器各自独立配置也已清理，保留报告、日志和必要截图。汇总output/validation/release-303/website/completion.json。实体手机及其他浏览器未验，未上传同期源码。

## 网页入口轻微漂浮与本地预览修复（2026-10-17）

网页版入口加入6秒周期、0到−4px的轻微上下漂浮，沿用首页离屏与后台暂停机制，减少动态效果时静态显示。尺寸、禁用状态及入口文案保持。仅修改官网CSS及现有动效专项，不更新音频、统计、站点路由或公开服务器。

`WEBSITE_MOTION_ONLY=1 node scripts/verify_website.cjs` 在独立Chrome155通过：实际入口位移采样范围1.2624px，处于4px设计范围内，仍disabled且无href；离屏暂停、动态切换减少动画、1440/800/390/320四宽度布局及无JS内容均通过。页面错误/失败请求为空。证据 `output/validation/website/verify-1792167217659-85728/`，截图已核对，scratch按回执清理成功。

本地5173端口原先HTTP可用，但浏览器实际空白。Chrome网络记录确认 `markdown-it.js?v=6d21f3ec` 返回504 `Outdated Optimize Dep`，随后AppShell动态导入失败；旧共享缓存当前已无有效元数据。不能用HTML返回200替代工作台实际加载。增加可选的 `PTB_VITE_CACHE_DIR`，本地启动器使用 `output/local-web-preview/vite-cache/`，保持默认开发命令与其他入口不变。只结束上轮本任务明确启动的旧Vite PID33856，未管理其他服务。隔离缓存重启后，Chrome实际显示18工具、21章和带6图的网页版入门，保留人工预览标签。

前端typecheck/build通过，构建仍有原大分块提示。当前5173提供源码界面预览，原隔离账号服务18765保持，不在本批添加代理或修改认证。官网动效的本地预览为8765。本批未部署官网/网页服务、未重建最终EXE，其他浏览器与长时间并发开发缓存行为未验。先前自动审批对账号代理/外部启动的拒绝记录保持，本轮通过标准项目命令和受支持的Chrome工具恢复查看，没有重试被拒代理操作。

## 统计滑条时间映射与四色（2026-10-16）

线上匿名快照site-stats-history.json确认下载在22:06:11变成243后维持243，采样记录总522条。实际曲线以时间映射，旧滑条以记录索引映射，采样疏密不一致导致位置误导，黄色点又固定最新位置。diagnosis.json保存22:08附近记录的两种横轴比例。当前实现统一使用timestamp并吸附最近真实采样，圆点/时间/数值/aria-valuetext同步；左右/Home/End/Page按真实记录移动，刷新保留历史选择或跟随最新。无记录不画点，单点禁用滑条，无插值/改计数/重算历史。四卡背景/曲线/点/滑条分别紫、珊瑚橙、绿、暖黄。

- 旧实现专项verify-1792163872221-69796失败：滑条min为0，曲线实际起点为时间戳。修复后verify-1792163924080-81568通过不等间隔、点/滑条位置、243平台、相邻键盘、刷新保留/跟随、三宽度、单点/空历史。第一次空历史夹具遗漏totals空数组处理，进程异常，verify-1792163903730-74668保存失败并确认所有子进程退出，scratch按项目入口清理成功；未修改产品断言。
- 最终四色冻结：轴专项verify-1792164162963-27560通过；完整统计verify-1792164162984-84888通过四条历史曲线、四个独立颜色且curve/marker/slider一致、七宽度1920至320、四时段/键盘、Star下降、失败时保留旧值/初次无数据和无JS说明。已目视statistics-1440.png。生成幂等通过。
- stats-time-axis-20261016冻结r0d81c4bf11f66d66，136资源仅index缓存键/site.js统计IIFE/site.css新增四色改变，133保持，新增/删除0。统计采集器、服务器计数/历史及配置不变，当前四路24音频及全部来源/锁不变。源JS中同期反馈改动未纳入，阶段脚本只替换当前服务器统计块。一次直接CSS写入OSError，原文件未变化，核对后定向补丁成功。已原子发布r0d81c4bf11f66d66，receipt website-511c560c33e5450183e09ffaeb46e930/deployment.json。公网verify-1792164333189-80812通过同项统计七宽度/四时段/四色/Star下降/错误恢复/无JS；另外实测真实时间轴选择22:25的243次记录，selectedTime=1792160726、markerX=459.75080841749167，与当前真实时间跨度公式完全一致。errors/requests为空。136资源完整大小/SHA、两下载HEAD、CSP及统计配置/采集器服务/论文目录/反馈PID/统计timer/nginx保持通过，回执stats-time-axis-20261016/https-readback.json。JS语法/生成幂等、文档及定向差异检查通过。其他浏览器与实体手机未验。

## 五路试听与悬浮语谱图（2026-10-18）

按最新要求保留前三路，第四路预测参数/整数ms/0.99动量加青木初始化，第五路读取第四路实际24k PCM16接80带NAFNet与BigVGAN。六例新增12WAV及30张实际输出语谱图，原73段与旧锁保持。模型仍stream-v5 7ac61f07、修复第6000步a60a3587与BigVGAN e2a93f92。RTX5090/已有pytorch_5090生成；首次指向同名训练6000检查点，其摘要dcbf357d与冻结模型不符，在写入前拒绝，改用既有repair-step6000副本a60a3587。生成末尾校验首次误把基础GL与accelerated_settings的元数据形状当作同一结构失败，核实原结构后修正校验字段，没有修改科学值或覆盖音频。修正后30轨/输入/预测/取整/相位/动量/模型/同PCM/73旧段及漂移/覆盖/重复生成拒绝通过。

输出显示统一24k/10ms Hann/1.5ms帧移/1024 FFT/60dB，同例共享频率和幅度参考，按实际等RMS播放电平作图。卡片五按钮无波形/下方小字，五路说明默认展开并突出重点。悬停/聚焦显示全部五图，当前行加粗、边框、当前标记，允许移入、Esc、窄屏避让/滚动。

- 生成器幂等、Python/JS语法、website_audio_baseline与verify_website_audio_baseline通过。文档检查2267文件/447来源/159任务，仍报同期来源计数447与total_records446不一致，未改动该登记；没有本轮新增链接错误，旧历史断链保持。
- Windows Chrome155源码verify-1792260019721-137584与最终冻结verify-1792260192591-109032均通过：30段HTTP SHA/完整播放、每按钮五图与对应高亮、无按钮下方小字、默认展开/mark、八档导航、五内容宽度/浮窗边界、键盘聚焦/Esc、互斥播放、实际快乐字体与无JS30原生播放器。errors/requests均为空；已目视检查1280/390浮窗截图。
- 从公网当前r5aaf8028a426a3a7冻结rbca3d6f3f94fcbd3，共180资产，5改变/43新增/132保持/0删除。仅下一步、本功能CSS/JS前置模块、两元数据变化，新增12WAV/30PNG/五路锁；旧73WAV、前三路、其他页面/原JS/下载/统计保持。scope.json与generation-checks.json在output/validation/website/five-route-aoki-20261018。
- 已原子发布rbca3d6f3f94fcbd3，回执website-ba1c5cacf3ad422d86f749f57c731702/deployment.json。公网verify-1792260368786-142484通过上述全部30轨/五图/默认展开/无小字/布局/键盘/互斥/字体/无JS检查，errors/requests为空。https-readback.json确认180资产完整大小/SHA、两正式下载HEAD、CSP/站点及统计配置/论文目录/反馈PID/统计timer/nginx保持，无配置写入/服务重载。
- 三轮浏览器scratch由各自finalizer清理；本轮冻结候选scratch在全部回读/播放后清理，183文件31230271字节，保留配方/清单/报告/截图/原子发布回执及正式网站资产。cleanup.json为deleted。未重建EXE，其他浏览器/实体设备未验。

## 四路32次试听与参数预测恢复（2026-10-16）

井井取消切回256，要求保留32次并恢复参数预测，四按钮无波形。前两路10ms/60dB的原始GL、动量GL沿用已锁定字节；后两路用参数模型stream-v5（7ac61f07）估计窗宽/dB，窗宽取整22/5/5/4/5/5ms，预测动态范围61.107/49.430/56.280/83.424/87.357/55.038dB。时间频率标定、Hann、FFT512、16kHz、hop24、seed20261015、随机相位、32次保持，均禁用Aoki。第四路读取第三路导出的实际24k PCM16，接同一第6000步NAFNet（a60a3587）和BigVGAN。没有重训练或选新权重。新增12WAV、61旧WAV原字节保持，四路音量等RMS衰减。旧三路锁保留，四路独立锁绑定预测值/模型/同PCM/实现来源及旧61段。

- GPU RTX5090完成六图12段。四路验证、32/相位/动量/模型/输入/预测参数漂移拒绝、覆盖拒绝、重复生成拒绝及生成幂等通过。首次测试mock旧三路read导致漏拦，改为mock四路实际read后通过，未放宽产品规则。
- 源码独立Chrome155：verify-1792163266802-42580通过24段HTTP SHA/完整播放、四按钮/无波形、同预测PCM与模型、音量/互斥/键盘、八导航/五内容布局、快乐字体及无JS24控件。目视next-1280.png确认文案、四阶段流程、六图四按钮完整。最终冻结CODE verify-1792163391832-40912三宽度92%字号/基线差0/DPR3通过。
- 从服务器当前r7fe41882f19f5f84冻结four-route-32i-20261016，候选rfe3d62f0a65cb82c共136资源，4改变/13新增/119保持/0删除；JS及当前全部旧音频、基础/三路锁、下载、其他首页区块保持。变化仅下一步、相应按钮CSS、两元数据及12新WAV/四路锁。最终冻结NEXT verify-1792163391832-56836同样通过24段SHA/完整播放、四按钮无波形、八导航/五布局、实际字体/互斥/键盘/无JS。已原子发布，receipt website-0a24e38bb6d640ad9215c5ccba62c384/deployment.json。公网NEXT verify-1792163565824-80376全部同项通过，errors/requests为空，包括本轮无JS完整加载与24原生控件；前版无JS超时仍保留为历史失败，没有扩大超时。136资源完整大小/SHA、两正式下载HEAD、CSP、目录摘要/配置/反馈PID/统计timer与nginx检查通过，回执four-route-32i-20261016/https-readback.json。文档与定向差异检查通过。平台Windows/Chrome155，CODE为DPR3，其他浏览器/实体手机未验；本轮不改EXE。

## 同参数三路试听、动量文案与本地EXE（2026-10-16）

- 后续32次迁移：井井明确由256改32，不是自动挑选参数。已核实上一版网页三路与此前六图导出为256次，桌面calibrated默认32次。使用显式--migrate-to-32入口，保留43段旧WAV，新增18段32i文件；新六段原始GL与动量GL同幅值/相位起点/10ms Hann/60dB/512FFT/16kHz/hop24/seed20261015，均32次，差别只有动量0/0.99。原始GL保持原有16kHz归一化后重采样呈现方式；动量GL与声码器呈现方式也沿用。第三路读取新动量24k PCM16，模型仍为第6000步a60a3587与既有BigVGAN。GPU RTX5090生成通过，43旧文件摘要逐字节保持。两锁及来源/页面同步32，旧256参数/摘要留在历史字段及旧文件，不覆盖。默认检查严格锁定32，显式迁移也不能重复执行。generation-checks.json记录六对32次误差，只用于频谱匹配核查，不替代听感评价。
- 行内代码对齐：原CSS inline-block+overflow:auto使内部文字基线比父行高5.15625px，CODE专项verify-1792159614936-50140复现并失败。只把upcoming-section code的overflow-x恢复visible，1280/390/320内部/正文基线差均为0，强制旧auto仍可重现−5.15625px。源码CODE verify-1792159807575-76652与冻结CODE verify-1792160019199-77612通过，保留DPR3局部图并已目视，字体/颜色/nowrap保持，也未修改Release长代码独立滚动。井井随后要求字号稍大，下一步code从正文82%改为92%，最终CODE verify-1792160205510-63440三宽度fontRatio0.92/基线差0，通过DPR3局部图目视。
- 32次初版冻结位于iterations32-inline-code-20261016，字体放大后的最终冻结iterations32-inline-code-final-20261016，候选r7fe41882f19f5f84共123资源，6改变/18新增/99保持/0删除，43旧WAV、JS、下载和用户定稿三段保持。page构建幂等、来源参数/迭代漂移及覆盖拒绝通过，最终冻结NEXT verify-1792160205510-72012通过八档导航/五档内容/18段完整播放与32次元数据、新文件SHA、互斥/无JS，errors/requests为空。已原子切换r7fe41882f19f5f84，receipt website-a8cb32f6217c4ddaa4207d6f5be10fa0/deployment.json；123资源完整大小/SHA、两EXE HEAD、CSP及目录/服务PID/配置保持通过。公网CODE verify-1792160459360-81956三宽度fontRatio0.92、基线差0，旧auto强制复现上浮6px，DPR3通过。
- 公网NEXT保留三次失败：verify-1792160459364-65404在无JS新上下文goto等待load时超过30秒；verify-1792160689647-79604在键盘跳转定位等待超过30秒，未取得足够现场数据确认原因；补充只读阶段/基线/待请求诊断后，verify-1792160925797-54756完成八档导航、18段SHA及完整播放、同参数/同PCM、互斥、五档内容/目录和实际快乐字体，随后仍在无JS新上下文goto等待load时超过30秒。该次nojsPending记录了屏幕截图、六图、图标与字体等29个未完成请求，主页面320px的nextTop129.890625px与目标130px一致，errors/requests均为空。没有扩大超时、删除断言或修改产品以绕过检查。当前字号/基线和32次音频已在公网验证，无JS全页冷加载的30秒门槛未通过；本地冻结无JS内容/18个原生音频控件检查通过，不能据此声称公网完整回归通过。

- 最终文案定稿：按井井提供的三段正文逐字替换，0.99以code显示，原有措辞、NAFNet/BigVGAN展开、括号与hhh全部保留。冻结approved-copy-20261016，rad6fb1720de5ff49共105资源，仅index.html变化，104保持。三段与用户文本逐字核对及生成幂等通过。冻结NEXT verify-1792158093070-76872通过八档导航/五档内容/18段完整播放/无JS，errors/requests为空，已目视1280长图。已原子发布，receipt website-8cf4a741a40440cfb911005504b46008/deployment.json；105资源大小/SHA、两EXE HEAD、三段公网逐字及服务/配置/目录保持通过。最终公网NEXT verify-1792158240436-66500同样通过八档导航/五档内容/18段完整播放/无JS与新文案排版，errors/requests为空，scratch已清理。

- 井井确认固定10ms/60dB比较，第三路直接取第二路动量GL接Mel修复/BigVGAN；EXE交付本地候选供检查，不公开上传。详见plans/2026-10-16-gl-three-way-comparison.md。
- 原始六段GL与其余25段旧WAV字节保持。新12段由共享研究GL及应用repair/vocoder适配器在RTX5090 Laptop生成，随机种子20261015、随机相位、Hann160样本/16kHz、FFT512、hop24、256次、momentum0.99。幅值谱与原始GL使用同一标定和解码实现摘要。第三路解码实际24k PCM16动量WAV作为输入，vocoder_input_sha256绑定第二路，修复权重a60a3587139ffdc79b69978615b2079c66a43282cb2b77ab99d276717b433ab2，第6000步，BigVGAN e2a93f92…保持；GPU与TF32关闭均记录。三路仅以衰减匹配播放RMS，不做额外听感挑选。
- build_website_gl_comparison.py为显式一次性生成入口，常规只verify。three-way-baseline.json锁定三路与旧31音频；verify_website_audio_baseline.py验证基线漂移、覆盖/重算拒绝，并验证第三路错接音频、错误模型步数、脉冲/预测混入及12段覆盖均被拒绝。无测试WAV副本留存。
- 文案独立说明青木初始化与动量迭代，三路试听关闭青木与预测；两张论文卡片及方法目录来源与应用406条投影一致。Perraudin原文作者PDF核对标题、作者及加速说明：https://perraudin.info/publications/perraudin-note-002.pdf 。首页日期、meta和FAQ旧日期删除。generator再次运行前后首页摘要相同。
- 源码NEXT报告verify-1792156263746-72452：18段全部播放至ended、HTTP SHA、等电平/互斥、六图来源、独立状态、8档导航/5档内容/无JS通过。冻结目录output/validation/website/gl-three-way-final-20261016，候选r0618fc40d775c978，105资源：4改变、13新增、88保持、0删除；JS使用服务器当前版本，排除本地反馈来源改动。冻结NEXT verify-1792156407887-51660、DPR3首屏 verify-1792156407887-65480、Release verify-1792156407899-61080通过，errors/requests为空，scratch清理完成。已目视1280px三路长图。
- r0618fc40d775c978已原子切换，receipt为website-b2ed73a0dbb2450393e1a85e7a3439af/deployment.json；不写配置或重载Nginx。https-readback.json验证105资源大小与完整SHA、两EXE HEAD、CSP，论文目录/站点与统计配置/反馈PID保持，统计timer正常。公网NEXT verify-1792156631455-70296、DPR3首屏 verify-1792156631477-73272、Methods verify-1792156631478-35028、Release verify-1792156631498-74292全部通过，errors/requests为空，scratch已清理。
- 最终EXE展开的实际core文件与冻结输入摘要一致，使用包内EGG科学Python运行随机/脉冲两种初始化的32次GL，输出均为momentum0.99，样本数/有限值/峰值检查通过，legacy仍为m09/1。报告local-package-72ed53fa/m09-final-runtime.json。初次误选不含SciPy的m05运行时失败，换用包内已含SciPy的EGG运行时后通过；未安装依赖或改变应用，失败日志保留。这项是实际展开字节/包内运行时检查，独立于GUI交互。
- 井井反馈256次版本两路听感接近后，当时从公网下载六对WAV核对SHA和PCM，全部文件/波形不同。生成时同幅值谱误差：原始→动量GL分别为Smith0.397001→0.392926、levantan0.157112→0.155441、observo0.225774→0.216800、modal0.177528→0.170744、pressed0.180954→0.169737、知乎图0.196433→0.188695，相对降低1.03%～6.20%。这衡量幅值谱匹配，不能等同于听感改善百分比；256次下两路听起来接近合理。公网差异核查为gl-three-way-final-20261016/audible-difference-audit.json，未改试听文件、参数或发布版本。
- 井井反馈IEEE DOI入口返回418后，将动量论文卡片主链接改为作者公开PDF，保留DOI书目信息；应用现有作者论文链接已为该PDF，应用字节不变。实际HTTP200、application/pdf、416417字节、SHA0809726aac561cfe1a44be448abdc79f4fe63d1e894595ba26691c935f8c0cf1，元数据与原文标题/作者已核对。链接修正冻结在gl-author-pdf-link-20261016，候选re75d238c3d229817仅index.html改变，104资源保持，音频与下载全保持。冻结NEXT verify-1792157137038-71828通过18段完整播放及宽窄/无JS，新链接断言通过；已发布为re75d238c3d229817，receipt website-2fa3d027fea142a0b03edb90ad6e906a/deployment.json，105资源完整大小/SHA、两EXE HEAD与服务/配置/目录保持通过。公网NEXT verify-1792157323503-77976再次验证18段完整播放、新作者链接、宽窄与无JS，无页面错误与失败请求。
- 本地候选EXE大小496333905字节；首次准备持久缓存实测逻辑文件长度3916182735字节，按NTFS硬链接身份去重1827540536字节，22893路径/12673独有文件，没有重解析入口。此数值不是文件系统分配簇大小，也不包含用户研究数据。cache-footprint.json保留测量，工程外实际退出已验证_MEI启动临时目录清理，未测瞬时峰值。最终QA本轮将清理其隔离缓存和副本，不清用户实际数据。
- 应用源码M09定向49项通过。首轮pytest误用根配置要求未安装的cov插件，改为workbench_source.configure及-c backend/pyproject.toml，未安装依赖/更改断言，49项通过。package_local.py --version3.0.2生成唯一候选PhoneticToolbox-302-20261016-210632-d8764d，496333905字节，SHA f281ad61e374820edfb2336b44fab46a3d94acc6ed4387763b9c2eaa9478448d，冻结d8c16c5f8a42219c84cd38140788baaa2de79e34b1104680cfb526d0f49c8c97。严格说明书校验/生成、类型检查、406项前端测试、来源投影、独立前端生成、冻结封装/审计已通过。最终包工程外分发、包内科学探针/MediaPipe、11托管计算/媒体帮助、中文深目录及超260缓存路径安装卸载、活动租约/资料保持、缓存进程/设置清理、两种真实换版/重启通过。默认GPU可见普通/最大化窗口、浅深主题、三字体七字号共168帧文字像素通过，原生窗口目视完整；verify_local_package.py整体成功，所有阶段退出码0，主窗口正常退出0。12个明确归属验收目标已清理，137767条生成文件路径逻辑长度约29.02GB，按报告口径不当作实际释放空间；报告/日志/截图及本次EXE/构建输入保留。完整回执output/validation/local-package-72ed53fa/report.json与cleanup.json。
- 历史打包故障适用性：唯一构建/新前端/同字节身份、科学探针和媒体帮助、中文深目录/长路径真实安装卸载、活动租约/研究资料、缓存进程与设置清理、两种实际换版、默认GPU可见音标滚动与普通/最大化窗口均适用，交由verify_local_package.py绑定最终SHA执行。此轮未改打包/恢复机制和字体/玻璃公共样式，因此首次/二次启动三轮性能对照、共享复制新机制、三字体七字号扩展矩阵不属本次必要扩展项；统一验收脚本仍实际执行了168帧完整矩阵并通过。不外推到干净系统或实体设备。

## 基础 GL 对照锁定与青木说明（2026-10-16）

- 按井井最新要求，先核查六段处理前，再完成修正并发布。初始公网r196722d9534d7cb4的六段WAV与2026-10-15 generation-checks逐字节一致，源参数均为random、momentum=0.99，没有变成脉冲。核查证据：output/validation/website/basic-gl-audit-20261016/before-audit.json。
- 为满足最基础GL、不优化，本轮显式关闭动量，使用相同输入/时频标定、10ms Hann、60dB（−60～0）、随机种子20261015、256次、16kHz/512 FFT/24样本帧移。复用研究共享reconstruct，sources=None、momentum=0，CUDA RTX5090 Laptop GPU生成六段，输出24k PCM16。其余25段音频字节保持。增益仅用于试听电平及防削波，页面只衰减较响侧，不加入相位或频谱优化。
- 新增temporary-site/AGENTS.md与assets/spectrogram-examples/basic-gl-baseline.json，锁定六段和输入SHA、固定参数与禁止项。website_audio_baseline.py由页面构建/音频导入/发布器调用。verify_website_audio_baseline.py实际验证pulse、0.99动量、帧移/seed漂移被拒绝，六段覆盖被拒绝且原字节保持，已有锁定时自动重算被拒绝。
- 青木说明按Aoki 2023原文186页第3节、187页图1–2核对：沿频率求和得到时间投影，提取峰值脉冲构造激励，以其相位初始化GL。文案及第二步卡片补齐该过程，明确处理前禁用脉冲和动量。原文：https://www.jstage.jst.go.jp/article/ast/44/3/44_E2297/_pdf/-char/en 。动量来源独立于青木方法，参见Perraudin等2013《A fast Griffin-Lim algorithm》第V节：https://perraudin.info/publications/perraudin-note-002.pdf 。
- 3.0.2 EXE核对：构建源snapshot SHA与build-info一致，冻结reconstruction.py与实际打包目录文件一致，只有原始GL投影循环、没有adaptive.py或动量项。当前开发adaptive.py写死0.99且界面无动量参数。本轮没有修改或重建EXE。
- 初次NEXT验证发现旧导航检查仍要求18px，与此前已获确认并上线的13–16px响应式规则冲突。更新检查为核对实际字号公式及文字边界，未修改页面字号或放宽页面几何检查。源码NEXT报告verify-1792154671383-11992与冻结NEXT报告verify-1792154903427-67656均通过八档导航、六图十二段完整播放/HTTP摘要/音量/互斥、五档布局/中文与IPA/无JS，errors/requests为空，scratch已清理。已检查最终1280px长图。
- 最终冻结目录：output/validation/website/ribbon-basic-gl-final-20261016。候选rb07973adb4ab0b99共92件，相对原公网10改变/3新增/79保持，处理后与其他音频、图源、JS、下载保持。增加三维图标/来源与基础GL锁定清单。冻结首屏DPR3报告verify-1792155011889-68188、Release报告verify-1792155011888-7620通过，错误及失败请求为空。
- 发布时序记录：初次图标版rd15cc62908f81009上传开始后收到先核查音频再上的新要求，本机上传停止，但远端已接手并随后切换。核查后立即以首页SHA并发保护恢复r196722d9534d7cb4，receipt为ribbon-release-20261016/publication-hold.json，配置和其他路由未改。最终完整候选通过检查后重新发布，最终rb07973adb4ab0b99已上线，https-readback.json验证92资源大小/SHA、两EXE HEAD、CSP及论文目录/站点配置/统计配置/反馈PID保持。公网NEXT verify-1792155175464-71204、RIBBON verify-1792155175483-67884、Release verify-1792155175488-69000均通过，无页面错误与失败请求。

## 首屏功能带与行内代码（2026-10-16）

- 黄色功能带主体缩至104px，缩小波浪和首屏留白，新增论文阅读、国际音标。按井井对首版的反馈，再将整条位置下移约40px，保留厚度。图标前两版SVG被否决，最终采用内置imagegen参照原主页插画生成的六个立体小物件，已获井井对风格的确认。
- 原图为1536×1024透明RGBA PNG，3×2等分、每格512px，无后期图像编辑；使用CSS分格展示，桌面68px、手机48px。源文件为temporary-site/assets/feature-icons-3d-v1.png，来源摘要与生成提示见同目录feature-icons.provenance.json。网页无CSS模糊或旋转。CSS为维护源，HTML由build_website_expansion.py生成，Release原文、音频及其他素材保持。
- `WEBSITE_RIBBON_ONLY=1 node scripts/verify_website.cjs`在Windows独立Chrome155.0.8059.79通过。1920×1080、1440×900、1280×800、1024×768四档桌面功能带主体均104px高，含上下波浪在首屏内完整可见。800、390、320px排版及全页横向边界通过，手机六项两列排列，无JS仍保留六项。原图HTTP回读字节及SHA一致。
- 真实更新小记中的0.1核对背景、字号、nowrap及单一行框通过。长代码限制在自身宽度内滚动。首轮波浪背景宽高比留边已修正。已目视核对最终桌面、手机和更新小记截图。
- 第一版预览DPR1，放大后图标及文字均出现像素锯齿；最终专项使用DPR3，1280px桌面PNG实际3840px宽，另保留ribbon-desktop-3x.png和synthesis-icon-3x.png。高清预览目视边缘清楚，此结论不等同于实体高DPI设备验证。
- `WEBSITE_RELEASE_ONLY=1 node scripts/verify_website.cjs`通过3.0.2下载元数据/路由、三个版本原文与摘要、鼠标键盘切换、四档排版及无JS。报告：`output/validation/website/verify-1792153929723-61768/report.json`，errors/requests为空。
- 最终报告与截图：`output/validation/website/verify-1792153914793-35252/`。errors和requests均为空，隔离scratch已清理。生成器再次运行前后首页SHA一致。后随rb07973adb4ab0b99上线并完成公网复验。此首屏专项未复验其他浏览器、Qt、最终EXE及实体手机。

## 六图最佳模型公开更新（2026-10-15）

经井井明确授权发布 r5f2b493b78bd750e，冻结候选位于 output/validation/website/six-best-rounded-public-20261015。变更6段处理后WAV、3项来源/对比JSON、首页，共10项变化/80项保持/0新增删除。旧首页私有备份，版本目录原子切换；不写Nginx配置、不重载，论文目录与反馈服务检查保持。回执：website-47a51543b8cf4f4e849931911b0b38d2/deployment.json。

公网专项 verify-1792057243525/report.json：Chrome154六图12段完整播放、HTTP SHA、整数窗长、对齐音量、8档导航/5档内容布局、无JS通过。整站历史截图数量28!=26断言未扩改，专项通过不代表整站默认测试通过。研究训练另在独立项目进行，不更新公网模型。

# 2026-10-15 六图最佳权重与整数窗长（本地）

parameter-regressor-stream-v5/best.pt，SHA256 7ac61f075259099e0c58e743818b2b61a1ec42d5a125eab6ef9fc9216182a86e。80带修复：mel80-online-mixed-v1/paused-best-20261015-162103/best.pt，第6000步最佳，SHA256 a60a3587139ffdc79b69978615b2079c66a43282cb2b77ab99d276717b433ab2。BigVGAN权重保持。

预测窗长 Smith22.098259、levantan4.538491、observo5.361847、Lu modal4.094801、pressed4.907186、上传图5.488895ms，按floor(x+0.5)取整得到22/5/5/4/5/5ms。先按整数ms转16kHz采样点，动态范围保留预测小数，256次脉冲GL，512FFT、Hann窗。固定hop24=1.5ms；四分之一窗长组hop88/20/20/16/20/20=5.5/1.25/1.25/1/1.25/1.25ms。图片时间重采样网格与STFT、iSTFT共同使用同一hop。Mel原生前端22.05kHz、1024窗、256帧移保持。

输出位于F盘randomized-auditions-20261015-v2的six-best-rounded-fixed-v1、six-best-rounded-quarter-v1和six-best-rounded-comparison-v1。合并页同图原主页GL基线／最佳固定hop／最佳四分之一hop三路共18段，RMS对齐且无削波，24kHz长度正确。原主页六段处理前保持；更新六段处理后与参数/波形/摘要。本地主页 http://127.0.0.1:8765/#next ，三路对照 http://127.0.0.1:8767/six-best-rounded-comparison-v1/index.html 。未公开部署。

验证：GPU实际推理，窗长采样点及四分之一hop断言、源图/权重SHA、有限非静音/输出峰值/长度/RMS通过；环境与迁移通过。合并页25资源字节回读通过，18段浏览器完整播放、1400/390布局通过。主页针对next验证verify-1792056304506通过六图12段完整播放/资源SHA、参数与实际整数窗长、音量、互斥、导航和5档布局/无JS，截图检查通过。导入与生成重复执行字节幂等。

此前误跑整站默认验证verify-1792056073019，末尾既有无JS截图数量断言失败（实际28，旧断言26），页面错误/失败请求为空；未修改该无关测试，不能声称全站测试全部通过。本轮受影响六例专项检查通过。研究试听完整播放证据在合并页browser-verification.json。未推送、未修改应用、未训练，未作听感提升结论。


# 官网改版检查

## 六组对比与10ms默认GL（2026-10-15）

井井明确更正所有处理前为10ms/60dB，继续不用脉冲方法，并加入Lu modal/pressed及知乎例图（2秒/6kHz），追加完整载瓦语论文引用。六图顺序为Smith、levantan、observo、Lu modal、Lu pressed、知乎例图，下一步区六图/12播放器，桌面三列两行，<=800px一列。小实验区保留原三图。

处理前均为随机相位初始化GL，10ms Hann、60dB（−60～0dB）20log10幅度映射、无预加重、16kHz/512FFT/24帧移/256次/动量.99/seed20261015。不接参数模型、Mel或声码器。研究入口generate_website_six_defaults.py登记独立六图清单后复用原生成函数，sources=None，不调用pulse_source，三组新增裁剪与研究输入字节相同，六例脉冲全零/峰数0、模型为空/training=false。输出F盘randomized-auditions-20261015-v2/website-six-defaults-10ms-20261015-v1/generated，已在RTX5090实际CUDA生成，环境/迁移通过。

六段24kHz样本35520/22402/17467/12245/12566/48000，有限非静音/峰值<.951。新增三段处理后逐字节取three-predicted-parameters-v5顶层comparison_audio.new-base80-repaired，与截图指向的预测参数80带路径相同。预测modal4.094800949ms/83.423828125dB，pressed4.907186031ms/87.357498169dB，知乎例图5.488895416ms/55.038040161dB，按16k量化66/79/88样本。原三段处理后仍为stream-v4导出，新三段为stream-v5最佳第26轮，逐对参数模型SHA明确记录，不能用单一模型ID覆盖六例。权重和处理后音频未重新生成。

全页31个WAV，旧25个仅3处理前改变，其他22个保持，新增6个WAV及3个谱图裁剪。每对播放时仅降低较响侧使RMS相同，无JS原生控制可手动调音量。原图引用/裁剪与公开音频provenance、缓存SHA同步，不公开本机路径。Lu两例作者/完整题名/卷和文章号/DOI位于对比下方，卡内短引用直达相同论文；知乎例图按用户命名，未知原作者不编造，第三方登记记录界限。

生成/音频身份回执output/validation/website/six-comparisons-20261015/generation-checks.json。本地verify-1792041142081通过12段HTTP大小SHA/时长/完整播放、相位/默认参数/实际音量、跨区互斥/局部状态、八档导航/五档布局/字体/目录键盘/无JS，页面错误及失败请求为空，1600截图已检查。最终命名及完整引用版verify-1792041375314本地与verify-1792041515354公网全部通过上述检查，并核对知乎例图名称、Lu完整题名/DOI及六张图片HTTP SHA。页面错误/失败请求为空，1600截图已检查，浏览器scratch清理状态deleted。

最终rbe930860a22563bb已上线，90项HTTPS完整大小/SHA均一致。相对任务开始的rb7d9b5f332919771为7项改变、9项新增、74项原字节保持，无移除；最终命名/完整引用相对中间r7067f210c837e052为3项改变、87项保持。范围仅首页next内容及CSS缓存版本、新引用的字号样式、三图/六WAV、原三段处理前WAV及来源清单。其他HTML区域归一化逐字相同，JS原字节保持。旧首页私有备份并原子切换，论文catalog SHA相同/反馈服务active，配置未写入/重载，未推送GitHub或重建应用。回执output/validation/website/website-989d8372c6dd43e7a0c834de6ebcc748/deployment.json，最终清单/范围/HTTPS/build检查在output/validation/website/six-comparisons-final-20261015。

六例导入/HTML重复运行幂等，源与冻结清单一致。旧三例导入入口禁止覆盖六例列表，实际调用验证在写入前拒绝，来源清单SHA保持；原三例处理后导入改为按ID更新和重新计算相应播放衰减，保留新增例子。UI数据80参数/400显示来源通过，文档2041文件/436登记/110任务errors=[]，JS语法及差异空白检查通过。实体手机、Safari/Firefox、长期压力及主观质量提升未验。

下节为此前三图20ms中间版，当前设置与发布结果以上节六组10ms为准。

## 处理前采用未知参数默认设置（2026-10-15）

井井要求处理前采用未知参数默认设置，并进一步指定去掉脉冲方法。最终三图为20ms Hann、60dB（−60～0dB）、20log10幅度映射、无预加重，固定种子20261015的随机相位初始化GL。16kHz/512FFT/24帧移/256次迭代/0.99动量及原裁图区坐标保持，不用参数预测、Mel修复或声码器。处理后仍保留原固定stream-v4预测参数与脉冲GL→80带NAFNet→BigVGAN导出。

在独立研究项目原脚本compare_five_pulse_vocoders.py增加--phase-initialization random，原专项pulse默认不变。随机分支不调用pulse_source，sources=None使用原reconstruct的随机相位。三例脉冲数组为全零占位、peak_times/heights为空、peak_count=0；参数模型为None、models为空、training=false。RTX5090实际GPU生成来源为F盘randomized-auditions-20261015-v2/website-unknown-random-defaults-20261015-v1。环境及13核心迁移兼容通过。

三段24kHz长度35520/22402/17467，有限、非静音、峰值小于0.951，25WAV仅三段处理前变化，三个处理后及其他19段原字节保持。来源/波形/缓存摘要和网页说明同步；沿用防削波导出，播放时按RMS仅衰减较响侧，处理后文件不改，无JS保留原生控制及手动音量。数值和身份回执：output/validation/website/random-before-20261015/generation-checks.json。

本地独立Chrome154 verify-1792040675875与最终公网verify-1792040778615全部通过：三例来源相位random、20ms/60dB/Hann、展开说明可见随机相位、六段HTTP SHA/时长/完整播放/按钮复位、逐例播放音量及RMS乘积、跨区互斥/局部状态、八档导航/五档布局/字体/键盘目录/无JS原生控制。页面错误/失败请求为空，1600和390宽度截图已检查，scratch清理回执deleted。首次新增说明断言读取折叠details的innerText而失败，修正为真实展开、验证可见文字再收起，未放宽断言。

rb7d9b5f332919771已上线，相对上一中间版r11086594fce3649b仅6项改变（HTML、三段处理前WAV、audio/provenance、comparison.json），75项原字节保持，无新增/移除，JS和配置保持。首页除next区外归一化逐字一致，生成入口及HTML重复运行幂等、当前源与冻结清单相同，81项HTTPS完整大小/SHA一致。旧首页私有备份并原子切换，论文catalog摘要相同、反馈服务active，服务器配置未写入或重载，未推送GitHub或改应用。发布回执output/validation/website/website-a35a1fc126484f07be4b7726d9edca4a/deployment.json，清单/范围/HTTPS及生成验证位于output/validation/website/random-before-20261015。上一中间版采用20ms脉冲GL，已由本次随机相位版替代。

文档检查2041文件、436来源、110任务，errors=[]；历史快照未解析链接保留原记录。JS语法与git diff --check通过。实体手机、Safari/Firefox及长期压力未验，本次没有主观听感提升的验收结论。

## 下一步与三图前后试听（2026-10-15）

本节记录此前10ms处理前对照和r8e4142c002c06805发布；当前20ms默认处理前及最新发布以本页上节为准。

本地将入口改为“下一步…”，放在更新小记之后，方法与引用紧接小实验。顶栏、阅读目录及正文一致，桌面导航18px，窄屏15–16px，1251px以上单行。介绍保留快乐体，改为韶子参考 Aoki（2023）短时窗语谱图研究，训练条件化 NAFNet 并级联 BigVGAN。官方论文 [J-STAGE](https://doi.org/10.1250/ast.44.186) 的第3节用幅度纵向投影峰脉冲提供 GL 初始相位，未提出 NAFNet 或 BigVGAN，网页分别归因且保留完整题名与书目信息。

按井井后续要求，当前流程为语谱图 → 参数小模型 → GL → Mel修复 → 声码器。固定stream-v4最佳参数模型，真实GPU预测与用户截图/原results.external逐项相同：Smith 21.731321 ms/66.803658 dB，levantan 4.654337 ms/55.789063 dB，observo 5.318752 ms/59.594707 dB。下限为−D、上限0dB，模型未预测窗型。按16kHz将窗长量化为348/74/85样本，实际21.75/4.625/5.3125ms，Hann/512FFT/24帧移/256次脉冲GL保持，随后采用原80带第5500步修复模型和BigVGAN。Mel条件保留原GL修复路线的未知标记，预测参数用于GL。

处理前保留原五图实验10ms/60dB GL的三个WAV，处理后在研究项目实际重新合成，源目录为F盘randomized-auditions-20261015-v2/website-predicted-parameters-20261015-v3，逐字节导入三段base80-repaired。GPU为RTX5090 Laptop、已有pytorch_5090/fp32，未更新权重或训练。六段24kHz，处理后35520/22402/17467样本、有限/非静音/峰值小于0.95，与保留的旧对照RMS误差小于5e-5；网页25个WAV仅三段处理后变化，其余22段保持。图片/坐标/出处不变，公开来源保留模型及导出SHA、不包含本机路径。生成代码及研究记录位于SpectrogramReconstruction，环境和迁移检查通过，未复制算法至应用工程。预测不作原图参数真值或听感提升结论。

`$env:WEBSITE_NEXT_ONLY='1'; node scripts/verify_website.cjs` 在独立Chrome154、自有回环服务/浏览器配置通过八档导航（320–1920）、五档介绍区（1600/1280/800/390/320）、导航文字与DOM顺序、原生键盘锚点、阅读目录/当前标记、真实字体、四步流程横向/单列、三图并排/单列与文本边界、无JS原生六音频控制及减少动态效果。六段HTTP SHA/时长/完整播放/ended与按钮复位、局部播放状态、互斥播放、全页ID唯一通过。页面错误和失败请求为空。已目视检查桌面导航、整块介绍及390px截图。浏览器scratch已清理，见 `output/validation/website/verify-1792038981970/report.json` 与cleanup.json。

井井审查后明确要求推到服务器，当前r8e4142c002c06805已部署。冻结候选与原ra956dd882260d399比较，8项改变（HTML/CSS/JS、三段处理后WAV、两份来源）、4项新增（三段处理前WAV及comparison.json）、69项原字节保持，无移除；排除审阅区域后其余HTML一致。发布清单补入comparison.json。沿用发布器，校验线上首页仍为准备时SHA，旧首页/配置私有备份，新目录逐文件校验后原子替换首页；断言CSP原字节一致，未写配置或重载Nginx。论文目录摘要保持，反馈服务活动。安装包、论文、模型/语料不在本次静态上传范围，未推送GitHub、启动训练或重建EXE。

发布候选 `verify-1792039765261`、最终公网 `verify-1792039846483` 均通过前述八档导航/五档内容、预测值及六段完整播放等检查，页面错误/失败请求为空，已核查公网整块截图，scratch回执均为deleted。81文件公网完整大小/SHA256一致。发布回执：`output/validation/website/website-2463720bd0e74f32856d52276534ad90/deployment.json`；配方、发布清单、范围、HTML区域核对和HTTPS回读在 `output/validation/website/next-step-deploy-20261015/`。实体手机/其他浏览器未验。25段音频摘要、导入/HTML字节幂等、JS语法与差异检查及生成回执保存在 `output/validation/website/predicted-parameters-20261015/`。早先纯介绍版证据保留在next-step-20261015，不代表本次重合成音频。

修复诊断：初次研究生成的模型元数据重复关键字已修正；最终输出单独匹配官网旧对照，避免同组其他路线峰值约束改变展示响度。首轮页面生成因局部变量遮蔽输出路径而失败，修正命名后重建。320px长页面第三张图片在原生懒加载阈值外，测试等待decode不触发加载；最终截图检查先显式请求图片，保留图片/边界断言，产品仍正常懒加载。仅终止已核对归属的该轮Chrome释放等待，失败报告和scratch清理回执保留。

## 语谱图试听替换为训练后 80 带结果（2026-10-15）

按井井指定取五图试听页第三路，Smith、levantan、observo 分别复制 `*-listen-base80-repaired.wav` 至官网三个试听按钮。导入逐例核对原图字节/摘要、来源、时长、频率标定及源导出SHA256，三段均为24kHz，采样数35520/22402/17467，保留原试听响度。路线为10ms Hann、60dB幅度映射、256次脉冲初始化快速GL → 第5500步80带NAFNet Mel修复 → 原BigVGAN。窗长/动态范围为假设，无新训练/重合成，不据此宣称听感提升。现行处理来源单独记录，旧GL参数明确保留为历史基线。首页实际参数说明与波形更新，音频链接加摘要防旧缓存；默认旧生成入口保护已确认导出。

Windows独立Chrome154本地 `verify-1792033359617`、发布候选 `verify-1792033497339` 和最终公网 `verify-1792033557151` 均通过三段HTTP字节摘要、正确时长、按钮完整播放结束及状态复位、无媒体错误、方法说明和1280/390宽度无溢出；本地截图已目视检查，页面错误/失败请求为空，浏览器scratch清理回执为deleted。音频导入与HTML生成重复执行字节幂等，22个音频均匹配来源清单。

井井明确选择同步到官网后，从当前线上r6a934e4e286065db建立候选，只替换首页对应卡片、三段WAV及两份provenance，其余71项资源摘要保持。本地尚未发布的“接下来……”未进入候选。沿用发布器，保留旧首页私有备份并原子切换，禁止同期首页变化时覆盖，断言既有CSP一致且不写配置/重载Nginx。当前发布ra956dd882260d399，77项资源公网完整大小/SHA256一致，论文目录摘要保持、反馈服务活动。首次公网浏览器检查因验证器未识别既有版本目录前缀失败，修正路径归一化后保持文件名/摘要断言并通过，无产品修补。

配方与回执：`output/validation/website/spectrum-80-20261015/` 中 prepare.py、scope-check.json、deploy.py、manifest.json、readback.py、https-readback.json；部署回执 `output/validation/website/website-cb544e4572774a1db3871e6c7a320dbc/deployment.json`。文档、JS语法及定向差异检查通过。实体手机、Firefox/Safari、Qt与最终EXE未验，本次未改主应用或推送GitHub。

## 接下来本地预览（2026-10-14）

当前本地页面在“小实验”后新增“接下来……”入口，区块紧接小实验、位于常见问题之前，阅读目录同步。卡片用现有完整 ZCOOL KuaiLe 快乐体，展示井井指定的 U²-Net 几何校正、条件化 NAFNet 图片转 Mel 谱、BigVGAN 合成音频与 KeSpeech / LibriSpeech / LibriTTS 训练计划。loss 曲线和例子截图留为等待实际成果的便签，没有生成占位训练曲线或冒充实际结果。该路线及声音质量写为未来目标，训练与科研性能未在本次验证。

定向入口：`$env:WEBSITE_NEXT_ONLY='1'; node scripts/verify_website.cjs`。本机证据目录为 `output/validation/website/verify-1791965349417/`。检查范围为 Windows Chrome 154、项目自有回环 HTTP 服务及独立浏览器配置：320–1920 px 八档导航、1600/1280/800/390/320 px 介绍区、原生键盘锚点跳转、阅读目录当前标记、三列/单列切换、文字边界、实际字体、减少动态效果及无 JavaScript 显示。最终 report.json 为 success=true，九项汇总检查通过，页面错误及失败请求为空；实际字体为 ZCOOL KuaiLe，截图经目视复核。

初轮修正：作者寄语字体仅有161个字符，改用7053字符的原有快乐体并检查介绍文字中文覆盖；锚点检查按既有 scroll-padding 与 scroll-margin 的实际留白计算；卡片间装饰箭头计入 scrollWidth，改为直接核对正文文字矩形及视口边界；无JS截图遵循减少动态效果以避免原生平滑滚动的稳定性等待；介绍区预留便签外伸位置，截图不裁掉标签。失败轮诊断保留，独立浏览器与服务由脚本收尾，scratch 清理回执单独保存。

其他正文、导航提示、试听资源、版本及下载内容保持。生成只改变样式缓存摘要、主导航、介绍区与阅读目录四处，重复构建核对字节幂等。仅本地预览，未部署官网、运行训练、重建 EXE 或推送。实体手机与其他浏览器未验。文档检查2041文件/435来源/110任务零错误、定向 git diff --check、JS语法及重复构建通过，回执见本机 `next-preview-20261014/checks.json`。

`WEBSITE-REDESIGN`：2026-10-12 六项导航、真实录音试听、音系归纳等截图、诗歌首图与匿名统计扩展已上线并通过公网核验，邮箱反馈、版本切换、精确计数时间及三张连续长图已完成验证和交付。

## 页面与素材

采用奶油色、黄色与紫色设计。首页六字于2026-10-18改为井井选定的童趣手写稿I，原PNG嵌入SVG，语音学下方加入贯穿三字、起伏与粗细变化的黄色手绘线，来源与配方见下方本次验收；外侧 [ʂ] [ɻ] [ɕ] 参考 Doulos 结构重新绘制。中央标注使用 Doulos，右侧语谱纹理为装饰。正文无研究教学副标题，网页版日期为 2026 年 10 月 31 日。导航为小实验、常见问题、更新小记、方法与引用、问题反馈、联系我们，删除首页浏览功能提示。首页左侧文字、日期和按钮小幅分层鼠标跟随，按钮移动不超过 3px。减少动态效果与触屏不启用该跟随。

功能展示共十六组二十五张中文截图，依次浅/深交替，保留主题名称和色点。音标叠三图、转写叠两图、EGG叠两图、语谱重建叠三图、变速变调叠两图、音系归纳叠三图，角度错落且露出可点区域，每张打开自身原图。原图提示仅第一组保留。十五个短波浪/星点分隔穿插其间。

- IPA/extIPA/VoQS 使用 Catppuccin 深色。Standard Chinese (Beijing) 与 UntPhesoca严两版诗歌使用 Rose Pine 浅色。诗歌全文252汉字36行，每行七字加一个标点，等宽排版、1.2行距，扩大IPA区域。
- EGG 为 GitHub 白底，补充实际逆滤波结果。
- Ayu 深色三张分别为图片重建、画三条曲线后实际重建、写入PTB后实际生成音频，均显示输入与结果对比，三张文件不同。
- One 浅色单音节取自用户指定 test 目录，基频 +15Hz 后合成。批量示例起点/拐点/终点各三基频，全连接生成27项。
- 新增 Gruvbox 深色发声类型合成、Monokai 深色 MFA、Material 浅色 TextGrid、Night Owl 深色 LPC、Gruvbox 浅色音系归纳。Tokyo Night 深色外观移至图标与说明书附近，随后为Dracula浅色图标与Temple深色说明书。其余为 Everforest 首页、Solarized 生理参数、Nord 唇形。

新截图均为2560×1440，应用缩放150%。原生Qt截图禁用额外的系统DPR叠乘，仅影响独立截图进程。用户唇形截图保持2559×1527，视频矩形 `(1891,333,601,340)` 永久覆盖可爱保密贴纸，区外像素完全相同。除用户明确授权的 test 目录试听片段外，不上传其他原始录音、任务库、模型、词典或无遮挡视频图片。

说明书截图后依次为自定义背景、更多功能、试听、常见问题、更新、方法、反馈和联系，再接使用说明与源码和作者气泡。大号省略号与“更多功能，等你在应用中发现。”SVG文字整体紧接自定义背景，保留六点、文字和样式。作者头像用用户提供的图片，只有头像链接知乎。气泡尾巴采用连续曲线与同色覆盖边框连接，桌面正文18px、手机16px。寄语用词为“需要保留的版权声明、许可证和第三方声明，请不要删掉”。修改字体的资源地址带内容摘要，避免旧缓存使新增汉字回退。

[字体核查](website-font-license-audit.md)：Doulos SIL 与 Phonetic Whisper 均遵循 OFL 1.1，后者是 ZCOOL KuaiLe 修改版，已重新命名并保留版权和完整许可证。系统字体仅本机引用。字体不受作者寄语中的项目非商业限制影响。页脚仍不显示素材入口，实际许可文件保留。

## 本地证据

- `capture-1791715879038/report.json`：首页、IPA、基础转写、外观。
- `capture-1791716120622/report.json`：IPA/extIPA/VoQS及全文等宽诗歌。
- `capture-1791716131879/report.json`：使用说明。
- `native-modules-d5a597dec2d44363a76d147fcead932b/report.json`：五组原生截图。
- `examples-a1d7fdc55e8f4848b0c4a924ca171ff1/report.json`：白底EGG及图片重建。
- `pitch-example-b25d3aecd75f4a0997687f09da36122a/report.json`：单音节+15Hz真实合成。
- `additional-stacks-8d14b53f849a4fff8e7c6b683e7921d5/report.json`：逆滤波与27项批量生成。
- `spectrum-three-1469d3d8298b42e9a057344e7305acfb/report.json`：最终三张不同模式的真实生成结果。

上述相对路径均位于 `output/validation/website/`，仅作为本机证据。各次scratch按清单清理，保留最终图、报告与回执。图形/字体/媒体来源分别记录于网站assets三个provenance文件。

`node scripts/verify_website.cjs` 在独立 Chrome154 中核查：25图尺寸与加载、16组深浅交替、15张叠图各自露出区域点击和原图身份、鼠标跟随、提示/导航删改、配色色点、头像唯一知乎入口、作者寄语字体、气泡与手机排版、发布日期、两个真实下载地址（测试拦截转移，避免重复下载大文件）、滚动入场/透视、弹窗Escape/焦点返回、黄页脚、1440/800/390/320宽度、无JS与减少动态效果。`删掉`两字另外通过浏览器实际字体报告检查为 Phonetic Whisper 的两个字形。

## 生产发布

发布脚本 `deployment/temporary-site/publish.py` 只取当前页面实际引用的资产与OFL许可，排除临时语料、来源本机路径、旧截图和安装包。资产先进入独立版本目录，再原子替换首页。保留服务器既有 papers 目录与反馈路由，旧首页和Nginx配置在私有目录备份。调整本站CSP允许同源脚本、样式、字体、音频与统计请求，检查nginx配置后reload，不改凭据、数据库、其他服务或系统运行时。

公网地址 https://www.phonetictoolbox.com/，当前发布`r9cd63448bcea7d8e`。`background-four-order-20261013/deployment.json`与`https-readback.json`确认76资源公网大小/SHA一致，服务器备份`/var/backups/phonetictoolbox-website/downloads-201ecab8971149f59dcb882830eeed84`。3.0.1双下载入口发布前核对服务器完整摘要，论文目录与反馈服务保持，nginx配置不改。页面/源码检查不代替Windows成品功能验收，Safari/Firefox、实体手机和长期压力未验。

## 本轮真实材料与读音

- `phonology-09257815345349eaa6a83bd4543eb6e2/report.json`：用户测试.xlsx从第一行导入，5453条记录保留，34声母、44韵母。调值1/2/5/6/7/8依次命名阴平/阳平/上声/去声/阴入/阳入，仅为演示，不作真实调类判断。示例文件显示名、网页说明与原图图注均明确标注。实际生成三份文件后拍摄调类、声母到韵母、二维表格，矩阵在网页最前。原表SHA256不变，临时输入及生成文件在进程退出后按清单清理，最终截图和报告保留。首次采集因检查选择器引号错误失败，修正检查后重新完整采集，不涉及应用逻辑修改。

- `expansion-b4624a3d713a40169e99cc816d20d088/report.json`：四组当前模块与浅色外观，2560×1440/150%。真人 ma 录音制作7步发声连续统，手工 TextGrid 独立副本。该次 MFA 中文词典截图随后按用户要求替换。
- `mfa-pinyin-e72b9f51d4854e2cb5f15300fd42e299/report.json`：原拼音LAB、用户指定 mandarin_pinyin_tab.dict 与相容 mandarin.zip，MFA3.3.8 实际对齐完成。原始 WAV/LAB、两份手工 TextGrid、模型和词典前后摘要一致。
- `lpc-view-e7756fff56ca4d7dad052062c1e920e6/report.json`：源码图高58dvh按页面缩放折算，避免150%再次放大。原生深/浅主题核对谱图、Hz横轴和全部操作均在可见区内。修改仅显示尺寸，8项LPC定向测试、typecheck/build通过，不改谱值。
- `capture-1791800539411/report.json`：诗歌首图改为全文，第二张为UntPhesoca严。36行、每行7字和标点。逐字核对252个映射值，处(chù)两处、扁(piān)、卷(juǎn)、还(huán)两处通过应用读音菜单选择。保留现代普通话本调，不应用连读变调或古音叶韵。[扁舟](https://www.zdic.net/hans/扁舟興盡)、[不胜](https://www.zdic.net/hans/不胜)与[诗文原文](https://ctext.org/text.pl?if=en&node=127868&remap=gb)为核对线索。其他多音字按所在语境复核，保留应用对应记音方案。

## 试听与统计

20段WAV：两段真人端点，7步发声连续统，1段真实原声与4种变速/变调结果，3段明确标注的生理模型元音，以及3段带图源引用的语谱图近似重建结果。发声连续统原声为用户指定test目录的dl_m01_0558_ma.wav与dl_m05_0558_ma.wav，变速变调仍用dl_m02_0558_ma.wav，原输入只读。音频provenance记录摘要、参数、时长与实测中位F0。MFA/TextGrid的女童录音只在截图中展示，不作为公开音频。生理参数合成本身是模型发声，页面明确说明这一性质。

匿名统计采用专用Nginx日志与Python标准库汇总，不记IP、Cookie、UA原文、查询参数或身份。仅GET成功页面和两个安装包种类进入日志，已知爬虫/测试UA排除。整文件或bytes=0开头的单范围请求计为一次下载发起，后续续传分块排除，重新发起首段仍计数。状态保留累计值和日志偏移，轮转、gzip和重复执行不重复计入。网页失败时不显示假零。每分钟systemd timer原子更新同源JSON，CSP仅增加同源media/connect许可。

`python deployment/temporary-site/test_stats.py` 两项回归测试覆盖GET/HEAD、200/206/错误、续传、零字节、增量、轮转、gzip、尾部半行、重复执行和损坏状态不归零。`verify-1791804400941/report.json` 本地及 `verify-1791807264208/report.json` 公网Chrome154检查四宽度、17音频、互斥/暂停/结束/键盘、FAQ/Release/邮箱反馈/联系方式、六导航、悬浮提示、统计成功与异常、原有功能。错误和失败请求均为空。三张音系归纳原图点击、二维表居前与示例调类说明通过；补充检查实际旋转矩阵，避免一般透视规则覆盖叠图错落角度。

## 三张连续长图

`long-images-1791811178260/report.json`：从最终公网页面以1600px宽度原生截图，三图依次为1600×9084、1600×8659、1600×7961，连续纵坐标为0/9084/17743/25704，无重叠或缺失。自动保护文字、旋转截图、装饰分隔与完整内容区，在留白处分界。三张成图与两处分界证据均已目视检查，未切断截图或文字。完整保留原声直显、邮箱反馈、联系卡片、展开的3.0.0更新小记与更多功能提示。独立浏览器临时目录已按清单清理。首次导出缺少fullPage选项导致超视口裁剪失败，修正截图参数与PNG尺寸断言后重导，失败及中途布局产物不作最终交付。

## 网页反馈、原声与版本小记

两个真人原始端点直接显示在连续统上方，中位F0为83.29和92.10Hz；七步为发声类型处理结果，约82.17–83.79Hz，保留A的基频轮廓。没有替换原录音或将原声称为相同基频。联系区域改为两张紧凑卡片，邮箱跳转邮件客户端，QQ群点击复制。

网页复用既有同源反馈接口和邮箱验证，首次输入6位验证码。支持最多4张PNG/JPEG/WebP、每张2MB，文字及截图发送、分页会话、管理员回复、邮件提醒、切换邮箱。身份仅存标签页sessionStorage，不改服务库结构、SMTP配置或管理工具。丢失发送回执保留内容和请求ID，重试不重复写入；会话内容以textContent渲染。`feedback-web-1791804522277/report.json`在真实FeedbackService、隔离SQLite和邮件捕获器下验证错误/正确验证码、图片、真实消息、丢失回执重试只存一条、回复文本转义、关闭邮件仍可查看回复、标签页重载、独立浏览器身份隔离及三宽度。测试专用第二版本正文验证鼠标及方向键切换，正式页面只显示v3.0.0。scratch已清理。公网health确认mailConfigured=true；本轮未发送真实邮件或生产测试消息。

更新小记按用户最后要求不自动同步，当前只有v3.0.0版本签，点击选择对应正文。正文及原文附加说明默认展开，表格正常排版，未来可加入版本快照切换。计数明确显示自10月12日北京时间18:23:30起累计，每分钟更新、不跨日归零；对应服务startedAt=2026-10-12T10:23:30+00:00。

最终检查：网页构建连续两次输出字节一致，所涉Git差异检查和文档验证通过。更新计数文案后，一条旧测试断言仍期待“起”，已同步为“北京时间时分秒起计数”并重新完整通过。公网62文件SHA256一致，服务返回的计数起点保持不变，既有论文目录与反馈服务正常。

## 本轮试听与阅读导航

当前本地新增三个指定语谱图引用例子，共20段音频。按坐标校正后的Smith/Levantan/Observo输出为1.4665/0.92175/0.71225秒，M09实际执行，64次迭代、20ms重建窗、16kHz输出、seed=0，参数与输入输出SHA保留。仅对必要图像区域作讲解性引用，权利保留，不将网上可读或用户提交误标为开放许可。顶栏改为轻巧便签导航，右侧增加进度环、板块目录和返回顶部。本地verify-1791806969330通过目录进度/当前板块/焦点归还/Escape/返回顶部，以及三段重建声音实际播放完成。底部授权文字按用户原文更新，祝愿分两行保持紫色和字号，重建字体覆盖新增汉字；官方许可文件保持。r70c928c30519b158已发布，69文件公网大小/SHA一致，原论文目录与反馈服务保持。公网verify-1791807264208通过20音频/三组重建播放完成、目录交互、四宽度、减少动效和无JS，错误与失败请求均为空。长图long-images-1791807727751及两处分界目视通过。文档验证1977文件/424来源/95任务，无错误；差异检查通过。

顶栏追加修正：桌面由13px升至20px，手机15–16px，恢复带图标的GitHub仓库入口。verify-1791807727783在公网验证1920/1600/1440/1250/1100/800/390/320八宽度的七项链接、文字实际边界、字号与仓库目标，错误/失败请求为空。r70c928c30519b158只追加顶栏调整，前一版本完整交互证据verify-1791807264208保留。

## 图片重建参数、品牌及提示框修正（2026-10-12）

对照v2源码，v2/v3核心均采用legacy /10灰度映射，默认−30–0dB、10ms。网站此前误用−60–0dB、20ms，压低浅灰弱成分。保持图像、时频范围、64次迭代、seed=0和16kHz输出不变，分别比较dB与窗长，最终恢复默认范围/窗长。observo末尾0.61–0.69s相对0.40–0.46s元音的RMS比由−32.27dB变为−18.06dB。数值支持弱段得到保留，不宣称恢复原录音或主观听辨等同v2。三段PCM无削波，17段其他音频摘要不变。证据：output/validation/website/spectrogram-parameter-audit-20261012。页面参数从生成清单读取。

黄底雾紫声波/深蓝末端品牌与favicon同步，首页、外观、31项图标页、说明书四图重新截图（capture-1791810675209）。导航提示从固定160px且禁止换行改为内容自适应、最大240px自动换行，首尾贴边定位。verify-1791810891275在八宽度检查七项导航和可见提示的真实文本边界，全部通过。完整本地verify-1791809371549验证20音频及三段重建实际播放完成。

最终发布r9e9f4fc0929d03a0，website-149aa34d220b4d74b25d9f8e81974d2a确认69文件公网大小/SHA256一致，论文目录与反馈服务保持。公网verify-1791811177900完整检查通过25图、16组主题、15张叠图、20音频、三组重建播放完成、目录/进度/返回顶部、四宽度、无JS和减少动效，错误和失败请求为空。verify-1791811178074在320–1920px八宽度验证七项导航及桌面提示框实际文字和外框边界。最新长图为long-images-1791811178260，连续分界0/9084/17743/25704，两处分界目视无文字或截图截断。文档检查1990文件/424来源/96任务通过。

## 自定义背景展示（2026-10-12）

按用户最新指定移除深浅配色整块，在使用说明展示后、小实验前加入两张用户截图，DeepSeek设置页在前、首页在后，沿用两卡错落叠图和原图查看。两份PNG逐字节复制，均为2559×1527，原始文件摘要不变；来源记录进入screens/provenance.json。展示总计16组26图，其余工具主题保留。构建重复运行内容摘要一致。本地verify-1791814659343通过26图加载、17叠图逐张点击和键盘返回、四宽度、20音频、无JS及减少动效。最终标题简化为两行，公网实际截图已目视检查。rdccdfc56fbd82ee1部署成功，70资源公网大小/SHA一致，论文目录摘要与反馈服务保持。三张新长图long-images-1791815035679为1600×9084、1600×8528、1600×7961，连续边界0/9084/17612/25573，两处分界目视检查均未截断文字或截图。首次完整公网检查的新展示及交互通过，后续短元音播放立即断言因异步加载竞态失败；验证脚本改为等待实际播放和按钮状态，并以0.25倍速覆盖暂停，最终公网verify-1791815875900完整复验通过，错误及失败请求为空，26图/17叠图、20音频和四宽度布局均正常。文档检查1996文件/424来源/98任务无错误。

## 3.0.1更新小记与首页背景图（2026-10-13）

按用户指定导入release/notes/3.0.1.md，快照逐字一致并记录源摘要。两版按钮按版本号降序，3.0.1默认显示、3.0.0保持可切换与全文展开。最新文案在公开GitHub发行页存在前不添加猜测链接。用户新首页截图逐字节替换后层图片，DeepSeek保持居前。verify-1791821830176本地检查完整文案、标题、列表、加粗、三行下载表格、鼠标和键盘切换、四宽度及无JS最新内容显示通过。首次测试未剔除链接附加箭头导致文字比较失败，产品文案完整；修正验证后通过。构建重复运行摘要一致，首轮raf9ef48ebc358b96上线。用户追加3.0.1双下载入口要求后，以publish_download_page.py在服务器核对两包完整SHA再原子切换r79b8a234f53c3e13；72资源公网大小/SHA一致，ptb-release.json版本为3.0.1，两个EXE的HEAD200及首尾Range206与本地最终字节匹配。下载版本号、提示大小同步为3.0.1、495.8/497.8MB。公网浏览器通过版本切换与新图展示，旧验证代码尝试滚动隐藏旧版本面板导致失败，验证改为逐一检查当前可见面板，后续完整复查verify-1791823229138通过。用户取消长图生成，取消前已完成的输出不作本轮交付。

### m11/m03与REAPER双变化连续统

用户指定原声A为dl_m11_0558_ma.wav、B为dl_m03_0558_ma.wav。两原声逐字节复制，原输入摘要不变。当前M07核心以REAPER分析源信号和LPC残差，ContinuumType.F0_AND_PHONATION生成七步，保持既有周期能量与输出幅度匹配和其他分析默认值。REAPER范围50–300Hz，1ms帧距，使用登记摘要的原生二进制，无后端回退；输出最终PCM也重新用REAPER测量。两原声中位F0为100.628929/85.561501Hz，七步为100/98.159508/95.238098/92.48555/91.428574/88.397789/86.021507Hz。七段摘要各异，F0逐步下降，九段均无削波，其他11段音频SHA保持。Scratch输入WAV在每次REAPER调用后清除，独立目录关闭后已移除。证据output/validation/website/phonation-reaper-20261013/verification.json。首次输出测量因Praat采样率为浮点数被AudioInput契约拒绝，适配器显式转换整数后生成成功。仅改网站示例生成脚本与文案，不改M07算法或科研默认值。本地完整verify-1791822976747通过26图/17叠图、3.0.1下载与版本切换、20音频及九段REAPER试听摘要、四宽度、无JS与减少动效。最终r1945daf3c7c521d9上线，72资源公网摘要一致，论文目录与反馈服务保持；最终公网verify-1791823229138完整交互复验通过，错误和失败请求为空，未生成新的三张长图。

最终证据：downloads-301-20261013/final-deployment.json、final-https-readback.json、downloads-readback.json确认最终页面/72资源、3.0.1包身份与论文/反馈保持。旧隐藏版本不属于当前可见滚动检查，但版本切换仍分别覆盖两面板；手机与桌面内容检查通过。文档验证2004文件/424来源/101任务，无错误。

## 知乎文章与自有音频（2026-10-13）

语谱图转音频展示正文添加文章按钮，跳转用户指定question/793458414/answer/4607998600，新标签并保留noopener/noreferrer，不显示裸网址。不转载文章内容，不根据无法读取的页面编造标题。A使用授权test目录现存dl_m02_0558_ma.wav，B使用明确提供的webedit_m02_0135_ma.wav；输入记录和页面均区分SCSC录音与自有音频。两原声逐字节复制，REAPER实测F0为128/109.589043Hz，七步输出为125/123.07692/124.031006/118.518517/114.285713/109.589043/103.896103Hz。小幅非单调的估计值保留，不篡改以制造单调结果。七段摘要各异，九段无削波、其他11段SHA不变，源文件不变，REAPER scratch已清。方法与默认参数维持上轮。证据phonation-reaper-m02-webedit-20261013/verification.json。HTML构建幂等，本地verify-1791824386658通过文章按钮目标跳转、来源标签、九段试听SHA、版本切换、26图/17叠图、20音频及四宽度/无JS/减少动效。文章按钮跳转测试对外部页面响应作占位，不声称第三方页面内容或可用性已验证；工具直读该文章为403。r8b0d86a8a7f10fcb已上线，72资源公网大小/SHA一致。公网完整verify-1791824688192通过按钮目标、来源标记、九段音频摘要、3.0.1版本/下载及四宽度等交互，错误与失败请求为空。按钮最终公网页面已目视检查，按用户要求不生成长图。

## 作者确认的九步应用导出（2026-10-13）

用户提供的完整M07导出有九段，continuum_type=3、正向，输入摘要与dl_m02_0558_ma.wav/dl_m05_0558_ma.wav一致。九段原样复制为phonation-1到9，输出摘要与m07.ptb.json逐一一致，原声也逐字节复制。导出记录为Praat/Parselmouth、40–300Hz、1ms，网页说明忠实记录其设置，复测输出中位F0而不将旧REAPER设置写入。导出目录全部输出与元数据、两个原始输入未修改，其他11段不变。九段最高峰值保留用户文件（约0.98），无削波，不实施网站旧0.65调幅。代码增加明确手工导出导入路径及防止自动重合成覆盖的保护，按钮/比例/步数随导出数量生成。22音频，本轮不生成长图。证据manual-phonation-m02-m05-20261013/verification.json，HTML构建幂等。本地verify-1791825782148通过九步按钮/文件摘要/来源与真实参数、22音频、版本/下载、26截图/17叠图、四宽度、无JS与减少动效。r787df59793c73aca上线，74资源公网摘要一致，最终公网verify-1791826137507完整交互复验通过，错误和失败请求为空，九步音频摘要、22WAV、四宽度与版本/下载均正常。

## 语谱图文字与原文核对（2026-10-13）

第一例按用户确认添加 `Hi, this is Steve Jobs.`，同步重建脚本的说明字段、既有图例清单和生成网页，不执行音频重建。[Smith原页](https://www.dsprelated.com/freebooks/mdft/Spectrograms.html)标题仍为 `Hi - This is <you-know-who>`，Steve Jobs姓名来自用户确认，未声称原页正文写出姓名。[Chappell作者上传的原文](https://www.researchgate.net/publication/349935398_s_weakening_in_Nicaragua)第225页明确记 observo 为 `[ok.séɹ.βos]`，图7.3在第226页，来自 Observo un par de islas；末尾插入的[s]是作者讨论的过度纠正，原文该段未独立解释[k]。第227页记 las olas 为 `[la.ʔó.las]`，第228页图7.4图注为 Levantan las olas，讨论/s/的喉部实现。两处词形与原文一致，本轮保持网页说明。

本地 `python scripts/build_website_expansion.py` 两次生成幂等，仅第一例说明增加英文句子。31个既有WAV/PNG的SHA256保持，未改变重建参数。独立Chrome154验证1280/390 CSS px下文字可见、无横向溢出及页面错误，窄屏截图目视通过。`python scripts/validate_docs.py` 检查2011文件/427来源/104任务，errors为空；定向 `git diff --check` 通过。证据为 `output/validation/website/spectrogram-text-20261013/verification.json`、`browser.json` 和两张定向截图。本次未部署，公网仍为上一轮字节。

## 背景四图与更多功能位置（2026-10-13）

用户提供四张2559×1527截图逐字节复制，前后叠层按最终要求为2、1、3、4，即首页、外观设置、国际音标表Plus、感知实验。四张采用不同角度和错位，原图查看保留。更多功能的六点、两行SVG整体移动到背景之后、小实验之前，底部不重复。修改构建脚本、首页、CSS、定向验证和网站README。初次重复构建留下迁移空行，归一化后再次重复生成字节一致。Windows独立Chrome154本地1600/800/390/320宽度和无JS顺序通过，四卡片各有可点击区域并查看正确原图，无横向溢出、页面错误和失败请求，证据verify-1791873092916。

初次发布前下载摘要校验发现3.0.1已被另一发行任务换包，旧报告被拒绝，页面未切换。只读计算服务器两EXE完整摘要后同步downloads.json，再发布r9cd63448bcea7d8e，76资源公网摘要一致，全部22WAV与上次发布清单相同，论文目录、反馈服务和nginx配置保持。第一次公网UI检查的固定等待不足，图片尚未加载完成；补上图片完成条件后公网verify-1791873435957通过四宽度、无JS顺序和四图逐张原图查看，错误与失败请求为空，未移除尺寸断言。Safari/Firefox、实体手机和EXE未验，不生成长图。

最终文档验证2011文件、427来源、104任务，errors为空；本次定向git diff --check通过。


## 3.0.2版本与下载同步（2026-10-13）

按用户恢复授权同步顶端3.0.2、免安装／安装版链接及默认更新小记。正文逐字采用已确认稿，来源字节SHA256保留，服务器发行清单notes与确认稿文本一致。服务器两EXE完整摘要核对：免安装495855100字节、SHA256 `0d0ab31d26eb4ec4563b03d246cacb3250778f71bbcce2ad6dd7e2faa6c518b9`；安装版497771194字节、SHA256 `19c55057d074c217a9d502ccd01d704dd72a1aa20c67e02a8570bd451c46e95e`。安装版摘要与公开ptb-release.json一致。两个公网HEAD返回200且长度一致，不重新上传包。

与上一站点清单比较，仅index.html、assets/downloads.json及新增assets/releases/v3.0.2.json变化，其他74项资源保持，不改变截图、试听或样式。沿用静态发布器，运行配方断言既有CSP字节完全一致，省略配置改写／Nginx重载，首页最后原子替换；旧首页保存在服务器私有备份目录。发布r6a934e4e286065db，77项资源与首页公网完整大小／SHA256一致，论文目录未变化，反馈服务保持活动。

Windows独立Chrome154本地及最终公网检查1600/800/390/320 CSS px：顶端版本、双链接、文件名和大小提示一致，三版本鼠标／键盘切换、确认稿段落／列表／标题／加粗与源摘要、无JS最新正文显示通过；无横向溢出、页面错误或失败请求。下载交互拦截为明确附件夹具，生产字节另以服务器完整SHA及公网HEAD核验。已查看桌面首页、窄屏首页及更新小记截图。实体手机、Firefox／Safari未验；本次未重建或验证应用。

证据：output/validation/website/verify-1791905405111/report.json；output/validation/website/verify-1791905709626/report.json；output/validation/website/version-302-20261013下download-precheck.json、scope-check.json、manifest.json、deploy-content.py、deployment.json、https-readback.json。定向差异及文档校验结果见该目录回执。

## 完整引用与语谱图教程第一集（2026-10-16）

- 范围为本地静态官网。按用户最终要求保持原导航及教程按钮字号，四张方法卡片继续展示，完整引用列表默认折叠，指定的写作/许可提示句已移除。列表直接读取桌面使用的 `frontend/src/generated/sources.json` 公开投影，55条学术来源及351条软件与代码来源全部保留，模块名称取当前注册表。
- 语谱图转音频按钮启用，第一集连接 `https://www.bilibili.com/video/BV1RuJm6gEWE/`。选择后加载官方iframe并显示分集，第二集待更新且禁用；介绍仍可切回。重复选择保留当前播放器，切换频道只留一个iframe，外链同步，无JS时两视频均有外链。
- 命令：`python -B -X utf8 scripts/build_website_expansion.py`，二次生成SHA一致；`node --check temporary-site/site.js`；`node --check scripts/verify_website.cjs`；`node frontend/scripts/generate-ui-data.mjs --check`。设置 `WEBSITE_METHODS_ONLY=1` 或 `WEBSITE_TUTORIAL_ONLY=1` 后运行 `node scripts/verify_website.cjs`。
- Windows独立Chrome155引用报告 `output/validation/website/verify-1792129423562-38124/report.json`：406条标题、引用正文、链接、模块及署名/许可/改写说明逐条匹配；四卡片不进入折叠容器；默认收起、键盘展开、搜索、按模块过滤、无结果恢复、实际剪贴板、1600/800/390/320布局、无JS全部引用通过。
- 教程报告 `output/validation/website/verify-1792129377468/report.json`：1920至320八档导航维持既有字号门槛，1600/1280/800/390/320五档教程无溢出；18模块名称一致、17禁用、两可用，第一集与待更新第二集、键盘激活、切回介绍/外链/分集状态、重复选择不重载、实际全屏及无JS通过。新视频起播后currentTime=1.94805秒、paused=false、readyState=4。
- 已目视检查桌面四卡片与折叠入口、390宽引用列表、桌面真实视频及320宽分集布局。两报告页面错误和失败请求为空，scratch清理回执为deleted。初次并行启动遇到同毫秒目录名碰撞，已在验证目录名加进程ID；引用URL检查改读原始href，避免浏览器自动补根路径斜杠造成字面不一致，保留源链接严格比较。
- 本地验收后井井明确要求上线，已发布 `r8402429e5829aa4c`。完整视频、实体手机及其他浏览器未验，第二集尚无链接；主应用及GitHub未改。当前发布结果见下段，旧版本验收保留后文。
- 文档检查2078文件、441来源、119任务，errors为空；历史快照36处失效链接保留原记录。10个本次文件UTF-8无BOM、台账JSON及定向 `git diff --check` 通过。


### 本轮官网发布与公网核验

- 2026-10-16用户明确授权上线。冻结首页SHA与上一轮本地已验字节一致，冻结输入及清单位于 `output/validation/website/references-tutorial-release-20261016/`。相对当时公网仅 `index.html`、`site.css`、`site.js` 改变，86项资源保持，无新增或移出资源。发布前核对首页与Nginx配置摘要，切换前再次防并发校验，旧版备份保留。
- 发布回执：`output/validation/website/website-d8f36e5a8b50433bb689137eb45ac118/deployment.json`。89项HTTPS实际资源完整大小/SHA全部匹配，两EXE仅执行HEAD且200/大小一致。论文目录摘要保持，反馈服务及Nginx正常，Nginx配置前后字节相同。证据为冻结目录中的 `https-readback.json` 与 `config-after.json`。
- 最终公网引用验证：`output/validation/website/verify-1792129740531-13540/report.json`，406条桌面来源逐条比对、四卡片常显、默认折叠、搜索/模块/空结果/复制、四档布局、键盘与无JS通过。
- 最终公网教程验证：`output/validation/website/verify-1792129740531-38332/report.json`，八档导航、五档教程、第一集与待更新第二集、两频道双向切换、重复选择保持播放器、实际全屏及无JS外链通过。第一集实测duration=1587秒，播放时间推进至1.911617秒，paused=false、readyState=4。两公网报告errors/requests为空，scratch清理回执均为deleted。

### 学术与代码库引用分开显示

- 2026-10-16井井要求将两类引用分开，并授权改完直接上线。四张方法卡片下方现为学术引用55条、软件与代码库引用351条两个独立折叠列表，两组各自搜索、按模块筛选、计数、复制和滚动，互不影响；默认均收起。来源归属及正文沿用桌面公开数据。
- 本地报告 `output/validation/website/verify-1792129969578-20524/report.json`，最终公网报告 `output/validation/website/verify-1792130022882-25796/report.json`：逐项核对两组来源/引用/链接/模块/署名，验证独立筛选、无结果恢复、复制、键盘开合、四卡片常显、1600/800/390/320布局和无JS，两报告errors/requests为空，scratch已清理。已目视检查桌面两组折叠入口与390软件引用。HTML生成幂等、JS语法与定向差异检查通过。
- 已发布 `rb2b3aaafb66dd89a`，仅首页与JS变化，87项资源原字节保持；冻结输入位于 `output/validation/website/references-split-release-20261016/`。发布回执 `output/validation/website/website-610cb244b31d433d811680562ff7a3c7/deployment.json`；同冻结目录 `https-readback.json` 核对89项HTTPS完整字节/SHA及两下载HEAD200/大小一致，CSP与论文目录保持，反馈及Nginx正常。本轮没有改视频接线或重跑完整视频，实体手机及其他浏览器未验。

## 使用教程小电视（2026-10-15）

- 本地新增顶栏和阅读目录使用教程入口，正文放在小实验前，保留正在施工中。淡紫色电视和黄色介绍沿用官网配色，18个模块按钮为原生disabled。模块文字与README逐项核对。按预览反馈移除更多功能整块、选一节文字/箭头和目录底部慢慢逛。九个大板块按内容采用手绘声波、撕纸边和胶带接缝，按钮改自然宽度、轻微倾斜与不规则双边线。
- 点击介绍才加载 `https://player.bilibili.com/player.html?bvid=BV1WWpr67EkX&page=1&autoplay=0`，重复点击保留播放器。iframe允许全屏，浏览器支持时显示独立全屏按钮，常驻B站视频链接在无JS时也可使用。
- 命令：`python -B -X utf8 scripts/build_website_expansion.py`（重跑SHA不变）；`node --check temporary-site/site.js`；`node --check scripts/verify_website.cjs`；设置 `WEBSITE_TUTORIAL_ONLY=1` 后执行 `node scripts/verify_website.cjs`。
- Windows独立Chrome154：八档导航、五档教程布局、18模块与README一致且禁用、黄色介绍、键盘入口、单次延迟加载iframe、实际全屏进入/退出、无JS外链通过；B站真实视频readyState=4，播放时间推进至1.90秒。JS语法和生成幂等通过。
- 本轮另验十处分隔的顺序、宽度层级及装饰语义，三处提示不存在，按钮采用自然宽度且无横向溢出。首轮发现手绘第二道边线伸出按钮造成内部横向溢出，改为内侧线后原断言通过。
- 报告和桌面/窄屏/播放器截图：`output/validation/website/verify-1792061776861/report.json` 同目录。已目视检查1280/320教程布局、390分隔与教程边界。浏览器页面错误与失败HTTP响应均为空，隔离scratch已由脚本清理。
- `python -B -X utf8 scripts/validate_docs.py`检查2043文件、436来源、110任务，errors为空；既有历史快照失效链接单独保留。`git diff --check`通过，本次10文件UTF-8无BOM检查通过。
- 多视频建议为模块选择后在电视下显示分集目录，单集直接播放；当前仅方案，未实现。
- 初次验证为本地静态站范围；本轮已按下方上线记录发布，主应用和发行包保持。完整视频、其他浏览器与实体手机未验。

### 分隔符三方案预览

- `temporary-site/divider-preview.html`以相同配色、宽度、留白和章节标题展示手绘声波、撕纸边、胶带接缝。井井已确认三种全部采用。手绘声波用于教程/小实验，撕纸边用于方法与引用/常见问题，胶带接缝用于更新小记/下一步/问题反馈/联系我们/作者寄语。
- `WEBSITE_DIVIDERS_ONLY=1 node scripts/verify_website.cjs`在Windows独立Chrome154验证1440/390/320宽度、SVG不出界和键盘目录，错误及失败请求为空。桌面、窄屏和单项截图已目视核对，报告：`output/validation/website/verify-1792062050368/report.json`，scratch已清理。
- 本地执行发布器build检查对照页未进入生产资产清单；生成HTML幂等通过。设计对照页保持本地，主页已按下方记录发布。

- 三种混用验证：`WEBSITE_CHAPTERS_ONLY=1 node scripts/verify_website.cjs`，Windows独立Chrome154检查十处分配、三种样式1440/390/320布局、目录锚点和无JS，errors/requests为空，scratch已清理；主页及对照页生成幂等通过。纸边桌面与胶带窄屏已目视核对。报告：`output/validation/website/verify-1792062209693/report.json`。

### 页脚入口、动效与本轮上线

- 正文使用说明与源码及目录项已移除，顶栏GitHub保持，页脚新增紧凑GitHub入口。九处分隔按已确认风格搭配。
- 首页漂浮在本机约2.4秒采样期间位移范围20.23px，悬停抬升/轻转/缩放/明暗变化与恢复通过；其他小节分组延迟进入，卡片适量悬停。首页离屏或标签页隐藏时暂停装饰动画，减少动态效果/无JS正文可见。
- 本地动效证据：`output/validation/website/verify-1792062797248/report.json`；最终候选教程/八档导航/真实B站起播与全屏：`output/validation/website/verify-1792062855696/report.json`；最终候选九处分隔、页脚/顶部链接、正文删除、四档页脚及无JS：`output/validation/website/verify-1792062870097/report.json`。已目视检查桌面悬停、390首页和页脚。
- 动态元素持续漂浮时，Playwright默认等待静止的hover步骤会超时；改为按实时中心坐标移动鼠标并验证真实:hover命中，没有停止动画或放宽效果断言。
- 发布输入冻结于`output/validation/website/tutorial-release-20261015/site`。相对当前公网仅首页、CSS、JS变化，86项资产字节不变；旧更多功能SVG不再进入新清单，旧服务端版本保留。设计对照页不发布。
- 用户明确授权本轮直接上线。部署器保留既有CSP，仅追加`frame-src https://player.bilibili.com`，完成配置语法检查与reload；索引/配置摘要并发保护、版本目录、备份和失败回滚保留。`r4698051b31a01f4e`发布成功，公网复查通过。

- 最终公网教程/导航报告：`output/validation/website/verify-1792062997351/report.json`，B站起播时间推进1.84秒、readyState=4、实际全屏进入/退出通过。动效报告：`output/validation/website/verify-1792062997346/report.json`。页脚/九处分隔报告：`output/validation/website/verify-1792063033135/report.json`。各报告errors/requests为空，scratch均deleted。
- `output/validation/website/tutorial-release-20261015/https-readback.json`核对89项资源完整字节/SHA及实际CSP；两EXE HEAD200与大小一致。论文目录SHA不变，反馈服务及Nginx运行正常。部署回执：`output/validation/website/website-9e7d8e1170714cafa8aeb90b021e396d/deployment.json`。
- 本轮未重建主应用、未推送GitHub或创建Release；完整视频、实体手机、Safari/Firefox与长期运行未验。历史文档中的仅本地范围以各段当时验证为准，本轮上述变更现已发布。

## 公开四项统计曲线（2026-10-16）

底部加入页面浏览、下载发起、网页版使用（次）和GitHub Star四个计数与四张独立阶梯图，支持24小时/7天/30天/全部和采样滑块。前三项每分钟更新，Star每5分钟采样，注明采样时间、离线与空缺。新工作区成功创建计一次，刷新/同工作区多标签不重复；不能解释为用户人数。

只读获取现有匿名Nginx日志和计数状态，恢复370个真实历史点。2026-10-16 18:44北京时间采样为页面1133、下载226（98/128）、网页版0、Star33。页面/下载历史用真实服务器记录，因此没有读取本地看板私人数据。旧时段网页/Star未观测值为null，不补造零值或历史；最新网页0反映尚未上线。稀疏Star样本保留原时刻，压缩不随其他指标覆盖。

5项Python计数、轮转、回填、稀疏采样检查通过。独立Chrome155实际历史展示，1440/800/390/320宽度、四时段、滑块、Star下降、刷新/失败保留值/首次失败、无JS说明通过。截图发现坐标文字继承描边和小整数重复刻度后已修正并重验。证据 `output/validation/web-launch/public-stats-ui-final.txt`、`public-stats-preview/site-stats-history.json`。生成入口仍为build_website_expansion.py。该记录为发布前本地验收；本轮获准发布后的状态见下段。

本批统计原始测试副本清理被自动审批拒绝（blocked by policy，无进一步原因），未重试删除，临时日志归档/scratch保留，回执output/validation/web-launch/stats-source-cleanup.json。最终管理EXE测试的工程外专用QA副本已停止，暂留待归属清理。

井井要求交给另一AI核验后删除，本轮只运行inventory_web_stats_copies.py：6件281171字节、5件明确保护证据，JSON/Markdown保存于output/maintenance/web-stats-copy-review.*。无递归删除功能、无删除授权扩展，先前审批拒绝回执保持。

2026-10-16后续获得官网统计及禁用入口的单独发布授权。以线上rb2b3aaafb66dd89a为基线，比对整个HTML，除统计区、入口及CSS/JS版本外的正文一致，89件资源按线上摘要冻结。site.js仅替换统计初始化，反馈来源上报的同期本地改动明确排除，反馈客户端其他代码逐字节保持。安装collect_stats.py与统计map，systemd仅给既有计数器增加GitHub所需网络地址族，保留原计数状态和分钟timer；站点Nginx配置/CSP、论文目录、反馈进程PID保持，未改变内测服务配置或开放/app/。

先发布r4aeb22b837a35311，89件公网完整GET摘要/大小、下载两EXE的HEAD大小、反馈健康、公开/app/404与工作区接口不可访问通过。当时公开数据页面1145/下载237/网页0/Star33，历史371点，仅为该次采样。公网独立Chrome四图、时段、滑块、Star下降、异常保留、无JS与四宽度检查通过，未发真实反馈或修改会话。

按井井随后反馈重排主页：大号浅紫网页版按钮置于双下载上方，整体同宽，标明内测中·暂未开放；实际disabled且无href。桌面顶栏字体与间距缩小、Logo和全部导航在1024/1280/1440/1920宽度保持同一行，800/390/320允许换行，全部无横向溢出。新版r196722d9534d7cb4已发布，最终公网独立可见Chrome七宽度与统计交互全部通过，无pageerror/失败请求。再次完整读取89件公网资源，SHA/大小一致；两下载HEAD、反馈健康/PID、论文目录/站点配置SHA、nginx -t与timer均通过。2026-10-16 20:11北京时间采样为1147浏览/238下载/0网页/33 Star，共375点，统计时间与历史点数较发布时前进，分钟刷新正常。数值仅为采样，之后继续变化。

证据集中在output/validation/website/stats-release-20261016，包括server-before.json、冻结manifest、包边界diff、安装/发布回执、public-verification.json及浏览器日志。旧日志副本和原始记录未删除。只公开匿名总数/时间曲线；正式网页使用计数保持0，私有内测不混入。

32次验收副本清理回执：系统自动审批拒绝移除已被最终候选替代的iterations32-inline-code-20261016/site，仅返回blocked by policy，没有更具体原因。命令未启动、删除0文件，该目录完整保留；未改用其他方式重试。当前源码/最终冻结目录/模型/用户数据不受影响，报告iterations32-inline-code-final-20261016/cleanup-superseded-site.json。


## macOS 下载统计接入（2026-10-17）

状态：`verified`。井井明确要求修改网站统计并接入Mac，实施范围见[计划](../plans/2026-10-17-macos-download-statistics.md)。

Nginx为Mac ZIP新增macos匿名类型，整文件及首个bytes=0范围计数，续传/HEAD/失败/空响应/测试与爬虫排除。公开downloads包含windows、macos及total；历史前五列原值保持，第六列为macosDownloads，第二列明确为windowsDownloads。下载图两平台共用时间/纵轴，各有颜色、实际采样点及滑条读数。旧五列接口兼容，未知Mac历史保留null，不显示假零。

保留的普通日志只补可证实的GET 200，206缺Range头无法区分续传，保持未补。匿名日志已有事件按时间/状态/字节的多重集扣除，轮转重复副本排除，回填完成标记防止重复。原日志不导出，新增状态及公开文件不存IP/UA。服务器预演补2次，Windows274/Mac2/总276，1条无法核实Range的响应不补。

已执行7项Python回归、JS语法和生成器两次幂等。源码Chrome155七宽度、双色/标记/合计、共享时间轴、零值与缺失、稀疏Mac事件、旧五列、失败保留及无JS通过。实际服务器隔离Nginx验证整文件/首段、续传、HEAD、测试UA和爬虫，得到Mac2/Windows1/总3，正式日志未写测试数据。证据：`output/validation/website/macos-statistics-20261017/preflight.json`及`verify-1792240552975-119060/`。

已发布r969968f235407e69，只有首页/CSS/JS三项变化，134资产复用。冻结候选统计及原下载区通过，公开统计专项通过；旧下载验收中的total=Windows断言未同步导致首次检查失败，保留原记录；修正为total=Windows+Mac后完整公开验收通过。未声称其他浏览器、实体手机或Mac/Safari通过。


最终公开验收完成：真实接口Windows274/Mac2/合计276；历史六列且旧Mac缺失区为null，补回1次/2次节点均可读。定时任务在部署后继续刷新，完整137文件服务器大小/SHA与冻结manifest一致，3项变化文件及根首页实际HTTPS摘要通过，其他134资产原字节复用。统计map与collector为授权更新，其他Nginx配置、论文目录、反馈PID保持，nginx -t和timer/nginx/反馈服务active通过。未重建或上传应用包。

冻结候选统计与原下载/安装浮窗通过；公网统计专项及原下载区七宽度、三下载名称/路由/HEAD、Mac浮窗整卡命中与三组真实交叠、鼠标/键盘/触摸/无JS通过。旧PLATFORM验收首次仍用total=Windows公式，失败原件保留在verify-1792240792057-106924；修改为total=Windows+Mac并分别验证Windows字段后，verify-1792240816223-107572完整通过，没有删断言或放宽数值。公网实际图截图已查看，Mac曲线与零基线接近符合当前数量。

最终回执：`output/validation/website/macos-statistics-20261017/completion.json`；冻结范围、页面发布、采集发布、服务器摘要和截图位于`output/validation/website/macos-statistics-381eab7c42f049dca7d89005feda264c/`；公网统计`verify-1792240781333-75396/`、原下载区`verify-1792240816223-107572/`。7份本轮独立浏览器/候选（含失败）scratch的cleanup.json均为deleted。其他浏览器、实体手机和Mac/Safari仍未验。
## 统一时间轴与首页标题排版（2026-10-18）

状态`verified`，范围为本地及公网Windows Chrome155。四图按同一历史首尾的真实秒数等比例映射，点序号与降采样不参与横轴计算。1848条只读快照从北京时间10/12 18:55开始，未知前缀留白，不补零。日期、滑块中心端点和绘图区1440/390/320宽度误差小于0.03px。查询吸附真实记录，键盘按相邻原始记录移动。

仅全部记录且单序列超过2048点时，在共同时间域的256个等时长桶内保留首/末/最小/最大点，并额外保留真实缺口两侧。先按原始间隔判定断线，避免抽样造成假缺口；值域、时间范围及滑条查询保持全量。最近24小时/7天/30天始终绘制全部有效点。最大值逐项归约避免参数展开溢出，大计数轴用万/亿避免裁切，读数保持完整。160001条不等间隔、跨约671天夹具：全历史页面浏览绘510点，本机切换约60ms；24小时/7天/30天分别全绘240/1680/7200点，首尾、孤立峰值、4天缺口和原始查询通过。计时只代表该夹具/本机。

首页标题约放大6%，四入口/黄色功能带下移，矮屏保留首屏完整功能带。网页版6px循环和3px悬停上浮实际通过，测得范围5.979px，离屏暂停/动态减少动画/无JS正常。发布前核对发现线上缺少本地已有动画，本轮随新要求一并补齐。下方按钮正常样式、HTML、下载路由及排列保持。按手写和质朴童趣反馈生成E–H、I/J/K标题概念稿，不重绘按钮。井井最终选定I，并要求黄线贯穿语音学三字，随后明确改为生动的曲线。当前采用有起伏、粗细变化与圆润收笔的原生SVG黄色波浪线，保持选定PNG原字节。来源/SHA见temporary-site/assets/hero-lettering-source.json及assets/provenance.json，源PNG SHA256为8120f6c0eefb53f71c20270fd5aaa66f739eefec1ebedfe95c63eeeeb5be48f9。专用build_website_title.py接入原资产生成器，--check核对确定性产物与来源，无字体软件嵌入。

本地WEBSITE_HOME_ONLY、STATS_ONLY、STATS_AXIS_ONLY、STATS_SAMPLING_ONLY、MACOS_STATS_ONLY、PLATFORM_ONLY、MOTION_ONLY、TITLES_ONLY均通过，包括七宽度统计/下载、未知/零值、刷新/故障保留、键盘、单点/空历史、原Mac浮窗与四宽度动画。音频基线、生成器幂等、JS/Python语法和定向diff通过。早期validate_docs.py曾报同期来源447项/声明446不一致，本轮未改该登记；最新全检2268文件/447来源/159任务零错误，历史快照断链单列，完整日志output/validation/website/selected-title-20261018/docs-validation-final.log。早期CSS查询串匹配、矮屏/窄屏、动画稳定等待和原生滑块伪元素读取失败留有报告，修正夹具或实际排版后复验，未降低断言。

获井井明确上线授权后发布r5aaf8028a426a3a7，只上传index.html/site.css/site.js三文件、106339压缩传输字节，复用134资产。冻结阶段在实时公网页面上只替换统计段、受影响CSS和统计IIFE，保留原反馈处理与其他正文。候选采样/下载/动效3组、实际公网统计/下载/动效3组通过，七宽度下载和四宽度动效截图已查看，页面错误/失败请求均空。服务器137文件完整SHA、3变化文件与根首页HTTPS完整字节一致，切换前两次核对首页SHA，原首页留私有回退，Nginx配置/论文目录/反馈PID保持，不重启服务。候选和六组浏览器自有scratch全部清理并留下deleted回执。证据output/validation/website/shared-axis-publication-20261018/completion.json及其out指向完整审计目录。

标题最终发布rc57cf2fda18a63c6。以当时实际线上内容冻结候选，只替换assets/hero-lettering.svg与index.html资源前缀，178资产原字节复用，完整180文件服务器大小/SHA、两变化文件和根首页HTTPS字节通过，压缩上传1113719字节。其他正文、CSS/JS、音频/语谱素材和统计保持，配置/目录/反馈PID不变。候选verify-1792261144979-109540及公网verify-1792261162716-115644分别验证1920/1440/1280/1024/800/390/320宽度：原PNG身份、可解码、六字无障碍文本、无溢出，四按钮位置/尺寸/文字/路由/背景/边框/阴影/字体前后一致。波浪线覆盖标题整行98.1%，原始坐标高度35.1px，实际桌面/窄屏截图已查看。页面错误/失败请求为空；专用生成器检查、Python/JS语法与定向diff通过。完整回执output/validation/website/selected-title-20261018/completion.json及其out指向审计目录，候选/浏览器scratch均deleted。

本机安装版output/release-staging/local-replace-303-20261017-163326-25f561/artifacts/PhoneticToolbox-3.0.3-windows-x64-setup.exe，产品3.0.3、497102829字节、SHA256 055bb0d1de1fa12eafc37f506f7544817f0f0d9441fafc8d3fac0420146623d4，与包报告一致。本轮未运行安装器、重建应用、推送或改写统计数据。实体设备、其他浏览器未验。统一汇总output/validation/website/shared-axis-20261018/verification.json保留各阶段报告与清理回执。


## 更新小记生理参数合成视频（2026-10-19）

状态：本地 verified，公网未部署。视频来自井井指定的桌面《春江花月夜.mp4》，逐字节复制为 `temporary-site/assets/videos/v3.0.4-vocal-tract-chunjiang.mp4`，SHA256 `c0f19521fdde5e2c34162588280e57b906975bb640744e405f0f4d827a7766ae`。FFprobe 核对 H.264 High / yuv420p、1280×720、AAC 48 kHz 单声道，时长 2.36 秒、662593 字节。封面由 FFmpeg 提取 0.4 秒画面，来源清单保留各自摘要。

生成器只在 v3.0.4 生理参数合成文字末尾插入 video/figure，不改变发行正文及 JSON 快照；controls、playsinline、preload=none，切换版本暂停，16:9 自适应。发布包支持 MP4 与 poster 引用收集/版本前缀，来源清单随包。生成两次 HTML SHA 一致，本地内存打包清单和路径核对通过，回执 `output/validation/website/release-vocal-video-20261019/package-check.json`。

首两次浏览器验证通过位置、宽窄布局和开始播放，但进度跳转超时。验证服务原来没有分段请求与文件总长度，已补 MP4 Content-Length / Accept-Ranges / 206 与无效范围 416，使本地验证对应静态服务器的视频行为；保持进度跳转断言并重验。早期失败和清理回执分别保留在 `output/validation/website/verify-1792383425623-142048/` 与 `output/validation/website/verify-1792383487260-79212/`。

第三次实际视频验证已通过，但旧无 JS 断言把版本号和更新二字硬编码为相邻，未匹配当前含 Windows 版的首段。已改为完整核对当前快照实际首段，未改变发行文字。最终 `WEBSITE_RELEASE_ONLY=1 node scripts/verify_website.cjs` 通过：独立 Chrome155 原生空格播放、进度跳至 1 秒、版本切换暂停、实际结束、52 帧解码和 28688 音频解码字节；1600/800/390/320 宽度的比例与边界、来源摘要、5版鼠标/键盘切换、无 JS 正文/控件/入口通过，页面错误和失败请求为空。宽窄截图已目视核对。报告 `output/validation/website/verify-1792383658438-84444/report.json`，清理回执为 deleted。

`python -B -X utf8 scripts/validate_docs.py` 检查 2338 文件、447 来源、168 任务，零错误，既有历史快照失效链接另列；Python/JS 语法与定向 git diff --check 通过。其他浏览器、实体设备和公网未验，本轮未公开上传。

## 方法示例紧凑四列（2026-10-23）

- 修改 `temporary-site/site.css` 与昼夜预览 `design.css`，桌面≥1024px四列、560–1023px两列、<560px单列。内边距18px、标题21px，引用自然展开。
- `WEBSITE_METHODS_ONLY=1 node scripts/verify_website.cjs`：Windows Chrome155通过，完整来源/引用、独立搜索筛选、复制、键盘、四宽度及无JS，页面错误与失败请求均为空。回执：`output/validation/website/verify-1792757362390-173064/report.json`。
- 独立Chrome以1600/1280/1024/800/390/320px检查首页与昼夜预览，共12组列数和无横向溢出通过，引用展开与夜间截图通过。首次800px昼夜仍受旧单列覆盖，修正后全部复查通过。几何与截图：`output/validation/website/method-cards-20261023/`。
- `python -B -X utf8 scripts/build_website_day_night.py`更新派生样式，文本/控件/媒体与当前首页保持一致。本次无公开部署，实体手机/Safari未验。

同日等高修订：Grid拉伸同一行卡片，纵向Flex将来源链接置于底部，取消第二/三张卡片旋转。首页和昼夜预览六宽度共12组同行高度/链接底线差小于1px，展开引用后整行高度一致，夜间截图通过，见 `output/validation/website/method-cards-20261023/equal-heights.json`。METHODS专项再次通过，回执 `output/validation/website/verify-1792757512742-179460/report.json`。文档校验无错误（保留历史快照既有失效链接提示），生成幂等与定向差异检查通过。

## 下载按钮下方安装说明残留（2026-10-24）

隐藏 `.macos-help summary`，安装提示保留下载按钮悬停与聚焦入口，Escape关闭时恢复下载按钮焦点。同步昼夜派生样式。`WEBSITE_PLATFORM_ONLY=1 node scripts/verify_website.cjs`在Windows Chrome155七宽度通过，下载路由、提示布局/交互、无JS下载及统计回归通过，页面错误/失败请求为空。首轮旧验证在Escape已回到下载按钮后再次focus，因无新focusin失败；新增焦点归属断言并移出/重新聚焦后通过。

回执 `output/validation/website/verify-1792846614094-252160/report.json`；首页/昼夜预览1280与390px四组无残留文字、提示聚焦/Escape、焦点归属通过，截图及回执在 `output/validation/website/macos-help-label-20261024/`。未公开部署，未验实体设备，昼夜全站测试未重跑。

本轮获得井井明确上线授权后，发布 `r550b51d09593c980`。从实时线上 `rfd524881c628735c` 的活动 `site-ime-013.css/js` 构造候选，仅替换 `.macos-help summary` 显示规则和Escape焦点，保留输入法及所有其他线上内容。首页仅换资源版本前缀。候选与公网独立Chrome155在1600/1280/1024/800/390/360/320px共14组确认无残留文字、无横向溢出、下载链接保持、hover/focus/Escape有效，页面错误为空。3文件上传、183文件复用，186文件服务器大小/SHA与变化3文件和根首页HTTPS回读一致。发布前及原子切换前核对首页摘要，Nginx配置/论文目录/反馈PID保持，无服务重启；私有回退保留。完整配方、差异、回执和截图见 `output/validation/website/macos-help-public-20261024/`，完成回执 `completion.json`。其他本地未发布改动未混入，实体设备未验。
