# macOS r6 官网下载发布

2026-10-17，状态`verified`。井井明确要求将已有Mac包上传服务器开放下载，并逐项确认安装说明拟稿。本轮最初按确认稿发布下载与安装页面，随后按追加要求将同一说明改为Mac按钮浮窗，最新交互见末节。不重建Mac应用、不修改Windows成品或GitHub Release。

## 成品身份与范围

使用已有`PhoneticToolbox-3.0.3-macos14-arm64-preview-r6.zip`，476461931字节，SHA256为`fc44db9ea9f29723c18393029fe97e11719a29ac9edfcdaeb464d64bcf9596a1`。本轮完整摘要与Mac最终构建回执一致，6771项ZIP CRC、单一`PhoneticToolbox.app`根、3.0.3版本及无个人图标偏好检查通过。包内展开1095807976字节。

同字节Mac既有回执记录579个arm64原生组件，最低二进制目标macOS14.0，实际M4/macOS26.7.1运行及中文空格路径解压后任务、页面、Qt、启动后签名复验通过。当前为ad-hoc签名，未做Developer ID/Apple公证。本轮在Windows检查归档与既有回执绑定，没有重新执行Mac运行或设备验收；不扩大到Intel、macOS14/15、另一台Mac或真实设备。

## 本轮实现与授权文案

服务器新增`/releases/macos-arm64/3.0.3/`下的ZIP、`install.html`、`download.json`和`SHA256SUMS`。断点续传仅用本任务的SHA命名片段，完整SHA通过才写正式文件；遇到同路径不同内容则拒绝覆盖。第一次传输因井井要求先审阅说明而停止，片段保留后续传。

公开安装页只包含已确认的Apple芯片/macOS14+要求、解压/拖入应用程序/双击三步、尚未公证及系统设置的首次打开方法、Apple官方说明链接、文件大小与SHA。本次安装说明沿用确认稿，Mac原开发记录与测试结果保持。

官网生成器根据`temporary-site/assets/downloads.json`的Mac元数据启用按钮与安装说明链接，CSS补足右侧提示定位。Windows大小/SHA同步已发布的新成品，统计图继续仅展示现有Windows数据，不生成Mac计数；网页版原状态保持。只从当前线上页面替换下载组、下载图图例及对应CSS/元数据，复用其他134项资产，不部署本地无关样式、修改Nginx或重启服务。

## 验证与证据

`scripts/verify_website.cjs`本地独立Chrome155通过七宽度下载布局、三个附件链接/文件名、Windows统计及时间滑条、Mac无JS下载。下载点击使用小型路由夹具，实际整包另以服务器完整SHA和HTTPS HEAD/Range核对，不从公网重复下载整包。

批准的安装页通过独立Chrome155四宽度、正文逐项匹配、SHA、返回/下载/Apple链接及无JS下载，已查看桌面与手机截图。源码及运行证据位于`output/validation/macos-server-release-20261017-r1/`，本地网页候选`output/validation/website/macos-publication-c3d88705307245a2afa7ad801ce37063/`及其manifest/scope/browser回执。正式上传及公开验证结果见下节。


## 已完成公开验证

Mac服务器4个正式文件完整SHA与本地一致；ZIP公网HEAD为476461931字节，Range 0–1023字节及总量一致，安装页/下载JSON/校验清单的实际HTTPS完整字节摘要通过。没有从公网重复下载整包。

官网已发布为`r9e0237ca1f770ed6`，仅增量上传`index.html`、`site.css`和`assets/downloads.json`，共92922传输字节；复用134资产，服务器完整137文件清单/SHA与冻结manifest一致，3项变化文件的实际HTTPS摘要及根首页回读一致。原首页留有私有回退，Nginx配置、论文目录及反馈服务PID保持。

实际公网Windows Chrome155的七宽度下载布局、三个附件名称/路由及HEAD、稳定频道Windows元数据、Windows统计/键盘滑条与Mac无JS下载通过；批准的安装页另以实际HTTPS通过四宽度、正文、SHA、返回/Apple/下载链接及无JS下载，截图已查看。浏览器下载点击使用小型路由夹具，正式ZIP身份由服务器完整SHA及公网HEAD/Range核对，两者分开记录，不声称重新在Mac/Safari运行。

公开入口为[官网](https://www.phonetictoolbox.com/)与[macOS安装说明](https://www.phonetictoolbox.com/releases/macos-arm64/3.0.3/install.html)。已按请求通过open_in_codex提交两个网页，工具返回queued，未将排队状态写成已在用户屏幕显示；最终答复同时提供实际公网截图。

发布回执包括本轮目录的`server-published.json`、`https.json`、`guide-public.json`、`completion.json`和`upload-cleanup.json`，以及对应website候选目录的deployment/server-after/public-browser/manifest。4个本轮服务器传输片段在正式文件和片段逐一SHA通过后删除；网站传输包由发布器清理，自有浏览器/候选scratch已清，正式Mac成品、原开发证据和旧网站回退保留。原开发记录及科学断言保持。

收尾定向`git diff --check`、Python/JS语法检查通过；`scripts/validate_docs.py`检查2238文件、446来源、151任务，错误为0，历史快照既有断链单列。5份本轮候选/浏览器scratch清理回执均为deleted。


## 同日追加：安装说明改为按钮浮窗

井井明确要求将安装页面改为Mac下载按钮悬停浮窗，不再单独跳转页面。沿用已确认的安装正文，主按钮继续直接下载原ZIP，移除独立安装链接；触摸和无JS由同一details/summary展开，键盘焦点可查看、Escape关闭，鼠标跨过按钮与浮窗间隙时保留短暂缓冲。浮窗按可用视口选择上/下方并限制高度，原下载文件不变。

涉及官网生成器、site.css/site.js及下载元数据，现有网站验证新增七宽度浮窗边界、鼠标移入内容/Apple链接可点击、焦点/Escape、触摸和无JS检查。现已公开发布r208f040f9772b085，仅4文件/105382B增量，133资产复用，服务器137文件完整清单/SHA和公网4变化文件/首页回读一致，Nginx/论文目录/反馈PID保持。实际Chrome155七宽度浮窗无裁剪、按钮到浮窗指针移动及Apple链接命中、焦点/Escape关闭、触摸开关、无JS展开及原三下载/Windows统计通过，公网截图已查看。旧安装URL替换为官网#macos-download的即时转向，相关download.json/校验清单同批更新，三小文件HTTPS完整SHA通过；Mac ZIP完整SHA/字节保持。旧/新SHA逐项保护，原三文件回退在私有目录保留。证据output/validation/macos-hover-help-20261017-r1。


首个候选在鼠标悬停后Escape检查失败，未发布；修正为仅在浮窗打开时由document响应关闭，并保留初次失败。发布准备另核实JS标记提取边界，仅插入Mac说明处理器，删除该区块后必须与线上原JS逐字节一致，未带入或重复后续反馈/统计处理器。重复候选与公网完整受影响范围通过，失败证据保留。

旧安装地址已用真实Chrome155验证自动进入官网同一浮窗，Escape关闭通过；无JS模式在减少动态效果且字体就绪后，原地址同样转回官网并可用原生summary展开。首次无JS夹具未等字体与原生锚点布局就绪导致点击稳定性超时，保留失败，补就绪等待后原点击断言通过，未改产品或使用强制点击。本轮4份成功候选/浏览器scratch回读均deleted，原失败及回退材料保留。


## 同日追加：浮窗遮挡与窄窗口定位修复

井井截图指出向下展开的安装卡片中部被黄色功能带覆盖。根因是hero的isolation建立独立堆叠上下文，卡片自身z-index无法越过其兄弟功能带。仅在原生details打开时提升hero外层至z-index 11，同时覆盖原先位于z-index 10的页头。定位处理器另按视口左右各16px夹紧卡片，修复825px窗口中原先left约-15px的越界；关闭后原页面层级恢复。安装正文、下载地址与包字节保持。

网站验证为整个可见卡片按16px行距、每行三点运行elementFromPoint命中断言，保留原七宽度视口边界与鼠标/键盘/触摸/无JS检查；新增1440/1024/825px三组向下展开，必须与黄色功能带真实交叠，并逐段命中浮窗。修改前公网复验捕获390px页头遮挡。首个本地修复候选又捕获825px横向越界，未发布；修正后另一个夹具因390px的插画将功能带隔开，无法满足预设交叠断言，失败报告保留。将该交叠用例改为确实交叠的1024px，保留手机七宽度及全区域命中检查，产品实现不为该夹具调整。

本地独立Chrome155受影响范围通过，已发布r4840ddd775cdf7c2，3文件105101字节增量，134资产复用。发布器严格断言仅新增CSS外层规则和JS两行水平夹紧逻辑，既有JS处理器及其他线上内容保持。公网Chrome155七宽度完整可见区域、1440/1024/825px三组向下展开与功能带实际交叠的整张卡片命中、原三下载/HEAD/统计/触摸/键盘及无JS通过，实际向下展开截图已查看。服务器137文件完整SHA、三变化文件与根首页的HTTPS完整字节一致，Nginx配置/论文目录/反馈PID保持，原Windows/Mac包及三份Mac说明元数据发布前完整SHA保护通过。证据output/validation/macos-hover-layer-20261017-r1及其completion.json对应候选/公网浏览器；未增加Mac/Safari或实体设备验收。

公网前两轮在load事件30秒限时失败，浏览器待加载清单指向原字体与图片。相同快乐字体1514968字节从Windows公网完整读取32.65秒，服务器HTTPS回环新/旧两发行均200且约0.08/0.06秒。仅此轮公网验收显式设置WEBSITE_NAVIGATION_TIMEOUT_MS=120000，默认仍30秒，原布局/遮挡/链接断言不变，失败报告与网络诊断保留；公网七宽度/交叠/三下载与触摸范围已通过，末尾无JS下载仍遇到事件超时。单独无JS检查又捕获点击稳定性超时，保留失败；仅等待fonts.ready不能保证样式和图片完成加载，补load事件、字体就绪、实际滚入视口及前台页面后，无JS关闭脚本后整页图片仍受慢链路影响，120秒load超时保留；单独检查使用本地同SHA字体/图片缓存，逐项与当前137文件服务器清单绑定，首页与CSS仍取实际公网，完成load/字体后普通滚动与点击。首轮缓存夹具的附件名/下载事件已通过，但新加的routeHits断言为0；浏览器原生下载未经过页面拦截，改核对实际HTTPS下载URL/附件名并立即取消，取消回执canceled通过，避免重下整包。静态缓存输入和真实公网传输范围分别记录，未宣称本轮重新从公网下载全部图片/字体或整包。最终为同一线上字节的两份互补证据，不将原整轮失败改写为成功；没有强制点击或修改产品绕过。生成器连续两次输出幂等，仅更新本地CSS/JS缓存摘要。JS语法、定向git diff检查与文档检查通过；本任务所有自有scratch清理回执已复核，正式包、截图和回退保留。
