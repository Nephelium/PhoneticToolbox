# 3.1.0 三端验收

2026-10-25，Windows、macOS与网页使用同一757e4f冻结，所有本地/私有成品门槛通过。井井已确认输入法0.1.5上线，并明确批准亲自修改后的完整3.1.0文案。网页、四正式下载、官网/频道、地图资料及公告已生效并通过下述公网后验；GitHub源码与Release为本轮最后收口项，完成状态以台账和发布回执为准。

## 冻结与文案

- 冻结：`PhoneticToolbox-310-20261024-224507-757e4f`，北京时间2026-10-24 22:45:46.834。944运行源清单SHA `d4f1c6df4a8a51f65f15e9f6c31f0f76881bff0748454146118eadd56ec39cae`；478前端/author输入SHA `ca2602a595eab5bb001e4c8ce06fe42370d27f2d5a8050c9a24602cb0c754213`。运行snapshot不包含原始前端TS，原始输入与编译dist分别绑定，最初错误QA布局假设已更正，不是产品改动。
- 三包/前端/内部依赖与协议core均3.1.0，API1.1.0保持。原3.0.5发布中止，原文件/断点与审计保留，未当作整版发布。前轮范围见[历史验收](release-3.0.5.md)。
- [正式更新文案](../../release/notes/3.1.0.md)保留井井编辑，只规范行尾，5747B，SHA `389a4c29fbc36406fb560c5ebeb60748b8995e85c6fc0b027b0aeb22550576da`。论文与Zenodo链接保留，应用更新弹层7项及官网定向renderer通过。GitHub/官网/两个更新频道使用同一正文。
- DataV冻结试用资料2、930地点、474300原始单元及5矩阵未改；17运行文件91473235B，源manifest SHA `e86ed70dfcb9f61eb61d601575db45a80bbc9a9b745746bd54db8a3c2fc5d7bd`。本轮没有进一步地理内容审计，没有整本PDF、旧OSM图或候选5入包。

## 实际验证

| 范围 | 已执行结果 |
| --- | --- |
| 共享源码 | 768前端/typecheck/build、30版本专项、20 native URL边界、13公告发布校验；公告Chrome19/源码Qt10通过。严格手册22章1128资产，公开投影22章/8新图；fresh Git读本辅助8项及32输出原字节通过。 |
| Windows最终字节 | 完整16阶段与7补充门槛通过。首次缓存选择/拒绝/重开复用、M19完整21/8图、默认GPU普通renderer11、手册音标168场景/336帧/21桌面章、3对实际2倍PNG/SVG、公告12+真实同profile重开2通过。完整8550science+418host复制/全SHA/同长mtime损坏修复通过。 |
| Windows启动 | 旧3.0.5三冷67.408/60.633/66.674秒，中位66.674；新3.1.0三冷66.888/65.614/61.822秒，中位65.614。新热8.875/8.793/8.640秒；真实跨版复用25.824秒。计时使用offscreen软件GPU，默认Intel D3D11可见UI另验，不混同两类结果。 |
| Mac最终字节 | 新源码26/冻结成品26/新Windows与Mac231数值比较通过。Qt20、热点14、M19完整21/8图，搬移后20+M19；5普通入口会话的公告12+重开2、renderer8+4、来源键与严格签名/退出0通过。ZIP6886成员CRC通过，221份证据逐SHA回收。 |
| 网页同源候选 | p757e4f真实3.1.0 wheels与完整RECORD、346浏览器源及未变资源闭合。新9项Linux科学/interactive/VTL音视频及取消通过。新构建Chrome地图17/下载5/读本8/公告关键3通过；19项公告源码证据逐SHA绑定，未冒称19项全部重跑成品。 |

Windows回执 `output/validation/release-310-20261024/windows-757e4f-final/receipt.json` SHA `9c634cd91d58b23a729430d3a5d0237ac47892977972a5afe30e3e0ed897f674`。Mac回执 `output/validation/mac-final-757e4f/final-audit.json` SHA `95a62a9f5b65acf43ed6f1ac11fbf773f94dff585efe5d0bea7f5cb12bd92523`。Web回执 `output/validation/release-310-20261024/web/private-ready.json` SHA `2437f653ee402cfab55a8b2423e7c50621892526584dad45f8449e563a36e67b`。

## 已执行公开发布与后验

- 四正式包公网全文200，总1986183913B，大小/SHA逐项等于最终QA原件；四HEAD200、64KiB Range206及本机前缀逐字节相同，无重定向或压缩，每包只完整GET一次。
- Windows EXE498154366B SHA `ab4a0c8d957aa81bcbb7200c743e32cd625ff5eefcd33338ca13b31d61029447`；setup497970004B SHA `d095e3207873c4a40181656f6aff165bb4fc888e5a8a593942f61df449dce987`；更新ZIP498154932B SHA `5c4fd69a9073b35f2d77d5e9d778b22121e86f111f0d5a6c35a16948ced2aeee`；三件均小于500000000B。
- Mac ZIP491904611B SHA `821ae524448143bdbade90c7cdc1f3e36339f4c2ad7f2aa36733b290324ed33b`，Apple Silicon/ad-hoc，未公证。
- 网页用户服务已激活p757e4f，实际health app/core3.1.0与/app17份入口/懒加载/手册字节通过。1362文件manifest SHA `f116014db7b1859c0a7e570b630b4eedda632c45812144294bfee80a33688b30`；803哈希profile129488B SHA `37ec8ee1b1d0814086f8984796d050e36e5beed5e0a26feb2f7c67025a7ae478`；fresh science receipt `84d856c0bda2473e3141edd4d5696009c3ec416ca9b0c37540c4ff0c76219f90`。原账号配置、旧runtime/receipt/manifest及回退材料保持，未数据库迁移。apply后准备补强因新active安全拒绝，仅隔离草稿，未再次切换。
- 地图/公告精确静态路由已部署。生产HTTPS/www SNI/TLS验证21 raw+21 gzip完整SHA与21解压SHA、42 HEAD/CORS；公网42 HEAD、manifest/来源许可10 GET及公告549B全文/HEAD/CORS、9拒绝通过。配置nginx-t与平滑reload通过，master身份保持，私有备份0700/0600。公告SHA `8a609edf3bf06b6268dd1c5a0ea2ff6b16191f2afc8dd01513dfc2f4c3f92c9d`，不上传已读或研究数据。
- 官网严格保留现行输入法0.1.5、81静态资产及2动态统计，只切首页、2新digest JS与stable/preview。5公网全文SHA通过；两频道notes均等于批准正文；独立Chrome1440/390的新文案/下载/IME hash/触屏/Escape通过，5截图已查看，0页面/HTTP错误，scratch已清。当前首页SHA `10e4e37eb10b0afc6efef829edac2704322b9cef44d1c734ff983824acd228e6`。

上述回执位于 `output/validation/release-310-20261024/publish/`，分别为 `public-download-check/report.json`、`web-activation/apply.json`、`static-routes/receipt.json`、`website-channels/completion.json`。公开操作均有实际写入和后验，不以本地候选替代上线。

## 清理与边界

新Windows、Mac、Chrome与科学QA自有scratch已清，报告/截图/配方保留。旧3.0.5回退和既往拒绝目录未触碰，逻辑字节与实测空间分开记录。

Mac实际26.6.2/arm64，未验证macOS14/Intel/Safari；Windows不是干净VM验收。未验证实体麦克风/扬声器/摄像头、OS默认浏览器真实打开或长期稳定性。最终地图结构依实际SVG、截图与导出核实，没有读取私有OpenLayers getter或新增地理内容审计。公开UI若使用合成账号响应，将单列而不冒充真实生产账号数据库E2E。

Git提交限定519项，64项混合官网/私有或候选内容排除，944运行源与478前端/author当前SHA仍匹配冻结。完整diff空白检查保留第三方ODbL原文36处、FFmpeg补丁必需上下文空行12处及已冻结Mac QA脚本1处文件末空行提示，不改原文/补丁或冻结输入来消除提示；其余候选无空白错误。REAPER保留100755。
