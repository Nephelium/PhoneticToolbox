# 网页模块设置保留验收

2026-10-17，任务 `WEB-SETTINGS-RETENTION`。当前证据覆盖源码、全新隔离 PostgreSQL 和 Windows 独立 Chrome/Edge。现有服务数据库未迁移，未部署。

## 实现范围

18模块的操作参数、显示与导出选项，以及工作台主题、字体、显示样式和栏宽，按账号保存在服务器的 `ptb_accounts.module_preferences`。该表只关联账号，不关联临时工作区、项目或资产，没有到期时间。退出登录与工作区清理仍回收上传、生成及编辑内容，保留设置。登录时先读取服务器设置，再打开工作台，换浏览器、换设备或清除站点数据后仍可恢复。

2026-10-16仅在当前浏览器按账号分区保存的实现未满足服务器保存要求，本轮已替换。首次升级只向服务器尚无记录的模块导入旧账号专属设置，已有服务器记录优先。匿名全局本地记录不自动归属账号。浏览器仅为当前标签暂存尚未同步的修改，服务器负责持久保存。

`GET /api/v1/preferences` 读取当前账号快照，`PATCH /api/v1/preferences/{module}` 按修订号保存变化的字段。账号、CSRF、同源、请求体大小与字段白名单检查均保留。修订冲突后读取较新记录，合并本地实际改动的字段，避免覆盖另一设备未涉及的修改。网络失败显示重试提示并继续重试。主动退出先补存，服务器确认失败时保留登录状态。

设置明确排除文件引用、分析结果、逐帧曲线、图像与笔画、调查字表、实验刺激和回答、PDF及批注。生理参数合成通过同源父页桥接，保留当前构型、声源、音量、显示与截面选择；音视频和关键帧工程继续按内容管理。论文自动获取及检查间隔在新工作区重新应用。浏览器设备权限仍由当前浏览器授予。

新增迁移文件 `backend/migrations/011_account_preferences.sql`，应用启动不会自动执行。已仅在本任务创建的临时 PostgreSQL 实例执行，没有迁移现有库、修改现有服务配置或公开发布。

## 实际验证

- `npm test`：442项通过，包含17项设置传输与页面绑定回归。覆盖刷新/工作区切换、账号隔离、数字输入表示、网络失败与重试、保存期间继续编辑、空队列补存、多设备嵌套字段合并及连续编辑不重写较新字段。
- `npm run typecheck`、`npm run contracts:check`、`npm run ui-data:check` 通过。API和字段清单由 `scripts/generate_contracts.py` 生成，再生成TypeScript契约。M10自身资源与既有登记中的当前摘要已定向刷新。
- 最后一次M10父页桥接修订后，17项设置定向、typecheck和隔离Vite构建通过；最终构建证据 `output/validation/web-settings/final-build-42bca500423c4e38b292ca842c9c8a4a/`，scratch回执deleted。文档校验2182文件/442来源/135任务零错误，36处既有归档失效链接保持；本次文件空白检查与M10/实际浏览器脚本JS语法通过。
- 全仓 `scripts/check_architecture.py` 未通过：既有index.html、presets.js、scene.js、floor-contact.mjs四项摘要失配，recording-capture.js与presentation-metadata.json两项未登记。本任务app.js、bundle-manifest及迁移证据中的当前摘要已按实物生成，未改原迁移身份，也未刷新无关条目掩盖既有差异。
- 后端、协议及安全定向154项通过，命令为源码绑定后 `pytest -c tests/pytest.ini backend/tests/test_account_preferences_postgres.py backend/tests/test_auth.py backend/tests/test_web_boundaries.py backend/tests/test_storage_boundary.py backend/tests/test_job_boundary.py tests/contracts tests/security -q`，并显式设置 `PTB_POLICY_FRESH_PG=1`。仅使用全新自有测试库。现有两项测试依赖弃用警告单列。
- 其中5项真实PostgreSQL测试覆盖APP与全18模块往返、新连接读取、账号/CSRF/请求边界、并发修订，以及主动退出和强制到期后实际清理上传/结果、设置仍存。到期用合成租约时间推进，未等待真实30分钟。
- 实际HTTP与构建后Vue使用相互独立的可见Chrome、Edge及第三个空Chrome配置。验证同账号恢复LPC与音标转换设置、两个设备交替和连续修改、断网提示与显式重试、清除本地数据后退出重登、另一账号默认值、全19条账号设置仍存。六组检查通过，pageerror为空。浅深截图保留，构建日志同时提供本轮隔离构建证据。
- 真实服务浏览器证据：`output/validation/web-settings/server-browser-6fa98210bd18455ea13f559d2a04e036/`。前后端日志位于同目录上层的 `frontend-tests.log`、`backend-tests.log`。自有浏览器/API/临时PostgreSQL均停止，可重建构建、浏览器配置、数据库和合成资产已按清理回执回收，报告、日志、截图及摘要保留。
- 说明书正文已改为服务器随账号保留、跨浏览器/设备恢复及保存失败重试，6张正式浅色图已重新采集。严格校验21章/1059素材，errors/warnings均为空；重新生成阅读资源后，独立可见Chrome确认新正文可读、21章/18模块、搜索及6图原始尺寸解码通过。最终阅读证据 `output/validation/web-manual/browser-13a4876077a44f41963428930efd7e7e/`，scratch回执为deleted。

首次真实浏览器验收发现原生fetch绑定方式和空补存留下已完成请求引用的问题，均已修复并加入回归。另有一次输入定位与页面实际无障碍名称不符，修正验证夹具后重跑通过。失败现场和清理回执保留，没有降低断言或绕过服务验证。

## 未完成与未验范围

现有私有内测实例执行011迁移并更新前后端后，当前服务才会具备此行为。该操作需要按根规则取得现存库迁移与实例更新的对应授权。本轮没有公开部署、最终EXE构建、实体第二设备/macOS/Safari、全部18模块逐控件或真实科研设备验收，M10完整原生会话设置往返亦未重验。跨浏览器证据来自同一Windows机器上的独立配置，服务器存储与接口不依赖浏览器本地数据。
