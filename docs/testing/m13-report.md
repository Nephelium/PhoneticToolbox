# M13 普通话转 IPA 迁移验收

2026-09-26，M13 开发态功能在 Windows Chrome、Windows Qt 桌面宿主及 WSL2 Ubuntu 静态托管的限定范围 **verified**。转换和图片导出完全在前端完成，没有创建 core/API、服务器任务、数据库变更或外部文本请求。生产部署、Linux 真实浏览器、macOS、EXE 及 `PENDING-IPA` 来源许可仍未验证。

## 功能覆盖

- F01：保留 10 个旧 IPA 映射列和旧汉语拼音声调规则，共 11 个选项；保留逐位置多音选择、旧数据第一条默认值、未知映射 `?`、标点/拉丁字符/空白/换行。
- F02：汉字与 IPA 独立字号、字音间距、行距、汉字粗斜体/下划线；IPA 固定内置 Doulos SIL。用户试用发现普通项与多音按钮高度不齐，定位为公共按钮 `gap: 7px` 泄漏，模块 token 已显式归零。Chrome 坐标回归小于 0.5 px，Qt 三项 IPA top 均为 225、汉字 top 均为 250.42709350585938。
- F03：字音同显/仅音标、左右/上下排布，切换不清状态；1,200 个已映射字符长文本完整渲染。
- F04：直接 Canvas 生成 PNG，运行时不含 html2canvas/CDN；断网下载成功，字体加载失败和 PNG 编码失败均保留文本并显示可恢复错误。
- 公共工作台：M13 按需加载，使用公共模块框架、工具栏、状态、浅深主题、标签关闭保护和本机草稿；无重复大标题、模块关闭按钮或音频条。公共接线另见 `docs/testing/p04-unify-report.md`。

旧说明书与源码的逐组关系、默认值、资源散列和限制见 [M13-source-map](../modules/evidence/M13-source-map.md)。

## 实际命令与结果

```powershell
npm --prefix frontend test
# 137 passed, 0 failed/skipped

npm --prefix frontend run typecheck
# vue-tsc --noEmit passed

npm --prefix frontend run build
# Vite 167 modules transformed; passed
# AppShell 470.86 kB / gzip 142.91 kB
# M13 lazy chunk 3,011.65 kB / gzip 149.07 kB

node tests/e2e/m13.cjs
# 10 Chrome groups passed

node tests/e2e/p04-registration.cjs
# shared AppShell storage-failure/retry and 12 viewport/theme/scale combinations passed

$env:PYTHONPATH='desktop/src;backend/src;packages/phonetic_core/src;scripts'
& '.venv/m09-ui/Scripts/python.exe' 'scripts/verify_m13_qt.py'
# 7 Qt stages passed

wsl.exe -d NInfer --exec perl /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/scripts/verify_m13_linux_static.pl
# Ubuntu 24.04 local static HTTP checks passed
```

仓库没有 `npm test:e2e`，没有把旧计划中的不存在命令报告为通过。生产构建仍按 Vite 默认阈值报告 M13 字表分包大于 500 kB；字表为 3.10 MB 原始 JSON、gzip 后模块包 149.07 kB，已与主壳分离，没有调高阈值或隐藏警告。

## Chrome、离线导出与截图

证据目录：`output/validation/m13-browser/1790434932018`。

- 11 标准对“妈”的实际结果依次为 `ma˥`、`ma̠˥`、六个 `mᴀ˥`、`ma˥`、`mä˥`、`mā`。
- `银行花` 普通/多音混合项的 IPA 与汉字行坐标分别完全一致，截图 `aligned-pinyin-light-1440x900.png`，SHA-256 `bad2a28b27bc1c510bd8adfbbfcb44375542ec2d64958857ce784ead639d627a`。
- 实际 AppShell 按需载入、dirty、保护关闭、保存并重开恢复通过；截图 `appshell-m13-light-1440x900.png`，SHA-256 `62def235b2b94e576e852934c2a2b139acbca4d38fc66f4f89f2cc77c11fc89d`。
- 公共接线补充证据 `output/validation/p04-unify/registration-1790435185843/report.json`：存储失败不关闭标签、重试成功、取消/放弃语义以及 12 组 viewport/theme/page-scale 组合通过。
- 离线后生成 `m13-offline.png`，1644×585，47,555 个非白像素、40,897 个蓝色像素，SHA-256 `9d5eb736443e145312dd10b048ab467daf0f02198b5835d4a794528ed881b2eb`。Canvas 调用记录确认 `ma̠˥`、`xa̝ŋ˧˥`、`xu̟a˥` 使用 `27px PTB-Doulos, "Doulos SIL", serif`。
- 深色仅音标/上下排布、长文本和浅色字音同显截图均已人工查看；`external=[]`、`errors=[]`。

## Windows Qt 桌面宿主

证据目录：`output/validation/m13-qt/dafabb33c9ea4e27bb270ab39caa9145`。

真实 `Workbench` 通过 `ptbapp` 本地协议加载生产构建，输入 `银行花`、切换汉语拼音、11 个选项、Doulos 已加载、浅深主题、无重复标题/关闭按钮/音频条均通过。报告明确 `database_operations: none`。浅色截图 SHA-256 `606208216edd44ccf2d72b582cabc1fb738bf2f46563ce66eccfb8877a288dd2`，深色截图 SHA-256 `052427750d2d695aa76e21ec6b37a5c60ef716678451a2d5f9f2f27f4539503f`。

Qt 启动日志保留既有素材的 libpng profile/transparency 警告，本轮没有通过修改素材或屏蔽日志来消除；页面、字体坐标、截图及退出码均通过。标签关闭保存/失败重试由真实 Chrome AppShell 专项覆盖，不把无窗口 Qt 弹窗探针扩大为已测。

## Linux 静态托管

证据：`output/validation/m13-linux-static/20260926-150243/report.json`。WSL2 Ubuntu 24.04 通过本地 `IO::Socket::INET` HTTP 服务逐字节回读 `index.html`、AppShell、M13 JS/CSS 和 Doulos 字体，5 项均为 200 且与磁盘 SHA-256 一致。没有外网请求或计算任务。此项只证明 Linux 可静态托管最终构建，不证明 Linux GUI 浏览器交互或生产反向代理配置。

## 文件与边界

M13 自有改动为：

- `frontend/src/modules/mandarin-ipa/{MandarinIpaPage.vue,state.ts,export.ts,generate-data.mjs,ipa-data.json}`
- `frontend/tests/{m13.test.ts,m13-live.html}`
- `tests/e2e/m13.cjs`
- `scripts/{verify_m13_qt.py,verify_m13_linux_static.pl}`
- `docs/modules/evidence/M13-source-map.md`
- `docs/manual/mandarin-ipa.md`
- `docs/testing/m13-report.md`
- `docs/plans/modules/M13-mandarin-ipa.md`
- `docs/modules/v2-manual-coverage.md`

共享 `frontend/src/app/AppShell.vue`、公共组件与 tokens 由公共 UI agent 修改和验证，M13 只遵循约定的 props/emits/expose 接口；平台后端与契约未改。没有修改 M08、M12、M14、V2、用户数据、数据库、系统配置或全局依赖，没有 push、发布或打包 EXE。`docs/modules/module-migration.md`、全局 ADR 和 task ledger 保留给统筹 agent。

## 未完成项

- `PENDING-IPA` 的 11 套映射原始书目、数据来源和再分发许可尚未闭合；当前只能称本地旧数据迁移一致。
- Linux 真实浏览器、macOS、生产域名/反向代理/缓存策略及最终发行包未测。
- 没有新增自动上下文消歧、语流音变或规范化规则；这些属于新科学语义，需要独立设计与授权。
