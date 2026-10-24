# M17 长例句介绍与全部显示

核对日期：2026-10-10。`M17-HOVER-ALL-VIEW` 为 `verified`，范围为 Windows 源码、独立 Chrome 与当前前端构建。

## 修改

- 长例句原先作为不可收缩的音标横排，名称栏被挤窄后把标题高度撑大。超过 12 个 Unicode 码点的例示改用独立音标行，按浮窗宽度换行；名称、说明与来源正常排列。悬停与右键详情共用修复，短例子保留原有横排样式。
- 点击播放暂时禁用。按钮悬停提示为：演示内容正在创作中，将在后续更新中开放。当前点击符号仍用于输入；既有媒体接口保留供后续制作。
- 王天恒 VoQS 引用的访问日期改为 `2026-10-10`，同步条目来源、译名数据、统一来源登记、生成来源与当前说明书。发表日期仍为 `2020-08-30`，本次未重新审计文献或许可。
- IPA 与 extIPA 的分区栏首项为显示全部，且为首次进入及重开后的默认选项。聚合原有分区，所有入口各出现一次；表区独立滚动，下方编辑区保持可见，原有单独分区继续可选。
- 模块帮助、README、操作说明与正式 M17 章节同步。早期播放接口截图明确标为早期画面，正式素材保留。

## 验证

| 实际命令或检查 | 结果 |
| --- | --- |
| `npm run contracts:check`、`npm run ui-data:check` | 通过 |
| `npm run typecheck` | 通过 |
| `node --test tests/m17-ipa-plus.test.ts tests/m17-r2-sources.test.ts tests/m17-r3.test.ts` | 22 项通过，含完整目录、Unicode 输入、撤销、来源及悬停定位 |
| `node tests/e2e/m17-hover-all.cjs` | 独立 Chrome 通过，页面错误为 0 |
| `scripts/manual/validate.py --project manual --strict` | 20 章、982 素材，错误和警告均为 0 |
| `scripts/manual/build.py --project manual --output frontend/public/manual --distribution software` | 通过，当前阅读资源已生成 |
| `npm run build` | 通过，保留既有大 chunk 提示 |
| `scripts/check_architecture.py` | 通过，错误为 0 |
| 受影响文件 UTF-8/JSON、当前引用日期一致性、本地链接与 `git diff --check` | 通过 |

浏览器检查覆盖 1366×768 与 1024×720 两种窗口、浅深主题、100% 与 150% 页面缩放。16 组全部显示检查核对 IPA 351 和 extIPA 209 个入口的完整性与唯一性，实际鼠标滚轮、末项可达、编辑框可见、文本保留、分区切换及重开默认值均通过。24 组长例句与短音标悬停检查核对标题和名称可读、无横向溢出、浮窗处于可视范围、鼠标可移入阅读。另验短组合 `pʰ` 保留原有横排、长例句右键详情、精确字符插入与撤销、禁用播放及实际鼠标点击标签后仍未启用、最新引用日期显示。

最终浏览器证据：`output/validation/m17-hover-all/2026-10-10T15-32-18-091Z/report.json`，截图保存在同目录。实际目视长例句介绍与 IPA 全部显示截图。独立浏览器配置在 `scratch/`，关闭自有进程后经项目清理入口删除，`cleanup.json` 为 `deleted`。首轮脚本对禁用标签调用 Playwright click，自动检查按预期拒绝点击；后改用真实鼠标坐标验证禁用效果，产品没有因此改为可点击。

## 范围

当前源码工作台重启后使用本次前端构建。本次未重新打包或验收最终 EXE、原生 Qt、系统 DPI 与其他平台，也未重验保留的播放接口。此前 R3 的播放开关开启流程属于历史行为，当前产品入口由本报告的禁用流程覆盖。未改科研字符、目录入口、原始语料、数据库或系统配置，未 push 或公开发布；同期任务差异保留。

## 2026-10-13 滚动重叠修复

`M17-CHART-SCROLL` 为 `verified`，范围为Windows源码/独立Chrome。分区栏取消 `position:sticky`，版本、右上角输入说明移入 `.m17-chart-viewport`，随表格一起滚动并由表区裁切。下方文本编辑框保持原位置。模块 README 同步滚动行为。

Windows 源码已执行 `node --test tests/m17-ipa-plus.test.ts tests/m17-r2-sources.test.ts tests/m17-r3.test.ts`，22项通过；`npm run typecheck` 和 `npm run build` 通过，保留既有大chunk提示。`node tests/e2e/m17-hover-all.cjs` 通过16组全部显示/实际滚轮与24组悬停布局，覆盖1366×768、1024×720窗口，浅深主题及100%/150%页面缩放。符号完整性与唯一性、末项可达、分区切换、编辑框与文本保留、输入/撤销、重开默认值均通过，页面错误0。`scripts/validate_docs.py` 文档检查及定向 `git diff --check` 通过。

最终证据：`output/validation/m17-hover-all/2026-10-13T11-52-08-656Z/report.json`，同目录保留滚动前后浅深主题截图；目视浅深滚动截图无重叠。自有浏览器关闭后scratch清理回执为 `deleted`。此前三份失败诊断及一份旧引用断言修正前的运行诊断保留，均已清理自有scratch。

滚轮用例记录滚动前后按钮、说明和编辑框的实际矩形，要求按钮与说明移动距离等于实际滚动量乘页面比例，并滚出表区上沿，编辑框保持位置。VoQS仅在内容溢出时验滚轮，否则核对说明归属滚动容器。初次VoQS在宽屏全部可见，测试错误要求必须滚动已修正；后续旧脚本直接设置DOM缩放未保持应用缩放状态，改用 `setPageScale`，原布局断言保留。旧来源断言同步现行 GB/T 格式与2026-10-11访问日期，不修改来源数据。失败报告保留。

本轮未验原生Qt、最终EXE、系统DPI及跨平台，未打包或发布。


## 2026-10-23 元音图透明背景

- 原因：`frontend/src/design/background.css` 将 `.m17-vowels>svg` 列入背景图片模式下的强制 `var(--panel)` 铺底范围。移除该选择器后 SVG 使用默认透明背景，其他图像/科学绘图的底色规则保持。
- Windows 独立 Chrome 浅/深主题实际背景图验证通过，元音 SVG 背景 rgba(0, 0, 0, 0)、路径无填充、28 个按钮保留且点击输入 i 正确，页面错误 0；typecheck、22 项 M17 测试、contracts:check、ui-data:check、build:workbench 通过。
- 执行命令：`npm run typecheck`、`node --test tests/m17-ipa-plus.test.ts tests/m17-r2-sources.test.ts tests/m17-r3.test.ts`、`npm run contracts:check`、`npm run ui-data:check`、`npm run build:workbench`。构建仅有大分块与插件耗时提示。
- 浏览器通过一次性 Node/Playwright 探针启动独立 Chrome 与项目 Vite，使用 `changeBackground` 导入内存生成的条纹 PNG，分别检查浅深主题计算样式、28 个按钮及输入结果，截图已目检。未使用用户设置/草稿；自有浏览器与服务已关闭，临时浏览器配置由 Playwright 管理。报告及两张截图保存在 `output/playwright/m17-vowel-transparent-20261023/`。
- Qt、最终 EXE/Mac 包与公开网页未更新或新增验证，已打开的源码窗口需重开载入。
