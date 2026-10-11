# GitHub 图文 README 检查

任务：`DOCS-GITHUB-HOMEPAGE`。日期：2026-10-11。状态：`verified`，限定本机文档、最终 EXE 界面截图、本地渲染和实际 GitHub 主页及发行检查。

## 交付内容

根 README 面向使用者介绍 3.0.0 工作台、Windows 下载与首次使用、研究流程、18 个模块、MFA 独立组件、帮助与反馈、数据保存及开发入口。官网入口采用本地静态 SVG 图片按钮，黄蓝配色与默认 W1 图标一致，不依赖动态图片服务。全文没有 HTML 折叠块。

当前主页展示五张同一份最终 Windows EXE 的原始截图：工作台首页、国际音标表Plus、汉字转国际音标、生理参数合成和外观设置。截图与按钮位于 `docs/assets/readme/`，来源、尺寸和 SHA-256 见[配图说明](../assets/readme/README.md)及[来源摘要](../assets/readme/provenance.json)。此前六张源码工作台图片及其原有输入授权保留为历史证据，根 README 已改用新图。

## 本次最终截图身份

| 对象 | 已核对的值 |
| --- | --- |
| 应用 | `PhoneticToolbox-windows-x64.exe`，版本 3.0.0，本轮候选 R7 |
| 文件大小 | 496,815,155 字节 |
| 应用 SHA-256 | `162c8fa542b7b3b1203ce535e3021235e261f0dfaeb095659f9946baec055edb` |
| 源码快照 SHA-256 | `3d1fd7bb865ca389df96ca0b90a3a39327267e90ca76df3fd01abc30e4ffa880` |
| 采集环境 | Windows 实际 Qt 进程，任务独立 C 盘配置与缓存，仅连接该进程自己的 Qt WebEngine 应用目标 |
| 页面与图片 | 1600 × 1050 CSS 像素视口，DPR 1.5，应用页面缩放 120%，原始 PNG 2400 × 1575 |
| 图片总大小 | 五张 PNG 共 2,220,286 字节，无裁剪、缩放、拼贴或重绘 |
| 输入 | 默认应用内容及本次编写的通用汉字示例，无私有研究数据、原始录音或私人路径 |

M10 使用默认 VTL `/a/` 构形，模型状态为 `ready`，构形等待状态为 `false`，129 个截面已就绪。该图同时显示器官视图、面积函数、截面与传递函数。没有把模型默认构形写成标准普通话准确率验证，也没有将本轮截图视为音频试听结果。

## 已执行的检查

| 项目 | 实际结果 |
| --- | --- |
| 截图采集 | 最终 EXE 正常启动，五个实际页面截图完成，进程正常退出，退出码 0 |
| 原始图片 | 对公开目录副本与采集原图逐一核对 SHA-256 与字节大小，PNG 尺寸一致；已逐张目视检查 |
| 模块入口 | 根 README 含 18 个唯一模块 README 链接，目标均存在 |
| 本地 README 渲染 | 既有 marked、独立无头 Chrome 和 Playwright 渲染 GFM，1200 px 与 390 px 两种宽度均通过；完整页及顶部视图已目视检查 |
| 图片与导航 | 五张 PNG 和一张 SVG 均完整解码；两种宽度无页面横向溢出；四个页内导航目标有效；无 HTML 折叠块 |
| 官网按钮 | 静态 SVG 无脚本、事件处理器、foreignObject 或外部资源；图片替代文字为“访问软件官网 ↗”，点击目标为 `https://www.phonetictoolbox.com/` |
| 文件引用 | 根 README 的 41 个本地文件引用均有效；连同配图说明与本报告，共复核 60 个本地引用。最终 12 件候选合计 63 个引用，在实际提交树中均存在 |
| 组件说明 | 官网 MFA 3.3.8-r1 元数据可读，ZIP 为 812,556,132 字节，安装后为 2,805,971,894 字节；README 采用约 813 MB / 2.81 GB，并说明主包不包含 MFA、模型与词典 |
| 文本与仓库边界 | 本次 Markdown、JSON、SVG 均为 UTF-8 无 BOM；公开候选仅包含 README、配图与必要来源/检查说明，采集脚本、配置、状态和预览留在忽略的 output 中 |
| 运行缓存清理 | 截图进程退出后按实际归属清理其解包缓存，活动进程为 0，清理完成，无剩余启动目录；小型隔离配置保留在任务临时位置，未进入公开候选 |
| 实际 GitHub 主页 | 公开 HTTP 200 页面在独立 Chrome 中按 1200 px 与 390 px 两种宽度检查，六图均实际解码；五张 PNG 为 2400 × 1575，按钮 SVG 为 300 × 58。两种宽度均无折叠或横向溢出，顶部截图已目视检查 |
| 远端导航与按钮 | GitHub 实际页面的 18 处标题及导航锚点引用均有目标；四个手动导航在两种宽度都实际点击并到达目标。官网按钮显示正常，链接为 `https://www.phonetictoolbox.com/` |
| 正式 Release | `v3.0.0` 已公开为 Latest，非草稿、非预发布。五件附件的名称、大小、服务端 SHA-256 全部匹配最终计划；两份小清单独立下载后与本地逐字节一致 |
| 真实 GitHub 更新源 | 使用实际 UpdateService 默认网络运输读取公开 GitHub 发布和 `ptb-release.json`，安装版与免安装更新 ZIP 两包解析成功，名称、URL、大小与摘要一致 |

## 本机证据与修正

- `output/validation/readme-release300/` 保存本轮采集脚本、最终采集报告、页面状态和缓存清理回执。成功采集报告中的应用身份、五张图片摘要及其原始字节已核对。
- `output/playwright/readme-release300/` 保存本地预览、两种宽度的完整页与顶部截图、渲染报告。该目录被 Git 忽略。
- 首轮截图脚本在插入多行示例文本时出现转义错误，已改为明确的 JSON 字符串编码；未修改业务代码。其所属进程结束并清理缓存后，最终一次采集完整成功。公开目录只采用成功运行的五张图片。
- 远端检查首轮按原始标题 ID 查找，未处理 GitHub 实际的 `user-content-` 前缀；检查器已按实际 DOM 修正，并增加真实导航点击验证。第二次运行的 1200 px 检查通过，重新请求页面时返回非 200；最终使用同一份公开 HTTP 200 页面切换两种宽度，完整通过。原两次回执分别保留为 `github-live-first-attempt.json` 与 `github-live-second-attempt.json`，最终 `github-live-report.json` success，检查进程退出 0，独立 Chrome 已关闭。
- `output/validation/release-300/github-release/` 保存五件精确资产计划、草稿及公开验收回执、独立下载的小清单和实际网络解析结果。GitHub 专用清单使用 GitHub 下载 URL，官网清单保持原样，三件应用包的身份不变。

## 发布目标与验证边界

本工作树的主线为 `main`，上游为 `origin/main`，远端仓库为 `Nephelium/PhoneticToolbox`。用户已明确授权本次 README 与公开配图同步，提交使用单独审阅的文件清单，不包含同一工作区的其他改动、安装包、私有研究数据或受限媒体。GitHub Releases 的历史发布保留，本轮应用下载入口为官网与 GitHub Releases。

首个图文及发行报告提交为 [`b669645ff779886b039f040f96cd7f113fbbbd3f`](https://github.com/Nephelium/PhoneticToolbox/commit/b669645ff779886b039f040f96cd7f113fbbbd3f)，严格包含批准的 12 件文件。远端 main、提交树与发布标签在发布时均已核对，文本按 Git 行尾规范比较，五张 PNG 原字节不变。[正式 Release](https://github.com/Nephelium/PhoneticToolbox/releases/tag/v3.0.0) 的标签固定在该提交，本报告后续验收补充仅更新 main，不移动发布标签。应用包来自前述冻结快照，标签仓库状态与冻结成品源码的对应范围已在 Release 说明中公开。

实际[GitHub 主页](https://github.com/Nephelium/PhoneticToolbox#readme)的图文、按钮与导航已经验证上线。最终 EXE 的其他发行验证见 [Windows 当前发行检查](windows-current-release.md)。本报告不外推科学准确率、物理音视频设备、长期运行或其他系统的验证结论。
