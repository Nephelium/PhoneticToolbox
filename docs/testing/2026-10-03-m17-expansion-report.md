# M17-R1：符号扩充与中文解释验证

日期：2026-10-03。状态：**verified，限定 Windows 源码前端、实际 Chrome 和构建后实际 Qt**。最终整包构建与Qt由本轮主代理统一验证；WSL仅核对目录再现，不外推GUI。未打包EXE，未执行DDL、环境安装或push。

## 已实现

- IPA 增加107个完整组合示例，独立于官方基础矩阵。extIPA增加12个部位号/例示、4个完整组合、2个明确标注的2002旧版入口。
- 现有总入口500→625：IPA351、extIPA209、VoQS65。入口包含组合、例示和范围工具，不等同于625个独立基本音标。全部265个所用字符由既有PTB IPA Plus覆盖，字体文件未修改。
- IPA/extIPA在各自体系内分区。井井追加反馈后，附加符号、韵律与补充组合改为自适应紧凑多列，平常仅中文名称和符号/例示并列，中文名称保留换行；1366窗口的IPA附加区4列、extIPA节奏区3列。名称悬浮显示简释，符号悬浮显示完整含义、用法、区别和来源，指针移入介绍后可继续阅读。默认开启悬停介绍，保留旧草稿明确关闭的偏好，名称提示始终可用。
- 分区导航固定在表区顶部，长内容有原生上下滚动条，底部编辑框常驻。基础图表和VoQS在两种默认尺寸单屏可见；长组合和范围例示自然换行，不缩小音标硬塞。
- IPA补充区的唇齿、齿唇、舌唇按钮直接进入extIPA对应结果，避免把已有入口遗漏或重复造作“新增”。[截图覆盖及差异](../references/m17-expansion-diff.md)按截图类别逐项列出落点与未确证项目。
- 中文参考吕佳、江荻（2013）的指定PDF4页实际图像，区分2002/2025版本。部分去鼻化保留新版含义，旧文术语用于别名。原PDF、截图未进入产品。
- VoQS全部65条数据和原图表结构与HEAD基线逐项相等，56个指定中文译名测试继续通过。

## 命令与结果

| 命令/核查 | 结果 |
| --- | --- |
| `python scripts/verify_m17_catalog.py` | 625入口、结构/清单可再现、码位校核通过；目录SHA256 `984981551985cb3fffdcc3aa0e58e668195baa97d90da8f2d99a155958c29cff` |
| `python scripts/verify_m17_catalog.py --cin C:/Users/13680/Documents/ipa.cin` | 显式只读资料路径复核通过，默认Windows行为保留；用于跨系统重现CIN别名，未改产品数据 |
| WSL 显式CIN核对 | 主代理使用既有 `/home/ninfer/ptb-m06-20260927/bin/python` 与 `--cin /mnt/c/Users/13680/Documents/ipa.cin` 复现625入口及相同目录哈希。初次不传参数时Linux默认家目录无该资料，差异不冒充通过；未修改HOME、复制原资料或安装环境 |
| `python scripts/verify_m17_fonts.py` | 265字符无缺失；原3944字形保留，字体可再现字节一致；未写入字体 |
| `node --test frontend/tests/m17-ipa-plus.test.ts` | 15项通过，包括组合身份、无圈项定义、分组无遗漏、旧版来源与全部链接可解析 |
| `npm --prefix frontend run typecheck` | 通过 |
| 主代理最终构建与前端回归 | 最新紧凑显示修改后生产构建、全前端231项测试和typecheck通过 |
| `node tests/e2e/m17-ipa-plus.cjs` | 17组通过，625字体CDP实查和625逐入口点击、Unicode选区/撤销/切表、真实剪贴板及UTF-8下载回读、IndexedDB、离线、存储/字体错误反馈 |
| `node tests/e2e/m17-expansion.cjs` | 最新紧凑布局36组通过；1366×768/1920×1080、浅/深色、每体系全部分区覆盖、基础单屏、编辑框可见、无横向裁剪、实际尾项位于视窗、自适应多列、仅名称/符号、名称提示、完整语义悬浮介绍、明确关闭偏好、组合/旧版身份、论文链接、搜索与3个跨表指引 |
| 配色专项 | 分区选中项显式使用accent/on-accent，两主题逐分区计算样式断言通过；目视浅/深截图通过 |
| `python -m py_compile scripts/verify_m17_catalog.py scripts/verify_m17_qt.py` | 通过 |
| 聚焦 `git diff --check` | 通过；仅仓库LF→CRLF提示，无补丁空白错误 |

实际Chrome证据：

- `output/playwright/m17/2026-10-03T06-20-10-462Z/report.json`：最终紧凑显示代码下625输入与实际字体，以及完整17组编辑/存储回归。
- `output/playwright/m17-expansion/2026-10-03T06-20-10-458Z/report.json`：最终36布局、配色、跨表按钮与完整悬浮介绍。
- 最终专项禁用Playwright默认的`--hide-scrollbars`，实际查看 `1366-dark-ipa-marks.png` 和 `1366-light-extipa-context.png` 可见原生垂直滚动条，`1920-dark-ipa-hover-explanation.png` 显示完整语义介绍。早期截图为简释行内显示或隐藏滚动条的中间态，保留证据但不作为最新外观。早期分区选中按钮配色已修为明确accent/on-accent。
- `output/validation/m17-expansion/lv-1.png`至`lv-4.png`：原文4页可读渲染。提取文本有乱码，未当作正确中文依据。原PDF SHA256：`b7b1f1fa261d52f44505ed4a9cf799676f823f6e785de1473ac2a856d0928e27`。

## Qt与平台边界

已更新M17专用 `scripts/verify_m17_qt.py`：每个体系遍历全部内部视图，聚合入口必须与目录完全相等，检查基础图单屏、每区无横向溢出、编辑框可见、尾项可达、原生文本保存，以及统一来源面板中的旧VoQS/新吕佳江荻引用。

主代理在最新紧凑多列构建后实际运行Qt通过：`output/validation/m17/qt-20261003-142313-21fffa/report.json`，`success: true`。包括两尺寸18分区及4组加载/输入保存/布局/来源检查，CJK、附加记号、非BMP字、点击插入、撤销重做、IndexedDB重读和真实原生UTF-8下载无损。主代理实际目视1366的IPA marks/extIPA context，确认4列/3列、原生纵向滚动条和固定编辑框。该证据是Windows Qt offscreen已构建前端，不是新EXE或物理设备测试；反馈前的Qt运行只作为中间证据保留。

M17无音频计算，本轮不需要真实语料输入。未做Linux GUI、实体DPI、物理输入法或新EXE验收。Ctrl/组合字符输入法检查仅包含已记录的Chrome事件测试。截图的特殊小舌/会厌拍音及未确认的卷舌内爆音字形未臆造，不声称完整复制第三方应用。
