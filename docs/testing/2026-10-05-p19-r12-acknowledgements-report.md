# P19-R12：底部入口、致谢署名与字体许可

日期：2026-10-05。状态：`verified`，限定 Windows 前端源码、独立 Chrome 与隐藏原生 Qt 宿主的本轮界面及来源登记。新 EXE 单列为构建完成，未启动或验收。

## 完成范围

- 侧栏底部左列为设置、关于，右列为使用说明、检查更新。列宽比例为 2:3，缩小左右内边距，长标签可换行。原入口行为和折叠侧栏的图标入口保留。
- 关于页两个按钮左右并排、等高，窄视口允许按钮内文字换行。
- 公共 `MethodReferences` 及账号页、各模块复用的来源列表省略缺失作者和版本占位文案。没有作者的引用仅复制题名，没有首尾分隔符。内部来源缺口和已有许可说明保留。
- 102 条来源补充作者、团队、版权主体或明确标注的维护者，附本次查验 URL、日期和归属依据。包含 22 条原中文作者占位记录及 80 条构建依赖的 `unknown` 记录。Weston Ruter 仅归属所列 IPA keyboard，未将其归为 ipachart.com 作者。Legorooj 明确标注项目维护角色。
- 四条本项目自有测试/集成/实现记录从外部致谢隐藏：`PROJECT-SYNTHETIC-WAV`、`PROJECT-M10`、`PROJECT-M05`、`METHOD-M16-SPECTRAL-SUBTRACTION`。保留内部工程来源登记及各自第三方方法/依赖。外部记录显示署名省略 PhoneticToolbox 适配者字段。第三方邮件许可原文中对被许可项目的指称保留。
- 字体许可弹窗统一标题为内置字体版权与许可。页签分别展示 Doulos SIL 7.000、JetBrains Mono 2.304 和 PTB IPA Plus 1.000。派生音标字体页同时显示 Doulos 与所用 Noto 字形的许可。点击或方向键/Home/End 切换，切换后阅读区回到顶部。

## 来源与许可

[逐条作者查验](../../third_party/evidence/p19-acknowledgements/author-audit.json)与[公共登记](../../third_party/source-registry.json)一致。官方手册标题页、上游项目版权署名、PyPI/npm 固定版本元数据、必要的准确版本包内版权声明用于署名确认。下载的包仅在内存读取文本，没有安装或执行第三方脚本。

仍未定位作者的本地 EGG/词典/11 套转换规则/音系归纳/旧编辑器/设计图/CIN 及 JavaScript argparse 移植分别保留内部未决项。主题原作者与许可的既有缺口不因本次显示精简而关闭。102 条署名查验不等于 102 条再分发许可审查通过。

JetBrains Mono 的当前字体文件和两份 OFL 文本与官方 v2.304 发布文件逐字节相同：

- 字体 SHA-256：`a9cb1cd82332b23a47e3a1239d25d13c86d16c4220695e34b243effa999f45f2`，92,164 字节。
- 许可 SHA-256：`30f0c136e3c88e422d0791acd97238870f9054a9729bc34cf2ff0d4ed8cac4ad`。
- [官方固定版本 OFL](https://github.com/JetBrains/JetBrainsMono/blob/v2.304/OFL.txt)允许在保留版权和许可等条件下随软件嵌入和再分发，继续内置，无新字体替换或下载。

公共登记仍为 391 条，其中四条历史退役、四条本项目记录不进入外部致谢，界面为 383 条。106 条记录仅作者查验或显示标记变化，其余 285 条保留。所有原实际版本、许可与发行状态未改。

## 实际验证

- `npm --prefix frontend run ui-data`、`ui-data:check`：383 条来源，生成一致。
- `npm --prefix frontend run typecheck`：最终通过。首次运行时同期 M09 的新契约字段尚未同步，未由本轮修改该模块，后来当前源码重跑通过。
- `npm --prefix frontend run test`：287 通过，0 失败/跳过。日志：`output/validation/p19-r12/frontend-tests.log`。
- `npm --prefix frontend run build`：通过。已有大分块提示保留，未放宽构建检查。
- 独立 Chrome 实际 Vue 页面：浅深两模式 × 导航宽 184/224/360 × 页面缩放 100/150，共 12 布局。底部列/行顺序、左列较窄、标签无溢出、关于页按钮水平等高通过。三字体页签/版权全文/键盘操作/阅读位置重置、两致谢分组、已核实作者搜索、无作者题名复制与四个入口原行为通过。
- 隐藏 Windows 原生 Qt 6 宿主：浅深两模式 × 100/150%，最窄侧栏共四布局通过。真实原生指针切换三字体页签、致谢分组显示和题名复制通过。复制终点为记录适配器，无用户剪贴板写入或系统浏览器弹出。
- JSON 完整性/去重、许可与版本保留、限定文件 `git diff --check`：通过。

证据：`output/validation/p19-r12/ui-report.json`、`qt-0f2b8ae8950e43a081f5a664f55ecda3/report.json`、`integrity.json`、`font-license-verification.json`。截图已实际查看，不把模拟页面缩放当实体 DPI 检验。早期验证脚本的旧主题入口路径和全局元素选择器已按真实页面范围修正。

## 本机包与限制

新入口：PhoneticToolbox-v3-Latest-20261005-R1.exe（历史本地产物，当前工作区不存在；原路径 `../../dist/PhoneticToolbox-v3-Latest-20261005-R1/PhoneticToolbox-v3-Latest-20261005-R1.exe`）。采用原有 `build_v3_local_preview.py --lean-qt` 配置，更新本机试用包。构建日志为 `output/build-PhoneticToolbox-v3-Latest-20261005-R1/build.log`。

EXE 构建已完成，296.77 MiB，SHA-256 `6b9c0a8c7253d224f6fae7155dd91dd549c9226d675a8e4792ce1c352bd5e68a`。EXE 尚未启动检查，沿用井井此前快速打包、不检查成品的要求。仍依赖既有本机科学环境，不能作为可搬迁发行物。科学任务、实体设备/DPI、Linux/macOS GUI、服务器与公开发行许可未在本轮验收。无 push、发布、安装、数据库迁移、系统配置修改或旧包清理，保留同期改动。
