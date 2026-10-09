# M13-R3：字体标签排版与实际 EXE 验收

2026-10-05，**verified，限定 Windows 本机 EXE 的 M13、源码与 Chrome 定向回归**。井井反馈汉字字体标签竖排、下拉框重叠。上一轮 [M13-R2](2026-10-05-m13-r2-report.md) 验证了最终源码宿主，但没有覆盖井井实际运行的同期打包成品，本轮补齐。

## 根因与修复

只读进程核对确认当前用户窗口来自 `dist/PhoneticToolbox-v3-M11-20261005-R2-Final/PhoneticToolbox-v3-M11-20261005-R2-Final.exe`。实际解包的 `MandarinIpaPage-CiD3xNA1.css` 中仍为 `.m13-settings-section .font-family-select` 的间距规则，缺少最终字体根标签的 flex 覆盖。新字体组件的根元素也是 label，受到本模块通用 label 两列网格布局影响。源码通过不能代表该中途快照包通过。

旧 EXE 摘要 `0af3488d787bd4f220f7f01b2974a9f219a196b1a6cabe29cdac0a208fdeceb0`，运行中的 CSS 与该包归档内容逐字节一致。只读诊断证据为 `output/validation/m13-r3-frozen-final/old-package-diagnostic.json`。

- M13 的通用 label 规则显式排除 `.font-family-select`，字体组件采用独立纵向 flex，标题与整宽下拉各占一行。
- 回看截图发现本机字体候选包含空名称，造成空白选项与系统默认的空值重复。只在 M13 入口对名称 trim、过滤空项并去重，系统默认始终显示且只保留一个。
- 字体下拉和自定义输入随基础字号增高，14px 时仍为 30px，24px 时为 44px，防止文字裁切。
- 增加固定成品入口 `--verify-m13-preview <独立目录>`，沿用既有预览检查模式。实际成品使用独立 SQLite、缓存与字体/浏览器 profile，保存框在测试进程内重定向，用户窗口和草稿未关闭或修改。

诊断中首个 `M13-20261005-R3` 候选发现空白字体项，仅保留为诊断产物，不交付。最终交付 **M13-20261005-R3-Final**。

## 成品

`dist/PhoneticToolbox-v3-M13-20261005-R3-Final/PhoneticToolbox-v3-M13-20261005-R3-Final.exe`

- 311,261,518 bytes，296.84 MiB。
- SHA256：`1b86822d853910ae29e1a7812efb4f0194138a8596b939c4a47c619f743dd2b3`。
- 使用既有本机 Python/PyInstaller 和 `--lean-qt` 构建，没有安装新依赖。`portable:false`，继续采用原本本机科研环境与 M11 组件路径。
- 保留当前快照中的同期模块修改。包内前端及三处 Python 源码目录共 **421 文件**与独立构建快照逐文件 SHA256 完全一致，成品运行时的两项 M13 资源摘要再次与快照一致。

## 实际验证

| 检查 | 结果与证据 |
| --- | --- |
| 全量前端测试、类型检查、构建 | 299/299；类型与构建通过。`output/validation/m13-r3-final-unit.log`、`m13-r3-final-build.log`。原有大分包警告保留。 |
| UI 数据与契约生成一致 | 两项通过，372 来源记录；没有改动 IPA 映射数据或 Doulos 字体。 |
| Chrome 定向回归 | 6 功能组、12 布局通过，声调/ü、默认值/恢复、字色/字体、两张离线 PNG、字体失败与多音位置修复。`output/validation/m13-r2/chrome-1791189652656/report.json`。 |
| 最终源码宿主 | 16 布局和三组原生交互通过，完整字体菜单 444 选项。`output/validation/m13-r3-source-final/report.json`。 |
| 实际最终 EXE | 从系统临时目录启动，移除外部 PYTHON/PTB/QT 环境变量，运行包内实际前端及 Qt。16 布局全部通过：1440×900/1000×700、浅深色、14/24px、系统默认/KaiTi。通过已有宽度控件的 Home 调整至真正 240px，并断言实际宽度；标题严格单行，下拉不重叠，默认文字精确为系统默认。`output/validation/m13-r3-frozen-final/results/report.json`。 |
| 原生交互与 PNG | QTest 实际点击声调及保存 PNG，输入真实按键。IPA 固定 PTB-Doulos，自选色的 PNG 精确像素回读通过。1821×474，SHA256 `d03daf0ecf374de39cfaf275984c96c6bab79222fb6e25cb952804b6041b353b`。保存草稿、关闭重开为空/Beijing、显式恢复通过。 |
| 静态快照与退出 | 421 文件逐一一致，成品 CSS 含通用 label 排除规则。四个本次子进程退出，无残留。`archive-report.json`、`process-report.json`。用户原窗口继续运行。 |
| 截图人工回看 | 实际 EXE 的 `results/light-1440-14-system-default.png` 与 `light-1000-24-system-default.png` 已回看，标签横排且系统默认可读。短窄大字号采用原有滚动。 |

可复核命令：

```powershell
& '.venv/m14/Scripts/python.exe' -X utf8 scripts/build_v3_local_preview.py --name PhoneticToolbox-v3-M13-20261005-R3-Final --lean-qt
python -X utf8 scripts/run_m13_package_check.py --exe dist/PhoneticToolbox-v3-M13-20261005-R3-Final/PhoneticToolbox-v3-M13-20261005-R3-Final.exe --snapshot output/build-PhoneticToolbox-v3-M13-20261005-R3-Final/snapshot --out output/validation/m13-r3-frozen-final
```

上述输出目录已存在。复核时指定新的包名和验证目录，不覆盖已有证据。检查运行器使用本机已有 Miniconda Python 的 psutil/PyInstaller，构建环境没有 psutil，不增加安装。

## 边界

实际成品验收限定 M13。未重验其他模块的科研任务、实体输入法/多显示器/DWM、Linux/macOS 原生界面或生产网页。隐藏 Qt 测试仅在自有进程禁用 GPU 合成和静音，既有图标 PNG 元数据与 GPU 回退告警保留，未声称修复这些告警。没有修改系统配置、现存库 schema、密钥或用户数据，没有 push、公开发布、安装或删除旧包。

先保存当前用户窗口中需要保留的编辑，再关闭旧窗口并启动最终包。两包沿用既有本机数据目录，文本仍需通过恢复本机草稿显式加载。
