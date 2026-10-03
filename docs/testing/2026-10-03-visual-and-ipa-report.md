# P18 / M17-R1 联合验收

日期：2026-10-03。状态：**verified，限定 Windows 开发态、下列 Chrome/实际 Qt 与 WSL 目录生成范围**。已包含同轮追加的紧凑排列与悬浮说明。本轮未生成 EXE。

## 实现范围

- M01–M09、M11–M16 共15个模块采用同一套默认300 CSS px侧栏、上下对齐外框、30px控件高度、工具栏与卡片间距。文件/刷新靠顶左，参数草稿/帮助/引用靠顶右，主要分析/生成位于右侧任务区。直接作用于图窗、采集或工程的操作保留在对象附近，逐项位置和例外见 [P18报告](2026-10-03-p18-layout-report.md)。
- 两栏模块将单个操作栏置左。EGG的参数/试听/任务整体移左，中央四图与总览保留。已有栏宽偏好继续有效，首次未保存宽度时使用300px。EGG旧右栏宽度有只读回退，并改为独立存储，避免继续与M01共用。
- M10声道工作台未启用此次统一外观，专属源码未修改。M17独立调整音标内容与显示，不使用公共三栏改造。
- M17新增125个输入入口，IPA107、extIPA18，总量500→625。含完整组合、示例、范围工具及两条2002旧版入口，不能称为625个独立基本音标。保留VoQS全部65入口、56个指定译名与原表结构，固定字体文件字节未变。具体条目和未确证截图字形见 [来源与差异](../references/m17-expansion-diff.md)。
- 中文解释参考井井指定的吕佳、江荻（2013）PDF，区分该文的2002表与当前2025 extIPA。新增两项出处已进入统一347项来源登记。未把第三方截图、PDF、音频或程序代码复制进产品。

## 最终构建和通用前端

本轮执行 `npm --prefix frontend run typecheck`、`npm --prefix frontend test`、`npm --prefix frontend run ui-data:check`、`npm --prefix frontend run build`。M17最新显示反馈后重新执行类型、完整测试与构建，231项测试全部通过，类型/来源生成检查与生产构建通过。构建有大于500kB的chunk体积提示，无构建错误。

P18源码浏览器矩阵92项、ScientificPlot尺寸/交互5组、未启用统一样式的旧工作台回归8组通过，证据路径及准确覆盖见P18报告。矩阵包括两主题、70%/100%/150%缩放及窄窗口。M13极窄时保留内部横向滚动。ScientificPlot修复了缩放时同步尺寸回写导致的ResizeObserver循环，坐标与科学计算公式未改。

## Windows实际Qt与自然录音

使用既有 `.venv/m09-ui/Scripts/python.exe`，源码 `PYTHONPATH` 指向 `desktop/src`、`backend/src`、`packages/phonetic_core/src`，显示采用当前任务拥有的offscreen Qt宿主，前端读取构建后的 `frontend/dist`。P18源码在其Qt检查后冻结，M17独立追加变更另重建并复验。未操作用户浏览器、已打开的应用或原有服务。

执行：

```powershell
python -B -X utf8 scripts/verify_p18_integration_qt.py --corpus 'C:\Users\13680\Desktop\project\音频数据\测试音频' --audio '00058肉.wav'
```

证据：`output/validation/p18/qt-c3c3eba4801b4dd981e1e3ddcee96b23/report.json`，success=true。

- 15模块 × 1920×1080/1366×768 × 浅/深主题，共60组实际Qt布局通过。检查整页无水平溢出、内容尺寸、默认栏宽和栏底对齐；M03操作栏在左，M11的Beam/Retry beam不展开环境详情也各出现一次。
- 参数估计、参数显示、LPC通过原生目录授权读取真实 `00058肉.wav`，原始44100Hz、单声道、183456帧、4.16秒。波形按实际录音显示，未生成伪造参数或发起科学分析任务。
- 该目录28个顶层原文件前后SHA256相同。音频SHA256：`81af7a4b87d47a640f315e2f7740279250d94765dc443fa72f630e6624e77717`。没有进入其他未授权语料目录。
- 人工查看M01真实录音深色、M03浅色四图、M06深色合成页面截图，栏框、按钮与图形区域符合最终规范。其余布局矩阵和截图保存在相同证据目录。

`scripts/verify_m16_m17_integration.py` 追加4组通过，证据 `output/validation/m16-m17-integration/qt-16c6d4639f1041098984953b7e5a90fa/report.json`。使用合成PortAudio后端验证两模块切换和共享工作台状态，不代表实体录音设备验收。

## M17追加反馈与复验

井井在同轮预览后要求默认仅显示名称与符号，完整解释改为悬浮显示，并增加紧凑列数及清晰滚动。已实现每个分类内自适应多列，1366窗口的IPA附加区4列、extIPA节奏区3列。名称悬浮显示简释，符号悬浮显示含义、用法、区别和来源；悬停介绍默认开启，保留旧草稿明确关闭的偏好。表区原生纵向滚动条可见，底部编辑框固定，长组合保留足够宽度并自然换行。

最终证据：

- `node tests/e2e/m17-ipa-plus.cjs`：17组通过，实际字体CDP检查625入口、全部入口点击、Unicode/组合字符、真实浏览器剪贴板、下载回读、草稿恢复与异常反馈。`output/playwright/m17/2026-10-03T06-20-10-462Z/report.json`。
- `node tests/e2e/m17-expansion.cjs`：36组浅/深主题、1366/1920尺寸和全部分区通过；末项可滚入可见区、无横向裁剪、原生滚动条、名称/符号紧凑列、完整语义悬浮与介绍框可持续阅读。测试专门关闭Playwright默认的隐藏滚动条参数，目视截图确认。`output/playwright/m17-expansion/2026-10-03T06-20-10-458Z/report.json`。
- 最终重新构建后运行 `python -B -X utf8 scripts/verify_m17_qt.py --require-single-screen`：4组检查、18个分区布局通过，success=true。实际桌面字体/资源、中文/附加号/非BMP字母、撤销重做与重新打开、原生UTF-8文本保存回读、全部625入口的分区覆盖、固定编辑框、基础表单屏及补充分区可滚动、指定两组来源均通过。证据 `output/validation/m17/qt-20261003-142313-21fffa/report.json`。
- 根代理另目视该Qt目录的 `1366x768-ipa-marks.png`、`1366x768-extipa-context.png`，确认实际桌面四列/三列、右侧滚动条和名称/符号呈现。详细音标来源、字体、VoQS不变证据见 [M17报告](2026-10-03-m17-expansion-report.md)。

此前 `output/validation/m17/qt-20261003-140819-732f09` 为反馈前中间证据，不作为最新外观；本次修改没有改变音标目录、字体或P18源码，未重复扩大无关模块测试。

## WSL与可复现性

在现有NInfer发行版、Linux x86_64和已有Python3.11.14环境执行：

```text
wsl.exe -d NInfer -- /home/ninfer/ptb-m06-20260927/bin/python -B /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/scripts/verify_m17_catalog.py --cin /mnt/c/Users/13680/Documents/ipa.cin
```

625入口目录与Windows生成内容一致，SHA256：`984981551985cb3fffdcc3aa0e58e668195baa97d90da8f2d99a155958c29cff`。首次默认命令因Linux用户目录无CIN别名文件而生成内容不同，已增加显式 `--cin` 输入路径，按相同只读资料复验通过。未修改HOME、原CIN或WSL环境。该项只证明目录生成与Unicode/来源清单的跨系统一致性，未验证Linux GUI或科学服务。

## 文档检查的既有问题

`python -B -X utf8 scripts/validate_docs.py` 检查1243份文件、347条来源和41项任务，发现4条既有旧EXE路径缺失：README中的20261001 LocalPreview和M10-R5，以及2026-10-01两份P17报告中的成品链接。已核实这些链接在HEAD中原已存在，本轮没有删除对应文件或修改检查标准。其余新文档/JSON/UTF-8/历史快照哈希检查未报告错误，V2等只读历史快照另有原记录中的悬空链接。此检查整体退出1，不能称为全库文档检查全绿。

## 边界与入口

使用 [Start-M16-M17-Workbench.ps1](../../scripts/Start-M16-M17-Workbench.ps1) 启动本轮源码工作台。旧EXE不包含本轮布局及音标更新。

验证范围为Windows开发源码、独立Chrome、最终构建的offscreen Qt和上述自然录音预览。未重验全部科学算法、长时任务、实体DPI/多屏、摄像头/麦克风/EGG硬件、Linux/macOS GUI或生产服务器。Qt offscreen日志有GPU上下文不可用提示，渲染和断言仍完成，不能据此声明硬件GPU正常。

无现存数据库DDL、环境安装、全局设置变更、V2改动、原音频写入、push或公开发布。所有测试进程和临时服务按其自身finally关闭，失败或中间证据保留。

最终聚焦审计证据：`output/validation/p18/final-20261003.json`，保存最终前端assets哈希。根代理独立与HEAD `5f30ae9024744de1ba218607bbc1089eb1fdcf29` 比较确认VoQS入口/图表逐项相同、字体逐字节相同、后端/桌面宿主/科学核心/M10专属源码无差异、无文件删除。最终 `git diff --check` 通过。
