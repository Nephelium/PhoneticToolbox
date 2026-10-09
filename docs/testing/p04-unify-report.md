# P04-UNIFY 公共布局与工具栏验证

2026-09-26。**Windows Chrome 的本轮布局、状态和定向交互范围 verified。** 不扩大为 Linux 服务、物理系统 DPI、所有桌面宿主或全部科学模块通过。全局主题保持 U2，原有未提交成果保留。

## 实现与边界

新增 ModuleFrame、ModuleToolbar、ModuleSection、ModuleStatus，接口及逐页操作迁移见 [公共接口](../design/module-layout-contract.md)。M01/M03/M04/M09/M12 已逐项移除模块大标题、副说明和重复关闭，名称与关闭仍由工作台标签承担。文件/刷新/批量进入统一工具栏，方法/帮助固定在工具组，M03/M04 草稿保存放参数区。M09 的相位缺失警示属于科研限制，继续显示。

M12 保留 R6 图窗第一、图窗内文件列表和 TextGrid 保存、下方强度/词典/词表/搜索、层级及唇偏保存。没有修改波形时间、轴、选择、拖动、剪贴或声学算法。仅为页面缩放增加按实际内容宽度折行的 container query，解决 1280×800/150% 下原三列溢出。

M13 模块 agent 交付真实页面与 Chrome 证据后，由本 chat 串行注册 AppShell，接入转换草稿 dirty/save、保存失败、取消/放弃及无音频条。模块目录未由本 chat 修改。M13 字表按需载入，避免主壳携带约 3 MB 本地字表。M08 缺生产宿主端口，未挂测试/空 port 页面。共享后端、科学核心、数据库、政策、系统配置、凭据、生成契约、总台账、全局 ADR 均不属于本次写入。

## 精确 changed_files

以下是本 chat 的修改，不能把整个 git status 归为本任务。M08/M13 模块文件和平台 agent 改动另行交付。

```text
frontend/src/app/AppShell.vue
frontend/src/design/tokens.css
frontend/src/components/ModuleFrame.vue
frontend/src/components/ModuleToolbar.vue
frontend/src/components/ModuleSection.vue
frontend/src/components/ModuleStatus.vue
frontend/src/modules/parameter-estimation/ParameterEstimationPage.vue
frontend/src/modules/egg-analysis/EggAnalysisPage.vue
frontend/src/modules/lpc-spectrum/LpcSpectrumPage.vue
frontend/src/modules/spectrogram-to-audio/Spec2WavPage.vue
frontend/src/modules/annotation/AnnotationPage.vue
frontend/tests/module-layout.test.ts
frontend/tests/p04-preview.html
tests/e2e/p04-unify.cjs
tests/e2e/p04-preview.cjs
tests/e2e/p04-registration.cjs
tests/e2e/m12-r5-flow.cjs
docs/design/module-layout-contract.md
docs/testing/p04-unify-report.md
```

初始分支 `codex/v3-rebuild`，初始 AppShell/tokens/WaveformViewport/M12 等已有未提交内容。修改前源码快照、SHA-256 和比对结果位于 `output/validation/p04-unify/baseline-files.json`、`preservation.json`。五个页面剔除新增组件 import 后的完整 script 与初始源码一致，tokens 的初始内容完整作为当前文件前缀保留。AppShell 仅在当前内容上增加 M13 注册及关闭语义。未 reset、checkout、stash 或整体覆盖已有成果。WaveformViewport、AnnotationTracks、editor、sequence 和科学核心未由本 chat 修改。

## 实际命令与结果

PowerShell，项目根目录，Node/npm 使用既有本机安装，Python 桥接为 `.venv/m09-ui/Scripts/python.exe`。浏览器为既有 Windows Chrome，独立临时 profile、headless，不控制用户浏览器。

| 命令 | 结果 |
| --- | --- |
| `node --test frontend/tests/module-layout.test.ts` | 4 passed。实际编译并渲染 Vue SFC，检查内容保留、可访问名称、原生按钮顺序、加载/错误角色、文本转义 |
| `npm --prefix frontend run typecheck` | vue-tsc 通过 |
| `npm --prefix frontend test` | 最后一次 137 passed，0 failed/skipped。含并行模块已落盘测试；早期本任务为 123 passed，不把他人新增测试归为本任务实现 |
| `npm --prefix frontend run build` | Vite 通过；M13 独立字表 chunk 仍有 >500 KB 警告，未放宽阈值 |
| `npm --prefix frontend run contracts:check` | OpenAPI 快照与生成 TypeScript 一致，未重新生成 |
| `npm --prefix frontend run ui-data:check` | 80 参数、333 来源，一致，未改生成数据 |
| `node tests/e2e/p04-unify.cjs --before` | 修改前 5 页与 M12 实际载入截图 |
| `node tests/e2e/p04-unify.cjs` | 5 页 ×12 空态组合，M12 载入/加载/错误 ×12；另含异步保存失败、关闭取消/重试、唇偏文件回读 |
| `node tests/e2e/p04-preview.cjs` | M01/M03/M04/M09 真实浏览器 File 输入，加载/载入/错误各12组合，共144组；M01 参数跨标签保留，读失败恢复 |
| `node tests/e2e/p04-registration.cjs` | M13 实际工作台按需加载、dirty、保存/放弃/取消、受控存储失败和恢复，12组布局截图 |
| `node tests/e2e/m03-draft-closeout.cjs` | 9组通过，原参数、LP 草稿与真实任务快照、存储失败、重开 |
| `node tests/e2e/m03-playback.cjs` | 4组通过，原音频/IF角色、连续播放及旧请求失败归属 |
| `node tests/e2e/m03-function-review.cjs` | 5组通过，EGG总览/批次取消，以及 M01 实 WAV、M02合成参数图默认选区手势 |
| `node tests/e2e/m03-result-feedback.cjs` | 10组通过，真实结果/下载回读、错误恢复、迟到读写归属 |
| `node tests/e2e/m04.cjs` | 24组通过，实际科学子进程/文件保存下载/历史/取消/迟到归属/参数保存与选区 |
| `node tests/e2e/m12-r6.cjs` | 6组通过，真实 Ctrl+X/C/V、Backspace、撤销、原生文字输入、布局及最终主题色 |
| `node tests/e2e/m12-r5.cjs` | 8组通过，三图选择/拖动/双击、窗长、150%坐标、原始 TextGrid 优先。测试前置状态修正见下 |
| `node tests/e2e/m02-png.cjs` | 3组通过，实际整幅PNG、共享字体/图表回归 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/check_architecture.py` | errors=[] |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/validate_docs.py` | 全局检查未全绿，两条已知 M10-R5 EXE 缺链；最后快照另有并行 M08 报告尚未落盘的临时缺链，见下 |
| `git -c core.safecrlf=false diff --check -- <本任务文件>` | 通过 |

不存在 npm `test:e2e` 命令，本报告没有使用或宣称通过该入口。真实桥接使用独立测试状态/旧空库 schema 的隔离副本，无 DDL。未创建 EXE、commit 或 push。

## 图像与真实交互证据

5 页 × 2 窗口（1280×800、1920×1080）× 2 主题 × 3 页面缩放（70/100/150%）× 4 状态=**240 组**。M13 另12组实际工作台组合。页面缩放通过项目 `setPageScale`，无缩小系统 DPI 冒充页面缩放。截图等待主题过渡结束，没有把暂停时钟下的过渡灰色认作主题色。

| 证据 | 路径（相对项目根） |
| --- | --- |
| 修改前五页及真实 M12 | `output/validation/p04-unify/before-1790431903463/` |
| 五页空态/M12载入与错误矩阵 | `output/validation/p04-unify/after-1790433643278/` |
| 其余四页实文件与受控错误矩阵 | `output/validation/p04-unify/preview-1790433653437/` |
| M13实际壳及存储错误 | `output/validation/p04-unify/registration-1790433663597/` |
| M03关闭/播放/公共选择/结果 | `output/validation/m03-ui/chrome-a0e06c5bcf1c426090d5487259e50235/`、`chrome-ede6e517f9ee4a9d8fc9465f3b154307/`、`chrome-16550ec873584211a749f77025942e6b/`、`chrome-256faaf13fcf4c1796c4dbdf5c102414/` |
| M04真实任务 | `output/validation/m04-ui/8d07db57da3248b7acfbc2fd367fabd2/` |
| M12 R6 / R5 | `output/validation/m12-ui/ddddc28fbfbb47dba7eb345edbfdb336/`、`3eedae28a108444890027d684f5ce028/` |
| M02 PNG | `output/validation/m02-png/chrome-1790432849815/` |

前后同场景示例：

![M12 修改前，1280×800，浅色100%](../../output/validation/p04-unify/before-1790431903463/M12-loaded-light-1280-100.png)

M12 修改后，1280×800，浅色100%（历史本地产物，当前工作区不存在；原路径 `../../output/validation/p04-unify/after-1790433643278/M12-loaded-light-1280-100.png`）

![M12 保存失败保留编辑，深色150%](../../output/validation/p04-unify/after-1790433643278/M12-close-save-failure-dark-150.png)

模块错误采用受控延迟/读取失败/浏览器 Storage 失败，不是自然磁盘故障、自然到期或生产宕机。M01/M09 布局矩阵使用真实本地文件读取，明确没有计算 provider，不声称完整声学任务链路通过。M03/M04 独立既有 E2E 才包括实际科学子进程与结果。

## 失败调查与未测范围

1. 首次 1280×800/150% 的 M12 宽度为623 CSS px，原三列 scrollWidth826，已通过 container query 修复并复验。失败证据 `after-1790432109728` 保留。
2. R5 旧测试两次撤销后恢复了待定起点，却立刻开始下一独立双击组。新布局的波形/Canvas像素取整使重复起点略晚于原值，产生极短空标注。使用只读 Vite load override 加载修改前源码，旧布局通过，说明旧脚本依赖像素取整的偶然分支。测试现先断言待定起点恢复，再按既有 Escape 清除，保留全部数值/保存断言；产品 editor/sequence 未改。失败 `549c75bc4d98484b9d6a8e66d0d238df`、诊断 `b0c39b78c27c4760aeb2c1b1f737d918`、旧源码对照 `87ab04d3e5e741d0b9c69a1bdf975701` 均保留。
3. 执行旧 `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m01_browser.py` 失败，保留 `output/playwright/m01-e-33564e0b45d24fdca2773a2c2a9dc4ac/` 与 `m01-e-4621e252f7fb4b9cb5f24197348780bd/`。脚本用首项假定00号带标注音频，实测首项是600秒16号文件；还依赖旧未展开标签列表、旧侧栏主题选择及旧参数显示样例入口。尝试显式展开后确认排序前提错误，未继续扩大修复旧全流程，本 chat 临时一行测试改动已撤回。该入口不计通过。M01 本轮覆盖来自新的实文件状态/草稿矩阵与既有公共手势入口，不涵盖完整托管账号与批处理。
4. 既有 M03 Vite 扫描脚本提示 vendor/OrbitControls.js 的 `three` 未解析，目标真实页面和断言通过，没有安装依赖或压制警告。M13 字表 chunk 大于500KB的构建警告保留，已按需拆包。
5. 文档全局校验仍报 `README.md` 与 `docs/plans/2026-09-11-m10-onset.md` 指向缺失的 M10-R5 EXE；历史快照缺链另列。最后一次检查（722文件）另发现并行 M08-source-map 指向尚未落盘的 m08-report.md，已发给模块负责人；不修改其文件。未重新生成 EXE 或改别人的历史证据。

| 平台/状态轴 | 本轮边界 |
| --- | --- |
| Windows | Chrome 实浏览器、前端编译/测试和报告中真实本机桥接 verified；252组截图含M13 |
| Linux / WSL 服务 | 本 chat 未运行，不借用平台 agent 的服务测试扩展 UI 结论 |
| 桌面宿主 Qt | 五个既有页面本轮未重跑Qt；M13由模块 agent 单独验证并在其报告归属，不把Chrome当Qt |
| 系统 DPI/多屏/声卡 | 浏览器 deviceScaleFactor=1；未变更或实测物理系统DPI、跨屏及实际听觉输出 |
| 小服务器资源 / 远程计算 | 未测，不改变能力准入或预算 |
| 发布 | 未 push、部署、生成EXE或公开发布 |

共享注册后续只接受完成的实际模块与生产 adapter。M08/M13 内部剩余项、平台服务、总台账和全局 ADR 由各负责人/统筹汇总。
