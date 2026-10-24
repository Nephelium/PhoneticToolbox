# 变速变调多参数与批次保存验收

日期：2026-10-21。范围：Windows 当前源码、独立 Chrome、实际 Qt、本机真实任务服务及 Praat。计划见[多参数实施](../plans/2026-10-21-m08-multi-variants.md)。

## 实现与边界

- 单文件与文件夹共用 `VariantSettings.vue` 和参数解析。语速与选定音高列表按所有组合展开，重复值去重，每批最多 256 个任务。音高倍率与 Hz 偏移互斥，未选字段不参与校验或计算。
- 单文件保留编辑 F0 的无声帧，只调整有声帧，再交原合成器处理速度与原始时间范围。非正的目标 F0 拒绝提交。当前视野处理仅调整并校验所选范围，范围外低 F0 不阻塞当前段。文件夹仍使用原 `transform`，拐点合成独立，Praat 默认与 0.01 阈值保持。
- 新任务仍为 m08/1 单配置请求，不迁移现存数据库。`m08_jobs.py` 只增加非默认参数的输出名标签。旧默认合成名、历史记录和外部副本保持。任务列表附原音名称、摘要及配置，重开页面后可按名称和 SHA-256 定位原音，不只凭同名文件关联。
- 全部任务结束后自动打开跨原音的保存选择。取消后从保存最近批次重开，桌面一次选目录、失败项保持勾选。网页支持时在点击保存后先调用保存选择器，单版本为 WAV、多版本为 ZIP；其他浏览器走既有下载。原始输入不进入结果管理操作。
- 每个任务的全部结果均可点击，不再把操作集中在列表尾部的首份结果按钮。跨来源时先处理未保存草稿，异步读取采用序号隔离，后台完成不抢占手动选择。

## 检查与证据

运行环境：Windows、项目 `.venv/m14`、Node 24、独立 Chrome、Qt WebEngine，均从本工作树加载业务源码。仅使用自建 16 kHz、1 秒、PCM16 的 150/220 Hz 纯音，不能据此判断自然语料听感。

| 检查 | 实际入口与结果 |
| --- | --- |
| 前端定向 | `node --test frontend/tests/m08.test.ts frontend/tests/m08-variants.test.ts frontend/tests/m08-export.test.ts`，16 项通过，连同设置持久化回归共 27 项通过；多值/去重/互斥、非法和预算、F0 缺口、原生保存报告、浏览器先选位置/取消/写入失败收尾及 ZIP CRC |
| 科学/任务/保存 | `python -B -m pytest -c tests/pytest.ini backend/tests/test_m08_jobs.py backend/tests/test_m08_wiring.py backend/tests/test_m08_preferences.py backend/tests/test_p17_m08_stream.py desktop/tests/test_m08_r1_exports.py tests/parity/test_pitch_manipulation.py -q`，75 项通过；包括既有 v2 精确数值门、真实子进程、摘要/任务发布/命名/编号/不覆盖和偏好白名单 |
| 独立 Chrome | `node tests/e2e/m08-variants.cjs`，生产页面/M08Port、实际 FileProvider/TaskBridge/LocalService/Praat；单文件四版本、两原音八版本、完成保存提示、取消/重试、外部副本复用、单项导出失败、跨来源草稿确认、刷新历史定位、六个浅深宽窄布局与网页实际 ZIP 下载，实际说明书新版正文/截图读取，共 11 组通过 |
| 实际 Qt | `python -B -X utf8 scripts/verify_m08_variants_qt.py`，4 组通过。QTest 原生多值键盘、原生指针选其他原音任务、八份真实 WAV 和选择目录/取消/重试、浅深布局；保存对话框目标由测试注入，未声称物理点选系统对话框 |
| 构建 | 前端 typecheck、contracts:check、ui-data:check、保留旧资源的 build:workbench；工作台源码需重新载入才能获得当前静态资源 |
| 说明书 | 更新 M08 正文、参数表、组合示例、任务切换和跨来源保存，替换当前引用的批量页截图，保留旧正式素材；21 章、1113 素材 strict 校验零警告/错误，阅读资源重新生成 |

Qt 实际导出的 .8 倍速为 1.25 秒，1.2 倍速为 0.8333125 秒。150 Hz 输入 ±20 Hz 输出中位 F0 约 130/170 Hz，220 Hz 输入约 200/240 Hz，逐文件误差小于 1 Hz。原始两个 WAV 的 SHA-256 保持。

必要报告、截图、配方与回执位于忽略目录 `output/validation/m08-variants/`。最终 Chrome/Qt 记录按本轮运行回执核对，临时音频、数据库、浏览器配置和下载进入 scratch，用后清理。最终 Chrome 为 `chrome-a7bc84ac6b884783ac98bc7a11555a6b`（11 组），Qt 为 `qt-2e190aba6d404ce986ebddbc2a475740`（4 组），两份 cleanup.json 均为 deleted。说明书另以 `node tests/e2e/m08-variants.cjs --manual-only` 复核真实解码与绘制，证据 `chrome-8a56ce5fdfe34885bcc325e2cd844952`，截图已查看，scratch 为 deleted。首张阅读截图只等待网络载入，图片使用异步解码而尚未绘制，验证入口已增加 decode/绘制等待，未修改阅读器。

## 诊断与未验项

- 首轮旧 wiring 单测复制历史任务库，其中已有成功任务但其结果文件不在新缓存，测试按第一行取结果失败。fixture 改为只在全新临时库执行既有 schema 定义，保持全部原断言并补充任务源名称/摘要/配置断言，12 项 wiring 重新通过。
- 首轮 Qt 设置额外缩放叠加系统 DPI，截图 DPR 为 2.25，未作为说明书素材。改为系统原生 DPR 1.5，最大化、页面 100%、收起导航，得到 2560×1600 正式截图。首轮进程内清理因 Qt 子进程尚存保留 scratch，进程退出后原清理入口核对并删除；最终入口在独立子进程退出后清理，回执为 deleted。
- Windows/macOS/网页共用前端、模块适配和任务执行修改已同步源码。Mac 实机/Safari、Linux 实际服务器请求、真实网页账号权限与资源到期、自然语料试听、最终 EXE/Mac 包未新增验收。未公开部署、push 或发布桌面成品。


## 2026-10-22 · 时间参考线与可选保存序号

Windows/macOS/网页继续共用 M08 页面与导出端口，两个桌面平台共用 M08Bridge。参考线方向新增纵向秒，按原始音频时间绘制，保留旧草稿 refs 横线数组，新增 timeRefs，网页账号字段白名单同步；无需数据库迁移。批量保存每次打开都不勾选序号，选中后按所选结果顺序添加 `1_`、`2_` 前缀，仅作用于导出名称。部分失败保持原序号重试；本会话的有序号/无序号副本分别复用，之后明确重命名仍保留各副本序号前缀。历史结果与科学计算参数保持。

本轮实际检查：

- `node --test frontend/tests/m08.test.ts frontend/tests/m08-variants.test.ts frontend/tests/m08-export.test.ts`：17 项通过，含网页实际 ZIP 条目与单 WAV 保存建议名、未勾选/勾选/取消编号、部分重试编号和历史名称不变。
- `.venv/m14` 使用 `workbench_source.bind_sources(Path.cwd())` 后执行 `pytest -c tests/pytest.ini desktop/tests/test_m08_r1_exports.py backend/tests/test_m08_preferences.py -q`：15 项通过。首次直接 pytest 未绑定源码导致收集时找不到模块，按统一入口绑定后通过。覆盖输出目录授权、同名冲突、重试、外部改动拒绝、编号副本管理及设置白名单。
- `node tests/e2e/m08-variants.cjs`：独立 Chrome 13 组通过，真实本机服务/Praat 与生产前端端口。新增横竖线并存、0.5 秒精确坐标、缩放、负时间拒绝、草稿重开/清除、两主题截图；八个真实 WAV 的可选序号、部分失败重试、摘要与原名副本相同、重开默认不选。既有取消/重复保存/跨原音/网页 ZIP 回归通过。
- `python -B -X utf8 scripts/verify_m08_variants_qt.py`：实际 Windows Qt 6 组通过，QTest 输入/点击复选框，横竖线坐标、八个编号 WAV 的字节回读、重开未选和浅深布局。原生目录选择的路径由测试注入，未声称物理点选系统对话框。截图已目视核查。
- 前端 typecheck、contracts:check、ui-data:check 与保留资源的 build:workbench 通过；说明书 strict 校验 21 章/1113 素材，零错误/警告，阅读资源重新生成并完成最终构建。`node tests/e2e/m08-variants.cjs --manual-only` 已在实际阅读器确认纵向秒与默认不编号的新正文，证据 `chrome-c83c1fc73a1d4b87aca0475c4d90ded0`，scratch 为 deleted。`validate_docs.py` 检查 2430 文件/456 来源/178 任务无错误，`check_architecture.py` 无错误，定向 `git diff --check` 通过。

证据目录：`output/validation/m08-variants/chrome-a1e8f27c0070455393c3df1c1ceaf1c9/`、`output/validation/m08-variants/qt-dea19f05cdc94bf2965868751073749f/`。两份 cleanup.json 均为 deleted，临时 WAV/任务库/浏览器配置已清理，报告与截图保留。

未新增验证：Mac 实机/Safari、实际线上账号设置往返、Linux 服务器及最终 EXE/Mac 包。三端已同步源码，未公开部署、打包发布或 push。
