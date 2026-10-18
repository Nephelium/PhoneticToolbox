# M12 P17 实测报告

2026-10-01。逐项规则：[M12-rules.md](M12-rules.md)。

M12修复输入框Ctrl/Meta+S保存保护问题，并完成真实录音编辑、下载、关联与大屏路径。完整复合手势/平台验收仍in_progress。

## 2026-10-19：删除、切换保存与历史版本

状态 `verified`，范围为共用源码、Windows Chrome/Qt 原生写盘和公开网页版。原行为有三处原因：重新打开优先读原始同名 TextGrid，切换时关联在保存回执前确定；删除音节分界后重合音素分界仍可见；删除整段时复用了剪切的完整音素限制。现已等待切换保存并重新解析关联，默认恢复当前工作版本，删除同步处理重合分界/跨界音素，Delete、Backspace 与按钮共用操作。音频外边界保留并提示。规则见[历史决定](../../decisions/ADR-M12-edit-history.md)。

本机历史使用既有 IndexedDB values 与 CAS，按账号/稳定项目/文件/音频摘要隔离，保留原始与最近 50 次不同内容。恢复产生新版本；视窗、声道、词表进度随文件恢复。已有自动保存稿首次接续，原始独立留存。写入失败不切换，原生 TextGrid 写盘失败时已写历史仍可恢复。无现存库 schema 变更，无研究文件上传。

实际验证：

- `node --test frontend/tests/annotation*.test.ts`：73 通过，含保留/恢复/隔离/并发拒绝/失败保留/50次上限和删除跨界、重合边界、撤销等 8 项新增检查。
- `$env:PYTHONPATH='backend/src;desktop/src;packages/phonetic_core/src'; .venv/m14/Scripts/python.exe -B -m pytest -c tests/pytest.ini desktop/tests/test_m12_annotation.py -q`：15 通过，含扫描跨会话身份稳定与不同目录隔离。
- `node tests/e2e/m12-overlap-endpoints.cjs`：18 组实际 Vue/Chrome 事件，覆盖前一轮首尾/四参考模式、切换保存、未按 Enter、写失败阻止切换、刷新恢复、历史恢复和 Delete。合成文件仅内存，见 `output/validation/m12-overlap-endpoints/report.json`。
- `.venv/m14/Scripts/python.exe -B -X utf8 scripts/verify_m12_history_qt.py`：实际 Windows Qt/QWebChannel/本机写盘 4 组通过，原 TextGrid/WAV 摘要不变。结果 `output/validation/m12/history-qt-e04d2771cd384335930b62795f792c35/report.json`，浅深截图已查看。测试指定自有目录，未覆盖人工系统文件选择；前几轮失败日志保留用于诊断，最终成功独立记录。6 份自有 scratch 已在进程退出后清理并保留回执。
- 前端 typecheck、contracts:check、ui-data:check、build:workbench 均通过，保留运行窗口旧资源。`scripts/validate_docs.py` 与 `scripts/check_architecture.py` 零错误，定向差异 whitespace 检查通过。

井井明确授权同步公开网页后，定向控制构建与线上 50 个编译块归一化一致。候选业务差异限 TextGrid 本轮主题/参考/首尾/删除/保存历史，以及共享文件端口的稳定历史项目标识；其余块仅编译引用摘要变化。15 文件、790673 压缩字节原子上线 `m12-edit-history-43a5390df3c044e098e43088738c3050`，保留旧静态资源及服务器回退。2067 发布文件、765 科学运行文件及 15 项真实 HTTPS 摘要核对通过，后端、配置、首页、文献目录、服务进程不变，无服务重启。候选与上线后各独立账号 5 组交互通过，包含真实账号/IndexedDB 导入、不按 Enter 切换、刷新新工作区恢复、历史恢复、Delete/下载/再次切换与零上传/零页面错误。两个成功身份及首轮脚本漏点击扫描的失败身份均已停用。证据 `output/validation/m12-web-20261019/{publication-scope.json,deployment.json,public-readback.json,candidate/report.json,live/report.json}`。

公开验收结束后，6497份自有临时文件已按路径归属/联接/活动进程核对清除，仅删除本轮 scratch。保留最小发布增量、源码冻结证据、报告、浅深截图、清理回执和服务器回退。该目录含硬链接，逻辑字节不代表实际释放磁盘量。

当前用户源码窗口此前已关闭，已由统一入口 `scripts/Start-Research-Workbench.ps1 -Module M12` 重新启动并置前，实际窗口确认历史入口与参考新规则均已加载。Windows/macOS/网页共用同一源码；最终 EXE/Mac 包未重打，Mac 实机/Safari、自然语料全手势与真实输入法组合仍单列未验。

## 2026-10-19：参考时间交集与首尾双击

本轮源码/独立Chrome范围 `verified`。按井井明确要求取消参考同名层整体起止时间的精确相等限制，改按四种复用模式的所选范围与双方层域交集替换，跨界区间截取，参考未覆盖处保留当前内容。点层保留原闭区间端点语义，时间位置不平移/拉伸，全部层成功后一次写入/记录撤销。无交集/类型错误不部分替换。规则见[专题决定](../../decisions/ADR-M12-reference-overlap.md)。

波形、语谱图、标注层的音频首尾双击增加 6 CSS 像素命中区域，仅实际可见文件端点吸附。普通与词表模式处理 WAV 和 TextGrid 已允许的最多1微秒域舍入差异，终点使用当前 TextGrid 原始值。

- `node --test frontend/tests/annotation-*.test.ts`：65项通过，新增15项覆盖四种模式独立预期、长短域、局部层、点层、失败原子性/撤销、内部视窗不吸附及三种终点舍入情况；原相同域 V2 对照继续通过。
- `node tests/e2e/m12-overlap-endpoints.cjs`：独立 Chrome 12组通过。真实 AppShell/AnnotationPage、32001帧/16kHz合成WAV（时长2.0000625秒）与2.000063秒TextGrid，普通/词表模式 × 波形/语谱/标注层共6组双击首尾附近1.5像素并保存精确0..2.000063秒区间；四种模式导入较短参考、较长参考裁剪、无交集提示通过。`output/validation/m12-overlap-endpoints/report.json` 保存范围与配方，浅深截图供核对。
- `npm --prefix frontend run typecheck`、`contracts:check`、`ui-data:check`、`build:workbench`：通过，保留活动窗口旧哈希资源。`scripts/validate_docs.py` 与 `scripts/check_architecture.py` 零错误。浅深截图已查看，无交集提示正常且当前标注保留。

合成音频/标注和保存结果仅在内存，浏览器与Vite在finally关闭。未重验Qt原生目录/本机写盘链、最终EXE、Mac或公网；当前用户源码窗口需重开载入本轮构建。

## 2026-10-18：文件状态主题色

本轮文件标记样式范围 `verified`。原 `has-grid` 状态与同名/候选判断仍有效，但共享 `buttons.css` 的玻璃、普通及选中按钮规则以 `!important` 覆盖文件按钮文字与边框，导致旧绿色状态消失。修复限于 `AnnotationPage.vue`：图例改用主题色，文件名跟随主题，左侧标记改为独立伪元素，选中行沿用公共可读文字；实心全部着色时采用主题对比色，标记边缘增加底色描边。

Windows 实际命令与结果：

- `node --test frontend/tests/annotation-*.test.ts`：50项通过。
- `npm --prefix frontend run typecheck`、`contracts:check`、`ui-data:check`、`build:workbench`：通过。工作台构建保留旧哈希资源，未清空活动窗口引用。
- `node output/validation/m12-theme-marker/verify.cjs`：独立 Chrome、真实 AppShell/AnnotationPage、内存合成 1秒/16kHz WAV 与两层 TextGrid；同名、有/无标注、自动保存及多候选状态正确，实际解码加载选中、悬停和筛选通过。Codex/Catppuccin/Nord × 浅深 × auto/plain/all × solid/glass × 背景开关共72组，通过检查状态色条、文字与背景区分，浅深截图已查看。报告及脚本在 `output/validation/m12-theme-marker/`。

浏览器与 Vite 已在 finally 关闭，音频和 TextGrid 只存在于内存，无磁盘研究素材副本。此检查使用合成文件能力，未重验 Qt 原生目录链、最终 EXE、macOS 或公开网站。共用源码和本地前端构建已更新，用户已打开的源码窗口需重启载入。

## 改动

保存命令先于editable保护处理，文本本身剪贴/撤销保留；删去顶部仅方法按钮占用的空行，方法和普通提示移右，错误/加载保持可见。专业三栏、可记忆右设置折叠、波形/语谱/标注高度随视口增长。

## 实测

`node tests/e2e/p17-m12.cjs` 18组通过，`M12/987cd529ceaf4bf8a30d1fba85c2ac8d/report.json`。输入是授权 `已标注音频/女-刘佳欣-已标注/音频 1-26.wav` 与原TextGrid，10.2609秒、44100Hz mono；所有编辑均在独占副本，另两个无标注/多候选场景也只复制这份真实WAV字节。

覆盖：中文IPA编辑/输入框Ctrl+S/独立回读，筛选扫描、音量、词典和原LAB按钮选择、顺序音节选项、全部6种语谱窗长、非法视窗/前后窗、四参考模式逐一撤销、覆盖取消、全部/单项替换与搜索前后，TextGrid真实下载；新建单层/双层，波形及语谱双击端点、Ctrl+C/X/V及撤销、Backspace、1ms右移、波形边界拖动、Esc取消、Ctrl/Shift/普通滚轮、负trim真实强度贴合。Space/P调度真实44100Hz选区并停止，未声称人工听到或物理时间正确。无TextGrid显式双层保存、多个关联候选明确选B、当前TextGrid切A及原件哈希通过。

一次加载257ms；20次真实视窗切换含输入和两帧等待 min/median/P95/max=50/59/76/83ms。未做冷/热各20次加载。

有内容大屏 `large-layout.json` 及loaded-1920/2560/3840.png：波形宽1098/1738/3018、高150/204/312。1920×1000 root920/920、中心908/939、右906/1206。保存、视窗、音量、波形/语谱/标注/文本、强度和词表默认可见；搜索末尾及右高级参考/唇形仍需区内滚动。1280×720末尾参考按钮可达。

浏览器预览实际打开语料文件按钮和多文件输入加载真实WAV/TextGrid；编辑中文IPA后点击保存TextGrid，下载名为音频 1-26_自动保存.TextGrid，落入独占验证目录，独立核心解析确认编辑内容及层数，原录音/标注哈希不变。第17–18组及同目录browser-save-readback.json记录此链；这是production previewFiles/portableAnnotation的本地浏览器导入与下载，不代表远端上传或服务器账号链。

未全验：服务器独立账号/文件入口、Meta/IME物理输入、Ctrl接续/整体拖动/Shift框选完整组合、自动定时保存及写失败矩阵、正trim和所有唇形操作。授权目录无真实配套唇形数据，不制造设备数据替代。

## 共同范围与复现

状态保持 **in_progress**。规则逐行 partial 只表示对应已执行路径通过，未全覆盖的输入边界/失败组合不得汇总为所有交互通过。所有证据路径相对 `output/validation/p17/`。

仅 Windows 源码和本机 Qt，原录音/V2只读；输出、临时任务库、组件运行目录均在独占 output。未安装环境、改现存库、改公共平台、commit/push 或构建EXE。本组产品源码已由主代理纳入 ab07736，最终EXE由主代理另验。

最终前端构建隐藏Qt17阶段：`qt-c/3a44621f31624425bfdcad3fe4944e9b/report.json`，WAV加载、M12 TextGrid保存后全部层和区间数回读、原文件哈希、M13实际文字通过。`frontend/dist/index.html` SHA256为 `2f5e7c053a04310ac1d0d81a4913413717ec15caca6a78b27ec57bcfa3646461`。隐藏窗口WA_DontShowOnScreen，inner1440×900；showMaximized标志不代表该隐藏窗口达到屏幕尺寸。此前真实可见最大化17阶段 `qt-c/a2d31781aeee478e8969d0fe22e3dea6/report.json` 为inner1707×996、QScreen1707×1067、DPR1.5，物理约2560×1600，属于中期构建。截图等待两帧与350ms。

最新五页布局/来源对话框/右栏收起重载记忆证据 `layout-c/1790855035363/report.json`。1920×1000、2560×1360、3840×2080、1280×720为模拟CSS viewport，覆盖1/1.5缩放；不能当作真实多显示器测量。大屏有内容证据在各模块段落；字号保持不变。

`node --test frontend/tests/m11.test.ts frontend/tests/m13.test.ts frontend/tests/m14.test.ts frontend/tests/annotation-r6.test.ts frontend/tests/annotation-r5.test.ts frontend/tests/annotation-r3.test.ts frontend/tests/annotation-sequence.test.ts frontend/tests/m15-runner.test.ts`：55通过。

`$env:PYTHONPATH='backend/src;desktop/src;packages/phonetic_core/src'; .venv/m14/Scripts/python.exe -B -m pytest -c tests/pytest.ini tests/parity/test_phonology_induction.py tests/parity/test_mfa.py -q`：28通过、3.59秒。纯状态/数值检查不充作本轮真实音频或物理设备验收。

Qt复现：同上PYTHONPATH下运行 `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/p17_m11_m14_qt.py`；M15运行 `scripts/p17_m15_qt.py`。两者使用现有dist、不构建。原生文件对话框由测试指定独占路径，未覆盖人工选择/取消的全部系统对话框。Linux/macOS/远程、人工听感、物理声卡延迟均未验。授权目录无>120秒录音，未生成替代音频。
