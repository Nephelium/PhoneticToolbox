# M12 P17 实测报告

2026-10-01。逐项规则：[M12-rules.md](M12-rules.md)。

M12修复输入框Ctrl/Meta+S保存保护问题，并完成真实录音编辑、下载、关联与大屏路径。完整复合手势/平台验收仍in_progress。

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
