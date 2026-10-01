# M13 P17 实测报告

2026-10-01。逐项规则：[M13-rules.md](M13-rules.md)。

M13全部11转换标准、多音字、排版、PNG/草稿及大屏所列路径通过。

保留专业三栏和上下布局，设置支持收起记忆，普通说明移右；未改逐字映射规则。

`node tests/e2e/p17-m13-full.cjs` 10组通过，`M13-full/1790852653036/report.json`：11确切旧标准输出、逐位置多音选择、全部字号间距/字体开关、1200字无截断、空输入导出禁用、实际离线PNG栅格和Doulos字体调用、字体/编码失败恢复、保护关闭草稿恢复。

`node tests/e2e/p17-m13.cjs` 最后通过 `M13/1790855668622/report.json`：两主题、缩放、布局/显示模式、多音弹窗位置/按钮/外部点击/Esc及滚动。同目录loaded-large-sizes.json使用实际文字，三档工作区增长且字体不变。来源帮助经真实AppShell弹窗另在公共布局报告通过。

20次汉字输入到两帧 min/median/P95/max=12/17/22/25ms，原始值 `interaction-timing.json`。包含输入自动化开销，不代替冷启动或不同机器性能保证。最终Qt有实际银行花输入，未验Qt PNG完整字体设备矩阵及跨平台字体。普通控件已逐项映射证据；所有极值/失败排列组合未穷尽。

## 共同范围与复现

状态保持 **in_progress**。规则逐行 partial 只表示对应已执行路径通过，未全覆盖的输入边界/失败组合不得汇总为所有交互通过。所有证据路径相对 `output/validation/p17/`。

仅 Windows 源码和本机 Qt，原录音/V2只读；输出、临时任务库、组件运行目录均在独占 output。未安装环境、改现存库、改公共平台、commit/push 或构建EXE。本组产品源码已由主代理纳入 ab07736，最终EXE由主代理另验。

最终前端构建隐藏Qt17阶段：`qt-c/3a44621f31624425bfdcad3fe4944e9b/report.json`，WAV加载、M12 TextGrid保存后全部层和区间数回读、原文件哈希、M13实际文字通过。`frontend/dist/index.html` SHA256为 `2f5e7c053a04310ac1d0d81a4913413717ec15caca6a78b27ec57bcfa3646461`。隐藏窗口WA_DontShowOnScreen，inner1440×900；showMaximized标志不代表该隐藏窗口达到屏幕尺寸。此前真实可见最大化17阶段 `qt-c/a2d31781aeee478e8969d0fe22e3dea6/report.json` 为inner1707×996、QScreen1707×1067、DPR1.5，物理约2560×1600，属于中期构建。截图等待两帧与350ms。

最新五页布局/来源对话框/右栏收起重载记忆证据 `layout-c/1790855035363/report.json`。1920×1000、2560×1360、3840×2080、1280×720为模拟CSS viewport，覆盖1/1.5缩放；不能当作真实多显示器测量。大屏有内容证据在各模块段落；字号保持不变。

`node --test frontend/tests/m11.test.ts frontend/tests/m13.test.ts frontend/tests/m14.test.ts frontend/tests/annotation-r6.test.ts frontend/tests/annotation-r5.test.ts frontend/tests/annotation-r3.test.ts frontend/tests/annotation-sequence.test.ts frontend/tests/m15-runner.test.ts`：55通过。

`$env:PYTHONPATH='backend/src;desktop/src;packages/phonetic_core/src'; .venv/m14/Scripts/python.exe -B -m pytest -c tests/pytest.ini tests/parity/test_phonology_induction.py tests/parity/test_mfa.py -q`：28通过、3.59秒。纯状态/数值检查不充作本轮真实音频或物理设备验收。

Qt复现：同上PYTHONPATH下运行 `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/p17_m11_m14_qt.py`；M15运行 `scripts/p17_m15_qt.py`。两者使用现有dist、不构建。原生文件对话框由测试指定独占路径，未覆盖人工选择/取消的全部系统对话框。Linux/macOS/远程、人工听感、物理声卡延迟均未验。授权目录无>120秒录音，未生成替代音频。
