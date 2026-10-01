# M14 P17 实测报告

2026-10-01。逐项规则：[M14-rules.md](M14-rules.md)。

M14真实桌面适配、归并、取消、正式任务和三文件导出所列路径通过。

三栏左输入策略/步骤、中归并和真实表格、右生成/保存结果；没有增造任务历史。

`node tests/e2e/p17-m14.cjs` 最后10组通过 `M14/51f809dcdb6e4dacb2c92351ff5df7d1/browser-report.json`：真实导入按钮文件选择5格式、帮助/来源、跳过行展开/收起、调值拖动/映射/取消确认、声韵Ctrl/Shift多选和拖动/链式归并/取消、实际生成取消后恢复、2DOCX+XLSX真实保存/下载、同名拒绝、坏/空/缺失输入恢复、两导入策略/空韵归并、草稿存储失败与关闭恢复。新库均为独占output，不修改现存库。

同目录loaded-large-sizes.json为真实16条表格的三档大屏布局，审阅区域随窗口增长；表行使用自然高度，字体14px。20次导入策略开关到两帧 min/median/P95/max=17/28/42/64ms；仅前端控件，不代表重新计算或DOCX任务耗时。

最终Qt仅空页导航/布局，完整任务和导出在Windows Chrome经生产FileProvider/TaskBridge/LocalService实测，传输替代QWebChannel。V2方法/文档产物对照另有下列28项定向测试。跨平台和全部异常输入/取消时序组合未覆盖。

## 共同范围与复现

状态保持 **in_progress**。规则逐行 partial 只表示对应已执行路径通过，未全覆盖的输入边界/失败组合不得汇总为所有交互通过。所有证据路径相对 `output/validation/p17/`。

仅 Windows 源码和本机 Qt，原录音/V2只读；输出、临时任务库、组件运行目录均在独占 output。未安装环境、改现存库、改公共平台、commit/push 或构建EXE。本组产品源码已由主代理纳入 ab07736，最终EXE由主代理另验。

最终前端构建隐藏Qt17阶段：`qt-c/3a44621f31624425bfdcad3fe4944e9b/report.json`，WAV加载、M12 TextGrid保存后全部层和区间数回读、原文件哈希、M13实际文字通过。`frontend/dist/index.html` SHA256为 `2f5e7c053a04310ac1d0d81a4913413717ec15caca6a78b27ec57bcfa3646461`。隐藏窗口WA_DontShowOnScreen，inner1440×900；showMaximized标志不代表该隐藏窗口达到屏幕尺寸。此前真实可见最大化17阶段 `qt-c/a2d31781aeee478e8969d0fe22e3dea6/report.json` 为inner1707×996、QScreen1707×1067、DPR1.5，物理约2560×1600，属于中期构建。截图等待两帧与350ms。

最新五页布局/来源对话框/右栏收起重载记忆证据 `layout-c/1790855035363/report.json`。1920×1000、2560×1360、3840×2080、1280×720为模拟CSS viewport，覆盖1/1.5缩放；不能当作真实多显示器测量。大屏有内容证据在各模块段落；字号保持不变。

`node --test frontend/tests/m11.test.ts frontend/tests/m13.test.ts frontend/tests/m14.test.ts frontend/tests/annotation-r6.test.ts frontend/tests/annotation-r5.test.ts frontend/tests/annotation-r3.test.ts frontend/tests/annotation-sequence.test.ts frontend/tests/m15-runner.test.ts`：55通过。

`$env:PYTHONPATH='backend/src;desktop/src;packages/phonetic_core/src'; .venv/m14/Scripts/python.exe -B -m pytest -c tests/pytest.ini tests/parity/test_phonology_induction.py tests/parity/test_mfa.py -q`：28通过、3.59秒。纯状态/数值检查不充作本轮真实音频或物理设备验收。

Qt复现：同上PYTHONPATH下运行 `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/p17_m11_m14_qt.py`；M15运行 `scripts/p17_m15_qt.py`。两者使用现有dist、不构建。原生文件对话框由测试指定独占路径，未覆盖人工选择/取消的全部系统对话框。Linux/macOS/远程、人工听感、物理声卡延迟均未验。授权目录无>120秒录音，未生成替代音频。
