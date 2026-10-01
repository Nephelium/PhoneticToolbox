# M15 P17 实测报告

2026-10-01。逐项规则：[M15-rules.md](M15-rules.md)。

M15设计器、四范式真实音频、恢复、保存失败重试和导出所列路径通过。**物理声卡延迟与人工听感未验证。**

三栏左导航/素材入口，中实际编辑，右预检/已有项目会话；正式实验保持专注运行模式。计时科研机制未修改。

实际录音 `test/webedit_f09_0135_ma.wav`，44100Hz mono、0.31786848秒，SHA256 `7257ae2773b5f448617c2464f446c1a29bd3f5c4c94204e057486e2ec5e98221`。各角色使用同一真实录音字节副本，仅验证角色顺序/调度，不验证刺激类别听辨。未生成测试音频；预检提示音是产品本身功能，自动化确认勾选不代表人听见。

## 命令和证据

- `$env:P17_AUDIO='C:\Users\13680\Desktop\project\音频数据\test\webedit_f09_0135_ma.wav'; node tests/e2e/p17-m15.cjs`：最后12组，`M15/1790855469227/report.json`。四范式离线/零ISI/早按键门禁/角色输出，JSON/XLSX/CSV实际回读，文本两帧/重复键，blur重呈现；新增多选问卷、问卷返回设计再进入、序言已有结果导出、受控一次IndexedDB配额错误后UI重试及仅一有效反应、手动暂停/部分JSON/复核试音/提前结束，单刺激删除/清空独占项目、真实持久存储答复。
- `node tests/e2e/p17-m15-design.cjs`：最后5组，`M15-design/1790855687463/report.json`。所有设计标签、序列操作、参数/六开关/按键范围、三问卷类型编辑、项目/序列文件往返和真实素材预览；同目录有真实序列loaded大屏截图与几何，工作区增长，字体14px和表行自然高度保留。
- `node tests/e2e/p17-m15-recovery.cjs`：8组，`M15-recovery/1790855132175/report.json`。缺资源门禁、同名文本/图像、目录XLSX、手动推进键隔离、真实第二标签锁、重载问卷/会话、图像解码/重呈现、导出确认关闭保护。
- 最终构建 `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/p17_m15_qt.py`：隐藏Qt13阶段，`M15-qt/4ecce2030aa143748ad400b1a4f1fd94/report.json`。真实音频、問卷、播放/反应、原生JSON保存与一个completed attempt回读。无现存DB操作，原生保存路径为独占output。

20次已有序列随机种子输入到两帧 min/median/P95/max=12/17/22/22ms，`interaction-timing.json`保留原始值。它不表示物理音频时延或实验反应精度。未验真实设备断开、外设物理计时、全部断电/存储失败/浏览器组合。

## 共同范围与复现

状态保持 **in_progress**。规则逐行 partial 只表示对应已执行路径通过，未全覆盖的输入边界/失败组合不得汇总为所有交互通过。所有证据路径相对 `output/validation/p17/`。

仅 Windows 源码和本机 Qt，原录音/V2只读；输出、临时任务库、组件运行目录均在独占 output。未安装环境、改现存库、改公共平台、commit/push 或构建EXE。本组产品源码已由主代理纳入 ab07736，最终EXE由主代理另验。

最终前端构建隐藏Qt17阶段：`qt-c/3a44621f31624425bfdcad3fe4944e9b/report.json`，WAV加载、M12 TextGrid保存后全部层和区间数回读、原文件哈希、M13实际文字通过。`frontend/dist/index.html` SHA256为 `2f5e7c053a04310ac1d0d81a4913413717ec15caca6a78b27ec57bcfa3646461`。隐藏窗口WA_DontShowOnScreen，inner1440×900；showMaximized标志不代表该隐藏窗口达到屏幕尺寸。此前真实可见最大化17阶段 `qt-c/a2d31781aeee478e8969d0fe22e3dea6/report.json` 为inner1707×996、QScreen1707×1067、DPR1.5，物理约2560×1600，属于中期构建。截图等待两帧与350ms。

最新五页布局/来源对话框/右栏收起重载记忆证据 `layout-c/1790855035363/report.json`。1920×1000、2560×1360、3840×2080、1280×720为模拟CSS viewport，覆盖1/1.5缩放；不能当作真实多显示器测量。大屏有内容证据在各模块段落；字号保持不变。

`node --test frontend/tests/m11.test.ts frontend/tests/m13.test.ts frontend/tests/m14.test.ts frontend/tests/annotation-r6.test.ts frontend/tests/annotation-r5.test.ts frontend/tests/annotation-r3.test.ts frontend/tests/annotation-sequence.test.ts frontend/tests/m15-runner.test.ts`：55通过。

`$env:PYTHONPATH='backend/src;desktop/src;packages/phonetic_core/src'; .venv/m14/Scripts/python.exe -B -m pytest -c tests/pytest.ini tests/parity/test_phonology_induction.py tests/parity/test_mfa.py -q`：28通过、3.59秒。纯状态/数值检查不充作本轮真实音频或物理设备验收。

Qt复现：同上PYTHONPATH下运行 `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/p17_m11_m14_qt.py`；M15运行 `scripts/p17_m15_qt.py`。两者使用现有dist、不构建。原生文件对话框由测试指定独占路径，未覆盖人工选择/取消的全部系统对话框。Linux/macOS/远程、人工听感、物理声卡延迟均未验。授权目录无>120秒录音，未生成替代音频。
