# M11 P17 实测报告

2026-10-01。逐项规则：[M11-rules.md](M11-rules.md)。

M11真实导入、参数、取消/失败任务、历史和日志所列路径通过；**成功自然录音对齐 blocked**。

## 改动与实际验证

共享三栏：左既有组件/模型，中真实配对与参数，右任务/日志/导出，右栏可收起记忆。没有伪造成功记录。

`$env:P17_LAB='C:\Users\13680\Desktop\project\音频数据\creak\老年组 15人\1.凌静梅\00141低_梯_题.lab'; node tests/e2e/p17-m11.cjs`：参数联动/0值拒绝/草稿、真实配对导入、词典切换、组件门禁、实际任务取消与m11_model_mismatch失败通过。证据 `M11-ui/98f4aa342a03406797f5729fee17323e/browser-report.json`。

`$env:P17_REUSE_M11='D:\PhoneticToolbox\PhoneticToolbox_v3\output\validation\p17\M11-ui\98f4aa342a03406797f5729fee17323e'; node tests/e2e/p17-m11.cjs`：重开本任务自有新库，取消/失败历史选择、非空阶段事件、非空脱敏日志展开、真实语料目录能力、输出目录、上传按钮文件选择、组件信息展开/收起、runtime/model/dictionary资源选择通过，见同目录 `history-report.json` 和 `failed-task-ui.txt`。此复用入口约束在P17/M11-ui下，不打开用户库。

20次Beam编辑到两帧结束 min/median/P95/max=20/24.5/29/34ms，原始值同目录 `interaction-timing.json`。仅前端参数，未宣称MFA实时。一次先前宿主验证校验阶段180秒未结束，测试宿主正常退出并取消自有任务；后续真实失败任务约69秒，不能隐去冷校验慢例。

最终构建隐藏Qt追加：`.venv/m09-ui/Scripts/python.exe -X utf8 scripts/p17_m11_m14_qt.py --m11-controls`，32阶段通过，`qt-c/d851c4dee0e54b15a7cff6cf5037f0fd/report.json`。真实Host/FileProvider入口记录了语料目录/输出目录取消后选中、运行时/模型/词典选中、archive/manifest取消、资源校验信息展开收起。原生文件对话框由测试返回明确路径或空值，未人工操作系统对话框，未运行组件自检或安装。

## 阻断

现有MFA3.3.8可启动，唯一声学模型`Documents/MFA/pretrained_models/acoustic/mandarin_mfa.zip`使用IPA音素；原LAB是dai1/tai2类键，现有`mandarin_pinyin_tab.dict`使用d/ai1/t/ai2。直接实际runner65.612秒返回m11_model_mismatch、峰值526880768字节，原件哈希不变，见 `M11/3661b812863c46a8b819f761548ff256/report.json`。

追加只读枚举4词典和90份配对LAB，现有mandarin_mfa.dict/mandarin_china_mfa.dict均无一份原转写完整覆盖；两个pinyin词典与模型音素冲突。`M11/existing-combinations.json`保留全部核对。V2未预置另一配套模型。未修改转写或下载替代。成功TextGrid、成功结果保存/下载因此未验。

组件自检自身生成合成音频，与本轮禁生成要求冲突，故没有执行检查或安装，只验证资源选择和按钮门禁。在线组件仍待发布禁用。

## 共同范围与复现

状态保持 **in_progress**。规则逐行 partial 只表示对应已执行路径通过，未全覆盖的输入边界/失败组合不得汇总为所有交互通过。所有证据路径相对 `output/validation/p17/`。

仅 Windows 源码和本机 Qt，原录音/V2只读；输出、临时任务库、组件运行目录均在独占 output。未安装环境、改现存库、改公共平台、commit/push 或构建EXE。本组产品源码已由主代理纳入 ab07736，最终EXE由主代理另验。

最终前端构建隐藏Qt17阶段：`qt-c/3a44621f31624425bfdcad3fe4944e9b/report.json`，WAV加载、M12 TextGrid保存后全部层和区间数回读、原文件哈希、M13实际文字通过。`frontend/dist/index.html` SHA256为 `2f5e7c053a04310ac1d0d81a4913413717ec15caca6a78b27ec57bcfa3646461`。隐藏窗口WA_DontShowOnScreen，inner1440×900；showMaximized标志不代表该隐藏窗口达到屏幕尺寸。此前真实可见最大化17阶段 `qt-c/a2d31781aeee478e8969d0fe22e3dea6/report.json` 为inner1707×996、QScreen1707×1067、DPR1.5，物理约2560×1600，属于中期构建。截图等待两帧与350ms。

最新五页布局/来源对话框/右栏收起重载记忆证据 `layout-c/1790855035363/report.json`。1920×1000、2560×1360、3840×2080、1280×720为模拟CSS viewport，覆盖1/1.5缩放；不能当作真实多显示器测量。大屏有内容证据在各模块段落；字号保持不变。

`node --test frontend/tests/m11.test.ts frontend/tests/m13.test.ts frontend/tests/m14.test.ts frontend/tests/annotation-r6.test.ts frontend/tests/annotation-r5.test.ts frontend/tests/annotation-r3.test.ts frontend/tests/annotation-sequence.test.ts frontend/tests/m15-runner.test.ts`：55通过。

`$env:PYTHONPATH='backend/src;desktop/src;packages/phonetic_core/src'; .venv/m14/Scripts/python.exe -B -m pytest -c tests/pytest.ini tests/parity/test_phonology_induction.py tests/parity/test_mfa.py -q`：28通过、3.59秒。纯状态/数值检查不充作本轮真实音频或物理设备验收。

Qt复现：同上PYTHONPATH下运行 `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/p17_m11_m14_qt.py`；M15运行 `scripts/p17_m15_qt.py`。两者使用现有dist、不构建。原生文件对话框由测试指定独占路径，未覆盖人工选择/取消的全部系统对话框。Linux/macOS/远程、人工听感、物理声卡延迟均未验。授权目录无>120秒录音，未生成替代音频。
