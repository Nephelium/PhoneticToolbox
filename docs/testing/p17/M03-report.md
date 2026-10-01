# P17 M03 执行报告

状态：产品源码冻结，限定 Windows，真实EGG会话、普通导出、IF与批次已验证。四档布局已验证。每个控件的实际状态见 [M03-rules.md](M03-rules.md)，未将缺素材/设备项当作通过。

改为两栏：中区2×2四图及总览，右窄栏参数/导出/历史。四图随剩余高度增长。音频与EGG微观图按各自可见峰值留8%边距，保留V2各声道独立归一化定义。修复非法微观0输入渗入显示导致NaN坐标/重复刻度：无效草稿保持，绘图沿用有效快照。77.249 s真实录音首显3.69 s；20次实际更新约78–115 ms。CSV501×5、3张1500×900 PNG、IF双WAV各22050样本、整幅及4张独立PNG均回读。

## 最终布局与证据

| CSS视口 | 中区主要实际图高（px） |
| --- | --- |
| 1920×1000 | 194–210 |
| 2560×1360 | 374–390 |
| 3840×2080 | 734–750 |

表中为四子图实际内部绘图区高度。默认与大屏模块/中区scrollHeight=clientHeight，小窗可滚动。右侧长记录允许内部滚动。大屏是模拟CSS视口，未伪称实体屏设备验收。

- 完整运行：`output/validation/p17/M01-M05-5cb4879d0220451da22cf911ce0043e1/report.json`，exit 0，页面错误/警告均0，包含本页真实科学主流程与产物。
- 引用/草稿/取消重试等普通交互：`output/validation/p17/M01-M05-f3d0eb386cd94cb3bc05ad939264ab78/report.json`，exit 0，错误0，X01–X08可逐行追溯。
- 额外关联/所有层/快照/频谱按钮/手动峰值/批量保存：`M01-M05-be70b5cb449347c4a6ca8cbe695bdb7e/report.json`的X09；该轮包含四页播放/暂停/继续/停止、EGG受控读取失败后重读、延迟字体预检时取消提交，完整exit 0、页面错误0。
- 最终构建隐藏Qt：`output/validation/p17/M01-M05-86b574624cd247ee8f349bfa4610a162/qt-report.json` success=true。首显3.136 s，20次实际更新75.8–157.6 ms。1920×1000 CSS模块无溢出。
- X12最后普通控件：`output/validation/p17/M01-M05-58f99d3ee30746759b579ca0bcd2d093/report.json` exit 0，真实双声道、M02两个原始文本层、EGG总览及指定LP20；M05最终空历史刷新：`output/validation/p17/M01-M05-3e2a15f6e9d3487d8217e3b5516ce1f7/qt-report.json` success=true。
- 每模块方便定位副本：`output/validation/p17/M03/evidence-index.json`及截图。原始录音SHA-256检查均true。公共80参数逐项及音量/空格/范围/进度追加：`output/validation/p17/M01-M05-4595abcd041e427cbf7d3b143a245dbc/report.json` exit 0。

## 实际命令

1. `node tests/e2e/p17-m01-m05.cjs`：完整主流程exit 0；后续`$env:P17_UI_ONLY='1'; node tests/e2e/p17-m01-m05.cjs`补普通控件，不替代首次完整批次。
2. `node --test frontend/tests/p17-egg-display.test.ts frontend/tests/m01-state.test.ts frontend/tests/m02.test.ts frontend/tests/m03.test.ts frontend/tests/lpc-state.test.ts frontend/tests/m05.test.ts`：39 passed，0 failed。M05算术固定样例仅属回归，未冒充本轮真实视频/录制。
3. `.venv/m14/Scripts/python.exe -X utf8 scripts/verify_p17_m01_m05_qt.py`：最终构建exit 0，隐藏窗口，不采集设备。
4. `.venv/m14/Scripts/python.exe -X utf8 scripts/verify_p17_m01_m05_readback.py output/validation/p17/M01-M05-5cb4879d0220451da22cf911ce0043e1`：exit 0。
5. `.venv/m03-compatible/python.exe -X utf8 scripts/verify_p17_lpc_v2_parity.py output/validation/p17/M01-M05-5cb4879d0220451da22cf911ce0043e1`：exit 0，2048数值精确相等。

统一typecheck/build/contracts/ui-data由主代理执行，子任务未另做全局构建。

## 限制

批准目录最长真实录音77.249秒，无超过120秒真实文件。无旧PKL、视频或lip.json，相关项blocked。M05摄像头/麦克风与物理同步由井井明确user-deferred至EXE手验。Linux/远程托管/不同实体显示器未验；本机无网页上传/逐文件下载入口的控件标not_applicable，实际原生保存已验。物理声卡可听性不能用headless WebAudio状态替代。没有修改V2、原录音、现存库、第三方环境、CI；没有自行commit/push/打包EXE。
