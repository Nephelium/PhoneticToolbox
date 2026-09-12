# M10 来源与操作映射

2026-09-11。相邻 v2 仅只读。第 15 章 15.1–15.11 与实际源文件分别核对。以下证据限定 Windows，平台状态与 EXE 结果见[验收报告](../../testing/m10-report.md)。前端路径根为 `frontend/public/vocal-tract`，核心为 `packages/phonetic_core/src/phonetic_core/vocal_tract`，进程/设备为 `desktop/src/ptb_desktop/vocal_tract`。

| 功能 | 旧操作及源码 | v3 目标 | 验收 |
| --- | --- | --- | --- |
| F01 观察 | scene.js/viewer.js 的二维、三维、器官/气腔/叠加、头/鼻/齿/标签、缩放平移 | 同源声道模块的 scene.js/viewer.js/anatomy.mjs | verify_m10_qt.py；01/02/05–09 截图，双列/三列实测矩形；闭塞截面边界 |
| F02 构形 | app.js/controls.mjs 的舌头、唇/颌、软腭、侧缘 | app.js 连续器官列与 controls.mjs；原生 engine/geometry_bridge | test_vocal_tract.py 的实际侧通路、手动根、口鼻面积对照；前端两轴/范围测试；Qt 实际 HY 参数落盘 |
| F03 历史 | PoseHistory、局部/整体重置 | controls.mjs、app.js | Qt /i/ → 撤销 → 重做 → 恢复 /a/ 实际参数往返；局部操作保留源码映射 |
| F04 音频 | engine.py/source.py/audio_output.py | 核心算法与桌面设备适配 | native 稳态闭塞/开放对照；Qt/EXE 真实动作及 PortAudio 流（测试音量 0）；禁止设备输出、停止和取消边界 |
| F05 图表 | app.js/monitor.js/monitor.py | 面积/传递/截面及实时展开 | 07/08 实际截面，11 波形/语谱图展开；读取实际 waveform/spectrogram 数组并检验停止；静音截图不冒充非零声谱 |
| F06 关键帧 | trajectory.py/animation.py/profile.py/keyframes.js/pitch.js | 无分页、重命名、F0放大及时间标注 | 03/04/10 截图；整卡载入/跨栏保存/HY JSON 回读/201 点 F0；22 卡滚动与 50 帧存储恢复、旧格式、超播放预算和取消测试 |
| F07 来源/退出 | aboutDialog、旧独立 HTTP 服务 | 公共致谢、client/worker/runtime/profile | 统一来源生成与资源哈希检查；真实两个独立 worker/配置、重启及关闭测试；EXE 非项目目录自测与退出结果见报告 |

R4 新增功能不反写旧 83 功能组基线：

| 新增操作 | 实现 | 验收 |
| --- | --- | --- |
| 关键帧文件/构形库 | document.py、files.py、profile.py、keyframes.js、presets.js | 有效文件往返、无效文件保留、命名构形持久化和声源保持 |
| 同步视频保存 | video.js、video.py、animation.py/runtime.py | 当前/六视图的真实编码与独立解码、时间戳/音频对齐、取消保留目标 |
| 原声缓存/150 Hz/1 秒 | runtime.py、audio_output.py、trajectory.py、app.js | 缓存命中与八类失效、缺省与显式旧值、48,000 样本 |
| 视觉与边界修订 | geometry.mjs、scene.js、pitch.js、controls.mjs、engine.py | 12 组网格、鼻腔三档开度、连续像素稳定及实际咽壁内限位 |

完整命令与限制见 [R4 报告](../../testing/m10-recording-features-report.md)。

发现：旧侧缘鼠标拖动被限制在 [-0.15,0.3]，不能到达 VTL 2.4 舌尖侧缘 <-0.2 的边音面积修正区。JD2 含 tt-alveolar-lateral(a)、tt-alveolar-fricative(a)、tt-alveolar-closure(a)，但旧页面只列八个元音。旧 JD2 automatic_calc=1，TRX/TRY 被自动计算覆盖，不能只增加滑块。旧 queuePose 丢弃有新请求排队时的返回画面，连续拖动可能迟迟不显示。

旧引擎实测：48 kHz，齿龈 lateral 最小口腔面积 0.20 cm²，fricative 0.15 cm²，closure 0.0001 cm²，均鼻口关闭。无振动 800 Pa 稳态闭塞 RMS 约 3.5e-11，已近静音。此处为模型数值，不能推出用户截图参数或人类听感。

VTL 2.4 源码 VocalTract.cpp:4306 起，正侧缘参数在 0.2–0.4 增加面积至 0.15 cm²，负侧缘在 -0.2–-0.4 增加 lateral 面积至 0.20 cm²。2.3 手册负向面积数值与当前源码有差异，以实际 2.4 引擎为准，两个版本明确区分。
