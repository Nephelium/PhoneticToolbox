# M10-R5 定向验证与修复

2026-09-27。状态：**verified，限定 Windows 开发态 M10 原生 worker、Qt WebEngine 页面和文件视频导出**。本次不生成 EXE；历史 R4 验收只适用于当时的 R4 包。R5 历史构建记录仍见 [R5 计划](../plans/2026-09-11-m10-onset.md)，但本轮检查 `dist/m10-recording/PhoneticToolbox-v3-M10-R5.exe` 不存在，因此没有 R5 冻结包运行证据。当前源码及 `frontend/dist/vocal-tract/keyframes.js` 与源资源的 SHA-256 相同。

## 核对范围与实际发现

| 项目 | 当前证据 |
| --- | --- |
| 首次起声、静音后起声、连续有声及短声段 | `sequence_envelope` 的 48 kHz 样本边界回归：首次和静音后均从零渐入，50 ms 声段渐入 25 ms；连续普通姿势中间无重新渐入。持续发声的 `attack_envelope` 跨 960 样本块连续。原始 `audio`、F0、姿势、时长及声源增益的既有路径未改，渐入在试听限幅后应用。未以此声称实际听感通过。 |
| 拖动排序 | 独立 Qt WebEngine 中用真实卡片几何向页面派发 drag/drop 事件，顺序由 `/a/、/i/、静音` 变为 `/i/、静音、/a/`，时段为 `0–0.20、0.20–0.25、0.25–0.45 s`。当前选择仍指向同一静音卡片，201 点手绘 F0 保留。配置文件、`.ptb-vocal.json`、准备好的合成帧和排序后视频时间轴的顺序及时长一致。该脚本验证页面事件处理，未覆盖实体鼠标在不同 DPI 下的拖动手感。 |
| 清空、撤销与重播 | 首次播放后停止重播命中缓存。清空后本机配置为零帧、空 F0，播放/视频导出禁用；旧准备音频 ID 无法回读。撤销恢复原顺序、选择及 F0，再清空后新建帧撤销入口消失。测试窗口关闭后仅其拥有的 M10 worker 退出。 |
| 当前与六视图视频 | R4 回归场景：当前视图 1280×720、0.45 s、14 帧、21,600 个 48 kHz 音频采样；六视图 1920×1080、0.60 s、18 帧、28,800 个采样。FFmpeg/ffprobe 实际解码 VP8/Opus，视频末帧结束时间误差不超过 1 ms，参考音频峰值偏移均为 0 样本，相关系数分别为 0.999687 和 0.999600。R5 排序后另导出 0.45 s 当前视图，14 帧、21,600 样本、0 样本偏移、相关系数 0.999713。解码中帧人工查看确认当前视图与六格、排序后的姿势标签、静音段和游标。导出取消保留已有目标文件。 |

基线发现两个 R5 后过时的验证预期：Python 导出音频回归和独立视频解码脚本仍把无包络波形作为期望值。解码最初相关系数为 0.921，换用播放端同一输出包络后达到上表结果，未放宽 `>0.95` 阈值或时序断言。另由新增失败回归确认真实缺陷：清空配置后原生运行时仍允许按旧 ID 读取先前动作音频。现仅在保存空关键帧时停止本窗口播放、清除本窗口准备结果和指纹，并使旧 ID 失效；恢复后的同序列须重新准备。不修改 VTL、构形插值、F0 算法或公共执行器。

排序后视频的首轮对照在测试脚本先额外执行一次 `animation/prepare` 后相关系数约 0.908。独立检查发现相同序列在同一 VTL 实例连续强制合成两次，波形相关系数约 0.923；移除这次额外生成后，视频与新引擎首次生成的参考波形相关系数为 0.999713。可确认当前缓存重播复用同一准备结果；强制重算的逐样本一致性尚未建立，本轮没有为追求逐样本一致而修改原生发声状态或 F0 语义。

## 命令与证据

- `.venv/m10-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini tests/parity/test_vocal_tract.py tests/parity/test_vocal_process.py tests/parity/test_vocal_recording.py -q`：24 passed。
- `node --experimental-strip-types --test frontend/tests/m10.test.ts`：5 passed。前端源码未改，`frontend/dist/vocal-tract/keyframes.js` 与当前源资源一致，未重做全项目构建。
- `.venv/m10-ui/Scripts/python.exe -X utf8 -c "from pathlib import Path; from scripts.verify_m10_recording_features import main; main(out=Path('output/validation/m10/r5-r4-regression').resolve())"`：通过，含文件往返、静音 F0 拦截、缓存重播、两种视频、取消、1 秒输出及视图静止检查。证据在 `output/validation/m10/r5-r4-regression/result.json`。
- `.venv/m10-ui/Scripts/python.exe -X utf8 scripts/verify_m10_video_decode.py output/validation/m10/r5-r4-regression`：通过；`decode.json`、两张解码 PNG 位于同目录。
- `.venv/m10-ui/Scripts/python.exe -X utf8 scripts/verify_m10_r5.py output/validation/m10/r5-ui-ordered-final`：通过，结果、排序后序列、音轨源数据及视频见该目录。`scripts/verify_m10_video_decode.py output/validation/m10/r5-ui-ordered-final` 的独立解码通过，截图为 `ordered-current-decoded.png`。一次早期脚本失败源于测试误把配置文件的 `version` 字段排除在 JSON 比较之外；一次重播超时源于脚本没有等待构形更新完成，均修正脚本后重跑通过。

所有交互脚本使用公开内置构形和独立随机配置目录。Qt 测试窗口可能短暂打开系统输出流，脚本将监听音量设为 0；未录制麦克风或摄像头，未改变默认音频设备和系统音量。这里的声卡调用、编码文件分析与人工听感是不同证据：本轮没有物理扬声器听感判断、实体鼠标拖动多 DPI 验收或真实窗口录屏。Linux 原生 ABI/设备、macOS、生产网页和旧 R5 EXE 未验证。当前可靠试用入口为项目内 `.venv/m10-ui/Scripts/python.exe -X utf8 scripts/m10_recording_entry.py`，使用本轮已验证的源码及现有 `frontend/dist`；它不是新发行包。
