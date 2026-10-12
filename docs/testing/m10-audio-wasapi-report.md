# M10 WASAPI 输出兼容

范围：2026-10-10，Windows 当前源码，sounddevice 0.5.3，本机六个 WASAPI 输出端点。没有改变系统设备设置。

UR22C Voice、Music、DAW 端点的默认采样率均为 44100 Hz，M10 直接请求 48000 Hz 时，格式查询返回 `Invalid sample rate / PaErrorCode -9997`。P40UG、Realtek、TYPE-C 默认 48000 Hz，可以直接接受原来的请求。启用 WASAPI 共享模式 `auto_convert=True` 后六个端点均接受 48000 Hz 流。另一个模块可播放与它采用的输出格式有关，不能据此认定声卡硬件损坏。

`LiveAudio._run` 对 WASAPI 使用 `WasapiSettings(exclusive=False, auto_convert=True)`，测试音、一次试听、连续发声与动画播放共用此处。科学引擎、PCM 样本、导出、动作时钟仍为 48000 Hz，不切换到其他设备，也不更改系统默认格式。[sounddevice 官方接口说明](https://python-sounddevice.readthedocs.io/en/0.5.3/api/platform-specific-settings.html)说明共享模式允许系统转换采样率。

命令：`.venv/m14/Scripts/python.exe -B -X utf8 scripts/verify_m10_audio.py`。证据：`output/validation/m10-audio/wasapi-90b270b6dba74fcdb24f0d2edec055df/report.json`。六端点 × test/preview/animation/live 共 24 路径成功，零监听音量，未录制输入或改写用户配置。临时目录按生命周期清理。

`desktop/tests/test_m10_audio_output.py` 覆盖 PCM 原样传入、声道、48 kHz 时钟、WASAPI 设置、非 WASAPI 路径、失败后停止；与录制测试合计 23 项通过，最终与本轮几何综合回归合计 55 项通过。新几何的实际 Qt 播放也完成，详见 [R16 报告](2026-10-10-m10-r16-report.md)。零音量设备实测确认开流、写入与结束，未代替实体耳机听辨。冻结 EXE 与跨平台未验。
