# P17 公共显示与本机语谱预览验收

2026-10-01。状态：本页所列 Windows 源码路径已验证；冻结 EXE 的独立结果见本轮总报告。不能把组件计时作为完整模块首次加载计时。

## 修改范围

- `WaveformViewport` 默认采用可见时间窗振幅、正零负刻度和细节连续连线。原始采样点每像素不超过 16 点时逐点连接，更密集时保留峰值包络。原音频、音量和科学输入不变。
- `ScientificPlot` 统一很小振幅和很窄时间窗的刻度格式，避免多个有效刻度都显示为 0。
- 公共语谱图与波形共用横向时间几何及缩放/双击入口。Windows 本机 API 复用有界 Praat 子进程，60 ms 合并视图更新；保留原计算与像素字节。
- `ModuleFrame` 提供可选高度约束。模块共用 10/8/30 px 内边距、区块间距和按钮高度，保留字体与色彩。模块负责布局与可达性，不能用裁切冒充单页。
- M10 不使用 `ModuleFrame`，未改其专属页面、模型、参数或录制。
- 按用户追加要求新增共享三栏 `ModuleWorkbench`，两侧可拖宽、右栏可折叠并记忆，小窗自动移栏。模块顶部上边距4px，项目返回入口并入标签栏。各模块新布局证据单列，早期截图仅代表早期版本。
- 成品首轮截图发现首次未选择目录时，公共桌面适配器发送空目录ID并误报授权失效。统一在`desktopFiles.list`对尚未选择的目录返回空列表；非空但失效的目录ID仍由原权限校验拒绝。增加真实Qt14页首次空态无该误报断言，原输入/科学任务不变。

## 真实输入与复验

来源仅为用户授权音频目录。931 个 WAV 完成只读格式清点，输入私有清单及 SHA-256 位于 `output/validation/p17/natural-inventory.json`。公共测试覆盖 0.318 秒普通语音、10.261 秒普通语音和 14.664 秒双声道 EGG。没有用生成音频替代本轮输入。

| 门 | 实际结果 | 可复核证据 |
| --- | --- | --- |
| 可见窗振幅与原采样点 | 三份真实录音，0.16/0.08/0.04/0.01 秒视窗独立计算峰值并与轴值/连线核对；输入不变 | `shared-waveform/1790852276209/report.json` |
| 大屏波形密度 | 3840视口使用50,000个真实原采样点连续绘制，未被旧1600像素显示上限截断；复验三真实输入及24次缩放，p95 15.81 ms | `shared-waveform/1790855238430/report.json`；该轮替代早期固定1600宽度上限 |
| 公共缩放反馈 | 最终轮24次真实 Ctrl 滚轮，等待状态变化和绘制；p95 15.81 ms，最大16.27 ms | `shared-waveform/1790855238430/report.json`；命令日志 `waveform-command-r2.log` |
| 语谱结果数值/像素 | 三输入、九种视窗/声道对照原单次进程路径，完整 JSON 和像素逐字节一致 | `spectrogram-session.json` |
| 语谱生命周期 | 非法声道失败后重建、受控极短超时、API 关闭、空闲退出、任务拥有的进程回收 | `spectrogram-session.json`、`spectrogram-http.json` |
| 鉴权与接口 | 未授权请求不启动预览进程；本机接口真实结果/缓存策略/关闭回收；服务器未创建复用会话 | `spectrogram-http.json` |
| 完整可见语谱反馈 | 20 次 canvas 上 Ctrl 滚轮到最新真实像素，p95 79.09 ms，最大 79.21 ms，无页面错误 | `shared-spectrogram/1790853257028/report.json` |
| 时间几何与选区 | 波形与语谱左右边界差 1 px（边框）；谱图拖动同步公共选区 | 同上 |
| 视觉 | 独立查看浅色/深色截图，振幅刻度可读、连线连续、两图时间对齐；小窗保存独立截图 | 两个共享证据目录的 PNG |
| 结构与流式边界 | 28 项架构边界、4 项临时写入边界和 10 项 M08 流式结构测试，共 42 项通过 | 下列命令 |
| 三栏与大屏公共布局 | 1920×1000 三栏与顶部4px、无整页溢出、折叠保留已挂载输入、记忆/键盘展开、键盘拖宽、小窗重排、可选侧栏；2560×1360和3840×2080真实SVG绘图区随高度增长且字号不变，共8组通过 | `shared-workbench/1790854845430/report.json`；仅布局数据/模拟视口 |

上述相对证据路径均位于忽略目录 `output/validation/p17/`。计时为本机样本，未声称所有设备固定延迟。语谱首次启动仍有科学运行时加载成本，复用只改善连续更新。

执行入口：`tests/e2e/p17-waveform.cjs`、`tests/e2e/p17-spectrogram.cjs`、`scripts/p17_spectrogram_timing.py`、`scripts/p17_spectrogram_http.py`。纯状态测试使用标量/数组，未作为自然音频结论。

结构检查命令：设置本进程 `PYTHONPATH` 指向当前 checkout 的 backend/src、packages/phonetic_core/src、desktop/src 后，执行 `.venv/m14/Scripts/python.exe -m pytest -o addopts='' tests/architecture backend/tests/test_managed_scratch_chunks.py backend/tests/test_p17_m08_stream.py -q`。初次直接用环境中旧安装 wheel 导致四个旧签名失败；改为正确源码入口后 42 项通过。未安装或覆盖环境内的 wheel。`addopts` 覆盖仅因该环境未安装旧项目配置所需的 pytest-cov，未跳过断言。

Linux 只读检查发现现有 NInfer WSL 无 Python/Node 可执行入口，本轮不安装运行时，未将 Windows 结果扩大为跨平台验证。实际声卡延迟、多显示器混合 DPI 和超过 120 秒真实录音仍缺对应证据。
