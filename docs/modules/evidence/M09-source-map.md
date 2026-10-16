# M09 说明书、源码与新入口映射

2026-09-11。旧来源为 v2 说明书 9.1/9.2、继承的 `gui/widgets/spec2wav_widget.py`、`services/spec2wav_service.py`、`models/spec2wav_models.py` 及 `core/spec2wav/{image_processing,griffin_lim,common}.py`。旧说明书窗长与源码默认不一致，以当前源码 10 ms 为迁移基线并明确说明。

| 功能组 | 原行为 | v3 入口 | 正常与边界证据 |
| --- | --- | --- | --- |
| M09-F01 | 图片导入、图内四点、透视校正 | Spec2WavPage / FileProvider / spec2wav_child | Qt/Chrome 图片载入，Chrome 四点操作，对照 OpenCV 的实际校正像素；拒绝交叉/退化角点、损坏图片与超限输入 |
| M09-F02 | 时间、频率、dB、窗长、迭代、输出率 | Spec2WavConfig / 页面标定区 | Pydantic 边界和非法范围，固定三组种子/尺寸/采样率精确对照；非零频带另测 |
| M09-F03 | Griffin–Lim、原图/生成图 | phonetic_core.spec2wav / 有界子进程 / 持久任务 | v2 数组与 WAV 字节精确对照，本机与实际 PG/Chrome 任务，设置变化显示旧快照提示 |
| M09-F04 | 播放、停止、保存 WAV、帮助 | AudioTransport / TaskBridge.save / 受控下载 | 实际 WAV/PNG/JSON 导出及回读，已有文件保护、哈希、服务/页面重开、取消/重试、跨账号拒绝 |
| M09-L01 | 桌面截图 | 用户点击并确认后，QScreen 捕获到一次性内存能力 | 实现保留，无后台截屏；不把本次合成文件输入测试冒充多屏截图设备验收 |

移植：NumPy FFT/STFT、Hann/OLA、幅度映射、相对 dB 比较图、线性重采样保留原计算。移除全局随机状态、GUI/文件路径/打印用户数据的耦合。原生解码/编码在适配子进程，科学核心只接数组。

单列修复：v2 的 freq_start 未参与频率映射。v3 非零起点采用等间隔原频带插值到 0–Nyquist 网格，低频补零，结果标记 `band-interpolation-zero-fill-v1`。0 起点路径保持旧数值。没有新增数据库 schema，复用 P06 jobs 和 P07 受控资源，未写入只允许 M01 操作的 batch 表。

限制：本次已验证 Windows 文件输入/校正/重建及 Windows 托管网页，不代表 macOS/Linux 运行或安装包发行；多屏与不同 DPI 的屏幕捕获另列设备待测。详见 `docs/testing/m02-m09-report.md`。

2026-10-14 可选声码器：按用户试听选择接入 BigVGAN base 22.05 kHz，来源 PKG-BIGVGAN-BASE。旧 Griffin–Lim 保留为默认，图片任务可显式追加 Mel 再合成；原始相位音频模式不使用模型。源码、CPU 组件、服务器固定下载及独立 Chrome 证据见[专项验收](../../testing/m09-neural-vocoder-evaluation.md)，Qt 与最终 EXE 尚未验证。本轮六组补充例音仅作为本机试听，不进入组件、官网或 GitHub。

内置预测：基础GL与完整修复共用 `m09_parameters` → `phonetic_core.spec2wav.parameters`，冻结v5数组与出处位于core/spec2wav/data。填写标定后显式点击按钮预测，实际重建按当前可编辑值计算；应用源码完成GL并用源码适配器调用组件权重。`m09_bridge`/`m09_runtime`负责已有Python选择、身份校验与GPU兼容性检查。`adaptive.comparison_image`按输入频带与像素尺寸生成结果图，默认输出率为频率终点×2；`SpectralCanvas`/`editing.pixel_brush`按校正图1–200px直径绘制。数值、Chrome及真实CPU/GPU/Qt证据见[m09-neural-pipeline](../../testing/m09-neural-pipeline.md)。
