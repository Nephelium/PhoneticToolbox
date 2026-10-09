# 语谱图转音频 · M09

本目录实现图片重建、图片涂鸦重建、音频藏信息三种模式。用户操作正文为结构化[使用说明章节](../../../../manual/chapters/m09.json)，阅读器按同一章节生成内容。历史操作摘要见[已有说明](../../../../docs/manual/spectrogram-to-audio.md)，名称及持久模块 ID 见[模块注册表](../../app/registry.ts)。以下根据 2026-10-05 当前源码整理。

## 模式、输入与输出

| 模式 | 输入与处理 | 输出与边界 |
| --- | --- | --- |
| 图片重建 | PNG/JPEG/BMP 灰度图，填写时间、频率、白黑 dB、窗长等标定，选四角或使用整图，Griffin–Lim 估计相位 | PCM16 WAV；幅值图不包含原始相位，不能唯一恢复原录音 |
| 图片涂鸦重建 | 先四点透视校正，再在校正网格绘图，继续使用图片 Griffin–Lim 路径 | PCM16 WAV；标定决定笔迹的时频与幅值含义 |
| 音频藏信息 | WAV/MP3/FLAC 解码，编辑指定声道的 STFT 幅值，保留原始相位直接 ISTFT | FLOAT WAV，原率/原样本数/原声道数，未编辑声道不参与算法修改 |

每个完成任务均含 `reconstructed.wav`、`calibrated.png`、`reconstructed.png`、`reconstruction.ptb.json` 四件套。JSON 记录源 SHA-256、参数、角点、完整笔迹、方法、网格与必要输出增益。当前没有把 JSON 重新导入为可编辑画布的入口。

## 最短流程

1. 选择模式。桌面选择图片目录或打开音频目录；上传宿主使用导入入口后在下拉中选源。图片也可截取屏幕，按左上、右上、右下、左下选四角，按 Enter 确认。
2. 图片核对真实标定，默认 0–1 秒、0–11025 Hz、白 −30 dB/黑 0 dB、10 ms、32 次、44100 Hz、种子 0。音频核对真实采样率与声道，默认编辑左声道，FFT 1024。
3. 图片涂鸦点击应用校正并绘图；音频选择源后自动载入频谱，改声道或 FFT 后点击载入频谱。使用黑白画笔、粗细、不透明度、缩放平移与撤销重做。
4. 点击开始重建、生成涂鸦音频或生成藏信息音频。任务接收后固定输入、参数和笔迹；从重建任务查看进度，完成后对照目标图和真实输出谱并试听。
5. 桌面保存重建结果一次另存四份文件，已有不同内容文件不覆盖；提供下载能力的宿主逐份下载。保存失败可再次保存同一完成任务，无需先重算。

## 控件与科学条件

- 黑色画笔增强、白色画笔减弱覆盖区域。默认粗细 12（原网格单位）、不透明度 100%，可调 1–100、0–100%。白色映射到显示最低幅值，不能当作完全置零；撤销才恢复上一笔。
- 画布显示缩放 100–800%，Ctrl + 滚轮缩放，Shift + 拖动平移，Ctrl + Z / Ctrl + Shift + Z 撤销重做。显示放大不增加实际频谱分辨率。
- 图片最小/最大 dB 是灰度端点，当前无独立二值化阈值。旧图片目标幅值映射 `10^(dB/10) × 10` 保留，不能直接当作标准幅值 dB 或声压级。
- 图片没有单独 FFT 输入。零频率起点时 `n_fft=2*(图高-1)`；非零起点先扩展完整频率网格并低频补零。窗长默认 10 ms，实际采样点换算后限制在 FFT 长度内。
- 音频 FFT 可选 512/1024/2048，Hann 窗长同 FFT，帧移为 FFT/4，75% 重叠，边缘补零。当前界面固定以原峰值以下 60 dB 显示，使用 `20log10` 幅值标度。
- 音频仅替换有效笔迹覆盖单元的幅值，未覆盖单元保留原值；原幅值为零时相位约定为 0。编辑声道峰值超过 1 时整体缩到 0.99 并记录增益，另一声道不参与算法修改。FLOAT 写出精度需与高精度源区分，不保证文件字节无损。
- 目标复频谱未必满足 STFT 一致性，输出重分析可偏离目标。图片对比两图灰度不同，只作结构比较；音频两图灰度相同。图像反演不证明恢复原音，绘图不保证隐写安全或压缩后保真。

## 状态、预算与保存

- 两绘图模式在当前标签中保留各自画布与笔迹。更换源、角点、编辑声道或 FFT 后须重载，成功重载清空对应旧笔迹。当前没有 M09 标定及笔迹的重开草稿恢复入口。
- 未提交笔迹关闭时可返回绘图、取消关闭或放弃笔迹并关闭。已接收任务的完整笔迹冻结到快照，任务重试沿原条件；改参数或新笔迹应重新生成。
- 图片限 16 MB、原图 2500 万像素、重建区 100 万像素、30 秒；像素×迭代次数最多 3200 万，FFT 最多 16384，迭代最多 128。
- 音频限 64 MB、8–96 kHz、单/双声道、30 秒，另受时频网格预算约束。最多 256 笔、总计 8192 点、单笔 4096 点。
- 底部全部、时间起终点、播放进度和音量只影响试听。当前页内结果默认试听左声道；编辑右声道后须导出并在支持选择声道的工具复核，显示两个声道不切换试听声道。
- 任务计算失败与用户目录保存失败分别处理：前者核对源、条件和预算，沿快照重试或改条件生成；后者保留完成任务，检查权限/空间后重选输出目录再保存。

## 源码结构

| 文件 | 职责 |
| --- | --- |
| [Spec2WavPage.vue](Spec2WavPage.vue) | 三模式、标定、源选择、任务历史与四件套保存 |
| [SpectralCanvas.vue](SpectralCanvas.vue) | 笔迹、缩放、平移、撤销重做与预算 |
| [image-header.ts](image-header.ts) | 图片尺寸预检 |
| [reconstruction.py](../../../../packages/phonetic_core/src/phonetic_core/spec2wav/reconstruction.py) | 图片历史幅值映射、Griffin–Lim 和重采样 |
| [editing.py](../../../../packages/phonetic_core/src/phonetic_core/spec2wav/editing.py) | 共用透视、笔迹混合和原始相位音频编辑 |
| [spec2wav_models.py](../../../../backend/src/ptb_api/spec2wav_models.py) | 参数与四产物契约 |
| [spec2wav_child.py](../../../../backend/src/ptb_worker/spec2wav_child.py) | 图片/音频解码、调用核心及四文件编码 |
| [m09_capture.py](../../../../desktop/src/ptb_desktop/m09_capture.py) | 原生屏幕四角选区与确认 |

## 验证与方法来源

从仓库根目录运行[源码启动器](../../../../scripts/Start-M09-Workbench.ps1)。前端检查为 `npm --prefix frontend run typecheck`、`npm --prefix frontend test`、`npm --prefix frontend run build`。定向入口为[功能检查](../../../../scripts/verify_m09_r1.py)、[界面检查](../../../../tests/e2e/m09-r1.cjs)、[Qt 检查](../../../../scripts/verify_m09_r1_qt.py)。

[R1 报告](../../../../docs/testing/2026-10-05-m09-r1-report.md)记载 Windows 源码、Chrome/实际 Qt、本地任务和完整笔迹的既有验证。Linux 预览、远程任务、实体听辨/DPI 和近期 EXE 需要独立验收，章节结构检查不代替科学任务。

Griffin 与 Lim（1984）[Signal estimation from modified short-time Fourier transform](https://doi.org/10.1109/TASSP.1984.1164317)是图片相位估计的论文来源；NumPy/SciPy、OpenCV、SoundFile 是分别登记的依赖。参见[来源映射](../../../../docs/modules/evidence/M09-source-map.md)、[R1 ADR](../../../../docs/decisions/ADR-M09-R1.md)、[统一来源登记](../../../../third_party/source-registry.json)，以及 [SciPy ISTFT](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.signal.istft.html) 和 [OpenCV 几何变换](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html)官方文档。
