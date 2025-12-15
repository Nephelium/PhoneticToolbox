# 唇形特征提取工具使用说明

## 1. 功能简介
本模块 (`lip_feature_analysis`) 旨在利用计算机视觉技术，通过摄像头实时捕捉并分析用户的唇部运动特征。工具基于 MediaPipe Face Mesh 模型，能够高精度地提取唇部轮廓，并计算多种唇形量化指标（如张开度、宽度、圆度等）。此外，工具支持同步录制音频，方便进行音视频联合分析。

## 2. 环境准备
在使用本工具前，请确保已安装 Python 环境（建议 Python 3.8+），并安装以下依赖库：

```bash
pip install opencv-python mediapipe pyaudio matplotlib numpy
```

**注意**：`pyaudio` 在某些 Windows 环境下可能需要通过 whl 文件安装或使用 `pipwin install pyaudio`。

## 3. 启动方式

### 3.1 直接运行
在项目根目录下，通过命令行运行主程序：

```bash
python lip_feature_analysis/lip_feature_analysis.py
```

### 3.2 加载历史数据
如果需要查看之前录制的数据并重新生成图表，可以使用 `--load` 参数：

```bash
python lip_feature_analysis/lip_feature_analysis.py --load
```
运行后会弹出一个对话框，请选择包含 `audio_recording.pkl` 的数据文件夹。

## 4. 操作指南

程序启动后，会打开一个名为 "MediaPipe Lip Tracking" 的视频窗口，显示摄像头的实时画面及面部网格。

### 快捷键控制
- **`r` 键 (Record)**：开始/停止录制。
    - **开始录制**：按下 `r` 键后，屏幕下方状态栏会变为红色 "RECORDING"，系统开始同步记录唇部特征数据和音频。每次开始录制都会自动创建一个新的数据文件夹（格式为 `lip_tracking_data_YYYYMMDD_序号`）。
    - **停止录制**：再次按下 `r` 键，录制结束。系统会自动保存音频文件 (`.wav`) 和特征数据 (`.pkl`)，并生成可视化图表。
- **`q` 键 (Quit)**：退出程序。
    - 如果正在录制中按下 `q`，系统会先停止录制并保存当前数据，然后关闭程序。

## 5. 输出结果说明

每次录制结束后，数据会保存在自动生成的文件夹中（例如 `lip_tracking_data_20251215_1`），包含以下文件：

1.  **`audio_recording.wav`**：录制的原始音频文件。
2.  **`audio_recording.pkl`**：核心数据文件（Pickle格式），包含所有帧的时间戳、原始坐标、归一化特征数据等。
3.  **`lip_metrics_plot.png`**：归一化后的唇形特征变化曲线图。
4.  **`raw_metrics_plot.png`**：原始像素级唇形特征变化曲线图。
5.  **`audio_recording_timestamps.pkl`**：音频数据块的时间戳记录，用于精确的音视频对齐。

## 6. 特征指标详解

工具会自动计算以下关键唇形指标，并进行归一化处理（除圆度外，通常通过除以面部宽度或高度来消除拍摄距离的影响）：

| 指标名称 (Metric) | 说明 | 计算方式 |
| :--- | :--- | :--- |
| **Lip Area Ratio** | 唇部面积占比 | (外唇面积 - 内唇面积) / 面部多边形面积 |
| **Normalized Lip Height** | 归一化唇高 | 唇部高度 (像素) / 面部高度 (像素) |
| **Normalized Outer Lip Width** | 归一化外唇宽 | 外唇宽度 (像素) / 面部宽度 (像素) |
| **Normalized Lip Openness** | 归一化张开度 | 上下唇内缘垂直距离 / 面部高度 |
| **Normalized Total Width** | 归一化总宽度 | (外唇宽 + 内唇宽) / 面部宽度 |
| **Lip Circularity** | 唇部圆度 | $4 \pi \times \text{Area} / \text{Perimeter}^2$ (接近1表示圆形) |

### 数据结构 (`audio_recording.pkl`)
对于开发者，可以通过 `pickle` 库读取 `.pkl` 文件，获取详细的帧级数据。数据字典主要包含：
- `relative_times`: 每一帧对应的相对时间（秒）。
- `landmarks`: 每一帧的 468 个面部关键点坐标。
- `height`, `outer_width`, `open` 等：上述各项指标随时间变化的序列数组。

---
**提示**：
- 请确保在光线充足的环境下使用，以保证 MediaPipe 面部捕捉的稳定性。
- 录制时尽量保持头部相对稳定，虽然算法包含归一化处理，但剧烈晃动仍可能影响数据质量。
