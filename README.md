# PhoneticToolbox (PyQt6) 

PhoneticToolbox 是一个基于 Python 和 PyQt6 开发的综合性语音分析与合成工具箱。本项目源自经典的 MATLAB 版 VoiceSauce，并在此基础上进行了深度重构与功能扩展，旨在为语音学研究者提供一个跨平台、易用且功能强大的图形化分析工具。

## 核心功能

PhoneticToolbox 不仅仅是一个参数提取工具，它集成了一系列用于语音分析、合成与感知实验的独立模块：

1.  **声学参数估计 (PhoneticToolbox.app)**
    -   **多算法基频检测**：除了集成 Praat 的基频算法外，还独创性地引入了 **REAPER** 算法，专门针对嘎裂声（Creaky Voice）等极低基频信号进行优化。
    -   **全面的声源参数**：支持计算 H1, H2, H4, A1, A2, A3 等谱幅度参数，以及 H1*-H2*, H1*-A1* 等经过 Iseli 校正的声源谱倾斜参数。
    -   **嗓音质量参数**：支持计算 CPP (Cepstral Peak Prominence), HNR (Harmonic-to-Noise Ratio), SHR (Subharmonic-to-Harmonic Ratio) 等嗓音质量参数。
    -   **共振峰与能量**：支持 Praat 算法估计的 F1-F4 及其带宽，以及基于 F0 同步窗口的能量计算。
    -   **可视化与校正**：提供交互式界面，允许用户查看参数曲线，手动修正错误的 F0 轨迹，并重新计算依赖 F0 的所有声学参数。

2.  **独立功能模块**
    本项目包含多个独立的子模块，每个模块都有其独立的功能和文档：
    -   **EGG 信号分析 (`EGG`)**：用于分析电声门图信号，支持双声道归一化、声门闭合/开放时刻检测（CQ, OQ）、以及生理-声学参数的联合分析。
        -   [查看 EGG 模块文档](EGG/README.md)
    -   **语音合成 (`klatt`)**：基于经典的 Klatt 串联/并联共振峰合成器，支持可视化调节 F0、共振峰、带宽及各种声源参数，并预设了多种发声类型（如耳语、气声、嘎裂等）。
        -   [查看 Klatt 合成模块文档](klatt/README.md)
    -   **基频修改 (`changeF0`)**：基于 Parselmouth (Praat) 的 PSOLA/重采样算法，支持对单个或批量音频进行基频曲线的精细调节与重合成。
        -   [查看基频修改模块文档](changeF0/README.md)
    -   **听觉感知实验 (`perception_experiment`)**：一个基于 Web 技术（HTML5/JS）的轻量级感知实验平台，支持 AX 区分、ABX 区分、情感识别等实验范式，支持本地保存实验配置与结果。
        -   [查看感知实验模块文档](perception_experiment/README.md)

## 环境准备与安装

本项目建议在 Windows 环境下使用，并推荐使用 Python 3.10 或 3.11 版本。

### 1. 创建虚拟环境
建议在项目根目录下创建一个独立的虚拟环境，以避免依赖冲突。

```powershell
# 进入项目根目录
cd c:\Users\13680\Desktop\project\【中山大学】\VoiceSauce

# 创建虚拟环境
python -m venv .venv
```

### 2. 激活虚拟环境
在 PowerShell 中激活虚拟环境。如果遇到策略错误，请先调整执行策略。

```powershell
# (可选) 解除脚本执行限制
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1
```

### 3. 安装依赖
安装项目运行所需的所有 Python 库。

```powershell
pip install -r PhoneticToolbox\requirements.txt
```
*注：依赖项包括 `PyQt6`, `numpy`, `scipy`, `matplotlib`, `praat-parselmouth` 等。*

## 运行指南

### 启动主程序 (GUI)
激活虚拟环境后，在项目根目录下运行以下命令启动图形界面：

```powershell
python -m PhoneticToolbox.app
```

**主界面功能简介**：
-   **参数估计**：选择输入 Wav 文件夹和输出 Mat 文件夹，设置参数计算配置。
-   **参数显示**：加载已计算的 `.mat` 文件，可视化声学参数曲线，并支持手动修正 F0。
-   **工具箱**：通过菜单栏或按钮访问 EGG 分析、Klatt 合成、基频修改等子模块。

### 命令行模式 (Batch Processing)
如果你不需要图形界面，或者需要在大规模服务器上运行参数提取，可以使用命令行接口：

```powershell
python -c "from PhoneticToolbox.controllers import run_parameter_estimation_once; \
run_parameter_estimation_once(wav_dir=r'你的Wav文件夹路径', \
                              mat_dir=r'你的输出路径')"
```

## 常见问题 (Troubleshooting)

1.  **ImportError: DLL load failed (PyQt6)**
    -   **现象**：启动时报错 `ImportError: DLL load failed while importing QtWidgets`。
    -   **解决**：
        1.  确保已安装 Microsoft Visual C++ 2015-2022 Redistributable (x64)。
        2.  尝试重装 PyQt6：`pip uninstall PyQt6 PyQt6-Qt6 PyQt6-sip` 然后 `pip install PyQt6`。
        3.  代码中已包含 `os.add_dll_directory` 自动修复逻辑，通常上述步骤可解决。

2.  **Praat 相关功能不可用**
    -   **现象**：无法计算 F0 (Praat) 或共振峰。
    -   **解决**：检查 `praat-parselmouth` 是否安装成功。本项目依赖该库调用 Praat 核心算法。

3.  **Matplotlib 中文乱码**
    -   **现象**：绘图界面中的中文字符显示为方框。
    -   **解决**：本项目已内置字体配置逻辑，会尝试加载系统中的 `SimHei` 或 `Microsoft YaHei`。请确保系统已安装这些中文字体。

## 联系作者

-   **作者**：井立文（知乎 @井韶子）
-   **邮箱**：jinglw3@mail2.sysu.edu.cn
-   **个人主页**：https://www.zhihu.com/people/jingshaozi

如果您在使用过程中发现任何 Bug 或有功能建议，欢迎联系作者。
