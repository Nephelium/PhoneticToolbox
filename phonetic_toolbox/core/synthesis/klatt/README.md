# Klatt 合成核心模块说明

本目录实现语音合成实验室使用的 Klatt 参数化语音合成核心，属于 **Core Layer**。

## 模块职责

- `tdklatt.py`: 时域 Klatt 合成核心与参数结构 `KlattParam1980`。
- `spectral_filter.py`: 谱滤波组件，用于合成信号的谱形控制。
- `input_parser.py`: 元音序列解析与分段工具。
- `klatt_config.py`: 默认参数、范围与基础常量。
- `smoothing_utils.py`: 参数平滑与过渡工具函数。
- `__init__.py`: 对外导出核心能力。

## 分层约束

- 本目录只放算法与参数定义，不放 Service 命名文件。
- 外部程序调用、临时文件和可执行文件探测等 I/O 逻辑放在 `phonetic_toolbox/services/`。
- GUI 层通过 `speech_synthesis_widget.py` 调用本目录导出的核心接口。
