# M07-A 原实现独立基线

2026-09-27，verified（限定当前Windows环境、公开合成输入与已列参数范围）。

`tests/support/m07_baseline.py`只调用继承原 service/core/models，不导入新 phonetic_core。为避开原包顶层无关EGG/GUI导入，测试仅提供包命名空间，未替换任何M07算法。原五文件与相邻V2 SHA256一致，具体值在 `tests/fixtures/m07/v2.json`。

原实现双轮结果：208数组、2,279,208数值完全一致。4输入覆盖8000/16000/11025/22050Hz、不同长度/幅度/谐波、首尾静音、默认及非默认帧长/帧移/LPC阶数/窗/预加重。捕获重采样、原音频F0、裁剪信号、分析边界、LPC系数、残差、残差F0、脉冲、两对齐模式控制点/回插。24生成配置覆盖三类型×双方向×两个幅度开关四组合，每组3步，峰值限制0.83；包含中间残差、浮点波形和逐步/拼接WAV完整字节。6长度覆盖不足一帧、整帧及非整帧尾部。

原错误行为：全静音、常量、极短输入拒绝。v3进一步明确有限值与资源错误，未修改原计算来消除异常。

命令：`.venv/m09-ui/Scripts/python.exe -X utf8 tests/support/m07_baseline.py`，PYTHONPATH指向源码目录。环境Python3.11.14、NumPy2.2.6、SciPy1.16.3、Parselmouth0.4.7。公开基准 `tests/fixtures/m07/v2.npz`/`v2.json`；完整双轮文件 `output/validation/m07/baseline/round1`、`round2`。

原生补充：`scripts/verify_m07_native.py`使用两份自建多谐波输入，原service与新原生port的5类分析数组及六组浮点音频逐位一致。证据 `output/validation/m07/native/report.json`。旧原生捕获只追加隐藏进程窗口参数，科学输入不改；原生二进制固定hash，Python后端未作为回退。

本报告不证明论文方法完全复现、自然语料普遍稳定、知觉等距或跨平台逐位一致。
