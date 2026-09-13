# M04 原 V2 合成基准

仅确定性公开合成输入，不包含用户录音。`arrays.npz` 包含12场景输入、返回谱线及5种WAV转换，`result.json`记录配置/错误/标签/PNG像素哈希，`manifest.json`记录原V2源码哈希及两轮对照。

生产核心不能动态导入相邻V2。仅测试脚本 `scripts/capture_m04_baseline.py` 在原V2独立进程中读取原代码。用项目 `.venv/m09-ui/Scripts/python.exe` 执行该脚本可重复捕获，当前基准已冻结，不要重复使用 `--freeze-public` 覆盖它。

两轮对照要求数组dtype、shape与数值字节一致，包含非有限输入。频率解析公式的一ULP检查与逐字节重复性检查分开，不能将数学舍入检查的界限套用到未来迁移对照。

完整范围见 [M04-A报告](../../../docs/testing/m04-baseline-report.md)。
