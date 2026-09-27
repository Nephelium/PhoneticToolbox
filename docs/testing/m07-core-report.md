# M07-B 纯核心与运行时对照

2026-09-27，Windows已列科学范围 verified；Linux同环境迁移对照 verified；Windows/Linux等价准入仍未通过。

核心独立文件 `m07_models.py`、`m07_legacy.py`、`m07_grid.py`、`m07_api.py`。数值步骤原样迁移与外围有限值、尺寸、资源检查分开。文件/进程/F0适配位于worker。核心无Qt、HTTP、数据库和固定开发机路径，未导入V2或M08。

- Windows：`pytest -o addopts='' tests/parity/test_phonation_synthesis.py -q`，44项通过。原数组和WAV按精确相等比较，未扩大容差。证据 `output/validation/m07/windows-core.xml`。
- 独立wheel：`uv build packages/phonetic_core --wheel --offline --out-dir <owned-output> --python .venv/m09-ui/Scripts/python.exe`，随后 `uv pip install --offline --no-deps --target <owned-output>/installed ...`。验证import确实来自target，42项既有科学测试通过。候选包路径 `output/validation/m07/wheel-25aca6fac2be46c2889adc5428e8ea4a`；后加的两项输入/控制点拒绝测试在源码44项中通过。未覆盖/安装到用户环境，无发行。
- 首次尝试 pip 时现有环境不含pip，未安装pip。随后使用现有uv及缓存离线构建。失败尝试保留，不算成功证据。

## Linux

复用现有NInfer环境 `/home/ninfer/ptb-m06-20260927/bin/python`，不安装依赖或修改系统。NumPy/SciPy/Parselmouth版本与Windows相同。

1. 直接对Windows基准严格比较：14通过、28失败。失败集中于分析与生成浮点数组，报告 `output/validation/m07/linux-exact.xml`。此前选用的P11环境不含NumPy，收集失败后改用已存在科学环境，未绕过测试。
2. `tests/support/m07_linux_probe.py`在Linux运行原实现双轮，再对迁移核心逐位比较：原双轮一致，迁移与同平台原实现一致。原Windows基准未覆盖。
3. Windows/Linux原实现共69个数组字段不同；控制点、脉冲、边界与本组量化WAV字节相同。最大LPC差约1.36e-7，残差差约2.30e-9，生成波形差约9.91e-12；完整逐字段值见 `output/validation/m07/linux-original/report.json`。

这里将差异定位到平台/科学运行时路径，不推断具体BLAS或指令级根因。少量WAV相同不证明所有音频可无差别跨平台。未接受新的等价容差，未开放Linux正式服务/REAPER/远程计算。后续门见 `docs/specs/m07-runtime-handoff.md`。
