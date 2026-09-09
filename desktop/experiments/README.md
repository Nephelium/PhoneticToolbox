# PhoneticToolbox P01 技术原型

本目录仅验证宿主、原始音频时间轴和 Windows 单文件包装。它不是完整 v3，现有科研算法尚未迁移。最终结果与限制见 [P01 报告](../../docs/testing/p01-host-probe-report.md)。

## 井井试用

双击仓库 `output/p01-probe/PhoneticToolbox-P01.exe`。首次启动会解包 WebEngine，因此启动时间包含解包成本。

1. 点击“载入双轨测试音”，或“打开 WAV”只读载入自己的文件。
2. 在波形上拖动选区，或直接输入起止采样点；终点不包含在选区中。
3. 试听、暂停、继续、停止；检查波形和听到的片段是否一致。默认音量 18%。
4. 切换浅深色、缩放到选区、改变窗口大小；选区保持。
5. 检查 IPA 行。该原型使用随包 Doulos SIL 7.000，而不是要求系统安装字体。

P01 仅接受 8–192 kHz、最多 8 声道、128 MiB 以内的 PCM16/24/32 或 float32 RIFF/WAV；压缩 WAV、WAVE_FORMAT_EXTENSIBLE 和其他输入格式仍待正式模块实现。当前不保存编辑、不写入所选音频。

## 复现

PowerShell 中从仓库根目录运行。下面所有 Python 路径都属于本项目，不能替换成 v2 的 conda 环境。

```powershell
# 首次建立独立 Python；不安装全局命令、不写注册表。
uv python install 3.11.14 --install-dir .venv/runtimes --no-bin --no-registry
uv venv --python .venv/runtimes/cpython-3.11.14-windows-x86_64-none/python.exe .venv/p01-standalone
uv pip install --python .venv/p01-standalone/Scripts/python.exe -r desktop/experiments/requirements-pyqt6.lock

# 生成有标注的合成音和原样字体/图标，再构建页面。
& .venv/p01-standalone/Scripts/python.exe -X utf8 desktop/experiments/prepare_assets.py
npm.cmd --prefix frontend/experiments/audio-viewport ci --ignore-scripts --no-audit --no-fund
npm.cmd --prefix frontend/experiments/audio-viewport run test
npm.cmd --prefix frontend/experiments/audio-viewport run typecheck
npm.cmd --prefix frontend/experiments/audio-viewport run build
& .venv/p01-standalone/Scripts/python.exe -X utf8 -m pytest -c desktop/experiments/pytest.ini desktop/experiments/tests -q

# 源码启动或自动验收（自测会短暂播放测试音）。
& .venv/p01-standalone/Scripts/python.exe -X utf8 desktop/experiments/host_probe.py
& .venv/p01-standalone/Scripts/python.exe -X utf8 desktop/experiments/verify_runtime.py --output output/validation/p01/reproduce-source --loopback

# 构建器限制本次进程 PATH，并审计原生 DLL 来源。
& .venv/p01-standalone/Scripts/python.exe -X utf8 desktop/experiments/build_probe.py
& .venv/p01-standalone/Scripts/python.exe -X utf8 desktop/experiments/verify_runtime.py --exe output/p01-probe/PhoneticToolbox-P01.exe --output output/validation/p01/reproduce-onefile --loopback
& .venv/p01-standalone/Scripts/python.exe -X utf8 desktop/experiments/verify_runtime.py --exe output/p01-probe/PhoneticToolbox-P01.exe --output output/validation/p01/reproduce-dual --instances 2
& .venv/p01-standalone/Scripts/python.exe -X utf8 desktop/experiments/audit_native.py
```

`verify_runtime.py --loopback` 只短暂读取默认播放端点的 WASAPI 数字回环以识别测试信号，不录麦克风、不保存混音原始音频。它不测量扬声器/耳机声学延迟。蓝牙、不同声卡和实际录音设备仍有各自的平台测试。

`prepare_assets.py` 从 v3 已继承的字体字节提取 OFL，生成 WAV；这些生成文件由脚本复现。主注册表与 [依赖清单](../../third_party/p01-dependency-inventory.json) 记录来源。原型没有捆绑个人语料或 VTL/REAPER 业务模块。
