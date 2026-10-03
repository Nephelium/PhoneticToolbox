# ADR-M03-R5：音频 F0 搜索范围与 REAPER

2026-10-04，accepted，依据井井本轮明确要求：EGG 增加 REAPER F0，REAPER / Praat 均采用 30–800 Hz。

## 实现决定

- 工作台新请求 `f0_policy=audio-f0/2`，保留原 Praat AC 方法和 10 ms 步长，调整搜索范围。REAPER 采用已登记哈希的原生可执行文件，10 ms、30–800 Hz、Hilbert 关闭、原生高通开启。沿用现有适配器的 16 kHz PCM16 输入转换。两者取 EGG 文件中的音频声道，GCI F0 仍取 EGG 事件。
- 复用原生适配器，未新增第三方运行时或 Python 替代后端。原生不可用或失败时给出错误，不生成替代算法结果。
- REAPER 使用原生 EST 时间，Praat 使用 `pitch.xs()`。单文件 CSV 保留各自时间网格，批次继续按既有 GCI 网格插值并应用原 CSV 静音掩码。原生无声 NaN 保留，短于 100 ms / 全零音频的 REAPER 输出为空，不运行原生进程。
- 动态右纵轴继续由勾选的有效 F0 决定。30–800 Hz 是音频 F0 的搜索参数，不是固定图轴，也不裁掉 GCI 离群点。

## 生命周期和兼容

正式任务通过既有额度接口预留 4,000,000 字节临时 WAV，科学子进程仅写入该指定资源，完成/失败/取消后由现有任务适配器释放。实时预览仅在已准入本机进程中使用父级拥有的有界临时资源，子进程组退出后清理。原生进程预算 1 GB / 25 s，仍受外层 EGG 进程组与 RPC 预算限制。该接线不开放未验证 Linux/服务器实时能力。

预览第一次选中 REAPER 时计算并缓存，关闭开关仅隐藏，选区变化从缓存裁剪；重新预处理/切换声道会失效。API 模型新增字段有默认值，旧任务无 `f0_policy` 时保持 `legacy/1` 和原 Praat 75–600 Hz，旧请求幂等哈希不包含新增默认字段。重新选择源文件使用新范围，历史产物保持不变。

结果 JSON 的 `f0_analysis` 分别登记 F0 修订、参数、原生二进制哈希和时间轴。`method_version=egg-legacy/1` 继续描述未改动的 EGG 检测/CQ/逆滤波方法，不能据此忽略新的 F0 修订。

## 依据

- [REAPER 官方仓库](https://github.com/google/REAPER)：16-bit 输入要求、可选 Hilbert、原生高通和 F0/epoch 说明。当前随带二进制的上游构建版本仍未知，本次不把仓库 HEAD 当作二进制版本。
- [Praat 原始自相关文档](https://www.fon.hum.uva.nl/praat/manual/Sound__To_Pitch__raw_autocorrelation____.html)。本轮保留 AC 方法，修改范围由用户明确指定。
- [验证结果](../testing/2026-10-04-m03-r5-report.md)。
