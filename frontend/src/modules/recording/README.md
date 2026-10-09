# 录音 · M16

本目录负责 Windows 桌面录音工程的前端页面与状态，覆盖设备选择与录前检测、自由/任务录音、多声道同步剪辑、可追溯降噪/增益、任务模板、WAV 批量导出和中断恢复。持久模块 ID 为 `M16`，入口见[模块注册表](../../app/registry.ts)，用户操作以[现行手册章节](../../../../manual/chapters/m16.json)为准。旧[独立操作文档](../../../../docs/manual/recording.md)保留历史记录，冲突时以当前源码与最新报告为准。

## 职责与架构

前端维护页面、任务草稿、整数帧选区、显示/试听配置和后台状态，不执行 Python 算法，不接受任意本机路径，不加入业务数据库或服务器任务。`RecordingPort` 经 PyQt6/QWebChannel 接本机服务。Qt 主线程选择目录并给对应 opaque grant。采集回调复制有界块，专有消费者确认落盘；处理子进程生成派生块，受控工作线程导出 WAV 并回读。

| 文件 | 职责 |
| --- | --- |
| [RecordingPage.vue](RecordingPage.vue) | 工程与任务草稿、设备、采集、版本、处理/导出、关闭保护 |
| [RecordingWave.vue](RecordingWave.vue) | 各声道最小/最大包络、Praat 灰度渲染、共同帧选区、图面焦点 |
| [defaults.ts](defaults.ts)、[types.ts](types.ts)、[state.ts](state.ts) | 48 kHz/双音频默认、单输入回退、UI 投影、任务和选区 |
| [task-import.ts](task-import.ts)、[shortcuts.ts](shortcuts.ts) | UTF-8 CSV/TSV、XLSX/JSON映射和预算；上下文限制的试听/编辑键 |
| [录音端口](../../platform/recording.ts)、[采集租约](../../platform/capture-lease.ts) | 本机能力和 M05/M16 采集互斥 |
| [Qt桥](../../../../desktop/src/ptb_desktop/m16_bridge.py) | 主线程目录选择与桥分发 |
| [本机服务](../../../../desktop/src/ptb_desktop/recording/service.py) | 单写者、任务快照、录音/版本、恢复和结果发布 |
| [存储](../../../../desktop/src/ptb_desktop/recording/storage.py)、[采集](../../../../desktop/src/ptb_desktop/recording/capture.py) | 原子清单、f32/SHA256前缀、有界原始采集 |
| [设备](../../../../desktop/src/ptb_desktop/recording/devices.py)、[任务](../../../../desktop/src/ptb_desktop/recording/jobs.py) | 身份/格式、单声道试听、派生处理与WAV回读 |
| [录音核心](../../../../packages/phonetic_core/src/phonetic_core/recording/) | 纯EDL、计量、谱减/线性增益和真实Praat显示 |

## 入口与操作链

[Start-M16-M17-Workbench.ps1](../../../../scripts/Start-M16-M17-Workbench.ps1)使用既有项目环境与构建页面，不自动安装依赖或创建数据库。

1. 在可写空目录新建工程，或打开含 `project.json` 的完整目录。
2. 明确选择输入与独立输出，核对采样率、同声卡通道数和角色。录前检测不保存条目。
3. 停止检测后录音，明确停止并等待内部保存。重录追加，最新成为该任务选用条目。
4. 试听、同步剪辑或显式生成处理版本，比较原始/当前/直接来源减结果差分。
5. 保存最新任务草稿，到工程之外目录导出WAV；完整备份保留整个工程。

确认导入和完成编辑只改页面草稿，保存清单才提交。删除任务直接提交并保留录音历史；开始检测/录音、切换工程和正常关闭尝试保存草稿。波形默认第一/左/唯一道，全部声道开关不改采集、EDL、处理勾选和导出。试听/语谱共用一个通道。重新选择输入设备重设双音频或单音频，须重新核对 EGG 角色。

## 数据协议与输出

[本地协议](../../../../contracts/recording/README.md)独立于公共 OpenAPI、P06/P07与现存库。

- 工程 `ptb-recording/1`、任务模板 `ptb-recording-tasks/1`、导出 `ptb-recording-export/1`。前端 `Project/Take/Version` 是去除音频路径和EDL的投影。
- QWebChannel `recording(request_id, JSON)` / `recordingReady` 返回 `{ok:true,value}` 或 `{ok:false,error}`，目录仅按用途通过 grant 授予。
- 同声卡1–8通道，`microphone`为界面音频（麦克风/线路），另有 `egg/other`。默认48000 Hz/双音频/float32，单输入回退1音频。原始为应用收到的PCM，系统处理记为 `unknown`。
- EDL采用每通道整数帧 `[start,end)`，所有声道同切点。剪贴板为当前工程片段引用，粘贴在起点插入，不替换选区；采样率/通道数/角色顺序须相同，不自动重采样。
- `takes/`原始与`derived/`派生为小端交错 `.f32`，配合帧索引/通道/hash。`project.json`为入口，`manifests/`为修订，`recovery/`为已确认前缀，单写者锁文件保留。不能改后缀当WAV。
- WAV为FLOAT/PCM24/PCM16，保留采样率和全部声道。新建 `录音导出-*` 子目录及 `manifest.json`/UTF-8 BOM CSV；JSON含完整任务、处理及质量记录，CSV有公式式文字防护。
- PCM超量化范围拒绝，FLOAT可保留超±1值。单段最多16000000帧且按声道数限制256 MiB采样负载，连续帧进清单，RF64未开放。失败/取消留成功项，未完成文件 `.wav.partial`。
- 任务选用批量只含现清单启用且未跳过的关联条目，自由/移出任务可用当前或全部范围。全部历史遍历所有版本，忽略原始/当前下拉。

采集与编辑/后台处理互斥。切页可继续采集并显示指示，空格仅试听。关闭等待请求、停止并提交，失败保持窗口和所有权。恢复只认已落盘且路径/尺寸/hash通过的连续前缀，不补零或删历史。普通网页无本地采集和隐式上传回退。

## 方法、来源与许可

| 来源 ID | 用途及登记 |
| --- | --- |
| `PKG-SOUNDDEVICE` / PortAudio | sounddevice0.5.3，MIT；PortAudio原生许可另列。设备查询/格式检查见[官方硬件接口](https://python-sounddevice.readthedocs.io/en/0.5.3/api/checking-hardware.html) |
| `SRC-PRAAT` | 真实Gaussian5 ms/50 dB相对灰度/6 dB每倍频程预加重。已记录Parselmouth0.4.7（GPL-3.0-or-later）和Praat6.1.38（GPL-2.0-or-later）；[Jadoul等2018](https://doi.org/10.1016/j.wocn.2018.07.001)，发行组合义务独立核查 |
| `METHOD-M16-SPECTRAL-SUBTRACTION` | 自有`ptb-spectral-subtraction/1`，周期Hann1024/hop256、平均噪声功率、0.15下限及三频点平滑，调用SciPy；不等价AU/标准Wiener |
| `PKG-NUMPY` / `PKG-SCIPY` | 既有M16环境记录2.2.6/1.16.3，BSD-3-Clause，实际wheel/原生组件另核查 |
| `PKG-SOUNDFILE` / libsndfile | SoundFile0.13.1，BSD-3-Clause；WAV写入/回读，libsndfile许可另核查 |
| `SRC-SHEETJS` | 复用本地[SheetJS CE0.20.3](../perception/vendor/README.md)，Apache-2.0，未联网导入 |
| `SRC-M16-VOICEVISTA` | 只读参考作者已有录音器安全设计，v3独立工程实现，无动态旧路径/数据库依赖，不把自有设计当第三方库许可 |

文件与发行状态详见[来源登记](../../../../third_party/source-registry.json)、[来源增量](../../../../docs/references/m16-source-additions.json)、[录音ADR](../../../../docs/decisions/ADR-M16-local-recording.md)。章节审计记录源码 SHA，论文引用、代码许可和完整发行包审阅分别核对。

显示版本 `m16-praat-display/1`，实时最多128、录后640个中心时间单元，0至min(5000,Nyquist)Hz。长视窗可能漏短事件，需放大；Gaussian显示和Hann降噪参数分开。`linear-gain/1`只处理勾选音频道，EGG/其他/未勾选保持。噪声样本3072帧至60秒，统计提示不自动识别语音。

## 定向验证命令

在仓库根目录使用既有环境。以下为完整回归的复现入口，按实际影响选择并记录结果。

```powershell
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
node --test frontend/tests/m16-recording.test.ts
$recordingPreviousPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = 'packages/phonetic_core/src;desktop/src;scripts'
    .\.venv\m09-ui\Scripts\python.exe -m pytest -o addopts='' desktop/tests/test_m16_r4.py desktop/tests/test_m16_r3.py desktop/tests/test_m16_recording.py desktop/tests/test_m16_job_publication.py desktop/tests/test_m16_host_integration.py packages/phonetic_core/tests/test_recording_core.py packages/phonetic_core/tests/test_recording_display.py packages/phonetic_core/tests/test_recording_praat_display.py packages/phonetic_core/tests/test_spectrogram.py -q
} finally {
    $env:PYTHONPATH = $recordingPreviousPath
}
node tests/e2e/m16-r4.cjs
.\.venv\m09-ui\Scripts\python.exe -X utf8 scripts/verify_m16_r4_qt.py
.\.venv\m09-ui\Scripts\python.exe -X utf8 scripts/manual/validate.py --project manual --strict --distribution software
```

E2E/Qt所需环境和启动服务以[R4报告](../../../../docs/testing/2026-10-04-m16-r4-report.md)及脚本为准，使用隔离合成设备/工程，截图/导出会产生测试产物。全书生成由主代理收口，不自动改 `project.json` 或其他章节。

## 说明书操作取证

[capture_m16_states.py](../../../../scripts/manual/capture_m16_states.py)使用隔离的 Windows Qt 工作台、合成设备和独立工程，记录窗口、原图摘要、导出及恢复结果。它不打开实体麦克风或扬声器，不改用户录音或现存数据库。合成第二道不当作生理 EGG，界面及回读检查不代替自然录音效果、听辨或墙钟连续采集。

报告和必要截图保存在本机 `output/manual-work/m16-captures/`，测试工程、重复 WAV 和可再生长音频在结束后按根规则清理。新素材遵循当前作者规范，不照搬旧截图尺寸。

## 已验范围与限度

[R4报告](../../../../docs/testing/2026-10-04-m16-r4-report.md)记录Windows源码、只读本机枚举、Chrome/实际Qt合成采集/播放、模板、多道图面、处理和WAV/恢复，WSL仅纯核心。[首版报告](../../../../docs/testing/m16-report.md)的受控加速60分钟是落盘/回读证据；[历史成品报告](../../../../docs/testing/2026-10-02-m16-m17-exe-report.md)不能代替后续源码EXE验收。

实体声卡/麦克风/EGG、热插拔、模拟削波/系统AGC、物理时延、墙钟长录音、自然气声/擦音处理听辨、实体DPI、Linux/macOS原生采集及近期EXE全流程须独立验收。dBFS/灰度未校准声压，float32不增加ADC位深。降噪可改变科研指标，应保留原始、方法和输入条件。每次修改仅声明实际重验范围，历史结果不自动扩大到当前源码和成品。
