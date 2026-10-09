# M01-E 目录、共同研究页与只读显示验收

2026-09-10，Windows，**verified（限定目录能力、草稿、试听与显示）**。井井在D后回复“继续”，随后追加默认单声道/可双声道、加高波形、Praat语谱图开关、紧凑文件列表/全选及长音频优化。起点为`codex/v3-rebuild`的`639bc6e`，当时工作区干净。完整M01/P08仍in_progress；参数批计算、持久取消/重试、双产物发布和切分保存属于F/G。

本轮范围与选择见[目录计划](../plans/2026-09-10-m01-workspace.md)、[追加显示计划](../plans/2026-09-10-m01-display.md)和ADR-024。源码/产物hash、实际测试路径见[本阶段证据](../modules/evidence/M01-workspace-migration.json)。

## 已实现行为

- 真正的Qt目录选择，输入/关联/输出授权分开；仅列顶层支持文件，取消保留原状态。目录/文件随机能力仅当前窗口可用；拒绝路径传入、重解析点/硬链接，读取时锁定并核对最终句柄路径和身份。重复选同一目录不重复关联，关联目录的WAV不混入输入批列表。输出只有安全目标预检，没有写入现存文件。
- 同一AppShell承载Qt目录与已登录网页项目，完整80参数、10+4设置有草稿/应用/取消。项目/账号隔离、标签切换保留、退出账号清除内存并停止试听。其余公共预览模块也按当前网页owner/project/module隔离；关闭重开M01后播放条重新绑定新状态，两个场景都已在Chrome实测。参数和设置可以保存；语料、目录句柄、账号凭据不写入草稿。全列表、当前试听、勾选切分集合和所选TextGrid层分别显示，分析结果保持真实空态。
- 默认显示试听声道一轨，可勾选两轨；桌面单轨172 CSS px，窄屏150 px。文件行约36–41 px，保留单选试听与独立切分勾选，并支持全选/清空。峰值绘图不会修改音频或影响后续参数计算语义。
- WAV在Web Worker解码并构建256样本块min/max，显示按视口像素聚合，放大保留原始细节与窄尖峰。切换文件撤销旧下载/解码并丢弃旧响应。文件上限64,000,000字节，解码上限32,000,000采样值；超限明确拒绝，未声称无限时长流式播放器。
- TextGrid两端共用C的IntervalTier解析器，原样移到轻量core入口；旧路径重导出。UTF-8/UTF-16、中文/IPA/引号和格式边界继续通过；预览保留合法负时间，试听按实际音频范围裁界；空白/sil/eps等标签不计入候选切分。服务器每次验证owner/hash/截止，解析或计算结束再查截止。
- 语谱图可开关，使用真实Parselmouth 0.4.7 / Praat 6.1.38，按当前试听声道和可见时间窗生成。Gaussian 5 ms、50 dB相对动态范围、6 dB/oct显示预加重、无动态压缩；默认上限5000 Hz并受Nyquist约束。保留Praat实际时频网格，绘图限制约1000时间步/250频率步。显示预算依据[Praat高级设置](https://www.fon.hum.uva.nl/praat/manual/Advanced_spectrogram_settings___.html)，分析和灰度规则分别见[分析手册](https://fon.hum.uva.nl/praat/manual/Sound__To_Spectrogram___.html)、[绘制手册](https://www.fon.hum.uva.nl/praat/manual/Spectrogram__Paint___.html)。这是自有前端绘制的实际Praat分析预览，不是Praat编辑器逐像素复刻或已校准SPL。

“仅保留浊音(ZCR)”已改为基频判定：源码实际使用Praat/REAPER有限正F0的并集；关闭时用能量静音mask。min/max F0影响Praat/REAPER/WM，WM jitter/shimmer实际窗口至少160 ms。这里只修说明，不更改现有公式、14默认值或黄金样例。

## 实际环境与验证

原工程环境`.venv/v3-dev`继续不安装NumPy。新建项目内`.venv/m01-ui`，使用[带hash锁](../../requirements/requirements-m01-ui.lock)，复用原科学版本和现有Qt/HTTP测试版本；安装三份最终wheel后共51包，`uv pip check`通过，core/API/desktop关键导入均来自site-packages。没有改变v2、全局依赖、系统CUDA、数据库schema或CI配置。

| 检查 | 实际结果 |
| --- | --- |
| 最终安装wheel的工程/契约/目录/Praat预览定向检查 | **113 passed**，含真实Praat子进程、超时清理、本地会话、服务器owner与计算完成后到期拒绝 |
| 最终wheel的C TextGrid回归 | **10 passed**；旧/新入口为同一函数，原解析源码仅相对import变化 |
| 前端测试 | **16 passed**，包括200万样本峰值、视口点数、草稿/全列表/关联/范围 |
| 生成/静态检查 | OpenAPI、TS、版本、UI来源、typecheck、build、架构、文档和diff检查通过 |
| Qt真实窗口 | 真QFileDialog选择17文件，单/双轨、真实Praat灰度、80参数、10+4设置、TextGrid两层；正常窗口关闭和所属服务退出码0 |
| 独立Chrome | 实际HTTP认证及共同页面；两个账号、三个项目、草稿隔离、试听终点/停止、显式不关联/刷新、快速切换、全选、浅深主题/390 px无横向溢出 |
| 长音频 | 600秒/16 kHz/单声道合成WAV，19.2 MB；浏览器本次加载约182 ms，波形≤1600峰值条；真实Praat在全窗与300秒放大窗均显示，独立受控计算600×160网格约0.75秒 |

耗时为本机这一次合成测试观察，不是跨设备性能承诺。Chrome账号/存储使用明确内存测试替身；HTTP、身份检查、WAV与Praat计算是真的，此处不冒充PG/配额/持久任务联合验收。短音频与双声道频峰由纯核心测试复核；播放检查验证WebAudio状态和时间，没有人工听觉或设备声压测量。

主要命令（项目根；Python分别使用标明环境）：

```text
uv pip compile --python .venv/m01-io/Scripts/python.exe --generate-hashes --only-binary :all: --output-file requirements-m01-ui.lock requirements-m01-ui.in
uv venv --python .venv/m01-io/Scripts/python.exe .venv/m01-ui
uv pip install --python .venv/m01-ui/Scripts/python.exe --require-hashes --only-binary :all: -r requirements-m01-ui.lock
.venv/m01-ui/Scripts/python.exe -m build --wheel --no-isolation --outdir output/validation/m01/e-wheels packages/phonetic_core
.venv/m01-ui/Scripts/python.exe -m build --wheel --no-isolation --outdir output/validation/m01/e-wheels backend
.venv/m01-ui/Scripts/python.exe -m build --wheel --no-isolation --outdir output/validation/m01/e-wheels desktop
uv pip install --python .venv/m01-ui/Scripts/python.exe --no-deps --reinstall --link-mode copy output/validation/m01/e-wheels/phonetic_core-3.0.0a1-py3-none-any.whl output/validation/m01/e-wheels/ptb_api-3.0.0a1-py3-none-any.whl output/validation/m01/e-wheels/ptb_desktop-3.0.0a1-py3-none-any.whl
.venv/m01-ui/Scripts/python.exe -m pytest -c tests/pytest.ini desktop/tests backend/tests/test_m01_preview.py backend/tests/test_m01_spectrogram.py packages/phonetic_core/tests/test_spectrogram.py tests/contracts backend/tests/test_api.py backend/tests/test_storage_boundary.py tests/architecture -q --junitxml=output/validation/m01/e-wheel-final.xml
.venv/m01-ui/Scripts/python.exe -m pytest -c tests/pytest.ini tests/security/test_m01_formats.py -k textgrid -q --junitxml=output/validation/m01/e-textgrid-wheel.xml
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run build
.venv/m01-ui/Scripts/python.exe scripts/verify_m01_workspace.py
.venv/m01-ui/Scripts/python.exe scripts/verify_m01_browser.py
.venv/v3-dev/Scripts/python.exe scripts/generate_contracts.py --check
.venv/v3-dev/Scripts/python.exe scripts/sync_versions.py --check
npm --prefix frontend run contracts:check
npm --prefix frontend run ui-data:check
.venv/v3-dev/Scripts/python.exe scripts/check_architecture.py
.venv/v3-dev/Scripts/python.exe scripts/validate_docs.py
git diff --check
```

最终Qt证据在`output/validation/m01/workspace-ffdd983001d043fdaae19e593206215f/`；Chrome功能与长音频证据在`output/playwright/m01-e-167cf4121fc14cd5b21b6ac3ed948178/`。两者均由独立测试进程生成，未使用Codex内置浏览器。来源登记仍为323条，更新实际使用位置、Praat显示参考和m01-ui锁的复用记录，未把未决再分发许可标成解决。

## 问题定位与保留边界

1. Windows scandir缓存的inode/link计数不适合文件身份校验，改为实际lstat；重复目录文件身份去重且关联目录不扩大音频集合。TextGrid严格响应模型原先误禁负时间，已按C解析语义修正。
2. 空`QApplication([])`使本轮独立Qt初始化进程以`0xC0000409`退出；传入正常程序名后，初始化探针、实际工作台、正常关闭均通过。该进程不是Codex，没有证据把它与此前Codex内置浏览器关闭问题认定为同一根因。
3. 10分钟语谱图最初触发子进程MemoryError；限定仅该预览进程的BLAS/OMP线程为1后，在保留1 GB Job上限下通过。未放宽内存限制。每API进程一个预览槽，20秒超时，已验证超时回收；预览没有临时磁盘音频或结果文件。Qt调用异步，隐藏/切换时旧结果不会写回页面，已发出的计算最多运行到其限时。
4. 既有Starlette/httpx、AnyIO弃用提示及Qt图片色彩配置提示仍存在；不为清除提示升级依赖。未验收macOS/Linux桌面、多实例持久浏览器profile竞争、完整EXE发行或生产负载；窗口文件能力互不通用由独立provider测试，不能扩大成所有多窗口生命周期都已通过。

保存性复核比较v2的7项状态/环境及427文件检查结果，36个P03/M01-A golden hash；结果见阶段证据。Git根限定D盘v3工作树，只进行已审阅路径的本地提交，无push、用户目录上传、v2修改或数据迁移。此次未观察到Codex闪退；没有修改Codex应用本身，也不能保证它永不退出。

下一项为M01-F：在已有方案上形成具体持久批次/配额/fencing/双产物提交实现和数据库审阅材料；任何实际DDL仍遵守独立、具体的授权门。
