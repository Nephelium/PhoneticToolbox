# M03-E3-B 有界长文件验收

2026-09-12。verified：限定 Windows 开发态最长 120 秒且 576 万帧的长文件路径。完整 E3/E/M03 仍 in_progress；下一项字体预检/剩余交互与来源收口，冻结 EXE 仍待。

## 行为与依据

输入同时受 64 MB、120 秒、576 万帧限制，8–96 kHz；逆滤波仍需显式不超过 1 秒、48000 帧的选区。完整文件先按原 V2 顺序独立声道峰值归一化、去趋势、滤波、检测，局部查看和导出遵守既有重复局部滤波规则。末尾微观中心现在可超过 60 秒。总览视窗仍限 60 秒，可平移导航到录音末尾，这与完整计算时长区分。

实测后将每个 EGG 子进程预算设为 3,000,000,000 bytes、240 秒，任务截止时间 600 秒；worker 并行度、64 MB 结果、配额、取消和原子发布机制保持。长 CQ/SQ/F0 序列上限为 120000 点，依据检测 1ms 最小峰距；超过 3000 点时以相同半径复合 SVG 路径呈现全部点，减少界面节点，不抽稀科研数值。核心源码/科学 wheel 未改。

## 测量与原版对照

- `scripts/probe_m03_long.py` 在独立科学进程中测试候选时长，测试脚本临时覆盖输入上限，不作为产品入口。120 秒/48kHz、渐增振幅谐波 PCM16，原全段 CSV+三 PNG 耗时 134.23 秒，峰值 Windows private commit 2,412,630,016 bytes，输出 2,226,016 bytes。证据 output/validation/m03-long/probe-120-single/report.json。
- `--chunked` 仅实验 Agg path chunksize=10000，峰值 2,157,641,728 bytes、141.64 秒，CSV 相同但波形 PNG 不同；未采用此绘图修改。该实验没有证明加速。证据 probe-120-chunked/report.json。
- `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m03_long_parity.py`：原 V2 环境只读双轮与现安装 MKL 核心比较；120 秒48kHz双声道锯齿合成，后半幅度加倍。20 组数组、23,242,942 个值的 float64 字节哈希精确一致，涵盖全文件时间/原始/滤波/音频/事件/GCI F0/Praat旧帧轨，以及完整、首部、末尾 CQ/SQ。Praat 实际帧语义另由既有回归覆盖。输入哈希 f86756fcefd284e4dbeb524e8d948628119c8b441e6922f4a8d6aeae78f67ee9。证据 output/validation/m03-long/parity-427754db761143458ccf74e72487aee1/report.json。

## 真实受限进程与页面

- `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m03_long_process.py`：实际 OwnedProcess/命名管道、3 GB/240 秒限制下，末尾 preview、整段 single、带三图 batch、末尾 inverse 均成功，耗时分别 33.34、15.89、7.88、3.94 秒，受到同机并发验证影响，不能作为稳定性能承诺。预览 WAV 实际保留全部 576 万帧，CSV 包含 119 秒末段，逆滤波 WAV 5760 帧；进程中途取消和121秒/5760001帧拒绝通过。证据 output/validation/m03-long/process-576ea1bbd31848c9a88c0ef76db0d5a7/report.json。脚本后来将输入从本轮原版捕获路径改为同配方生成，强制检查相同 SHA-256，便于复跑。
- `node tests/e2e/m03-long.cjs`：3 组通过、pageerror=[]。真实本地服务/核心，120秒末尾119.75秒中心、完整曲线保留17000以上CQ点、持久完整导出和受控目录实际保存5文件。证据 output/validation/m03-ui/chrome-fddafa1b874541d8b08a997293c622cb/report.json。
- `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m03_long_qt.py`：3组通过，真实Qt文件夹入口、同样末尾/全段显示、实际保存CSV及三PNG。证据 output/validation/m03-ui/qt-long-18f390b512a74f96b2d067874376b729/report.json。使用测试数据库副本，不运行DDL。
- `node tests/e2e/m03-ranges.cjs`：6组通过，5/5000ms保留，66.4秒文件的65秒末尾已可计算；120.8秒明确拒绝且不建任务。证据 output/validation/m03-ui/chrome-60035e0975244dc29c80a167b9262049/report.json。
- 原生单文件CSV与Qt保存CSV逐字节相同，SHA-256 ca71820a7c0476b05934d05a289c1082f715285229172b7c0dc51dbded5fc3af。科学环境Pillow实际解码10张PNG（预览PSD、single/batch三图及Qt三图），并目视检查全段波形PNG和Qt全段页面。

## 回归、失败与范围

后端新增限制用例先在原60秒/微观60秒限制下失败，实施后通过。旧导出测试的60.000125秒拒绝项按新边界改为120.000125秒，保留超限拒绝语义。

`Invoke-M03-Python.ps1` 下运行 test_m03_exports/preview/ranges/export_names：59 passed。m09-ui 环境 test_m03_long_limits/contract：16 passed，保留两条已有依赖弃用提示。前端 `npm --prefix frontend run test` 50 passed，typecheck/build/contracts:check 通过。契约由后端生成，未手写生成类型。docs/architecture/diff 检查通过。

保留试验失败：首次原生脚本使用宿主未安装的Pillow，在启动计算前失败；改为宿主检查PNG头，随后在既有科学环境Pillow完整解码，没有安装新依赖。首次Chrome在生成契约时被Vite热更新重置到首页；第二轮等待首张图后立即断言三张图导致失败；固定源码后等待三图都到达，最终通过。首次Qt测试保存目录选择器误把合成输入目录当输出，产品按授权目录正确保存，脚本断言失败；明确将测试输出选择切到saved后复跑通过。失败目录 chrome-555f717e21c8441680d862cdff7881b6、chrome-db83e971b2e64db3a1c9e9494795ca21、qt-long-d2f0914714f943c39fad24bce5cd4a6f 保留。

此次没有新增外部代码或算法来源，没有更改科学核心、V2、用户录音、全局环境、数据库schema、CI或旧EXE，未push。新增长录音验收是公开合成输入；自然录音既有证据不扩充为120秒自然录音验收。没有验证所有信号在预算内必然成功、10账号并发负载、跨平台或冻结发行。完整M03保持in_progress。
