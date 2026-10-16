# 语谱图自动预测与 80 带修复组件

2026-10-15，Windows x64 当前源码与独立 CPU 组件。此前完成独立组件本机验证。本轮按最新要求只修改模块，参数预测已内置并覆盖基础GL，预测值允许手动覆盖；不改版本，不构建EXE或独立组件，不公开上传。

## 内置参数模型（本轮）

压缩float32数组303856字节，80666个参数。权重SHA256 `e6edbf5c450fc08f4ff4c39d2828a15e24c4dd7286c619308c671ed4df7dffdc`，原检查点与下方v5一致，完全不量化。主应用复用NumPy/Pillow，无需Torch或组件下载。参数资源在core包中，已配置package-data，现有源码冻结也会包含该资源；本轮不进行包构建，权重保持Git忽略。

28组与原CPU Torch的图像/线性和对数标定对照通过，最大物理输出绝对差 `8.20159912109375e-05`（ms或dB），所有整数窗长一致，参考容差rtol/atol均2e-5。本机NumPy预测中位数约3.76ms，仅计算调用，不含窗口及进程启动。证据 `output/validation/m09-parameters/numpy-99244e3ed5fc4df48bf7d90740eed99a/report.json`。

89项定向Python通过，覆盖无需Torch的预测、GL真实生成、旧快照默认值、手动值不回退旧算法、线性/对数、Aoki自动及显式开启/关闭、256次、采样率/比较图坐标与精确1/2/3/74/200像素画笔、已有Python身份与失败注册保护。执行命令见下方。

## 当前源码交互与推理验收

- 标定未填完时预测禁用，填写后点击按钮才预测。预测只回填一次，重建提交当前值；手动修改与较早异步预测返回相遇时保留修改。模型不预测频率轴，保留线性/对数选择。
- 两个算法共用Aoki自动/强制脉冲/随机相位选项；默认32、最高256次。新任务使用calibrated算法，手动编辑不改变时间/dB映射。旧请求仍保留legacy兼容默认。
- 结果频谱按最终音频重新分析，取输入频率区间、线性/对数坐标及相同宽高。默认输出采样率为频率终点×2，非整数时向上取整Hz；使用实际输出率记录WAV。画笔单位为校正图像素直径，缩放只改变显示。
- GPU入口为原生选择Python→检查兼容性→选择GPU。检查真实CUDA运算与现有修复/声码器模型，不依赖CUDA可见标志单独判定。成功注册解释器身份，使用前复核摘要；未通过时不注册、不修改用户环境。GPU用于修复与声码器，预测和GL仍用CPU。
- 独立Chrome浅深主题、1280/800宽度、标定/按钮/编辑及过期响应、算法/Aoki、CPU提示/GPU选择检查、下载重试、GL256/257与对数轴、画笔1/200px通过。最终追加浏览器回归为headless，证据 `output/validation/m09-vocoder-ui/chrome-1792072308320/report.json`；此前可见Chrome证据保留，不混同最终无头检查。
- 真实CPU/GPU完整推理通过，证据 `output/validation/m09-parameters/devices-955807580b3d44cab1c25f15849b424d/report.json`。GPU为NVIDIA GeForce RTX 5090 Laptop GPU，Python3.11.13/Torch2.10.0.dev20250930+cu128。0.12s输入在0–5000Hz下输出10000Hz、1200样本，强制随机相位有效，对比PNG形状48×96与输入相同；CPU/GPU最大PCM差为1/32768。启动和推理合计GPU9.19s、CPU8.90s，此短片段不支持GPU吞吐更快的结论。
- **真实可见默认GPU Qt通过**，证据 `output/validation/m09-parameters/desktop-24b994be07884d9fac423c257f52891a/report.json`。隔离任务库中验证无组件内置预测与基础GL保存，再验证原生Python选择、兼容性检查、GPU完整生成及手改14ms/随机相位/256次后再生成保存。0–8000Hz默认16000Hz，0.2s输出3200样本，两张PNG均96×144，最终结果截图已检查。
- Qt测试发现并修复两处状态竞态：声码器状态查询纳入公共串行任务队列；GPU检查成功后直接等待最新组件状态，避免碰到进行中的轮询时按钮继续禁用。失败证据保留，最终通过轮次与两份失败轮次的自有scratch已按归属清理，清理回执均为completed。用户现有应用窗口和任务库未关闭或清理。
- 最终typecheck、contracts:check、ui-data:check、前端build、说明书严格验证与生成通过。构建保留既有大chunk提示。最终EXE、跨平台与完整组件公网下载不属于本轮验收；没有构建或发布成品。

## 应用与官网结果差异核对

Lu modal原图正确标定为0.5102s、0–5000Hz，当前NumPy模型预测4.094800472ms、83.423828125dB，取4ms，与官网4.094800949ms、83.423828125dB一致。同一原图改成1s、0–11025Hz，窗长会变为1.979877949ms。用户截图还含裁剪差异，不能把原图测试的动态范围91.835762dB当成截图预测值。

另有已修复的产品问题：此前改预测值会切回legacy GL，造成时长/dB映射变化。当前手动编辑始终保持calibrated实现。官网修复模型是第6000步，当前已安装组件仍为第2000步；官网GL256/内部16kHz与512FFT/固定1.5ms帧移，应用默认GL32/内部至少24kHz与1024FFT/四分之一帧移。本轮没有擅自更新修复权重或覆盖用户指定帧移，因此不宣称两条完整音频逐样本相同。

## 致谢与参考文献同步、源码重开（2026-10-15）

按用户后续要求，补齐Aoki完整GB/T 7714书目及加速GL、NAFNet、BigVGAN三篇论文，保留原GL引用，合计五篇方法文献。书目分别核对J-STAGE、作者原文、Springer出版页、作者arXiv与NVIDIA官方仓库；OpenReview页面本机工具遇浏览器验证，其会议信息另由作者材料支持。软件来源单列本项目预测/修复模型与NVIDIA BigVGAN预训练模型，更新实际CPU/GPU用途。同步source-registry、coverage、生成来源、BibTeX及说明书项目/章节，没有改权重或科学算法。

3项来源定向测试、ui-data:check、说明书strict验证/生成、前端build通过。Chrome浅深1440/900四组全局55篇引用逐条显示及复制、语谱图五篇论文和三项模型归属、旧模块覆盖与无横向溢出通过，证据 `output/validation/source-citation-normalization/chrome-f46e870e82074a2b8dd472d1a02acfe7/report.json`，使用自有临时资料并自动清理。文档检查2066文件/441来源/112任务零错误。

公共源码入口 `Start-Research-Workbench.ps1 -Module M09 -CheckOnly` 已核对当前工作树绑定；随后按用户要求启动更新后的源码应用，启动器PID65948、实际窗口PID102664，窗口标题PhoneticToolbox 3.0、非零窗口句柄且Responding=true。启动日志位于 `output/validation/m09-parameters/user-relaunch-20261015-221140.*.log`，仅有既有PNG色彩配置警告。此次正常应用留给用户试用，不执行测试退出或清理。未构建EXE或公开发布。

## 先前完整组件体积（本轮未重新打包）

此前组件构建的体积记录保持如下，来源显示的补充不触发组件或主程序重打包。

十进制 MB，1 MB = 1,000,000 字节。

| 内容 | 字节 | MB |
| --- | ---: | ---: |
| 窗宽/动态范围推理权重 | 329781 | 0.330 |
| 80 带修复推理权重 | 7664589 | 7.665 |
| BigVGAN base 权重 | 56010917 | 56.011 |
| 三个模型合计 | 64005287 | 64.005 |
| 完整 ZIP | 303736862 | 303.737 |
| 安装后文件 | 856300543 | 856.301 |

本地构建 `output/build-m09-neural-20261015/`，ZIP SHA256 `92ace32cf1e542d1d806a2bcb3e40c9dd3ef34bf3269fe773b63fe9e9002b242`。相对原 BigVGAN CPU 包增加 7.333 MB 下载量。安装时还需 ZIP、解压和安全余量，建议至少预留 2 GB。GPU 版未构建，不能把 CPU 体积当作 CUDA 包体积。

参数模型冻结 v5 第 26 轮／累计 6656 步最佳，原检查点摘要 `7ac61f075259099e0c58e743818b2b61a1ec42d5a125eab6ef9fc9216182a86e`。80 带冻结本轮增强续训的第 2000 步最佳，原检查点摘要 `705e9ac5a3eb17ce4b74f3f3eaa03b6b6c9654579471748f64c5033ce204f522`。仅提取 state_dict，剔除优化器、训练配置与语料路径。实际包含推理文件摘要在组件 manifest 和 `models/provenance.json`。不会跟随训练目录后续变化。

## 先前组件构建的已执行与边界

- 独立组件 Python 的真实预测、GL、80 带修复及 BigVGAN 自检通过。
- 62 项定向 Python 检查通过。普通 GL 核心仍逐样本对照旧算法，导出 WAV 按独立计算的新增整体增益核对字节；并验证 15 ms 边界、四分之一帧移、样本数、线性/对数轴、静音/脉冲增益和 256 次。
- 独立可见 Chrome 浅深主题的下载失败/重试、旧声码器、完整算法选择、自动预测、状态轮询不重复预测、GL32/256/257 与对数轴提交检查通过。证据在 `output/validation/m09-vocoder-ui/`。
- 初次 ZIP 安装测试因测试根路径过长触发既有 240 字符原生库保护，已将测试改为独立短路径；未降低安装器标准。
- 最终 ZIP 已通过独立短路径安装、自检，真实预览预测与两条有界 worker（自动线性/手动15ms对数）、采样率/时长/有限值/峰值/四分之一帧移/实际取消通过，证据 `output/validation/m09/n-8569561d/report.json`，scratch 已清理。
- 本机正式组件根目录安装完成，pipeline_ready=true，旧版本保留。该次构建没有Qt证据；当前源码Qt证据见上方，最终主程序EXE及公网完整组件下载仍未验。
- 类型、契约、UI数据、前端构建、说明书严格校验/生成通过。构建只有既有大chunk提示，未改变阈值。

## 方法与公开分发

[Aoki 2023 原文](https://www.jstage.jst.go.jp/article/ast/44/3/44_E2297/_article/-char/en)第 3 节支持频率投影、峰脉冲及抛物线插值初始化。15 ms 为用户指定策略。NAFNet 为本项目研究结构的推理适配，未把后接神经网络归入 Aoki 论文。

BigVGAN 当前模型目录内 MIT LICENSE 与 alias-free-torch Apache-2.0 LICENSE 已核对并保留。训练来源包括 KeSpeech、LibriSpeech 及已授权的本地训练语料。2026-10-15 核对 [KeSpeech dataset_license](https://github.com/tzyll/KeSpeech/blob/main/dataset_license.md)第 1–3 条：非商业、禁止演绎及禁止向第三方分发数据；原文未单独明确训练后模型权重的发布条件。此处不将数据条款自行升级为权重许可结论。公开组件前须落实该条件与本地训练语料的模型发布范围。当前候选仅本地，不含原语料。

## 执行入口

- `scripts/build_m09_vocoder_component.py --model-root … --output … --repair-checkpoint …（后续r2仅含修复与声码器，本轮未执行）`，使用已有 m09-vocos-eval CPU 环境。
- `scripts/verify_m09_pipeline.py --build output/build-m09-neural-20261015`，使用 m14 解释器及当前源码路径。
- `scripts/install_m09_component.py --archive output/build-m09-neural-20261015/PhoneticToolbox-Spec2Wav-neural-r1-windows-x64.zip`，仅本机安装。
- 当前89项Python定向：使用 `.venv/m14/Scripts/python.exe -B -X utf8 -m pytest -o addopts='' -q`，PYTHONPATH指向backend/src、core/src及desktop/src，运行 `backend/tests/test_m09_parameters.py`、`test_m09_pipeline.py`、`test_m09_runtime.py`、`test_m09_vocoder.py`、`test_m09_r1.py`、`test_m09_boundary.py`，`tests/parity/test_spec2wav.py`、`packages/phonetic_core/tests/test_m09_editing.py`、`desktop/tests/test_m09_bridge.py`。
- 当前真实模型：`scripts/verify_m09_runtime.py --python <已有CUDA解释器>`；可见Qt：`scripts/verify_m09_parameters_qt.py --python <已有CUDA解释器>`，使用独立任务库、浏览器资料与运行时登记。源码应用入口仍为 `scripts/Start-Research-Workbench.ps1`。
- 前端：typecheck、contracts:check、ui-data:check、build，`node tests/e2e/m09-vocoder.cjs`。

全项目架构检查未通过：`frontend/public/vocal-tract/` 的 app.js、index.html、presets.js、scene.js、floor-contact.mjs 共5项摘要与登记不符。这些文件本轮未修改，保留同期工作，不修正其登记来制造通过。最终文档检查2066文件、436来源、112任务零错误；历史快照链接另列，不作为当前错误。文档和架构日志分别为 `output/validation/m09-parameters/docs-validation.log`、`architecture.log`，定向git diff --check通过。
