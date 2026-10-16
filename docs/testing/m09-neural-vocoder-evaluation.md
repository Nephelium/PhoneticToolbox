# 语谱图转音频：声码器试听与可选组件

2026-10-14，Windows x64、当前源码、CPU。井井先要求比较官网三张图，试听 Vocos 后认为改善不明显，随后要求其他模型。HiFi-GAN 与 BigVGAN 对比后选择继续接入 BigVGAN，同时注明改善不多，并要求增加试听材料。本轮保留现有 Griffin–Lim 为默认，BigVGAN 显式选用。

## 试听方法与范围

原始三图来自官网的出处清单 `temporary-site/assets/spectrogram-examples/provenance.json`，没有改动图片、标定、官网音频或网站。Vocos 基线重算与官网 PCM 逐样本一致。比较现有重建后再合成，以及两种明确 dB 指数适配的图片直转。组内试听匹配 RMS，静音候选保留失败判定，不将静音放大冒充结果。

| 模型 | 权重大小 | 输入与本轮结果 |
| --- | ---: | --- |
| Vocos 24 kHz | 54,365,991 B | 图片直转六个候选接近静音；再合成可出声，井井认为改善不明显，未接入 |
| HiFi-GAN v3 LJ Speech | 5,865,403 B | 三图与 BigVGAN 同组比较，未选择接入 |
| BigVGAN base 22.05 kHz、80 Mel | 56,010,917 B | 三图可再合成；井井认为有小幅改善，选择可选接入 |

BigVGAN 原权重 revision `760f0c694aba68b8c200dd52ffd3b4ecafa7028c`，SHA256 `e2a93f92971e1b6750d8f9b701121597cd271e473e6c189c8072be7080047735` 与官方 LFS 身份一致。模型来源、许可、其余权重身份及依赖在 `third_party/source-registry.json`、评估报告和 `requirements/requirements-m09-vocoder-eval.lock` 登记。代码/模型许可与试听音频来源分别记录。

独立实现的 Mel 前端与上游前端最大误差：HiFi-GAN 9.54e-7，BigVGAN 0，均低于 1e-5。本轮用 GL→Mel→声码器保留现有图片映射，未将论文截图的线性频谱直接解释为训练 Mel。三图频谱收敛度（越低越接近目标）现有方法为 .464/.257/.264，BigVGAN 再合成为 .545/.576/.415。它衡量频谱结构，不能代替听觉质量判断，也没有据此宣称声码器恢复原声。

评估记录：

- Vocos：`output/validation/m09-vocos/website-ce898c7a563b461c8d166a9791f1f9c1/`。
- HiFi-GAN/BigVGAN：`output/validation/m09-neural-models/website-9675487133394e8ab8c44e00318a968d/`。
- 六组补充：`output/validation/m09-examples/public-f4842bdda567490ba86f220c4f6908ae/`，`comparison.html` 包含原声及两种结果，另交付六段 A/B WAV，先现有方法、0.6 s 间隔、再 BigVGAN。

六组补充为普通话 5.590 s、粤语 5.145 s、英语 A 7.150 s、日语 7.200 s、韩语 4.605 s、英语 B 5.865 s。前五组来自 [sherpa-onnx 官方 SenseVoice 测试例音](https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html)，固定 HF 仓库 `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` revision `2365baeacb507f821a0c8120fcee3d484dba7a07`。英语 B 为 [BigVGAN 官方 LibriTTS 示例](https://github.com/NVIDIA/BigVGAN/blob/7d2b454564a6c7d014227f635b7423881f14bdac/demo/examples/libritts_24k.wav)。实际下载 URL、源文件 SHA256/字节与处理设置均在 results.json。

补充例子从已知录音以独立 SciPy STFT 生成灰度 PNG：16 kHz、Hann 25 ms、帧移 5 ms、FFT 1024、相对 10log10 幅度标度 -30/0 dB，然后两路使用相同图片、32 次 GL 迭代、种子 0。尾部去掉不足一帧的样本，结果记录处理后的时长。其输入条件比未知论文截图完整，不能混同两种条件下的效果。页面默认 10 ms 保持。18 段单独音频及六段 A/B 为用户本机试听交付，未上传服务器、官网、GitHub，也未加入组件包。原下载与浮点中间量随 scratch 清理。

2026-10-14 按井井追加要求，六组试听页现在直接并列输入、现有重建与 BigVGAN 输出语谱图，共 18 张带坐标的 PNG，可点击查看大图。输入使用原来的量化 PNG，两张输出从已有试听 PCM 以相同 SciPy STFT 参数重新分析，参考原声的频谱峰值为共同零点，统一 -30～0 的 10log10 幅度色阶，不逐图归一化。上方音频与输入摘要在绘图前逐项重核，保持原字节，未重跑模型、重建或下载例音。`spectrograms.json` 记录分析参数、图片尺寸与 SHA256。

只补图的复现命令：`.venv/m09-vocos-eval/Scripts/python.exe -B scripts/evaluate_m09_more_examples.py --render-existing output/validation/m09-examples/public-f4842bdda567490ba86f220c4f6908ae`。完整评估结尾也调用同一渲染入口，绘图依赖留在已建立的本地评估环境。

独立 Chrome 检查六组/18 图片实际解码尺寸与 PNG 头一致、18 音频元数据与组内等时长、1440/800/390 宽度三列/单列及无横向溢出通过，桌面截图已核查。证据 `output/validation/m09-examples/plots-1791912982230/report.json`，自有浏览器 scratch 清理完成。首轮检查错误地假定浮点英寸×DPI 的尺寸正好为整数，修正为核对实际 PNG 头与解码像素，没有修改图片或放宽加载要求。本次仅更新试听页和渲染脚本，不扩大为应用或模型重新验收。

## 可选组件与任务链

### 0–5000 Hz、短片段与论文图条件补充

按井井追加要求，后续声码器实验默认只使用 0–5000 Hz 输入图，旧 0–8000 Hz 试听作为此前条件保留，不改写其来源或音频。`evaluate_m09_more_examples.py` 后续生成的默认频率终点已改为 5000，渲染按实际记录频率范围读取，旧报告仍显示原范围。

本轮 `scripts/evaluate_m09_paper_conditions.py` 从上一轮六段公开参考 PCM 读取，逐项核对原摘要，无新增音频下载和私人素材。48 组分为：

| 条件 | 材料 | 时长 | 窗宽 | 组数 |
| --- | --- | --- | --- | ---: |
| 已知参数对照 | 六种既有例音 | 完整4.605–7.200 s及活跃语音1.000 s | Hann 5/10/25 ms | 36 |
| 论文插图模拟 | 同样六种例音 | 同一1.000 s片段 | Praat Gaussian有效5/10 ms | 12 |

全部频带为0–5000 Hz，帧步长2.5 ms，GL 32次迭代/种子0、输出16 kHz。采用完整长段与同一源文件的活跃1秒片段，起点按20 ms RMS首次达到本例最大帧RMS的12%后前留50 ms、对齐2.5 ms网格选取，记录实际起点。该选取保证短例包含语音，不能将其解释为随机抽样的音质评价。5 ms Hann仍保持50%重叠，避免用无重叠条件人为削弱重建。核心由图片高度与频率终点推导FFT，并从图片宽度/时间推导整数步长，所有实际结果均核对native_sr=10000、hop=25、输出时长/样本数一致；为避免浮点恰好落在整数下方，标定增加1e-8 s的边界守护，未修改核心算法。

此前模拟使用与现有反算相容的10log10幅度灰度、准确已知Hann窗和FFT网格，属于较理想条件。新对照仍采用该映射以单独考察窗口和时长，先以16 kHz/1024点独立SciPy STFT分析，再截去5000 Hz以上频带，输入为321行。论文图模拟实际调用Praat 6.1.38/Parselmouth 0.4.7的Gaussian PSD，设6 dB/oct显示预加重、50 dB相对功率显示范围、压缩0，重采样成241行/约400像素每秒并加入JPEG质量80压缩，最后以PNG交给当前实际重建入口。Praat有效窗长与Hann物理窗长的定义不同，Gaussian使用两倍物理长度。上述分析与显示差异见[官方分析说明](https://www.fon.hum.uva.nl/praat/manual/Sound__To_Spectrogram___.html)和[官方高级设置](https://www.fon.hum.uva.nl/praat/manual/Advanced_spectrogram_settings___.html)。这是有明确参数的插图模拟，未冒充特定论文的原图，也没有调整产品反算方法来补偿这些差异。

源端边缘反射补齐并记录，Praat实际时频步长随窗宽由该版本决定，频谱插值到指定图网格。组内及同材料/时长的全部窗宽/配置使用共同RMS目标，参考音频保留原频带。所有展示的图均只画0–5000 Hz，模型输出的额外高频成分未画在该范围内。每组并列输入和两种输出图及音频，支持语言/时长/窗宽/配置筛选和A/B连续试听。

结果 `output/validation/m09-examples/paper-bb430824418b47c58c7d05010b32e2ee/comparison.html`。48组/96次实际当前spec2wav_child计算全部输出有限非静音，30组短片段均为1.000 s，两种方法保持参考样本数；实际phase_method分别为griffin-lim与griffin-lim+bigvgan。原三列144张展示图、144路逻辑试听（共用同组参考）及48段A/B交付。PNG、音频和图摘要、实际配置、源时间、模型元数据与耗时均记录。

独立Chrome核对48组筛选、全部144张图像解码尺寸/摘要、144音频元数据与每组三路等时长、1440/800/390三列/单列和无溢出通过，桌面截图已核查。证据 `output/validation/m09-examples/paper-browser-1791914716848/report.json`。实际计算和浏览器自有scratch均已清理，试听交付保留。本轮不重建主程序、组件或发布新版本。

### 原图直接估计 Mel 再合成

井井指出现有链路先经GL会限制比较，明确要求原图→Mel→BigVGAN，随后要求再跑一批，并豁免后续检查。`scripts/m09_image_mel.py` 是独立纯图片适配器，仅接收PNG像素、标定、已知dB定义和预加重设置，无波形或GL参数/依赖。按10log幅度或PSD解码成相对功率，白色截断单元置零、DC置零，撤销已知预加重，对频率/时间功率插值到模型网格，5000Hz以上补零。绝对幅度用图片能量和Parseval关系选定RMS=.1，再换算模型未归一化FFT幅度，套用官方Slaney 80带、0–8000Hz滤波和自然对数截断。模型设置22.05kHz/1024点/256步长不变。短窗到46.4ms训练窗仍只是插值近似，未声称恢复原始Mel或原始相位。

`scripts/evaluate_m09_direct_mel.py` 对同一48张图使用固定官方BigVGAN权重、CPU常驻推理，无音频读取参与Mel生成和合成。输出裁回图片标定网格推导的时长，并转16kHz，48段直接路线均已生成。计算时的Parseval误差最大6.94e-18，固定纯频谱的官方滤波/对数步骤对照误差4.44e-16，输出有限非静音和长度断言通过；这些是豁免前已完成的计算内约束。原始直接结果及48份真实模型Mel NPZ在 `output/validation/m09-examples/direct-ff47f50952c94a87abd157396fb81124/`，图片、Mel与WAV摘要、模型版本和增益已记录。

同一试听页增加第四列原图直转Mel→BigVGAN，并可展开查看实际模型Mel输入。原输入/三列生成音频保留，展示用四路线副本按语言/时长统一RMS及.95峰值预算重新匹配，避免直接路线瞬时峰值限制导致响度不同。副本/增益记录comparison-rms.json，处理脚本 `output/validation/m09-examples/rematch-direct.py` 保留。本次48组共192路展示音频、192张线性语谱图及48张模型Mel图。用户当前试听认为GL与GL后BigVGAN接近，模型仍保留可选，未改产品算法为直接路线。

### 指定粤语短窗的自动网格与时间汇聚

井井反馈直接路线听感异常，要求仅重做 `yue-short-matched-5`。检查现有算法确认旧版已经输出80×87帧，尺寸匹配并未解决5ms输入与46.44ms模型窗的频谱差异。本次使用明确1秒标定与401时间采样点，按22050Hz/256采样点帧移自动计算87帧；周期Hann平方对每个模型窗内的图片功率做时间汇聚，边界反射，最终仍经模型513频率格点和80带Slaney Mel/自然对数。适配器增加可选 `duration_seconds` 与 `temporal_mode='hann-energy'`，旧48组默认point模式保持。时间汇聚是局部能量近似，不能恢复短窗缺失的窄带谐波及跨频相位关系，不能称为精确换窗。

实际命令：`.venv/m09-vocos-eval/Scripts/python.exe -B -X utf8 scripts/evaluate_m09_single_mel.py --source output/validation/m09-examples/paper-bb430824418b47c58c7d05010b32e2ee --model-root output/validation/m09-neural-models/bigvgan-base`。仅推理一条，生成结束后才读取已有比较PCM用于绘图，不作为模型输入。输出16000Hz/16000样本、有限非静音及图片能量约束在运行内通过，展示沿用旧粤语短段RMS目标/.95峰值预算，若触发峰值限制以JSON实际RMS为准。结果与独立试听页：`output/validation/m09-examples/yue-grid-1b62b8c1f7d94e35bfa5c3034da111b4/`。依井井免检指令未运行独立测试、浏览器、文档或差异检查。仅本机实验，听感改善待反馈，主应用未更换算法。

按井井最新要求跳过第四列及新Mel详情的浏览器、文档和差异复核；此前三列Chrome证据不扩充为四列验收。新生成的试听音频和图片已交付，但音质等候井井试听判断。未修改组件/主程序构建，未重新上传或发布。

图片重建与图片涂鸦重建提供下载和显式选择。下载后仍保持现有方法为默认。固定 HTTPS 地址下载后先检查整包 SHA256、逐文件清单、路径/预算与自检，再原子登记。不存在组件时明确失败，服务模式没有组件安装接口。重复下载已就绪组件复用现存安装。桌面下载走独立处理锁，任务/状态查询继续响应。

主应用不导入 Torch。构建入口原先会收集整个 ptb_worker；本轮明确跳过只在外部组件执行的 vocoder_child 隐式导入，并排除该模块、Torch/TorchAudio 的主程序分析，防止未来开发环境装有 Torch 时扩大主包。外部 runner 的实际迁移自检与任务已通过，最终 EXE 大小仍须未来重新打包验收。可选包有独立可迁移 Python/CPU Torch 及必需闭包，保留许可/NOTICE；不含 CUDA 编译、训练检查点、Vocos、EnCodec、TorchAudio 或试听音频。BigVGAN 从已有 GL 音频计算 Mel，在模型内部按帧补齐后裁回长度并重采样，保持原结果采样率与样本数，记录模型版本、输入/输出增益。相对重建图来自最终音频，任务/结果来源追加 PKG-BIGVGAN-BASE。音频藏信息拒绝启用。默认快照排除 vocoder=none，旧任务哈希保持。Windows 父 Job Object 控制全部子进程；选用模型时总进程内存预算 3 GB，默认约 1 GB，时限 240 s。

组件 r1 的固定身份：

| 项目 | 值 |
| --- | --- |
| ZIP 字节 | 296,404,057，约 296.4 MB |
| 安装字节 | 848,216,175，约 848.2 MB |
| ZIP SHA256 | `935c2774220cc168b74f75a30466ec7ff64d0cfd158d8a582541bdd5a3ea8165` |
| 清单 SHA256 | `5940bfe8ccc437217454eca5eba649ee3a631430cf82520dc393e611ca29897a` |
| 下载 | [BigVGAN base r1 Windows x64 组件](https://www.phonetictoolbox.com/releases/components/vocoder/bigvgan-base-r1/PhoneticToolbox-BigVGAN-base-r1-windows-x64.zip) |

按用户要求仅发布上述独立组件，服务器文件完整 SHA256/字节与本地最终 ZIP 一致。实际应用 ComponentManager 从公网 HTTPS 下载到本机，完整字节与 SHA256 再次一致。证据 `output/build-m09-vocoder-final-r1/upload-plan.receipt.json`、`https-download.json`。当前用户组件已从相同最终 ZIP 经逐文件校验、实际路径自检安装。真实本机 GET 状态与已安装时重复 POST 均为 200/ready，current.json 字节保持，见 local-api.json。此前较深 QA 路径超出共用 240 字符保护，换到正式组件根目录后最长路径 219 字符，自检通过，未放宽保护。初版构建缓存与失败补充例音的定向删除被自动审批拦截，只有 blocked by policy 结果，无进一步原因，文件保留；没有改用其他删除途径。最终构建输入、环境、固定权重与用户安装保留。

## 验证

### 图片到Mel自训小原型

井井授权先用小样本试训，后续再提供大批量语音。采用方案 `docs/plans/2026-10-14-m09-image-mel-prototype.md`，新增原创条件残差U-Net `scripts/m09_mel_network.py` 及完整配对/训练/推理/绘图入口 `scripts/train_m09_image_mel.py`。原图解析投影的80带Mel、窗长、Hann/Praat窗型和灰度动态范围为网络输入；频带输入覆盖比例从官方Mel矩阵计算，输出标准log-Mel。BigVGAN固定。原声仅进入训练标签与离线比较，图片推理不接原声或GL。

仅用此前六条已下载公开例音：zh/ja/ko/libritts四整录音23.26秒用于训练，en7.15秒验证选模型，yue5.145秒整录音留作测试。源文件先划分，图像变体及时间裁片不得跨集合。Hann/Praat Gaussian×5/10/25ms形成36全段配对，训练24、验证6、测试6。Praat图50dB/6dB每倍频程/JPEG80，Hann图沿用30dB的10log10幅度定义；所有原图0–5000Hz。目标由同一16k源录音重采样22050Hz后按固定官方BigVGAN前端产生，最终帧反射补足，源录音RMS统一0.1。该幅度约定下粤语全段峰值1.316使官方前端打印范围警告，未削波，未放宽既有约束；不作为幅度保真结论。训练数组存自有scratch，正式模型/配方/摘要及活动试听保留。

实际命令：`.venv/m09-vocos-eval/Scripts/python.exe -B -X utf8 scripts/train_m09_image_mel.py --source output/validation/m09-examples/public-f4842bdda567490ba86f220c4f6908ae --paper output/validation/m09-examples/paper-bb430824418b47c58c7d05010b32e2ee --model-root output/validation/m09-neural-models/bigvgan-base --steps 1200`。固定种子20261014，CPU PyTorch2.8.0+cpu，4线程，8裁片/批次、64帧/裁片，Adam学习率0.001，L1加时间/频率差分约束，零残差初始化。训练1200步/82.1秒，验证集选择800步，119345参数，模型494254字节。验证原始log-Mel MAE2.021130→0.616159。整段粤语Hann5/10/25ms分别2.424329→0.746116、2.345117→0.666776、2.396901→0.631825；Praat5/10/25ms分别2.649123→0.718570、2.588262→0.634128、2.462625→0.545478。MAE单位为自然对数Mel，包含缺失频带，不能等同听感或低频保真提升。

试听仅生成粤语1秒：Hann5/10/25ms及Praat5ms，每组原声、标准Mel→BigVGAN参照、解析直转、模型转换共16路；组内共同RMS/.95峰值预算，图包含实际原图、三种Mel及四路输出频谱。指定Hann5ms测试裁片MAE1.962488→0.735434。输出 `output/validation/m09-examples/trained-mel-06063f10d3924575b45eaf1f3db83eea/`，含模型、progress/results、训练曲线、试听页、Mel数组与音频。首轮绘图原图色阶反向已在renderer修正，并用 `--render-existing` 重绘，未重训或改音频；该最终renderer与首次run.json脚本摘要不同，运行配方仍记录首轮训练身份。

训练/生成运行内互斥划分、源摘要、有限值、非静音及16000样本等约束通过；用户此前免检要求继续有效，不做额外测试/UI/文档检查。每类仅一条源录音，语言与说话人混杂，不能据此认定跨说话人泛化；后续大语料必须另明确来源授权/训练划分/科研保真验收。本地例音及衍生训练模型公开分发许可未核实，本轮无公开发布。主应用/已有BigVGAN组件仍保持GL后再合成。

实际执行：

```powershell
$env:PYTHONPATH='D:\PhoneticToolbox\PhoneticToolbox_v3\backend\src;D:\PhoneticToolbox\PhoneticToolbox_v3\packages\phonetic_core\src'
& '.venv/m14/Scripts/python.exe' -B -m pytest -c tests/pytest.ini backend/tests/test_m09_vocoder.py backend/tests/test_m09_boundary.py backend/tests/test_m09_r1.py backend/tests/test_m11_delivery.py -q
& '.venv/m14/Scripts/python.exe' -B scripts/verify_m09_vocoder.py --build output/build-m09-vocoder-final-r1
& '.venv/m14/Scripts/python.exe' -B scripts/evaluate_m09_more_examples.py
npm --prefix frontend test
node tests/e2e/m09-vocoder.cjs
```

- 34 项后端检查通过，涵盖旧快照、音频模式拒绝、缺失/路径越界、服务端禁用、公开错误、共用 MFA 默认与重复安装；仅既有 Starlette 弃用提示。首次追加复跑因未设置 PYTHONPATH 导致收集失败，修正环境后通过。
- 真实最终组件迁移自检、三张官网图的六路有界任务、模型来源/时长/样本数与真实取消通过。记录 `output/validation/m09/v-aceb1ddd81c14f40a4e543ec46d87d9a/report.json`。该首次安装脚本保留现存安装保护，当前已有组件时不直接重跑安装。
- 六组补充全部使用当前真实 spec2wav_child 与已安装组件，18 个输出有限非静音，采样率和样本数匹配参考。对齐组内试听 RMS，不按候选单独拉高音量。
- 397 项前端测试全部通过。新增来源最初漏配 M09 致覆盖检查失败，补入覆盖表后全量复跑通过。
- 独立 Chrome 浅/深、缺失组件、默认选择、下载失败/重试、下载后默认保留、显式 BigVGAN 快照与 1280/800 宽度通过，截图已核查。证据 `output/validation/m09-vocoder-ui/chrome-1791910456388/report.json`。初轮 Vue 标签、可访问名称和原生 option disabled 断言已修复；没有通过改变禁用规则绕过检查。
- 应用阅读器运行时校验 `node frontend/scripts/verify-manual-runtime.mjs frontend/dist/manual` 通过，20 章/1053 素材/21313 节点；这是当前生成资源与渲染结构检查，未作为最终 EXE 打开证据。
- 本轮评估/真实任务/Chrome 自有 scratch 均有清理回执。试听交付和必要输入、模型、当前最终组件构建缓存保留。补充脚本首次仅摘要辅助函数不接受 bytes，修复后六组全部成功；失败运行的重复试听产物和旧组件缓存的额外清理被自动审批阻止，保留并记录。

typecheck、contracts:check、ui-data:check、前端 build 通过。说明书 strict 校验与 software 构建通过，20 章/1053 素材，无警告或错误。文档校验 2042 文件/431 来源/110 任务，errors 为空，未改历史快照的既有断链单列；定向 git diff --check 通过。新增主包分析排除后，4 项已有打包缓存回归和构建 CLI 解析通过，未执行完整 PyInstaller 构建。架构检查仍有同期 M10 的五个生成资源 hash 差异（app.js/index.html/presets.js/scene.js/floor-contact.mjs），本轮未修改这些文件或放宽检查；未出现新增声码器边界错误。源码、本机组件与 Chrome 已列范围的验收不覆盖最终 EXE、真实 Qt 下载交互、Windows ARM/macOS/Linux、实体音频设备或长时间高负载。本轮未重打主程序，未发布新版 EXE、Release 或推送源码。


## 授权语料扩量与80／100带双前端试验

2026-10-14，井井反馈小模型已有提升，授权暂时使用其指定数据目录，并要求加入BigVGAN v2 24k/100带、源图频率上限2–22.05kHz（主要在8k以下）、对数轴、正式训练前大纲、训练后loss曲线和可选PESQ。随后授权的KeSpeech仍留在E盘，本轮未读取或复制。主应用、下载组件和发行包未改。

- 来源只读当前training_audio的webedit子集。清单24605项中有1项缺文件，不采用缺项或removed目录，不追踪清单中工程外的原录音路径。选2612个唯一SHA256：训练2408条1208.3455秒/11说话人，验证104条50.1152秒/2说话人，测试100条50.4510秒/2说话人。四调均衡，三组说话人互斥。详细路径与选择清单仅留Git忽略的dataset-private.json。抽样是一个语料域内的跨说话人小试验，不代表广泛语言/设备泛化。
- 同一44.1k原音统一RMS0.1且峰值上限0.95后分支生成图像与两种目标。源窗5/10/25ms、Hann或Praat Gaussian，源频率上限抽样15%固定5k、75%均匀2–8k、10%均匀8–22.05k；约30%对数轴从50Hz起，其他线性轴从0起。实际全部2612图中272图超过8k，820图为对数轴。Hann采用20log幅度60dB，Praat采用PSD50dB及6dB/oct显示预加重，裁剪保留母谱标度；区别于历史Hann10log幅度图，旧模型对照仅作诊断。
- 目标分别锁定BASE22050Hz/80带/fmax8000与v2 24000Hz/100带/fmax12000，FFT/窗1024、hop256。v2官方revision c329ede9e9bbc100ddf5c91e2330a61921262370，generator450088331字节，SHA256 6f9c5715550c9d0f11159ceb8935638da5aeb19e27d1e63677632df095e376f5，官方LFS身份、weights_only严格CPU加载、有限输出通过。复用已锁定官方BigVGAN源码，未引入另一代码副本。
- 小型残差U-Net保持119345可训练参数，扩展80/100带覆盖buffer及逐例观测支持度。两网络独立从零残差训练，相同划分、种子20261015、每批8个最多64帧裁片，真实帧mask排除补齐；AdamW初始0.0007、余弦到0.00003、weight_decay0.0001、梯度裁剪5。每250步验证，分别3000步，验证MAE选择最佳模型。简单逐频带常数偏差从训练集拟合，作为无时间结构对照。图片推理仅使用像素和已知标定，没有原音/GL输入。
- 准备阶段因scipy重采样ceil与标定时长round差异发现少数一帧错位，统一最近目标采样数后再补齐hop，重生成全部配对，未放宽shape检查。失败批次只保留failure/recipe/cleanup证据，scratch清理deleted。修复后准备约116秒。先完成12个不保留权重的资源试跑，向井井交付实际数据/配置/预计8分钟纯训练与15–25分钟含交付的正式大纲，再开始正式训练。

| 前端 | 验证解析/常数偏差/训练后MAE | 测试解析/常数偏差/训练后MAE | 最佳步 | 实际训练含评价 |
| --- | --- | --- | --- | --- |
| BASE80 | 1.7407 / 1.5462 / 0.4301 | 1.9835 / 1.7793 / 0.6238 | 2750 | 192.5秒 |
| v2 100 | 2.1157 / 1.5319 / 0.4701 | 2.0657 / 1.8533 / 0.8068 | 3000 | 234.4秒 |

MAE为自然对数Mel、逐样本平均，两个目标前端不同，不能直接以0.62和0.81排名声码器。测试完全受输入频率支持的滤波带误差，80带为1.4750→0.4842，100带为1.4713→0.4970；完全缺失频带分别4.2124→1.3514与3.5910→1.7487。每例缺失组不存在时不纳入该组均值，部分支持组单列。仅作数值映射证据，不等同F0、声调或共振峰恢复证明。

输出位于 `output/validation/m09-examples/corpus-dual-9e63c8116fa64dcba5489b6f8d0e3076/`：两个checkpoint、原始history/metrics、loss-curves.png、12组72路24kHz试听、原图/两Mel/输出谱图及comparison.html。每组为原声、旧小模型+BASE、新80带+BASE、新100带+v2、两个真实Mel声码器参照。24kHz展示副本匹配组内RMS与峰值，BASE重采样不增加原带宽；2k/8k/22.05k及对数轴附加条件复用第一条测试录音，不能视作额外独立说话人。

模拟照片脚本组合纸张底色、光照、噪声、印刷网点、模糊、JPEG、平面透视和平滑曲页位移。保存已知双向采样场及有效区，校正示例使用合成真值，未训练自动几何模型，未把照片混入当前声学训练。`augmentation-preview/` 包含实际频率/窗口六格图、照片六格图、母图及可复现recipe。用户当场查看样图。此算法尚未经真实打印拍照分布检验。

实际命令（源目录仅记录在忽略的私人清单）：

```powershell
& .venv/m09-vocos-eval/Scripts/python.exe -B -X utf8 scripts/train_m09_corpus_pilot.py --phase prepare --source <authorized-training-audio> --minutes 20
& .venv/m09-vocos-eval/Scripts/python.exe -B -X utf8 scripts/train_m09_corpus_pilot.py --phase benchmark --run <run-directory>
& .venv/m09-vocos-eval/Scripts/python.exe -B -X utf8 scripts/train_m09_corpus_pilot.py --phase train --run <run-directory> --steps 3000
& .venv/m09-vocos-eval/Scripts/python.exe -B -X utf8 scripts/train_m09_corpus_pilot.py --phase preview --run <run-directory>
```

Windows/PyTorch2.8.0+cpu/4线程。定向检查通过：线性/对数轴的1kHz峰落入相同物理Mel带；两目标前端多组重采样边界帧数一致；零残差初始化；2612源摘要唯一和说话人隔离；72WAV均24kHz有限非静音且每组精确等长，实际时长取整误差0采样；95个HTML引用存在；PNG/JPEG可解码，loss及增强拼图人工视觉检查通过。verification.json记录范围，训练临时数组scratch清理deleted。preview在训练进行时新增，训练执行源码摘要与新增预览后的源码不同，仅预览代码追加，已训练权重与配方不变。

PESQ未计算。原环境没有pesq；核对PyPI官方0.0.4源码身份与setup后，独立venv安装在PyPI TLS连接阶段失败，未关闭证书验证、未安装系统编译器或改全局配置。没有以其他指标冒充PESQ。本轮未验证浏览器交互、主应用、最终EXE、F0/共振峰保真或真人听感，未上传录音或衍生模型。


### 噪点与长窗增强追加（2026-10-14）

按井井要求，在模拟照片中追加纸面随机暗污点（高斯半径0.5–2.5px）及相机黑白脉冲噪点（概率均匀0.002–0.008乘退化强度）。独立随机流保持已有几何参数不变，保存纸面透明度mask和照片脉冲mask。clean数字母版无额外噪声。新增prepare窗长采样v2：50%常用5/10/15/20/25/30/40/50ms，50%连续均匀3–50ms；manifest记录版本和采样分布。

命令：`.venv/m09-vocos-eval/Scripts/python.exe scripts/train_m09_corpus_pilot.py --phase preview-expanded --run output/validation/m09-examples/corpus-dual-9e63c8116fa64dcba5489b6f8d0e3076`。生成独立augmentation-preview-v2/photo-contact-sheet.jpg及window-sweep.png，后者同一录音、0–5kHz、Praat Gaussian5/7.5/12.7/20/25/30/40/50ms。保留旧预览及checkpoint，未重训或生成新试听。

Windows既有独立环境定向检查通过：10000次抽样落于3–50ms并覆盖全部常用值及小数；Hann/Praat各3/7.5/12.7/25/30/40/50ms的14组配对均满足80/100带shape和有限值；同seed噪声复现、跨seed变化、噪声开关不改变几何场、clean母图不变。证据在新版预览verification.json；两张拼图已视觉核查。未验证未知实拍几何校正、噪声训练后听感或GUI/EXE。当前自动展开路线仅写入方案。


### 有框／无框独立几何前端与外部图冻结（2026-10-14）

按井井授权继续级联研究，独立实现PlotGridNet，输入96×144RGB，输出9×17的二维反向采样网格。216266参数；有框、淡框、无框各占三分之一，随机图区位置、上下/无刻度、外围框、正负曲页及第二谐波位移，混合噪点、扫描、透视，窗长3–50ms。推理仅接收图片，不接收生成参数。未下载或执行UVDoc/DewarpNet模型，现成模型比较仍待进行；本机原型未接入应用。

继承已授权清单的说话人分组，120条源录音且源摘要互斥。训练88条44.8416秒生成480张，验证16条7.7384秒生成96张，测试16条8.2363秒生成96张。合成672张约356.6秒。训练前已提供实际数据、21.6万参数、1500步和预计1–2分钟大纲。固定CPU4线程，批16，AdamW0.001余弦到0.00005；损失为网格MSE加0.2局部网格增量MSE；每100步以验证像素EPE选模型。展示曲线为网格坐标误差，训练曲线采用固定前96张，不冒充全训练集loss均值。

实际1500步25.906秒，最佳验证700步。测试平均EPE21.392px，训练平均网格基线39.655px；完整框19.939px、淡框23.634px、无框20.605px，分别32张。像素按900×600照片计算；96张网格无折叠，checkpoint873471字节。训练误差继续下降、验证平台期约18px，存在过拟合/域覆盖不足。固定前六测试例对照显示标尺混入和边界裁偏，不能接受为科学标定或正式自动音频重建精度。该结论仅适用于当前合成分布，无真实照片、跨语料泛化或声学准确性结论。

输出：`output/validation/m09-examples/geometry-b463b1dde9e14703ada2764e136c31f8/`，含comparison.html/jpg、loss-curves.png、geometry.pt、history.json、results.json、dataset-private.json、verification.json。训练缓存由finally清理，cleanup.json为deleted。两阶段命令：

```powershell
& .venv/m09-vocos-eval/Scripts/python.exe -B -X utf8 scripts/train_m09_geometry_pilot.py --phase prepare --source-manifest output/validation/m09-examples/corpus-dual-9e63c8116fa64dcba5489b6f8d0e3076/dataset-private.json
& .venv/m09-vocos-eval/Scripts/python.exe -B -X utf8 scripts/train_m09_geometry_pilot.py --phase train --run output/validation/m09-examples/geometry-b463b1dde9e14703ada2764e136c31f8 --steps 1500
```

独立外部验证冻结在`output/validation/m09-examples/figure-holdouts-028d8e274b0c4dc594a51f0931466c19/manifest.json`。网页三图沿用原工程图像和坐标，窗口10ms仅为既有重建假设，不是原作者分析参数。Lu、Liang与Kong（2025）两图采用用户最后提供的2022×917截图，左谱图区裁框[138,417,929,741]、0–0.5102s；右[1136,414,1931,741]、0–0.5236s；均线性0–5000Hz。窗口和动态范围未知。新版无鼠标遮挡，manifest仅以新版为当前输入，旧摘要列入superseded_paper_inputs。用户撤回的calibrated.png和reconstructed.png未加入。五图均禁止训练及checkpoint选择，已检查裁图/坐标/摘要和来源，未生成新音频。后续人工核对ROI的声学结果与自动几何结果分别评价，不以粤语作为任务目标。

脚本`prepare_m09_figure_holdouts.py --paper-image <latest-user-image> --update-run <holdout-directory>`可复现该更新，完整来源截图仅本机保存。论文DOI10.1016/j.wocn.2025.101413已对照出版方核实，已登记本机evaluation-only来源。Windows定向检查、语法和JSON检查通过；文档检查2054文件、435来源零错误（原有历史快照缺链单列）。未测试产品UI/EXE，未公开上传。


### 多架构详细训练计划核对（2026-10-14）

按井井要求，将现有`docs/plans/2026-10-14-m09-image-mel-prototype.md`原位扩为完整执行方案，几何与Mel分线，保留原小样本配方作为历史。候选为U²-Net定位/网格、分层Transformer/预训练文档展开，以及Mel用条件U-Net、更强CNN、Restormer/SwinIR和MambaIRv2；均为待适配候选，无新源码/权重引入。本轮只做计划和只读核对，不启动新训练或安装GPU环境。

源码核对：`batch`每例最多64帧，短例edge padding并对loss应用mask；BASE约0.743s、v2约0.683s。`infer`及后续声码器都整段，尚无分块拼接。GroupNorm/padding及全图能量标定需要纳入未来整段对分块差异检查。几何模型输入为整张96×144RGB，批处理split32不是单图分块。

硬件只读查询：`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`返回RTX5090 Laptop GPU、24463MiB、616.92；当前评估环境torch2.8.0+cpu且cuda.is_available=False。E盘9个压缩分片合计144387352346字节，仅流式读取第一个WAV头部，16kHz/单声道/PCM，无文件解压。未做全库采样率断言或MD5核验。16kHz源最多8kHz实际带宽，与2–22.05kHz图上限要求分别处理，高频例子须来自足够高原采样率的授权录音。

新计划明确数据清点/分组防泄漏、1/10/50–100小时声学阶段、2000/2万/20万几何合成阶段、窗长/退化/轴条件、未知元数据、长图全局几何和带上下文Mel裁片、可选组件路由、独立评估/失败条件以及实测耗时换算。拟定数据量和阈值不代表实验结果；五个用户图仍排除训练和模型选择。未实现组件开关/分块推理，未生成这五例新试听。

本轮方案交付验证：`scripts/validate_docs.py`检查2054文件、435来源、110任务，errors为空；未修改历史快照缺链。相关文件`git diff --check`通过，台账JSON可解析。


井井补充本机已有GPU PyTorch后，使用`conda env list --json`定位pytorch_5090，通过`conda run --no-capture-output -p <actual-prefix> python -c ...`执行CUDA矩阵乘、损失和反向传播，再synchronize。实测Python3.11.13、torch2.10.0.dev20250930+cu128、CUDA12.8、RTX5090 Laptop、capability12.0，loss和梯度均有限，BF16支持查询为true。NumPy/SciPy/PIL/Matplotlib/soundfile/librosa/Parselmouth存在，einops缺失；未安装包或修改任何环境。旧CPU venv的事实仍成立，不能据此推断机器没有GPU环境。方案已改为复用该Conda环境，实际训练脚本device适配、BF16训练与各架构测速仍待执行。


### 独立训练项目迁移（2026-10-14）

按井井指定，训练研究源码已实际迁至 `C:/Users/13680/Desktop/project/【个人项目】/SpectrogramReconstruction`。13个训练/增强/评估Python文件逐个核对路径与移动前后摘要；原scripts位置不保留算法副本。详细训练/几何方案迁出并保留文档跳转；研究来源摘录迁入新项目，原来源登记保留应用用途及此历史证据引用。应用推理、组件构建及验证代码留在本工程。

新项目使用configs/local.json引用现有权重、历史结果和授权语料；KeSpeech留E盘，新训练输出默认E:/SpectrogramReconstructionData。复用既有Conda pytorch_5090，未创建venv、安装包或修改环境。新训练脚本无需应用业务算法；历史GL/应用对照脚本仍显式依赖原应用并沿用旧结果目录，不冒充完全独立的对照套件。

在新项目执行scripts/Run-Python.ps1 scripts/check_environment.py及scripts/verify_migration.py通过。三核心算法字节不变，四个CLI入口、80/100带checkpoint严格加载与CPU/GPU FP32数值对照、几何checkpoint前向及照片增强确定性通过。首次默认TF32与CPU对照失败，关闭TF32后以原1e-4容差通过，未放宽检查。完整范围和逐文件摘要见新项目docs/validation/migration-report.md、migration-files.json、migration-checks.json。未重训、未生成新音频、未运行全部旧评估脚本，未验证应用UI/最终EXE或公开上传。GPU训练适配与KeSpeech清点在新项目继续。

迁移收尾文档验证：validate_docs.py检查2041文件、435来源、110任务，errors为空；历史快照原有缺链未变化。新项目UTF-8/语法/JSON/本地文档链接检查通过，修复迁出方案对几何ADR的相对链接；相关Git差异空白检查通过。
