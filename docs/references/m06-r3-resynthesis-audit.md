# M06 参数提取与重合成自然度核查

2026-10-04 R3 历史核查。下文保留当时的代码事实和算法建议。随后井井授权按报告修改，R4 已接入可选 WORLD/PSOLA 和 Harvest，见 [实施与验证](../testing/2026-10-04-m06-r4-report.md)及 [ADR](../decisions/ADR-M06-R4.md)。GlottDNN 仍未接入。未对自然录音进行听辨实验，不能宣称自然度已验证改善。

## 1. M06 与参数估计 M01 是否一致

**共用多项底层声学函数，完整流程与输出语义不同。** M01 输出测量结果，M06 将部分测量转成能驱动 Klatt 的有限值曲线。

| 项目 | M01 参数估计 | M06 语音合成提取 |
| --- | --- | --- |
| F0 | Praat CC 与可选原生 REAPER，分别保留 pF0/rF0 及衍生参数 | 修改前固定 Praat CC。本轮可选择 CC、AC、原生 REAPER，每次选一条驱动提取 |
| 帧移、范围 | 随声学设置；核心默认 5 ms、60–880 Hz | 固定 10 ms，范围取左侧已应用的 F0 上下限，初始 50–500 Hz |
| REAPER 选项 | 核心默认 Hilbert=true、no_highpass=false，可配置 | 本轮固定 Hilbert=false、no_highpass=false，结果元数据记录 |
| F0 时间轴 | 原始时间对齐到固定秒数网格 | 旧实现按数组长度拉伸；本轮复用同一时间对齐函数，不跨无声空缺补出测得 F0 |
| 清音、静音 | 有声掩码加相对能量静音阈值，可保留 NaN，按设置平滑 | 未使用 M01 完整静音阈值/平滑策略；由所选 F0 形成掩码，连续编辑曲线填空，但无声帧 AV=0 |
| 共振峰 | 同一 Praat Burg 函数，可配置上限/阶数，带槽位限制 | 固定 6000 Hz、5 个候选，只使用函数实际返回的 F1–F4/B1–B4；再裁到合成参数范围 |
| HNR/频谱 | 随设置计算不同频带与多项校正值，保留各字段 | 周期窗固定为5，四个频带 HNR 做 dB 算术平均，H1–H2 使用原始 H1−H2 |
| Jitter/Shimmer | 公共测量函数，内部另有脉冲/F0流程，结果为百分数 | 只取 PPQ5/APQ5；本轮修复 APQ5 百分数到 Shimmer 内部比例换算，Jitter 单位不变 |
| 能量 | 核心默认 40 ms，测量值 | 固定 20 ms，相对有声能量中位数对应 AV60。AH 初始化0 |
| 结果 | 科研测量表、配置与后端记录 | 24条可编辑合成曲线；不是 M01 全部测量结果的直接副本 |

代码入口：`services/acoustic.py::analyze_audio`、`models/acoustic.py::AcousticConfig`、`synthesis/klatt/engine.py::extract`、`acoustic/f0_praat.py`、`acoustic/alignment.py`。以上均在 `packages/phonetic_core/src/phonetic_core/` 下。

算法选择针对主 F0 及依赖该轨迹的声学计算。Jitter/Shimmer 公共函数仍有自己的脉冲定位过程，不能描述成全链路全部切换为 REAPER。REAPER 缺失或失败时明确报错，无静默回退。

## 2. 发现并修正的问题

1. **F0 帧时间丢失。** 旧实现使用 `compute_praat_f0` 的数组，按0到1的相对位置插值成目标长度。本轮保留 Praat/REAPER 真实帧时间，使用 M01 的 `align_track_to_grid`；编辑曲线采用实际10 ms帧点，末值保持到音频尾部。原始 F0 空缺与有声掩码另存于提取元数据，图上为编辑填补的 F0 不作为测量证据。
2. **Shimmer 百分数被当作比例。** `compute_pq` 返回均值相对偏差×100，APQ5=0.5表示0.5%。合成配置的内部合法范围是0–0.1，合成端再×100转换成百分数。旧提取把0.5直接传入并裁成0.1，实际成了10%。本轮在写入前除以100，0.5%保存为0.005。旧文件不做无依据的自动转换，重新提取才采用新行为。

这两项作为 `m06-extract/2` 记录；合成核心仍 `klatt/2`。单位契约测试不能替代主观听辨。

## 3. 剩余自然度限制

- **清音/气流噪声不足。** 提取后 AH 全为0，未检出有声的帧 AV=0，无法靠此参数组合完整重建原录音中的擦音、气声噪声或其他非周期成分。全段未检出有声时界面明确提醒。
- **测量值与控制量缺少经过验证的逆映射。** 当前 HNR 的合成实现是对谐波间隙施加经验增益，不能保证输出重新测得的 HNR 等于输入值。H1–H2 与谱倾斜的频域修改彼此影响，不能作为独立生理控制量。
- **少量共振峰不足以保存整个谱包络。** F5/B5/A4/A5使用默认值，部分共振峰与带宽会裁入编辑器范围。对高F0、不同声道长度或共振峰缺失的录音尤须逐项检查。A1–A5虽然有测量/输入字段，当前固定SW=0的串联声源路由也不等同于完整的并联幅度拟合。
- **原始声源细节未保留。** 合成器重建周期声源、扰动和噪声，原录音逐周期波形、相位及频带非周期性未被完整编码。F0正确只是其中一环。
- **算法对不规则发声会分歧。** 本轮旧双正弦夹具在REAPER下全无声，多谐波150 Hz夹具可检出。该事实不能推广为对真实气声、嘎裂的优劣结论，须用实际研究语料和必要的人工/GCI证据核对。

## 4. 后续路线建议（planned）

| 目标 | 建议路线 | 理由及边界 |
| --- | --- | --- |
| 导入录音后只改F0或时长 | Praat overlap-add/TD-PSOLA，可优先复用已有变速变调能力 | 重用原信号的短片段，保留大量音色细节；依赖正确的周期定位，极端变调/不规则发声仍需检查。不能自由设定全部声质参数 |
| 导入录音并尽量自然地重合成，同时改F0/谱包络/非周期性 | **优先试验 WORLD**：F0 + CheapTrick + D4C | 保存完整谱包络与分频带非周期性，适合作为复制合成基线；不能把D4C参数直接叫AH、HNR或Shimmer，也不保证精确保留嘎裂的逐周期结构 |
| 研究气声、嘎裂与声门源形状 | 保留Klatt可控元音路线；另评估声门源/残差模型 | 先定义周期形状、开放程度、脉冲间隔和噪声调制等控制量。GlottDNN等可作研究参考，但涉及模型/训练数据与泛化，当前不建议直接替换主模块 |

建议用相同自然录音做四组盲听对照：原始、当前修正Klatt、WORLD未修改重合成、PSOLA未修改或小幅修改。先验证无修改往返损失，再验证研究操纵量。覆盖常态/气声/嘎裂/假声、男女/高低F0、清浊交界与静音；响度匹配用于听辨而不改保存的科研原件。记录F0倍频/半频、清浊判断、谱包络偏差和主观自然度，避免仅以频谱误差代替听感。

2018年的比较研究在4位说话人、40个含辅音四拍词、14名听者的MUSHRA条件下发现WORLD(Harvest)优于该实验中的WORLD(DIO)、STRAIGHT、YANG。它支持优先做WORLD基线，不证明WORLD对本项目语料或非模态发声必然最优。[论文全文](https://www.jstage.jst.go.jp/article/ast/39/3/39_E1779/_pdf/-char/en)

## 5. 核实来源

- Google / David Talkin. [REAPER 官方项目](https://github.com/google/REAPER)：F0、清浊状态及epoch/GCI估计。当前项目二进制仍以原登记SHA-256锁定，原始构建commit未知。
- Boersma / Weenink. [Praat音高方法选择](https://www.fon.hum.uva.nl/praat/manual/how_to_choose_a_pitch_analysis_method.html)：现行手册区分filtered AC、raw CC和raw AC。本模块调用锁定Parselmouth已有的CC/AC接口，不能把AC标签当作新版filtered AC。
- Boersma / Weenink. [Praat overlap-add](https://www.fon.hum.uva.nl/praat/manual/overlap-add.html)：原无声片段复制与周期同步片段重排的实际步骤。
- Morise, M., Yokomori, F., & Ozawa, K. (2016). WORLD: A Vocoder-Based High-Quality Speech Synthesis System for Real-Time Applications. *IEICE Transactions on Information and Systems*, E99-D(7), 1877–1884. [DOI 10.1587/transinf.2015EDP7457](https://doi.org/10.1587/transinf.2015EDP7457)。
- Morise, M. (2016). D4C, a band-aperiodicity estimator for high-quality speech synthesis. *Speech Communication*, 84, 57–65. [作者文献清单](https://www.isc.meiji.ac.jp/~mmorise/world/publications.html)。[WORLD官方实现与许可](https://github.com/mmorise/World)为modified BSD；具体绑定依赖尚未引入或审计。
- Morise, M., & Watanabe, Y. (2018). Sound quality comparison among high-quality vocoders by using re-synthesized speech. *Acoustical Science and Technology*, 39(3), 263–265. [DOI 10.1250/ast.39.263](https://doi.org/10.1250/ast.39.263)。
- Airaksinen, M., Bollepalli, B., Juvela, L., Wu, Z., King, S., & Alku, P. (2016). GlottDNN — A Full-Band Glottal Vocoder for Statistical Parametric Speech Synthesis. *Interspeech 2016*, 2473–2477. [DOI 10.21437/Interspeech.2016-342](https://doi.org/10.21437/Interspeech.2016-342)。论文的TTS听辨结果不直接覆盖本项目。

上述新增论文在统一来源登记中标为 `reference-only`，未移植代码、安装依赖或打包论文。
