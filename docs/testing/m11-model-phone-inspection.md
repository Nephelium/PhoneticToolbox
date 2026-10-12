# MFA 模型包与配套词典音素核对

核对日期：2026-10-12。范围为 Windows 本机官方资源获取、ZIP 完整性、包内元数据与配套词典标签核对，以及提供样例在 MFA 3.3.8 下的源码原生对照。未更改业务实现、应用登记资源或默认模型。

## 实际资源

| 对象 | 发布版本 | 获取方式 | 文件长度 B | 模型音素标签 / 词典条目 |
| --- | --- | --- | ---: | ---: |
| `english_mfa.zip` | 3.1.0 | 官方 GitHub Release 下载 | 92,170,811 | 100 |
| `english_mfa.dict` | 3.1.0 | 官方 GitHub Release 下载 | 1,078,195 | 42,353 |
| `mandarin_mfa.zip` | 2.0.0a | 从现有 3.3.8-r1 独立组件 ZIP 读取指定成员 | 92,275,957 | 133 |
| `mandarin_mfa.dict` | 2.0.0a | 从同一独立组件 ZIP 读取指定成员 | 8,709,372 | 163,066 |

英语下载入口为 [模型发布页](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-english_mfa-v3.1.0)与[词典发布页](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/dictionary-english_mfa-v3.1.0)。普通话来源对应[模型发布页](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-mandarin_mfa-v2.0.0a)与[词典发布页](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/dictionary-mandarin_mfa-v2.0.0a)。两个 ZIP 本身为声学模型包，`.dict` 为发音词典。

本机交付目录为 `output/validation/m11-model-inspection/`，按语言与发布版本分目录保留原包、词典、原始 `meta.json`、可读元数据和完整模型音素表。`resource-manifest.json` 记录各文件长度、SHA-256、发布来源与检查范围，`模型音素对照.txt` 提供完整列表与常见标签对照。原始字节保留，未将外部模型纳入公开源码或安装到全局环境。

## 验证与发现

- 英语下载文件长度与官方 GitHub Release API 资产长度一致。两份模型 ZIP 的全成员 CRC 检查通过；普通话模型与词典 SHA-256 与现有运行环境审计登记一致。英语 SHA-256 为本次本地计算摘要，未将其表述为上游发布的校验值。
- 两份词典可按 UTF-8 读取，跳过可选发音/静音概率列后核对发音标签。计数包含实际文件中的特殊词条、重复词条及发音变体，不等同于不同普通词数量。两份配套词典均无超出模型 `phones` 及登记静音/OOV 标签的音素。
- 完整模型音素集合直接来自各 ZIP 的 `meta.json`。英语 3.1.0 登记 `a`、`b`、`d`、`ɡ`、`ɹ`，未登记 ASCII `g` 或 `r`。普通话 2.0.0a 登记裸 `a`、`i`、`u` 与带声调标签，未登记 `b`、`d`、`g`、`ɡ`、`ɹ`。`g`（U+0067）与 `ɡ`（U+0261）须按实际字符区分。
- 官方词典页面所列音素不能替代声学模型包内的完整集合。此前依据词典页面推断普通话模型没有裸 `a`，已按实际包内元数据修正。
- 英语元数据的 `version` 为 `3.1.0`。普通话发布标签为 `2.0.0a`，包内 `version` 为训练所用 MFA 的 `2.0.0rc4.dev19+ged818cb.d20220404`，该字段与发布标签及当前 MFA 3.3.8 运行环境版本分开记录。

## 提供样例的实际对照

用户提供的压缩包通过 CRC 检查，SHA-256 为 `7b9925aef569c9f3bf7379572a3c65b229f150b9c9f461202aab4a493abedc95`。词典只有 `s→s`、`ʂ→ʂ`、`i→i`、`a→a`、`u→u` 五个条目；四份 LAB 实际使用 `s/a/i/u`，每份 18 个词条，未使用 `ʂ`，无未登录词。四份音频为 24 kHz、单声道、16-bit PCM，时长 9.83、9.22、12.20、10.60 秒。

测试复用 v3 工作树中已经独立登记的 MFA 3.3.8 环境，使用本页已核实的英语 3.1.0 与普通话 2.0.0a 模型。原始输入只读，WAV 与同名 LAB 临时合并至每次拥有的 `scratch/corpus`；Beam 100、Retry beam 400。直接原生对照使用上游 `PretrainedAligner`，工具箱对照执行当前未修改的 `backend/src/ptb_worker/mfa/child.py`。

| 组合与路径 | 实际结果 |
| --- | --- |
| 当前工具箱子进程 + 普通话模型 + 原五行词典 | 四份均成功，每份 words/phones 层各 18 个非空标签，无未知词/音素标签 |
| 上游原生 MFA + 英语模型 + 原五行词典 | 排除唯一不支持的 `ʂ` 发音条目，四份均成功，每份 words/phones 层各 18 个非空标签 |
| 当前工具箱子进程 + 英语模型 + 原五行词典 | 重试产生 `m11_model_mismatch`，在归一化阶段中止 |
| 当前工具箱子进程 + 英语模型 + 仅删去 `ʂ→ʂ` 的四行词典 | 四份均成功，words/phones 层每份各 18 个非空标签，所有标签序列与原 LAB 完全一致 |
| 当前工具箱子进程 + 普通话模型 + 同一四行词典 | 重跑四份均成功，words/phones 层标签序列与原 LAB 完全一致，四份新 TextGrid 已交付到用户指定桌面文件夹 |

英语组合首次工具箱运行在收到 `runtime_checked` 前超时，保留失败诊断，不将其记为音素不匹配。相同输入与未修改的子进程重试后明确复现 `m11_model_mismatch`；首次启动超时原因尚未确定。

单行删除对照仅从测试副本移除 `ʂ→ʂ`，其余词典内容、四份 WAV/LAB、模型、Beam/Retry beam 与业务子进程保持相同。当前工具箱英语路径实际通过，四份结果的 words/phones 层标签序列均逐项等于 LAB；原压缩包、模型与业务子进程 SHA-256 前后一致。可交付四行词典与四份 TextGrid 保留在 `output/validation/m11-feedback-1012/english-compatible/`，原词典未覆盖。

普通话交付对照沿用同一四行词典、四份 WAV/LAB 和搜索参数，只切换到普通话 2.0.0a 模型，实际重跑并生成四份结果。交付目录原先只有四份 WAV，摘要均与压缩包内对应音频一致；本次以独占创建方式写入四份同名 TextGrid，未覆盖文件，交付后原音频摘要不变。绝对交付路径、各 TextGrid 长度/SHA-256、运行日志和清理范围记录于本机 `output/validation/m11-feedback-1012/mandarin-delivery-summary.json`。

结论：英语模型不支持词典里的 `ʂ`，当前四份语料不使用这个条目。上游 MFA 会排除该条目后继续，当前工具箱在 `normalize_text` 内只要 `excluded_phones` 或 `excluded_pronunciation_count` 非空就拒绝整个组合，因此把未使用词条也纳入了阻断范围。普通话模型支持这份词典的全部标签，原生实际任务通过；仅删掉未使用条目后英语工具箱路径也通过，确认该条目是本例阻断触发因素。实际需要的不兼容发音仍须核查，不能从本例外推全部自建词典均可安全忽略排除。

压缩包将 LAB 与 WAV 放在不同文件夹。当前桌面目录读取在 WAV 所在目录寻找同名转写，实际导入前需将各对文件放在同一目录。此次只在测试副本中合并，原压缩包保留。

证据在 `output/validation/m11-feedback-1012/summary.json`、各运行目录的 `inputs.json`、五组对照结果与日志中；项目验证目录保留十二份新生成的 TextGrid，用户指定桌面目录另交付四份普通话重跑结果。本机 `reproduce.py`、`verify_without_sh.py` 与 `verify_mandarin_delivery.py` 保留配方。原复制 WAV/LAB、运行缓存、SQLite 与中间输出均由项目清理入口核对后清理，五个 scratch 目录有 `deleted` 回执。未删除原始压缩包、模型资源或独有结果。

## 跨语言句子 IPA 样例的实际检查

用户另提供目录与同目录 `ipa_dict.txt`。本次词典为 29 条逐符号自映射，SHA-256 为 `1430ff0f3935a32ee10919b8a6a85dde7b1640bab062109343be975cde046302`；四份同名 WAV/LAB 分在两个子目录。WAV 均为 24 kHz、单声道、16-bit PCM，时长约 3.24–4.09 秒。每份原 LAB 19 个空格分隔标签，其中长音号 `ː` 与重音号 `ˈ` 分别成为独立标签。

沿用相同已核实环境、两个模型、Beam 100 / Retry beam 400，临时合并四组 WAV/LAB 后将原词典字节原样交给当前子进程。英语和普通话两次实际运行均返回 `m11_model_mismatch`，未生成完整 TextGrid。未修改、精简或重映射输入词典与转写。

| 模型 | 词典中不支持的音素标签 | 其中由当前 LAB 实际引用的标签 |
| --- | --- | --- |
| English MFA 3.1.0 | `ɑ̃ ɕ ʁ ː` | `ː` |
| Mandarin MFA 2.0.0a | `b d ð ɑ ɑ̃ ɔ ɛ ɜ ɡ ɪ ʁ ʊ ʒ ː` | `d ð ɑ ɛ ɜ ɡ ɪ ː` |

这组样例包含实际使用的不兼容发音，不能套用前一组仅删掉未使用 `ʂ` 的修剪方案。英语模型登记有 `ɜː`、`ɑː` 等元音加长音的完整标签，未登记独立 `ː`；如需重写词典/LAB，应按具体模型标签和研究目标制定映射，不能在本次只读诊断中静默改变音素序列。原 LAB 还包含词典未登记的独立 `ˈ`，这是原始文件静态检查结果，本次在音素不匹配阶段中止，未单独验证其规范化/OOV 行为。

证据与复现配方在 `output/validation/m11-feedback-1012-2/summary.json`、相应运行目录的输入摘要、英语/普通话结果及日志和 `reproduce.py`。原目录所有文件摘要前后一致，隔离 scratch 已有 `deleted` 回执。本次未测试自动映射、其他语言模型或直接上游忽略不兼容后的输出。

## 未完成范围

反馈者实际使用的模型 ZIP、MFA 版本与旧 TextGrid 未提供，不能确定与本次版本相同，也未核对其过去输出。本次验证限当前源码子进程与直接上游 MFA，不代表 GUI 导入、最终 EXE、边界准确率、实体听辨或跨平台验证。未实现兼容性检查修正。
