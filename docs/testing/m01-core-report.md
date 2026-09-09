# M01-B 科学核心迁移验收

状态：verified（限定Windows数组核心与独立wheel）；149项定向检查通过。M01/P08 整体仍为 in_progress，完整 UI、原生输出预算和批次任务未接入。

## 已完成的计算迁移

`analyze_audio(AudioInput, AcousticConfig, associations, backends, cancellation)` 接收原 dtype/声道采样数组、不可变配置、已解析关联数据和显式后端。核心不导入旧 Python 包、Qt、HTTP、数据库或进程 API。旧版原生二进制仅由合成测试适配调用，正式原生适配仍待 M01-C。

- 原 80 参数目录、服务 None/空数组与显式全选/子集语义、SOE 扩展、14 设置、声学缺口平滑、唇形时间、TextGrid 半开区间标签保留。
- WM 仍用 IRAPT 优先、Praat 回退，记录实际分支；Python REAPER 跟踪器与 native 明确区分。
- 10 个纯算法文件保留表达式和函数体；其他文件仅拆分 I/O、导入、配置及后端记录。逐文件原 hash、迁移 hash 与不变 AST 清单见 [迁移证据](../modules/evidence/M01-core-migration.json)。
- Sinc 查表数据与既有来源 hash 一致，包内包含资源、NOTICE 和已审计的上游许可文本；仍不等于完整再分发许可已解决。

## 已执行验证

全部使用新项目内 `.venv/m01-science/Scripts/python.exe`，不是旧 conda 环境：

| 命令（共同前缀 `-X utf8 -m pytest -c tests/pytest.ini`） | 结果 |
| --- | --- |
| `packages/phonetic_core/tests` 的最初配置/对齐/工程探针 | 14 passed |
| `packages/phonetic_core/tests/test_acoustic_audio.py` | 37 passed：30 组真实 WAV→Praat 解码与数组构造精确相同，6 组非法数组/采样率，1 组量化解析值 |
| `packages/phonetic_core/tests/test_acoustic_associations.py` | 5 passed：三种唇形时间、标签边界/重名层、Sinc 资源实际使用 |
| `tests/parity/test_parameter_estimation.py` | 34 passed：26 个 M01-A 计算用例、7 个 P03 声学用例、连续配置与阶段取消 |
| `tests/parity/test_m01_conversion.py` | 30 passed：原函数独立捕获的 PCM 与完整 WAV SHA-256 精确一致 |

数值按原比较器 `max(atol, rtol*scale)`，atol=1e-10、rtol=1e-7；字段、shape、时间轴和 nonfinite 掩码精确。不改变 golden 或放宽容差。

补充转换基准由 `scripts/capture_m01_conversion_baseline.py` 在原解释器独立执行原转换函数，停止在跟踪器入口；只读已转换 WAV 的字节，不运行原生程序。30 个输入覆盖 16000/22050/44100 Hz、单/双声道与 uint8/int16/int32/float32/float64。这个证据证明转换，不冒充原生算法验收。

初始缺失实现检查为 `ModuleNotFoundError: phonetic_core.acoustic`；新环境在安装核心包前也出现过包未安装的收集错误，后者不作为算法失败用例。开发过程没有更改原计算公式来让测试通过。

## 环境、来源与边界

科学运行包 8 个，测试/构建包 10 个，见 [依赖清单](../../third_party/m01-dependency-inventory.json) 和 requirements-m01-science/test 锁。8 个科学版本与原环境审计一致。没有安装 openpyxl/xlsxwriter 或修改原环境，导出库在 C 时单独处理。

来源注册表与软件致谢共用生成数据。历史 VoiceSauce/OpenSauce 移植来源和许可未决项保留，未把方法参考写成所有代码的作者。所有测试音频均为自有解析合成信号；真实自然持续元音与配套唇形仍未新增验证。

下一项：进入 M01-C，验证原生输出的强制资源限制、格式安全与双产物原子发布。数据库实际迁移与 M01 页面联合验收尚未发生。

独立 wheel 基线验收：新建 `.venv/m01-wheelcheck`，按测试锁安装 18 包并安装本地 wheel，在 `output/validation/m01` 以 `-I -m pytest` 执行核心与 P03/M01 定向测试，**143 passed**。JUnit 证据为忽略的 `output/validation/m01/b-wheel-before-metadata.xml`。wheel 的独立 site-packages 路径与无 Qt/FastAPI/旧包导入均已核验。


## 最终结果与独立修正

- 数值迁移提交 `55bc27e` 保留旧元数据默认值，并通过143项独立wheel测试；随后单独修正 M01-D01：`AnalysisResult.sampling_rate` 取实际输入采样率，不再固定16000。22050/44100失败用例先红后绿，P03/M01-A完整数值对照同时明确断言这个元数据差异。
- 后端状态修正：空输入、Praat调用失败、Praat有效点不足分别记unavailable及原因；只有取得足够有效点才记Praat回退成功。3个专门失败用例先红后绿，声学数组未改。
- 最终独立wheel：**149 passed**（核心62、M01/P03数值34、转换30、原P03/M01-A冻结保护23）。运行于m01-wheelcheck的site-packages，命令与前述一致；最终JUnit为 `output/validation/m01/b-wheel-final-metadata.xml`。前端仅更新生成的致谢数据，`npm --prefix frontend run typecheck` 和 `ui-data:check` 均通过。
- `scripts/check_architecture.py`、`scripts/validate_docs.py`、Git差异检查与环境依赖兼容检查通过。318项来源登记已与生成致谢保持一致；历史许可缺口不改变。
- 保存性：v2的427个基线文件、HEAD/index/status、原包元数据hash和环境路径与本阶段前一致；P03的8份及M01-A的28份golden逐项hash不变。Git根已确认在本项目D盘目录，本轮没有push、现存数据库操作或Codex内置浏览器关闭操作。

核心取消只在计算阶段间检查；不能宣称原生子进程即时取消、网页配额控制或任意文件解码已验收。损坏WAV检查验证外层SciPy解码拒绝，核心仅接已解码数组；正式文件入口仍在C。完整M01/P08保持in_progress。

最终交付wheel位于忽略的 `output/validation/m01/wheels-handoff/`；收尾只纠正差异编号注释，AST与149项测试的wheel完全一致，重建后48项输入/后端/资源检查通过，安装内容逐文件与最终源码一致。最终wheel SHA-256：`917ee84b43645cbe5ecff6817ca28ed0d1809edcbb168b32b848d6d55a66d467`。
