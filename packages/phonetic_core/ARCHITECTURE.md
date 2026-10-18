# 科学核心架构

phonetic_core 是桌面与服务端共享的科学包。输入为明确模型、数组和配置，输出为科学结果及必要元数据，外部能力通过 ports 注入。约束见 [AGENTS.md](AGENTS.md)，跨进程关系见[总架构](../../ARCHITECTURE.md)。

原音系读取/归纳/文档渲染、LPC与Klatt生成/合成也由浏览器运行原代码。Pandas只在旧表格读取、DataFrame生成或对应平滑真正调用时导入，类型注解使用前向引用，数组计算与缺失值语义保持；纯生成/合成不因导入分析模型而加载整套Pandas。浏览器绑定、输入预算与字体由外层固定执行器负责。

## 源码分工

| src/phonetic_core 下的位置 | 责任 |
| --- | --- |
| models/、ports/、services/ | 数据结构、外部科学能力接口与算法编排 |
| catalog.py | 声学参数的稳定身份与定义 |
| acoustic/ | 参数估计与轨迹处理 |
| egg/、lpc/、lip/ | EGG、LPC 和唇形相关纯计算 |
| synthesis/、manipulation/ | 合成、发声类型及音高/时长变换 |
| spec2wav/ | 语谱图重建、原音语谱编辑、内置参数模型的纯 NumPy 推理与固定权重 |
| vocal_tract/ | 声道模型、数据与原生能力边界 |
| annotation/、textgrid.py | 标注及 TextGrid 解析/变换 |
| transcription/ | 转写/音系相关规则与数据处理 |
| browser/、../../browser/ | 原版Praat WASM的内存适配与C++构建绑定，科学函数仍由原实现计算 |
| recording/ | 录音数据的显示、编辑和处理计算，设备采集在桌面层 |

Python 科学核心并不囊括所有客户端算法。M13 的浏览器转换与 M15 的实验调度各自有前端实现边界，不能为了目录统一复制到 Python 或声称它们已经走 worker。

## 调用与 I/O

backend/ptb_worker 负责请求快照、受控文件、原生适配及计算进程，把明确输入交给 core；desktop 的专用能力调用相应核心函数或原生 ports。core 不反向调用工作台、账号、网络路由或任务数据库。

解析和科学格式变换可以位于本包，用户路径授权、上传/下载、持久化事务、原生进程与设备生命周期在外层。共享函数不能通过 cwd、环境中旧项目包或固定开发机路径寻找隐式输入。

M09 的 OpenCV 使用范围限定为数组计算：LINE_8、circle、line、getPerspectiveTransform、warpPerspective。核心以明确 from-import 导入这五个符号，维持既有透视插值和画笔像素语义。架构检查仍禁止 cv2 整包、通配符、动态导入、图像文件编解码、窗口、摄像头与视频接口；编解码由 worker 适配。这是静态依赖约束，不充当安全沙箱。

新涂鸦以 `size_unit=px` 显式标记1–200像素直径，在校正图像网格上使用圆端点与线段距离栅格化，旧快照省略该字段时保留OpenCV画笔语义。显示缩放不参与实际笔迹计算。

音频编辑图窗保持用户选定高度，以独立画布显示当前频带。新笔迹可带 `canvas_width`、`canvas_height`、`frequency_fraction`，核心按二维像素覆盖面积投到原STFT网格后混合灰度，保持未覆盖单元、原相位和另一声道；旧快照缺字段时沿用原网格。改变图高和显示频率上限仅重新投影笔迹，独立于FFT分析和最终样本数。

浏览器通过同源Worker在Pyodide执行本包，Praat来自Parselmouth 0.4.7原始源码内的6.1.38。适配只传递Float64数组和原时间网格，不复制算法。NumPy/SciPy等WASM依赖的微小浮点差异由独立跨平台对照报告记录，整数窗长、形状、采样率、声道、掩码和算法语义保持；旧原生回归不放宽。固定版本与构建摘要见resources/manifests/browser-science-runtime.json，执行/数据持久化位于平台与worker外层。

## 科学结果约定

声道原生资源选择显式支持 emscripten/wasm32，仍与Windows/Linux/macOS目录隔离。浏览器通过相同ctypes ABI调用VTL动态库，复用Engine/Runtime与m10/r21几何。授权扩展的pressure_curve只覆盖声源Pa，新增微扰为可选m10/source-1，空设置保留上游声源路径；当前文档写8版，旧1–8版继续读取，旧几何导入明确提示重算。相邻完全一致姿态复用几何，声源独立更新不重算几何。纯设备、音频排程与取消恢复在外层，算法/存储决策见[ADR](../../docs/decisions/ADR-M10-pressure-perturbation.md)，初轮Wasm隔离和对照见[报告](../../docs/testing/m10-wasm-report.md)。

- 使用真实采样率、声道角色、每道帧数与整数半开选区；返回轨迹保留真实时间网格。
- 单位、默认值、缺失值、无声区及异常范围属于结果语义，不在重构或界面缩放中改变。
- 原始输入数组不被无意原地覆盖。结果记录实际算法后端、版本和 source_id，发生降级须显式说明。
- 显示降采样与正式计算分开；WAV 量化、TextGrid 起点、CSV/XLSX 数值和采样率转换各有边界检查。
- 可变原生状态不跨账号/任务共享，Windows DLL 不冒充其他平台的科学实现。

## 来源、环境与回归

来源/版本在 third_party/source-registry.json 和模块映射中维护。论文、代码授权、许可文件和数值一致性分别举证。变更科学行为需独立说明，与纯 import/I/O 重构分开。

使用模块对应的既有科学环境运行本包 tests 和根 tests/parity。包结构变化另验独立 wheel 安装；预期值来自独立基准，不把当前函数输出回填为通过值。合成输入覆盖算法边界，自然语料准确率、实体采集和听辨需各自证据。

vocal_tract/session.Runtime是桌面与网页共用的声道状态机，注入engine、profile、live和设备目录端口，不自行打开设备、HTTP或账号存储。envelope.audition_samples统一播放/导出增益，document记录实际截面数并提示异采样重算。Engine.enable_web_sections显式启用33/65策略；桌面原选项保持。

R21在原生接触计算前平滑局部舌曲线，跨平台由同一补丁构建。jitter/shimmer使用既有声源ABI并接入全局/区段控制，按区段边界更新且保留随机序列，计算退出时恢复原设置。Mac本轮仅同步源码与构建入口，原生R21待编译，边界与审查见[报告](../../docs/testing/m10-smoothing-review.md)。

M10 preset_library集中名称、套装和构型校验，Runtime负责新建/复制/切换/命名及原子导入策略。旧同名记录允许读取和逐项更名，新保存拒绝规范化重名。preset_pack写v3并读v1/v2/v3，R18–R20只有模型摘要和参数顺序一致且明确确认重算才允许导入。切换套装不改动已有关键帧或科研默认值。

M10 stop-timing/2以帧内释放开始为VOT零点。负VOT浊音起始前声门内收，避免开放声门提前充压；声道闭塞保持，浊音起始2ms过渡可跨普通后续帧。animation在时序帧起止边界切块，使用左极限终点与零长度边界声源更新，避免相邻动作提前混入。显示、音频及导出共用轨迹。旧method/1保留数值并提示按新方法重算，未知方法拒绝。
