# 科学核心架构

phonetic_core 是桌面与服务端共享的科学包。输入为明确模型、数组和配置，输出为科学结果及必要元数据，外部能力通过 ports 注入。约束见 [AGENTS.md](AGENTS.md)，跨进程关系见[总架构](../../ARCHITECTURE.md)。

## 源码分工

| src/phonetic_core 下的位置 | 责任 |
| --- | --- |
| models/、ports/、services/ | 数据结构、外部科学能力接口与算法编排 |
| catalog.py | 声学参数的稳定身份与定义 |
| acoustic/ | 参数估计与轨迹处理 |
| egg/、lpc/、lip/ | EGG、LPC 和唇形相关纯计算 |
| synthesis/、manipulation/ | 合成、发声类型及音高/时长变换 |
| spec2wav/ | 语谱图重建、原音语谱编辑及相关变换 |
| vocal_tract/ | 声道模型、数据与原生能力边界 |
| annotation/、textgrid.py | 标注及 TextGrid 解析/变换 |
| transcription/ | 转写/音系相关规则与数据处理 |
| recording/ | 录音数据的显示、编辑和处理计算，设备采集在桌面层 |

Python 科学核心并不囊括所有客户端算法。M13 的浏览器转换与 M15 的实验调度各自有前端实现边界，不能为了目录统一复制到 Python 或声称它们已经走 worker。

## 调用与 I/O

backend/ptb_worker 负责请求快照、受控文件、原生适配及计算进程，把明确输入交给 core；desktop 的专用能力调用相应核心函数或原生 ports。core 不反向调用工作台、账号、网络路由或任务数据库。

解析和科学格式变换可以位于本包，用户路径授权、上传/下载、持久化事务、原生进程与设备生命周期在外层。共享函数不能通过 cwd、环境中旧项目包或固定开发机路径寻找隐式输入。

M09 的 OpenCV 使用范围限定为数组计算：LINE_8、circle、line、getPerspectiveTransform、warpPerspective。核心以明确 from-import 导入这五个符号，维持既有透视插值和画笔像素语义。架构检查仍禁止 cv2 整包、通配符、动态导入、图像文件编解码、窗口、摄像头与视频接口；编解码由 worker 适配。这是静态依赖约束，不充当安全沙箱。

## 科学结果约定

- 使用真实采样率、声道角色、每道帧数与整数半开选区；返回轨迹保留真实时间网格。
- 单位、默认值、缺失值、无声区及异常范围属于结果语义，不在重构或界面缩放中改变。
- 原始输入数组不被无意原地覆盖。结果记录实际算法后端、版本和 source_id，发生降级须显式说明。
- 显示降采样与正式计算分开；WAV 量化、TextGrid 起点、CSV/XLSX 数值和采样率转换各有边界检查。
- 可变原生状态不跨账号/任务共享，Windows DLL 不冒充其他平台的科学实现。

## 来源、环境与回归

来源/版本在 third_party/source-registry.json 和模块映射中维护。论文、代码授权、许可文件和数值一致性分别举证。变更科学行为需独立说明，与纯 import/I/O 重构分开。

使用模块对应的既有科学环境运行本包 tests 和根 tests/parity。包结构变化另验独立 wheel 安装；预期值来自独立基准，不把当前函数输出回填为通过值。合成输入覆盖算法边界，自然语料准确率、实体采集和听辨需各自证据。
