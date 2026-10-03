# 浏览器计算与服务器分工

2026-10-03，针对井井本轮提问的可行性说明。以下区分现有代码事实和建议，不代表已把全部 Python 核心迁入浏览器。

网页打开的位置与计算发生的位置是两个问题。服务器可以发送界面、固定资源和模型，用户浏览器加载后直接读取用户选择的文件，计算过程不必上传媒体。桌面版也可以用同一套网页界面调用本机 Python/原生组件。

| 功能 | 当前事实与建议 |
| --- | --- |
| 普通话转 IPA、国际音标 Plus、感知实验 | M13 字典转换、M17 符号/草稿、M15 刺激呈现和结果存储已有客户端实现。它们的这些主流程无需服务器替用户计算。实验定时仍受设备/浏览器条件影响。 |
| 波形、选区、频谱显示、TextGrid 编辑与试听 | 相当一部分已有前端实现。文件读取、长文件预览、工程保存还应区分桌面/浏览器平台适配。进一步实现本地文件全流程可减少上传。 |
| 唇形实时检测 | 当前 Web Face Landmarker 已在本设备 Worker 推理。本轮只修复本地保存和输入选择。legacy FaceMesh 逐帧结果仍经本机 Python 任务；两模型不能合并为同一科学方法。 |
| FFT、滤波、部分声学参数、EGG/LPC、合成和变速变调 | 原理上可评估 JS/WebAssembly/WebGPU 实现。现有 Python/native 算法需逐模块移植或编译，并核对窗口、时钟、边界、NaN、滤波、精度和实际性能。不能把可移植写成已实现。 |
| 音系归纳 | 文本规则适合客户端；当前 M14 导入/导出仍调用平台任务接口。Excel/Word 格式处理也需移植，不能只看表面操作轻就声称已无后端计算。 |
| MFA、大型模型、VTL 等原生组件 | 保留桌面本机运行很合适。网页可提供用户明确选择的远程计算，或将来经授权评估本机桥接。MFA/原生依赖/模型内存使纯浏览器移植成本较高，不必作为第一批。 |

推荐方向为客户端默认计算，服务器提供账号、可选同步、共享资源和明确选择的重任务。这样能够降低服务端 CPU、内存和用户文件传输需求。实际节省比例取决于功能使用频率、输入时长、并发和客户端性能，当前没有压测数据，不给出可承载人数或固定降幅。

计算放在浏览器后，成本转移到用户 CPU/GPU/内存。长录音应采用 Worker、分块/流式和预算保护；后台节流、设备权限及不同浏览器编码支持仍需处理。算法和模型可以从同站点下载，不等于把用户音频上传。服务端必须只在用户明确选择远程计算时接收媒体。

核对的现有代码：`frontend/src/modules/mandarin-ipa/`、`ipa-plus/`、`perception/`、`annotation/`、`lip-extraction/`、`frontend/src/platform/m14.ts` 及共享科研任务接口。

官方能力依据：

- [MDN Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)：浏览器内音频图、输入流和采样处理。
- [MediaPipe Face Landmarker Web](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/web_js)：Web/JavaScript 面部关键点推理，同步检测需通过 Worker 避免阻塞界面。
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)：浏览器设备端推理，WebAssembly 与 WebGPU 等执行路径。此为技术可行性依据，本轮未引入 ONNX 依赖。
