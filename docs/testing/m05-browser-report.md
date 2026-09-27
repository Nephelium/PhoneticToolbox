# M05-B 浏览器可行性与检测后端比较

2026-09-27。**结论：本机 Chrome、实际 Qt Workbench 的专用 Worker 路径可执行；Web 模型未通过新旧等价门，仅作为 candidate 预览。正式分析保留 legacy FaceMesh 离线后端。**

## 固定资源与隔离

`@mediapipe/tasks-vision` 0.10.14；官方 face_landmarker float16 revision 1；本地静态资源及 SHA256 见 `resources/m05/resources.json`。无运行时 CDN/latest，没有用户媒体上传。CJS 官方文件原样保存，并以字节一致的 `vision_bundle.classic.js` 别名满足 Qt 现有静态宿主的 JS MIME。

模块初始化之前没有请求模型/WASM。Dedicated Worker 执行检测，主线程只转换轻量坐标/指标；一次至多一帧推理。模型 VIDEO 单脸内部平滑与旧 LandmarkStabilizer 是两层，元数据分别记录。[官方 Web 指南](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/web_js)说明同步调用阻塞线程及单脸平滑条件。

## 实测

| 平台 / 路径 | 证据 | 结论 |
| --- | --- | --- |
| Windows Chrome、CPU/GPU requested delegate、IMAGE/VIDEO | `output/validation/m05/chrome-feasibility.json`；三组×两模式×两 delegate，共360帧 | 12 组均完成；无外域请求；初次报告初始化117–144ms；10ms心跳最大约11.6ms |
| Windows 实际 Qt Workbench、自定义 ptbapp scheme、同 12 组 | `output/validation/m05/qt-bc6781d3a78a4a5dbeb2f41511accce1/report.json` | 12组均执行，完全本地静态资源；无摄像头/音频 |
| Qt 首次失败 | `qt-ea6464db58314c97b240801176ce3774/report.json` | `.cjs` 未以 JavaScript MIME 返回；资源别名修复，未绕过安全策略或转主线程 |

requested GPU delegate 成功不等于物理 GPU 加速已证明；驱动、renderer、显存和其他设备另测。上述耗时不能推广为摄像头固定 FPS。网页浏览器缓存/持久离线安装与桌面离线资源也不能互相替代。

## 等价门结果

测前门槛见 [实施计划](../plans/2026-09-27-m05-implementation.md)。同一无损解码帧、分辨率、处理序列分别检测。用同一个 Python 指标实现计算两个 detector 的关键点，排除 JS 公式误差。

- 12组中 **0组通过等价门**。本有限素材上检测 mask 差异为0，不能推断其他姿态也一致。
- 平均关键点欧氏差约0.80–2.18px，各场景最大约3.52–12.21px；指标逐项 mean/p95/max 见 `output/validation/m05/backend-comparison.json`。
- IMAGE/VIDEO 与 CPU/GPU 分开比较；没有把模型内部平滑关闭伪装为可配置，也没有将模型误差和公式移植误差混用容差。
- Python/JS 同关键点指标/防抖测试通过与模型不等价是独立结论。浏览器导出必须保留 candidate backend_id，不能写 legacy 标签。

```powershell
node tests/e2e/m05-feasibility.cjs
& '.venv/m09-ui/Scripts/python.exe' -X utf8 scripts/verify_m05_qt.py
& '.venv/m05/Scripts/python.exe' -X utf8 scripts/m05_compare_backends.py
```

## 后续接线边界

允许继续候选预览、原始录制与 legacy 离线分析。真实摄像头/麦克风、自然语料、后台录制、写盘最终状态、长录制、回放/导出、云端任务与其他平台仍按独立验收推进；本探针页面不是产品交付。

## 产品与实机追加证据

已接正式工作台，Chrome 和 Qt 均完成本机三种录制模式及完整本地保存。点位绘制已按 V2 完整网格纠正，异步显示绑定同一输入图像，停止后保留实际尺寸；`Invalid state` 的两次 Qt 测试原因为验收脚本错误关闭权限框，修正实际点击后通过。最新边界见 [完整报告](m05-report.md)。

`cache-report.json` 验证网页缓存并非持久离线安装：已初始化 Worker 可离线继续，新 Worker 离线启动失败。Qt 本地静态资源另行通过。真实摄像头 60 fps 请求只协商到 30 fps，实际文件约 28.43 fps，不能沿用演示推理耗时作为设备采集性能。
