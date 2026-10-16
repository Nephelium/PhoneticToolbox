# 唇形离线批量与媒体导出验收

日期：2026-10-13。范围：Windows 当前源码、独立 Chrome、真实本机 API／worker、冻结 FaceMesh 0.10.14、实际 MP4／GIF 编解码。未重打 EXE、未发布。

## 改动与行为

- 顶部模式菜单增加离线分析，操作按钮为选择文件与分析所选文件，原独立离线区移除。支持多文件，单项失败保留诊断并继续，取消停止后续项。
- 每个成功视频自动进入历史结果，显示输入文件名。新任务记录受管源视频引用，重开后可恢复视频和测量；每次只保留一份历史视频 Blob。
- 结果保存位于右侧最上方，默认高清 1080、视频与动画均勾选、多文件保存本批全部成功结果。保存（应用偏移）使用各结果自己的值，保存（不应用偏移）写零人工偏移。整批选择一次目录，各视频独立子目录，完成后才清对应未保存标记。
- 完整保存包括既有科学数据、WAV／时间信息及所选视频与 MP4 网格动画。单独导出动画 MP4／GIF 只向选择目录新增该动画，不重新识别或创建分析任务。
- MP4 与 GIF 直接复用已分析完整帧，使用相同面部范围、冻结完整网格、30 fps 显示采样与音频覆盖范围，GIF 无音轨。GIF 使用白底到网格线的专用 256 色抗锯齿调色板。
- 网格表来自既有 resources/m05/metric-spec.json，逐边核对与冻结 MediaPipe 的无向 tessellation／contours 完全一致。编码不再为读取连接表导入模型，保留原 1 GiB 子进程内存上限。
- 单独动画的预算只计算本次新增输出，完整结果计算当前独立结果目录，不把用户目录已有文件计入保存预算。

## 实际验证

| 检查 | 命令与结果 |
| --- | --- |
| 前端类型 | npm --prefix frontend run typecheck，通过 |
| 前端唇形回归 | node --test frontend/tests/m05*.test.ts，16项通过，含目录取消、批量部分失败、逐项确认及单独格式路由 |
| Python | .venv/m05/Scripts/python.exe -B -m pytest -c tests/pytest.ini backend/tests/test_m05_offline_exports.py backend/tests/test_m05_r1.py backend/tests/test_m05_r2.py backend/tests/test_m05_r3.py backend/tests/test_m05_exports.py backend/tests/test_m05_video.py -q，36项通过，10条既有 protobuf 弃用提示 |
| 浏览器和本机链路 | node tests/e2e/m05-offline.cjs，5组实际流程、12组浅深／1440、900、800宽度／100%、150%布局通过；无页面错误 |
| 独立编解码回读 | 四份完整保存对应 first／second 视频各自 .125／-.075 与零偏移，原时间数组保持；两份单独动画均为34帧、末帧1.10s，MP4一条音轨、GIF零条；中央内部网格像素检查及PNG目视确认通过 |
| 协议与数据 | npm --prefix frontend run contracts:check、ui-data:check，通过，未改公开API/schema或迁移现存数据库 |
| 构建 | npm --prefix frontend run build，通过；保留已有大chunk提示 |
| 说明书 | strict validate、software build，通过，20章／1053素材；本章正文与章节目录、模块README、开发操作页已同步 |
| 文档／架构 | validate_docs.py：2016文件／427来源／106任务，无错误；check_architecture.py仍报生理参数合成app.js、index.html、presets.js、scene.js、floor-contact.mjs五处既有资源摘要差异，本次唇形修改未新增架构错误 |

最新浏览器证据位于忽略目录 `output/validation/m05-offline/chrome-c00b5e60ef614685bdf91744f24cce74/`，保存 report.json、codec-readback.json、浅深截图与两个解码PNG。上一份同功能最终媒体证据在 `chrome-1c212e55a8eb463188a164f122d02f69/`。

公开合成输入使用既有 NASA astronaut 测试图生成两个6帧／10fps的 FFV1 视频，均配1秒／8000Hz正弦PCM；独立网格测试使用478点规则网格与正负偏移。原始测试图只读。浏览器配置、合成媒体、独立数据库与输出在 scratch，服务和浏览器退出后按归属清理，最终 cleanup.json 为 deleted，报告／截图保留。

## 修复过程与范围

首次实际保存暴露1080画布分配失败，原因是1 GiB编码进程额外导入模型，仅加载冻结邻接表后完整保存通过，未扩大预算。首次GIF固定 RGB8 与 H.264 解码网格交集未达既定0.75门槛，换用线条调色板后原断言通过，未放宽阈值。历史菜单等待用例先等待了不可见的原生 option，改为等待其已挂载后恢复流程通过。上述失败日志、截图及后续验收都保留。

旧任务若没有受管源视频引用，仍可读测量并导出动画；恢复原视频或完整保存中勾选视频，需要重新选择源文件分析。其他长文件／自然语料准确性／实体音视频同步、实际Qt、最终EXE、跨平台及本次动画取消的原生时序尚未扩大验收。既有说明书截图未替换。
