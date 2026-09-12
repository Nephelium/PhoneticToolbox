# M01 收口与 M02/M09 定向交付

2026-09-11。井井授权只完成 M01 收口、参数显示和语谱图转音频，然后停止。M10 录制版及其并行任务的未提交成果保留，不推进 M03 等模块，不重复 DDL、push、公开发布或生产部署。

## 完成范围

- M01：39 项最终审阅已关闭，见 `m01-final-review.md`。Windows 有界桌面与网页工作流 verified，P08 整体仍 in_progress。
- M02：原参数 XLSX/SQLite 读取、默认一图窗内同区叠加曲线、多个独立图面板、批量分配/合并、搜索/筛选、共同时间轴与播放、主题、SVG 保存，Windows 桌面与 Windows 服务/Chrome 定向 verified。
- M09：有界图片输入、四点校正、标定、Griffin–Lim、持久任务、双图/声音预览、完整四文件结果与原始参数快照，Windows 文件输入和托管 Chrome 工作流定向 verified。桌面主动截图入口已实现，多屏/DPI 捕获设备验收未包含，不声称完整跨平台或正式发行。

## 真实验证

环境为项目内 `.venv/m09-ui`，依赖由 `requirements-m09-ui.lock` 锁定，安装本次 core/API/desktop wheel。独立于既有 m01-ui、m10-ui 和 v2 conda，未改变全局环境。

执行命令：

```powershell
& '.venv/m09-ui/Scripts/python.exe' -X utf8 -m pytest -c tests/pytest.ini backend/tests desktop/tests tests/contracts tests/security packages/phonetic_core/tests tests/parity -q
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run build
& '.venv/m09-ui/Scripts/python.exe' -X utf8 scripts/verify_m02_m09_local.py
& '.venv/m09-ui/Scripts/python.exe' -X utf8 scripts/verify_m02_m09_qt.py
& '.venv/m09-ui/Scripts/python.exe' -X utf8 scripts/verify_m02_m09_web.py
```

本轮安装的wheel回归 Python **472 passed，116.59 秒**（含当时既有 M10/R4，不将其扩展到并行任务后续R5），同图叠加修订后前端 **32 passed**，typecheck/build 通过。新增旧 SQLite 精确显示及非零频带物理频峰用例均已通过。仅有既有 Starlette/AnyIO 弃用警告，未放宽断言或跳过测试。

| 证据目录 `output/validation/m02-m09/` | 已核验事实 |
| --- | --- |
| `local-0dac4626d78741a88b5dce21565a2595` | 本机真实服务/任务/子进程，参数表、WAV/PNG/JSON回读，幂等、防覆盖、服务重启、取消/失败重试，输入与旧数据库行保留，无残留活跃临时文件 |
| `qt-413d47d151a140698a79ec31c80c47f7` | 实际 Qt 15 步，两个参数图窗、原生选择/保存对话框、SVG 文件及完整重建四产物回读，正常关闭 |
| `web-a065f87a8c4b415a977658c1e4127e28` | 真实 PostgreSQL/API/worker、独立 Chrome，实际上传与两窗、四点重建、四份下载哈希与 WAV 回读、刷新恢复、跨用户拒绝，1440/1000/390宽及浅深主题 |
| `web-db0f0fcd262443d99d3820e30673b81a` | 最终共同前端与新wheel复验通过；增加第三图窗、合并再分配、旧结果标定变更提示、图窗2实际截图库与像素断言，下载哈希/音频回读与账号隔离再次通过 |

## 图窗尺寸问题与修正

井井指出图窗 2 过小，同时截图中图窗 1 曲线也被压缩。根因为公共 `svg { width:20px;height:20px }` 图标规则，新参数图只覆盖宽度，空窗仅一行文字。修复为独立图面高度、响应式 viewBox、固定可读字号、空图面至少 280 CSS px。当时Qt/Chrome新增像素高度断言，但两窗630×290 px的堆叠设计仍不符合用户同图语义，不能作为最终交付依据。

首次 Qt 测试 `qt-f8f0871432c54053bbb21e0675d50b60` 超时于第5步。它在同一 JavaScript tick 连点两个 Vue checkbox，第二次覆盖尚未刷新的选中数组；测试现按独立用户动作逐步等待，不算成功证据。此次 UI 问题也说明仅确认 SVG 存在不够，新的尺寸断言已补入。

## 科学与安全边界

- v2 起点 0 的数值与 PCM16 WAV 通过三组固定种子/尺寸/采样率精确对照，没有容差扩大。局部 RNG 不改变全局随机状态。
- 非零频率起点的旧缺陷独立修复与标记。真实/请求时长、采样率、FFT/步长/窗、种子、削波数进入 JSON，原标定改变时不伪装旧结果已更新。
- M02 XLSX 禁公式/外链，SQLite 只读内存导入，受限子进程；M09 图片解码在约1GB/240秒 Windows Job 中，前端先校验图片头尺寸。网页输入/临时/结果复用配额与 owner/到期/fencing/原子发布。
- 结果重建只代表图像近似声学重建，不代表原声音、全部科研有效性或生产负载测试。桌面合成声音经解码/播放控件验证，不代表所有声卡设备。
- 来源在 `third_party/source-registry.json` 与生成的模块致谢同步。SoundFile 0.13.1 / OpenCV 4.13.0.92 实际锁定；原生 wheel 的完整再分发许可审计保留未决项，没有把论文引用当作代码授权。

## 使用与停止点

开发入口：`scripts/Start-Research-Workbench.ps1`，复用已审阅的现有本地状态库，不执行迁移。操作见 `docs/manual/parameter-display.md` 和 `docs/manual/spectrogram-to-audio.md`。

本轮到这三个指定任务结束。本任务不再修改M10；其他模块与跨平台打包/公开发行均等待井井下一项明确指令。

## 同图叠加纠正与最终复验

井井随后澄清同一图窗是曲线画在一起。此前独立纵轴子图的设计和对应完成声明撤回。重新核对说明书2.2全文、图2-9/2-11与旧`_plot`，按同一坐标区叠加、共享纵轴及旧50倍/100量级自动双轴规则修正。波形单独位于上方，文字边界与标注同步波形和参数区。原值/原帧不改变、不归一化。

- `qt-d4c046f5a44548549d92eba2a95e8d4a/report.json`：15步全部通过，默认三曲线同绘图区、第二窗两曲线同绘图区、实际SVG及四份重建结果保存回读、正常关闭。
- `web-d0d309f9b88c48cd84c3094a9450852b/report.json`与`web-report.json`：每窗一个shared-plot-area、曲线共用刻度/图例、波形标注、滚轮缩放/左拖平移/双击复原、三窗批量分配/合并、主题/窄窗/下载/账号隔离通过。PNG截图已人工查看，`m02-default-overlay.png`和`figure-2-readable.png`为最终叠加效果。
- 实际SVG画布630×433和630×410 CSS px，文字包围高度16 px；共同曲线坐标区高300 px，空窗最小300 px。
- 新增3项自动双轴/原帧统计/共用尺度回归，前端32项通过，typecheck/build通过。旧堆叠图截图仅留作历史问题证据。

交互/导出适配差异见M02-source-map：复选框与显式批量分配替代旧即时多选绘图，列宽滑块替代拖分隔条。当前导出参数图SVG，不包含上方波形，不声称已提供旧整幅PNG导出。M09截图设备与跨平台验收仍保留原限制。

收尾静态检查：check_architecture.py 返回 errors=[]，contracts:check通过，git diff --check 无空白错误。ui-data:check首次提示来源版本漂移，定位为并行M10任务将PROJECT-M10由R4更新到R5；按当前登记重新生成公共致谢后检查和构建通过，未修改该任务源码/EXE，也未将既有wheel回归扩展为R5验收。
