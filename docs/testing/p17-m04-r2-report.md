# P17 M04-R2：紧凑布局与直接 PNG 保存

2026-10-01。限定 Windows 源码工作台、真实录音和实际 Qt 保存链路通过。产品基线为 `ea89397`，文档基线为 `e3d02cc`。本报告覆盖井井追加的 M04 图高、右栏操作和 PNG 保存反馈，不扩大为其他模块、托管网页或跨平台验收。

## 根因与修改

原波形高度使用 `100dvh - 430px`，LPC 容器使用 `100dvh - 235px`，大屏时两种视图都接近整屏拉伸。现在分别按 `28dvh` 和 `58dvh` 增长，保留可读的最小高度与小窗规则。时间起止、清除选区、开始/取消分析和预算提示集中到右栏谱图参数内，左栏保留文件、标注和试听。没有修改 LPC 计算、采样边界或 48,000 样本预算。

读取 V2 实际实现：`phonetic_toolbox/gui/widgets/lpc_spectrum_widget.py` 的分析结束流程调用 `_create_export_figure`、`LpcService.save_plot_figure`，自动写入输出目录或源文件目录；图为 8×4.5 英寸。`phonetic_toolbox/services/lpc_service.py` 使用 `fig.savefig(..., dpi=300)`，得到 2400×1350 PNG。V3 本机任务端提供 `result/saveJob`，没有浏览器任务端的 `download` 方法；旧页面因此只显示完整目录保存，缺少专门的 PNG 入口。本轮没有复现完整目录保存失败，不能将其声称为已定位的同一故障。

新增“保存 PNG 图片”，读取该任务已生成并校验哈希的 PNG，验证 PNG 签名后交给现有 Qt `blob:` 下载保存窗口。保留完整 PNG/WAV/JSON 目录保存。导出保存任务快照，不依赖当前图形缩放或窗口尺寸；取消不写文件，页面只提示打开保存窗口，不提前声称写入成功。未新增公共桌面或后端接口。

另将本页 `.spectrogram-view` 的 `margin` 改为 `margin-block`，避免覆盖公共横向对齐。波形/语谱/TextGrid 的公共坐标修复及辅助测试由统筹负责，见 [共享报告](p17/shared-m04-r2-report.md)。

## 最终证据

Chrome：`output/validation/p17-m04-r2/def71954a0b14c6caaa3be3003237b8c/report.json`，exit 0，errors/warnings 均空。真实录音为批准目录内的 3.849 秒、44,100 Hz 单声道文件及同名原始 TextGrid，实际分析 0.2–0.4 秒。该轮任务提交至可见结果为 5.343 秒，仅为本次墙钟实测。

| CSS 视口 | 波形 SVG 高 | LPC SVG / 内部绘图区高 | 开启语谱图后模块 / 中区溢出 |
| --- | ---: | ---: | --- |
| 1920×1000 | 280 px | 493 / 431 px | 无 / 无 |
| 2560×1360 | 380.8 px | 702 / 640 px | 无 / 无 |
| 3840×2080 | 582.4 px | 1119 / 1057 px | 无 / 无 |

以上为模拟 CSS 可用视口，未冒充三种实体屏幕验收。开启语谱图的截图为同目录 `M04-spectrogram-{1920,2560,3840}.png`，结果图为 `M04-result-{1920,2560,3840}.png`。1920 结果态右栏有 42 px 的内部溢出，任务记录可滚动，主区和常用分析操作保持单页；1280×720 小窗可滚动，见 `M04-small.png`。人工复核 1920 语谱图及 3840 结果截图确认布局。

统筹辅助检查覆盖 12 组真实模块坐标：三档视口各 1×/8×/16× 缩放和平移、原始 TextGrid 区间点击、较大图表字体与 88 px 轴留白、Praat 拖选及 Ctrl 滚轮。波形/标注/三条时间轴外边界完全相同，canvas 边框造成 1 CSS px 内缩；选区与真实标注边界最大差小于 0.29 px，波形与语谱选区最大差小于 0.025 px。检查阈值保留原辅助脚本的 1.05 CSS px，未放宽。该轮完成后统筹又统一了语谱与标注时间文字的单位和小数精度；这两处最后模板微调未包含在本轮截图中，已由统筹在公共最终模板补测中验证，成品文件核对与 PNG 操作见[EXE 报告](2026-10-01-p17-m04-r2-exe-report.md)。

| 操作 | 实际结果 |
| --- | --- |
| 右栏起止输入与开始分析 | 左栏无重复时间输入；右栏 0.2–0.4 秒提交真实 LPC 任务成功 |
| 直接保存 PNG | Chrome 接到真实 PNG 下载，文件实际写出 |
| PNG 读取失败后重试 | 单次受控读取错误可见；解除注入后真实读取成功，未伪造结果字节 |
| 完整结果保存 | 原生目录保存 PNG、WAV、JSON 三个文件仍成功 |
| 三种 PNG 回读 | 直接下载、错误恢复下载与完整结果 PNG 的 SHA-256 完全一致 |

实际 Qt：`output/validation/p17-m04-r2/705aab9348aa4f3fbb54d145fbaa7ae4/qt-report.json`，exit 0，success=true。运行真实源码 Workbench 与已构建前端，隐藏窗口避免干扰用户桌面；只替换本次测试进程的文件对话框选择值，不替换保存、下载或计算实现。

- 首次 PNG 对话框取消，实际没有图片落盘；第二次确认后收到 `DownloadCompleted`。
- 实际文件 `saved/R2-direct.png` 为 133,715 字节，2400×1350，白底，读回约 299.9994 DPI；与完整结果 PNG 逐字节一致。
- SHA-256：`351fe3398ed4e0379860937a1d743cdb3db17cb364424efa5ad26bd233452507`。
- Qt 测量为 1920×1000 CSS、DPR 1.5，模块 `clientHeight=scrollHeight=920`；系统 screen 报告 1707×1067，不将此当实体 1920 显示器证明。
- Chrome 与 Qt 各自的 `originals-unchanged.json` 均为 true。输入仅由指定真实目录逐字节复制，全部产物和新建隔离测试库位于本轮输出目录；未改原音频、原 TextGrid、V2 或现存数据库。自有进程已退出。

## 实际命令与边界

1. `node tests/e2e/p17-m04-r2.cjs`：最终 exit 0，含上述 12 组坐标、布局及真实保存。
2. `.venv/m14/Scripts/python.exe -X utf8 scripts/verify_p17_m04_r2_qt.py`：最终 exit 0，Qt 取消与真实 PNG/完整结果回读通过。
3. `node --test frontend/tests/lpc-state.test.ts`：5 项通过。
4. `npm --prefix frontend run typecheck`：通过；统筹后续统一构建、类型检查及 191 项前端测试也通过，详见[EXE 报告](2026-10-01-p17-m04-r2-exe-report.md)。
5. `.venv/m14/Scripts/python.exe -m py_compile scripts/verify_p17_m04_r2_qt.py`：通过。
6. `git diff --check -- frontend/src/modules/lpc-spectrum docs/manual/lpc-spectrum.md docs/testing/p17-m04-r2-report.md docs/testing/p17/M04-report.md docs/testing/p17/M04-rules.md`：通过。

保留中途失败证据：`c0e2910b41ab4eaba4e06a537df01362` 在字体变化触发语谱重新读取时，辅助脚本提前取 canvas；补充等待后最终 12 组通过，未改坐标阈值。Qt `931d1f6f84764fab8bb1c0d77d304a88` 实际保存已成功，但测试将 `st_size` 属性误写为函数导致报告失败；修正测试并从新目录完整重跑通过。两次中途运行均不计最终通过。

本轮子代理未打包、提交或 push；最后两处公共时间文字微调及最终 EXE 由统筹单独记录。网页认证、Linux/macOS 原生保存、本机实体多屏与物理听感不在本轮证据范围。
