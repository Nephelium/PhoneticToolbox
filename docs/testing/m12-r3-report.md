# M12-R3 编辑、绘图联动与 P04 页面缩放

2026-09-14。状态 verified（限定本机 Windows、独立 Chrome、实际 Qt 及下列 R3 单文件 EXE）。范围包括井井本轮初始修改与两次追加反馈，不推进其他模块迁移。

## 实现

- 手工边界移动和插点取消原 20/15 ms 门槛，删除操作撤去 5 ms 匹配容差。密集边界选择最近一条，删除精确相邻边界，拖动后键盘选择跟随新边界。保留正时长、顺序与 TextGrid 六位小数精度。只有原本重合的词/音素边界联动，独立近邻不再被 40 ms 搜索吸附。
- 清空/删除按钮移除，Backspace 清文本，Alt＋Backspace 合并选中边界，Ctrl＋Z 撤销。输入框内仍按正常文本编辑处理。
- 普通模式可在已有 TextGrid 的空白音节区或波形双击起终点，建立标注并输入文字，无需词表。顺序模式保留原拼音推进、Ctrl 连续和声母/完整韵母切分。
- 首次音素切分可选首个边界在双击处，或按音素数等分，默认点击位置且设置记忆。多于两音素时后续等分剩余时长。音素层已有内部边界时只增加点击位置边界。旧主动音素自动填充仍为词典均分。
- 新自动保存采用 `_自动保存.TextGrid`，原 `_webedit` 可读并保留，新版本优先。升级旧默认偏好，自定义后缀保留；中文目标仍做源/目标版本冲突检查。
- M12 波形与语谱 canvas 同为 175 px。波形接入 Shift＋滚轮，三图共用时间窗；短视窗按真实采样位置连接原数据，长视窗保留峰值包络。M12 通过可选属性启用，其他模块默认绘图/手势保持。
- 语谱图支持左键前向/反向拖动与鼠标捕获，选区同步波形/TextGrid，空格试听该范围。选择操作不修改标注。
- 整个 v3 通过设置的外观区域调整页面 70%–150%，每步 10%，可恢复 100%。阻止主文档及同源嵌入文档的 Ctrl＋滚轮整页缩放，同时图内事件继续分发。公共弹窗按实际缩放后视口限制尺寸，标题/关闭/底部操作保留。未修改声学 RMS、FFT、强度贴合算法、采样或播放增益。

## 实际验证

| 命令/入口 | 结果 |
| --- | --- |
| `npm --prefix frontend run typecheck`、`test`、`build` | 类型检查和生产构建通过，103 项前端通过 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini desktop/tests backend/tests/test_frozen_workers.py backend/tests/test_local_workspace.py tests/contracts tests/architecture -q` | 134 项通过 |
| `node tests/e2e/m12.cjs` | 原 18 组通过，包含原语料哈希、保存冲突、IME、引用复用与唇偏 |
| `node tests/e2e/m12-r1.cjs` | 6 组短录音顺序/框选/微调回归通过，本轮没有复跑外部长录音路径 |
| `node tests/e2e/m12-r2.cjs` | 12 组布局/实际层名/空文件/选区试听/自动振幅/主题及小窗口通过 |
| `node tests/e2e/m12-r3.cjs` | 13 组通过，见下列证据；实际鼠标、键盘、保存/回读和 WebAudio 参数检查 |
| `node tests/e2e/m03-overview.cjs` | 3 组公共波形默认布局/双声道/缩放回归通过 |
| `scripts/research_entry.py --local-root <独立 state> --verify-m12-r3 <results>` | 实际 Qt 50 步通过，原生 Ctrl/Shift 滚轮、真实鼠标语谱拖选、中文保存/下载、首次切分选项与精确回读 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/build_m12_preview.py` | PyInstaller 单文件构建通过，171.7 秒 |
| `scripts/run_research_repair_check.py --exe dist/m12-preview-r3/PhoneticToolbox-v3-M12-R3.exe --verification m12-r3` | 实际冻结 EXE 50 步与保存/下载回读通过，退出码 0，无可见测试窗口，无自有残留子进程，3 个非法/缺失 worker 入口拒绝 |

本轮仅重建并重装项目自己的 `ptb_desktop` wheel 到 `.venv/m09-ui`，`--no-deps`，未升级第三方或全局环境。首次 Python 回归的 1 项失败来自该环境仍安装 R1 默认后缀，更新本项目 wheel 后 134 项全部通过。

## 原始证据与修正过程

- `output/validation/m12-ui/af49569932fb45ccaf9a7683d794950b`：原 18 组。
- `output/validation/m12-ui/225e95986e6149c1a369c05f60766be5`：R1 六组。
- `output/validation/m12-ui/27825547d98c428b879b8aea85bc8c83`：R2 十二组。
- `output/validation/m12-ui/795927f4ab554e0bb7fd5c7126a7fc34`：最终 R3 十三组，0.1 ms 单元保存、110% 实际密集拖动/删除、普通双击、定时/切换中文保存、旧文件优先级、同源嵌入滚轮、70/150% 与 800 px 弹窗、10/40/80 ms 原采样曲线、三图平移/选区/首边界设置/持久偏好/顺序模式。
- `output/validation/m03-ui/chrome-389b6c51ff0546ad8094253748396210`：最终公共波形回归。
- `output/validation/m12-r3-qt/db7a2404a55f4485914a6be800864afc/results`：实际 Qt 50 步及浅深截图。
- `output/validation/m12-r3-v2-hashes.json`：V2 六份原文件 hash 全部不变。
- `output/validation/m12-r3-wheel`：本项目桌面 wheel。
- `output/validation/m12-r3-package.log`：单文件构建日志。
- `output/validation/m12-r3-exe/frozen-87cda2d7fd9748238ff1bbe4af49b8a1`：实际冻结 EXE 及进程结果。测试从系统临时目录启动，PATH 仅系统目录，移除开发环境变量，独立新测试状态。

视觉检查发现最初 150% 设置弹窗标题超出窗口，已修复缩放后的视口计算，并将标题/关闭/底部实际位置加入断言。保留初始图片 `m12-ui/e142c646e03440a89f2934c8a163b39d`，最终小窗口截图见 R3 最终证据。

Qt 验证脚本修正了相对输出路径的文件夹选择、R3 新文字进入音素层后的预期、拼接 JS 缺少分号，以及原生滚轮量与 DOM delta 的区别。当前 Qt 一格轮动产生 `deltaY=60`，最终按实际 delta 严格检查平移公式，不按固定 120 冒认。早期失败保留在 `m12-r3-qt` 的根目录及 `78d4acff...`、`5c6d8b...`、`f343566b...` 子目录。Chrome 新脚本也修正了文本框严格选择器歧义和用放大按钮达到 40/10 ms 的测试步骤，未改变原 80 ms 手工视窗输入门槛。

M03 旧静态声道资源的 Vite `three` 预扫描提示和 Qt 原 PNG profile 提示仍存在，实际定向检查与生产构建通过，未绕过检查或修改资源来消除提示。

## 边界

沿用 ORIGIN-WEBEDITOR、PENDING-DICTIONARY、SRC-PRAAT，没有新增第三方来源。R3 为用户要求的交互适配，不扩大为公开发行许可确认。未改 v2/原研究数据，未改现存数据库、全局依赖、CI、生产服务或 push。旧 R1/R2 EXE 保留。

证据限此 Windows/Chrome/Qt 环境。未复验实体声卡听感、所有 DPI/跨平台或生产服务。临时包范围沿用 M12 试用包，不包含 M03/M04 独立兼容运行环境；原长录音能力参考 R1/R2 报告，本轮冻结检查使用合成短录音。

## 本机临时包

`dist/m12-preview-r3/PhoneticToolbox-v3-M12-R3.exe`，326663305 字节。

SHA256：`b093da7f952f72ad5b92aec747c3ee438a6429dfe5679d18f2e465a306968866`。

同目录附使用说明。R2 原包 hash 复核为 `5b67817ceb08bf82eca351c7eeb7c0b64388c21f204d71bfb66a31b4b8614006`，未变。
