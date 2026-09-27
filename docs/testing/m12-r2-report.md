# M12-R2 · 层名、布局和三图联动

2026-09-14。**verified，限定以下 Windows 开发态及临时 EXE 实测范围**。对应井井六张试用截图与明确修改要求，计划见 [M12-R2](../plans/2026-09-14-m12-workspace-r2.md)。

## 已实现

- 文件列表按钮不再被 flex 纵向压缩，只显示音频名，长名称省略并保留完整悬停提示。绿色及左边线表示有关联 TextGrid，顶部有图例。多个歧义候选改在点击后的对话框中选择。
- 删除编辑器内部嵌套的 main，保留单个工作区滚动与文件列表滚动。无需滚动整个外层页面。自动生成选项合并到词典/词表工具栏，并解释词典映射与词表顺序的区别。
- 两个层名选择器读取文件中所有实际层名，点层/非全域层标注类型并保留。过期的 words 偏好不再产生找不到层。缺文件、空白文件和零层文件只加载音频，由使用者输入名称显式创建层，支持重复名校验、追加单层及撤销；创建前不自动落盘。
- 音频总览/平移滑块与音量移至波形上方，去掉底部播放条与选区输入。空格/P 播放当前选区，再按暂停、续播至选区末尾。未选区时提示选择，改变选区后从新起点播放。
- 波形选区、单个音节/音素和多音节范围共用 Workspace start/end，语谱与标注层同步高亮。重新拖选后清除过期的编辑器选择，避免之后的工具操作把范围改回旧区间。
- 波形叠加当前音节/音素边界。按实际可见采样峰值缩放振幅及正负 Y 轴刻度；静音有有限显示范围，低幅度使用科学记数，音频样本与播放增益保持独立。三图使用同一左轴留白。
- 公共波形的顶部总览和振幅轴均为显式可选属性。保留默认 SVG 直接子节点结构，使现有参数显示整幅导出继续捕获波形。

## 验证

环境：Windows、Node 24.13.0、项目 `.venv/m09-ui` Python 3.11.14；未安装或升级依赖。

| 命令 | 结果 |
| --- | --- |
| `npm --prefix frontend run typecheck` | 通过 |
| `npm --prefix frontend test` | 94 passed，含层名/显式创建/整组选区/实际振幅及原有功能 |
| `npm --prefix frontend run build` | 通过 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini desktop/tests backend/tests/test_frozen_workers.py backend/tests/test_local_workspace.py tests/contracts tests/architecture -q` | 134 passed |
| `node tests/e2e/m12.cjs` | 原 18 组通过，含自然录音、全部参考模式、IME、自动保存、保存期间编辑及冲突保护 |
| `node tests/e2e/m12-r1.cjs`（传入授权 prepared 目录） | 8 组通过。按 R2 显式建层后复验完整用户拼音序列、内部切分、Ctrl 连续/普通新起点、鼠标整组拖动、1 ms/2.5 ms、长录音完整保存及原文件 hash |
| `node tests/e2e/m12-r2.cjs` | 12 组通过。51 份录音布局、真实层名与点层、空白自建/撤销/重名/单层追加/歧义关联、三图对齐、实际 WebAudio 源节点范围与暂停续播、0.02/0.8 视窗峰值、浅深色及 1280/1000/800px |
| `node tests/e2e/m03-overview.cjs` | 3 组通过，EGG 默认波形在上、总览控件在下与双声道/缩放语义保持 |
| `node tests/e2e/m02-png.cjs` | 3 组通过，5 份真实 PNG 下载逐块 CRC/300 dpi/背景及波形整幅导出通过 |
| `scripts/research_entry.py --local-root <独立 state> --verify-m12-r2 <results>` | 实际 Qt 28 步通过，原编辑/下载/唇偏回读、显式中文层名、顺序标注和 1 ms 保存、三图坐标/选区一致 |
| `scripts/verify_m12_long_qt.py --source <用户 prepared 目录>` | 实际 Qt 6 步通过，13784432 帧及尾部显示、全部四层回读，原始文件 hash 不变 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/build_m12_preview.py` | PyInstaller 6.22.2 单文件构建通过 |
| `scripts/run_research_repair_check.py --exe <R2 EXE> --verification m12-r2` | 真正冻结 EXE 28 步与保存/下载回读通过；临时目录、系统 PATH、移除开发环境变量后启动；退出码 0，无可见测试窗口，无自有残留子进程，3 个非法/缺失 worker 入口拒绝 |

## 原始证据

- `output/validation/m12-ui/5cd694aed363438ba2d366ffd5481912`：原 18 组。
- `output/validation/m12-ui/129972de97de4bfca788695ab2749ed2`：顺序/整组/长音频 8 组，长文件一次加载 3.66 秒，仅表示此机此轮测量。
- `output/validation/m12-ui/0690729d55e34fd59c55eba50be22db1`：最终布局及联动 12 组、浅深色和小窗口图、文件几何与单滚动容器数据；波形拖动后直接按空格验证，不额外设置键盘焦点。
- `output/validation/m03-ui/chrome-19a1eb58878d4aa5a8fb587305a6a51c`：公共 EGG 布局回归。
- `output/validation/m02-png/chrome-1789384738694`：参数显示完整 PNG 回归。
- `output/validation/m12-r2-qt/bf285d33ad6643ae8a72677882d5bfa3/results`：最终实际 Qt 28 步与中文输出。
- `output/validation/m12-r1-long-qt/745f85806a3b4d8c907907f3f54170d5`：当前 R2 构建的长录音 Qt 6 步。
- `output/validation/m12-r2-v2-hashes.json`：V2 六份来源 hash 不变。
- `output/validation/m12-r2-exe/frozen-8f72cacd42ea48279b9d7d69a72ce68e`：冻结 EXE 28 步与进程检查。
- `output/validation/m12-r2-package.log`：完整构建日志。

Qt 首轮最后一步 `stage_timeout_27` 来自测试把理想点击时间写死为 0.701/1.101。Qt 模拟 MouseEvent 将小数像素坐标取整，实际选段为 0.700703–1.099117。修正验收为语谱选区逐值等于实际标注区间，仍严格检查保存后的 0.001 秒平移。失败证据保留在 `m12-r2-qt/edfdf47ad76f48d69ef92139cae37d77`，没有放宽平移精度。

M03/M02 的测试 Vite 自动入口扫描报告旧静态声道资源的裸 `three` 导入无法预打包，页面按现有配置运行，实际检查通过；生产前端构建通过。Qt 仍输出原有 PNG profile 提示。没有为消除提示修改资源或依赖。

## 来源和边界

沿用 ORIGIN-WEBEDITOR、PENDING-DICTIONARY、SRC-PRAAT。新增显示/交互为用户要求的适配，无新增第三方来源；原科学算法及 v2 保留。未执行现存库 DDL、push 或公开发布。测试只使用独立本机状态及已授权的自然录音副本。

测试验证播放节点实际选区与增益控制，未评价实体声卡的听感或音频设备延迟。当前桌面与 Chrome 证据不代表生产网页、跨平台或所有 DPI 设备通过。临时包范围仍不包含 M03/M04 独立兼容运行环境。旧 EXE 保留。

## 临时包

`dist/m12-preview-r2/PhoneticToolbox-v3-M12-R2.exe`，326658041 字节。

SHA256：`5b67817ceb08bf82eca351c7eeb7c0b64388c21f204d71bfb66a31b4b8614006`。

R1 原包哈希复核仍为 `5da4988f2d87bf75a900aa098593f3d1a6286025bb0449c86f344e18c536efb2`。冻结 EXE 使用受控短录音验证；长录音证据来自当前前端构建的实际开发态 Qt 和 Chrome，二者范围分开记录。
