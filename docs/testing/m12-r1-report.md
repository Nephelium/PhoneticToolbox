# M12-R1 · 空白标注、顺序标注和整体微调

历史 R1 记录：空白文件自动建立两层和旧播放布局已被井井后续要求的 [M12-R2](m12-r2-report.md) 显式命名建层及共享选区替代（Superseded）。本页保留原验收和 R1 EXE 的证据。

2026-09-14。状态：**verified，限定 Windows 本机开发态及下述临时 EXE 路径**。只处理井井在 M12 试用中反馈的功能，其他模块保持原检查点。

## 原因与实现

- 无 TextGrid 的 WAV 原先被页面列表排除，空文件直接进入严格格式解析。现在 WAV 均可打开，缺文件/纯空白/合法零层文件在内存建立真实两层，首次保存以绑定的 WAV 版本校验。非空损坏文件保留错误，结束加载提示。
- 井井指定的 prepared 录音是中文文件名的 FLOAT32 单声道 WAV，44100 Hz、13784432 帧、312.57215419501136 秒、55137784 字节。标注总域匹配，包含 sentences/recovery_status/words/phones 四层，分别 85/85/1147/2236 项。原先 `_初始定位` 未在匹配规则中，M12 的 800 万帧上限也阻止载入。现增加唯一同名前缀匹配和有歧义时的关联选择，M12 使用已有公共 64 MB/3200 万采样值预算。未改变 WAV 解码样本或标注时间。
- 新增显式启用的拼音顺序编辑：普通双击起点→终点，两层外边界一致；内部双击按点击时间分声母/完整韵母；Ctrl＋双击沿上一音节实际终点继续。可粘贴/上传词表、选择下一条、Esc 取消起点、撤销恢复进度。人工切分后单区间改字不均分时间。普通模式原填充规则保留。
- 新增一次框选工具及 Shift 框选。鼠标整体拖动或左右键微调移动词和对应音素，默认 1 ms，可设 0.001–1000 ms。平移不缩放内部时间，整体校验越界/未选标注重叠/跨越所选外边界的音素，失败整次不修改。拖动和键盘共用平移规则。单击选中不再产生无效撤销步骤。

## 已执行检查

环境为 Windows、Node 24.13.0、项目 `.venv/m09-ui` Python 3.11.14。只重建/重装自身 desktop wheel，未修改第三方依赖或系统运行库。

| 命令 | 实际结果 |
| --- | --- |
| `npm --prefix frontend run typecheck` | 通过 |
| `npm --prefix frontend test` | 90 passed，包括新增空白、顺序、整组微调和手工边界保留用例 |
| `npm --prefix frontend run build` | 通过，当前共同前端已更新 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini desktop/tests backend/tests/test_frozen_workers.py backend/tests/test_local_workspace.py tests/contracts tests/architecture -q` | 134 passed，包含 M12 原生版本保存及绑定 WAV 首次保存的检查 |
| `node tests/e2e/m12-r1.cjs`，传入用户指定的长录音目录环境变量 | 8 组独立 Chrome 实际页面检查通过。包含用户给定完整拼音序列、双击/内部切分/连续段/独立起点、框选/拖动、1 ms 与 2.5 ms 微调、失败保留、空文件/零层/非空损坏，以及真实长录音的四层完整保存 |
| `node tests/e2e/m12.cjs` | 原 18 组 Chrome 回归通过，含拖动联动、参考四模式、试听、自动保存、输入法/迟到保存保护及原自然语料 |
| `scripts/research_entry.py --local-root <独立测试 state> --verify-m12-r1 <results>` | 已安装 wheel 的实际 Qt 22 步通过。原 TextGrid/唇偏及两类下载回读通过，追加空白 TextGrid、拼音序列、点击切分、连续下一音节与 1 ms 键盘微调保存回读通过 |
| `scripts/verify_m12_long_qt.py --source <用户指定 prepared 目录>` | 实际 Qt/QWebChannel 6 步通过，55 MB FLOAT32 录音完整读取及末尾显示，保存四层后逐字段对照（数值采用原 TextGrid 的 1 微秒格式精度），原文件哈希一致 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/build_m12_preview.py` | PyInstaller 6.22.2 单文件 EXE 构建通过 |
| `scripts/run_research_repair_check.py --exe <R1 EXE> --verification m12-r1` | 实际冻结 EXE 在系统临时目录、仅系统 PATH 且移除 Python/Qt 开发环境变量启动，22 步及全部输出回读通过，退出码 0、自有残留子进程 0、三个非法/缺失 worker 入口拒绝 |

原始证据：

- `output/validation/m12-ui/466f37c03d74432b87d46086a0fd0675`：新增 8 组与长录音首尾/浅深色/小窗截图。该次长录音加载 3.49 秒；这是一次本机测量，不作为其他设备的性能承诺。
- `output/validation/m12-ui/9e7d9ad4ef924324a2ea1cd3fd1e76a8`：原 18 组回归。
- `output/validation/m12-r1-qt/3afbe1db32604020a21608769c0f053b`：22 步已安装 wheel 的实际 Qt。
- `output/validation/m12-r1-long-qt/570d5fdb89fc433a94de467ef975f226`：长录音实际 Qt 的 6 步与完整文档回读。
- `output/validation/m12-r1-exe/frozen-2d0a533b52814850a8e32174f7c93568`：真正冻结 EXE 的 22 步与自有进程生命周期。
- `output/validation/m12-r1-{frontend,python,build-frontend}.log`：命令输出。

首次新增浏览器用例揭示了数值输入的 Vue 自动数字转换：默认字符串步长能用，修改后变成数值，原 `.trim()` 校验报错。已修为先转字符串验证空白、再作有限数值校验；实际 2.5 ms 用例重跑通过。失败记录保留在 `output/validation/m12-ui/0e09dd64dc104a49b91106511f7dd166`，未放宽步长精度检查。

## 来源与边界

沿用 ORIGIN-WEBEDITOR、PENDING-DICTIONARY、SRC-PRAAT 登记。本次为用户明确提出的新交互和编辑规则，没有引入外部库或新算法来源。相邻 V2 的 6 份来源文件按迁移清单重新核对，哈希全部一致。用户指定的长录音仅复制到忽略的测试输出目录处理，原 WAV/TextGrid 哈希复核不变。测试选点在真实 UI 受像素定位精度影响，数值算法单测用确切秒数，保存后的 1 ms/2.5 ms 检查采用 TextGrid 六位小数格式的微秒精度。

未扩大到生产网页/跨平台/所有录音规模。临时包仍不包含 M03/M04 的独立兼容运行环境，不代表其他模块发行验收。未执行现存库迁移、修改 v2、覆盖旧 EXE 或 push；仅使用原已授权的隔离测试工作区初始化机制。

## 临时包

`dist/m12-preview-r1/PhoneticToolbox-v3-M12-R1.exe`，326651083 字节。

SHA256：`5da4988f2d87bf75a900aa098593f3d1a6286025bb0449c86f344e18c536efb2`。

原 `dist/m12-preview/PhoneticToolbox-v3-M12.exe` 及此前其他包保留。长录音已在安装环境的实际 Qt 验证，冻结 EXE 验证使用受控短录音；不把这两种证据混为同一运行。
