# 三端共享生理参数合成验收

日期：2026-10-18。任务：M10-SHARED-SYNTHESIS。语义与授权见 [ADR](../decisions/ADR-M10-shared-synthesis-controls.md)。当前结论为共享源码、Windows 原生/Qt、独立 Chrome Wasm、macOS ARM64 原生通过定向验收。最终 EXE、Mac 应用、公开网站未更新，不能引用本报告代替发行验收。

## 交付行为

- 同一前端/核心/C++ 补丁实现模型切换、IPA 时长分类、全局及区段特征、独立上下唇、闭塞/释放/送气/正负 VOT、构型包分享。Windows DLL、Wasm、Mac dylib 分别构建为 m10/r20。
- 男性 JD2 / 女性 W02 使用原 speaker 解剖，新姿态 F0 为 120/200 Hz，旧文件已有 F0 保留。构型、常用项、序列按模型隔离。共同显示头壳不代表 W02 MRI。
- 交换文档写 v8、读取旧 1–7 版并提示按当前几何重算；构型包写 v2、验证模型/几何/speaker SHA/参数顺序，预览后按选择保留或替换同名构型，立即供 IPA 调用。新包没有本机 ID 或可执行路径。
- 帧范围特征使用稳定 ID，跟随排序；时间范围保持秒数。全局先作用，相对偏移相加，最后添加的固定值优先，然后加偏移并限幅。原始姿态不重写。
- 曲线空态提供添加姿态入口，非空可实际绘制/撤销/放大。短试听等待姿态完成，修复浮点插值边界溢出。桌面/Wasm 完整播放和视频准备上限 60 秒；受限服务器科学回退仍为 5 秒。

## 实际验证

| 平台/入口 | 结果及范围 |
| --- | --- |
| Windows x64 原生、核心、桌面进程、Web 接口定向 pytest | **146 passed**，87.67 s。两条已有 Starlette/AnyIO 弃用警告，不新增依赖。含模型/存储/文件、声源/抖动、舌/牙齿/气腔、源更新、监测、播放、长缓冲及新增时序/特征/唇部测试。 |
| macOS ARM64 原生真实编译 | **19 passed**，33.57 s。共用 shared-controls 与 gestures 测试，包含 60 秒完整声音/画面、W02、正负 VOT、模型不同唇部位移、摩擦噪声、文件/区段特征。最低目标 macOS 14.0；不是 Mac Qt 或最终 app 验收。 |
| Windows 独立 Chrome、真实 Wasm/指针/音频调度 | **17 组通过**。实际上下唇/舌面下压与反向/舌叶/舌底拖动、重复短试听、双曲线绘制和撤销、重叠区段、连续修改全局值、v8 导出、男女切换恢复、构型子集导出/导入预览/保留/替换/IPA、负 VOT 编辑、60 秒和取消恢复。 |
| Windows 实际 Qt/原生 IPC/文件适配 | **3 组通过**。独立唇部、原生构型包 v2 文件读写、IPA 塞音/v8 特征完整导入导出、模型隔离和 120/200 默认、浅深主题。测试明确锁住物理音频设备，不能当作 WASAPI 听辨证据。 |

Windows 命令使用 `.venv/m14/Scripts/python.exe -E -s -B -X utf8`，先 `scripts/workbench_source.configure()`，运行 `pytest -c tests/pytest.ini -q`，文件集合为 `tests/parity/test_vocal*.py`、`test_m10*.py`、`backend/tests/test_vocal*.py` 和 `desktop/tests/test_m10*.py`。八个新增关键验收位于 `tests/parity/test_vocal_gestures.py`。Mac 使用任务已有独立环境与候选覆盖路径，没有修改原 Mac 源码交接树、系统或全局依赖。

浏览器入口：`scripts/verify_m10_wasm.py --work output/validation/m10-shared-20261018/source-build --science output/validation/browser-compute-20261017/browser-science --frontend output/validation/m10-shared-20261018/frontend-final --vocal-runtime output/validation/m10-shared-20261018/wasm-r20/vocal-wasm --shared-controls --port 18785`。实际浏览器运行使用独立进程/配置，不调用 Codex 内置浏览器。Qt 入口：`scripts/verify_m10_shared_qt.py`。

前端类型、14项定向测试、契约/UI数据一致性、514模块构建通过。架构与文档检查零错误（2296文件、447来源、162任务），未修改历史快照断链。资源登记71项及重复执行无差异通过，修复登记器对同轮新bundle摘要读取旧值的问题。统一入口 `scripts/Start-Research-Workbench.ps1 -Module M10` 已启动当前源码工作台，实际加载正式 R20 DLL，证据为 `output/validation/m10-shared-20261018/source-workbench.json`。

## 数值与科学边界

- Chrome 固定同一姿态从 120 改到 300 Hz，播放 PCM 稳态 RMS 为 0.0668623 / 0.0692936，差 **0.31024 dB**。Windows/Mac 两模型 a/i/u、60–350 Hz 矩阵的最大电平跨度均 <1 dB。测试同时核对气压/微扰独立性与分析计算不打断原生声源状态。时变补偿使用锚点近似，没有证明任意姿态的感知响度完全一致。
- 浏览器 60 秒固定姿态产生 **2,880,000 采样、1,801 幅画面**，最后一次准备 **29,090 ms**，完整末帧和播放停止通过。长计算取消会重建 Worker 并恢复已经确认保存的 7 帧/模型/曲线/特征；测试通过。此性能只适用于本机固定姿态，复杂连续构音会更慢。
- 闭塞不自动归零。原生闭塞期预发声仍保留，显式静音帧继续严格静音。t→a 测试保持闭塞至 59 ms、68 ms 后达到释放目标，释放区间 RMS 超过闭塞区间两倍。负 VOT 测试在闭塞几何不变时开启振动。字段定义是声门指令相对释放起点，实际声学 VOT 与爆破可辨度需另测。
- JD2 下唇后收 0.6 cm/抬高 0.6 cm，W02 后收 0.6 cm/抬高 0.8 cm，均形成最前端约 **0.15 cm²** 狭窄；清声输出 RMS >0.0001，1 kHz 以上功率占比 >85%。这是可调唇齿狭窄及原生摩擦输出证据，未做 /f/ 盲听认证。零偏移几何保持，上唇/下唇单独移动时对侧唇面不变。
- 舌面下压和舌叶原生光滑连接进入面积/声学。舌底仍是原有显示重建，本次修复可见锚点与拖动死区，不声称建立新的舌腹组织力学模型。

## 修复与保留证据

测试实际发现并修复：短试听浮点越界、零时长绘图区不可操作、爆破详情点击冒泡导致卡片重绘折叠、播放完成后独立唇位未保留、桌面切页误弹服务器能力提示、长渲染 iframe 超时未同步。旧格式/几何断言按升级语义更新，未放宽数值门槛。

一次早期浏览器直接预览出现全局喉位 0/0.5 断言失败；后续完整运行及新增连续修改 0.3/−0.2/0.5 的保存/重新计算检查通过，未再现，早期失败回执保留。没有将它解释为已确认的声学算法根因。

本机证据：

- `output/validation/m10-shared-20261018/windows-final-tests.json`、`macos-r20-tests.log`、`lip-acoustics.json`。
- `output/validation/m10-wasm-20261018/browser-dd61e352cf2e4470be12f0106524a17d/shared-controls.json` 及宽/窄截图。
- `output/validation/m10-shared-20261018/qt-21077358a1d341f58d330b4afbf4c396/report.json`、浅深主题和导入预览截图。
- Windows 几何 SHA256 `bfb7450a6d5c6fc60be240dcb5e3592650452a0dbb0d5027c0423f30a3417d67`；API/Analysis 均为 `2d3f928b8fa33cb7dba4a577071c3a65e6219450805656ebba6fe623a89ba489`。
- Mac 几何 SHA256 `c446192c6bc57903ccdb4b98a8dec92d7fac3e5f8d6f819b9339ab0ab2672b92`，两份不同 install_name/签名的声学库由 `native-pair.json` 核对。Wasm 对应源码/运行包见锁定清单，未公开部署。

语音在内存中测量，不保留可重建 WAV。独立浏览器/Qt 测试结束关闭自有进程并清理 scratch，保留日志与截图。Mac两个本任务pytest目录的内部current链接已核对后解除并清理，回执为macos-scratch-cleanup.json。当前原生和前端候选是源码验收输入，继续保留。

未验范围：Mac Qt/Safari、实体 iPad 与物理设备听辨、完整参数空间、长时间负载、60 秒视频完整转码及最终 EXE/Mac app。视频时长/包数/字节边界经过接口回归，不据此声称长视频端到端验收。旧网站仍使用已发布版本。
