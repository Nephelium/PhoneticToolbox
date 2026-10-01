# M03 自动更新、空白与耗时修复

2026-09-29。verified：限定本次 Windows 源码宿主、独立 Chrome、实际 Qt 和所列输入。完整 M03/跨平台/冻结发行保持原阶段边界。井井要求对照 V2 手册与源码修复，并追加 review 后一并处理发现的问题。

## 原因与修复

1. **完成后空白**：原页面在任务运行中允许修改总览选区，却只拒收旧快照，没有自动处理最新选区。实际本机两项成功任务 ROI 均与截图选区不同，均有完整结果。现在加载后自动提交，参数/手势 180 ms 防抖，运行或读取期间的操作合并，完成后更新到最新配置。
2. **管道人为限速**：每收到 4 KiB 仍 sleep(0.005)。现在仅空读时等待，保留每轮取消、上限、超时和进程检查。真实 13,830,119 字节结果的受限子进程用时从 20.93 秒降到 3.78 秒，再经过按需计算降到 3.07 秒。
3. **临时文件小块事务**：正式路径 profiling 显示 470 次写入，scratch 占 11.56 秒，完整执行 16.85 秒。仅本机 EGG 使用适配器原已允许的 1 MiB 块，完整执行约 3.96 秒。每块仍执行租约、预留、fsync 和事务；托管平台与其他模块的块大小不变。
4. **无用全段事件检测**：未请求 GCI F0 的 preview 只需局部事件，省略未消费的全段检测。全文件归一化、去趋势、滤波不变，局部 CQ/SQ、微观信号、GCI/GOI 仍用原函数。GCI F0、CSV/图像导出和 IF 保留原完整路径。CSV 依赖改为按需导入。
5. **历史恢复误失效**：恢复时用采样点对齐后的边界覆盖原浮点配置，会触发 stale。现在保留原 ROI 和微观中心，实际采样边界继续留在结果元数据中。
6. **更新和取消竞争**：结果读取也属于当前请求，完成读取前不释放 pending。明确取消会清除待更新意图并作废迟到结果。切换文件后旧响应或错误不影响新选择。
7. **重复音频传输**：同源 hash、声道顺序、采样率、帧数相同的当前预览，复用已验证的内存试听音频。ROI/参数更新只读取 JSON/PSD。历史恢复仍完整读取，文件切换清空缓存。

旧图保留时坐标、色阶与显示开关均使用旧快照，界面标明尚在更新，当前参数未完成时禁止试听、导出和 IF。失败仍显示并允许显式重试。设计见 [ADR-M03-RT](../decisions/ADR-M03-realtime.md)。

## V2 与数值

直接读取相邻 V2 `Phonetic_Export/index.html` 3.1–3.4，及 `egg_widget.py` 的 `_on_load_finished`、`update_roi_plots`、`update_zoom_plots` 和参数回调，核对自动更新、三种局部处理范围和初次加载行为。原手册 10–200 ms 与后来源码 5–5000 ms 的历史差异不倒退。

实际输入为截图对应的 77.249342 秒、44.1 kHz、3,406,696 帧录音，保留原件。修改前后完整 bundle **逐字节相同**，SHA-256 `756dcfe78194aafad47ae156709a376314c27e0127ca8b4d0e6e728bbd46b673`。交换声道、30 Hz 高通场景另对三个旧结果文件逐字节比较，也一致。既有 V2 独立冻结数组回归继续通过；8 份 V2 源码哈希与迁移登记一致。

## 验证命令和结果

- `scripts/Invoke-M03-Python.ps1 -PythonArguments @('-m','pytest','-o','addopts=','backend/tests/test_m03_preview.py','backend/tests/test_m03_ranges.py','backend/tests/test_m03_exports.py','backend/tests/test_pipe_drain.py','backend/tests/test_managed_scratch_chunks.py','-q')`：66 passed。`addopts=` 沿用独立 v3 环境运行方式，隔离根 v2 遗留的未安装 coverage 插件选项，不跳过测试。
- `npm --prefix frontend run test`：最终共享工作区 183 passed。`run typecheck`、`run build` 通过，保留既有大 chunk 构建提示。
- `node tests/e2e/m03-realtime.cjs`：12 组实际服务/计算路径检查，pageerror=[]。环境变量 `PTB_M03_TEST_INPUT` 仅显式提供本机私有测试输入，不写入脚本或公开 fixture。涵盖初次自动绘图、参数更新与音频复用、读取期间连续修改、静音、非采样对齐历史、取消/重试、切文件迟到错误、当前错误恢复、F0/声道/滤波、单声道拒绝及实际长录音。
- `python scripts/verify_m03_realtime_qt.py`：7 组真实 Qt 检查，含自动显示、参数、浮点历史、CSV/三 PNG 查看与原生保存、IF 双角色读取和实际长录音。浅深截图已查看。测试仅关闭自身窗口。
- `python scripts/verify_m03_jobs.py`：6 组原正式任务验收通过，含所有权/hash 拒绝、排队及真实子进程取消、不可变重试、写入故障完整回收、过期租约 fencing、未提交结果不可读、服务重启恢复、三种导出及 60 秒完整带图任务。无残余 temporary，原测试行和输入保持。
- `git -c core.safecrlf=false diff --check` 通过。新增 PowerShell 入口语法解析无错误。

Python 宿主验收使用 `.venv/m09-ui/Scripts/python.exe`，`PYTHONPATH` 指向本项目 core/backend/desktop 源码。科学子进程保持 `.venv/m03-compatible/python.exe` 固定 MKL 环境，无第三方升级。

## 实际整条 GUI 时间

下表为最终一轮实际测量，Chrome 与 Qt 验收存在同机并行，不作为所有设备的性能承诺。原用户任务库只读记录为 39.75 / 36.67 秒，ROI 不同，不作严格倍数比较。

| 路径 | 首次选择到四图可用 | 后续选区更新到可用 |
| --- | ---: | ---: |
| Chrome + 本机正式任务适配 | 7.818 s | 5.232 s |
| 实际 Qt + 正式本机服务 | 7.656 s | 5.719 s |

自动更新已恢复，长文件仍有数秒等待，未声称达到 V2 内存交互速度。历史任务和完整受管结果继续保留。

证据均在忽略目录：`output/validation/m03-realtime/`、`output/validation/m03-ui/chrome-bebc0313199249fb981150eab23eec91/`、`output/validation/m03-ui/qt-realtime-564fe2944e394cddb04a550c2fc3db42/`、`output/validation/m03-jobs/local-9c3b69ab1f414369826db7f49d5eef4d/`。

## 失败记录与边界

管道回归先确认 1024 次多余 sleep；按需检测回归在旧代码明确失败后修复。首次 E2E 受并行工作台编辑的临时不完整 Vue 标签及旧测试宿主 core 安装影响，未作为产品成功；随后使用完整源码路径。前两次长文件完整页面仍超过 20 秒验收阈值，未放宽阈值，继续 profiling 并修复 scratch 写入，最终在相同阈值内通过。profiling 首次脚本名与标准库 profile 冲突，已隔离脚本目录后重跑。

`check_architecture.py` 检查报出并行工作区声道模块 `frontend/public/vocal-tract/{app.js,index.html,v3.css}` 的三项资源哈希不匹配，M03 没有更改这些文件或登记，不冒称全仓检查通过。WSL 的 NInfer 未提供 python3，未安装系统依赖，Linux 科学链保持未验证；未连接生产服务器。

当前入口为 [Start-M03-Workbench.ps1](../../scripts/Start-M03-Workbench.ps1)，直接使用统一源码宿主和已构建前端，避免旧安装包代码掩盖修复。**旧 EXE 未重新打包**。未改 V2、用户原件、现存库 schema、全局环境、CI、密钥；未 push 或公开发布。保留其他并行任务的改动。
