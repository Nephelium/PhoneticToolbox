# M16 本地协议 recording/1

此目录是独立本地工程 JSON Schema，不加入现存 OpenAPI、P06/P07 或数据库。`project.schema.json` 是审阅快照。工程结构及路径、文件尺寸、帧索引、同声道约束还由 `recording/storage.py` 运行时验证。前端 `types.ts` 是去除音频路径和 EDL 的 UI 投影，未手改公共 generated client。

`microphone` 是内部兼容角色名，界面表示音频（麦克风 / 线路）。默认 2 个音频角色，EGG 仅用户显式指定。单输入设备明确回退 1 个音频通道。

传输为宿主独立 `recording(request_id, JSON)` / `recordingReady`，响应 `{ok:true,value}` 或 `{ok:false,error}`。目录只通过 Qt 主线程 `m16_choose` 授予 opaque grant。前端不得传任意路径。所有采样索引为每通道整数帧，区间 `[start,end)`。

| 操作 | 语义 |
| --- | --- |
| capabilities / devices | 设备身份与本地能力，Windows 外原生采集未准入 |
| open / project / save / tasks | 工程单写者、清单提交、任务快照 |
| start / probe_start / stop / status | 同声卡原始采集、检测、落盘收尾与有界预览 |
| preview / play / play_stop | 只读显示和所选输出设备试听；支持当前/原始/处理差分 |
| edit / undo / redo / restore / version | 同步多通道 EDL 与永久版本历史 |
| noise / process_start / job_cancel | 固定噪声样本、本机子进程处理、取消后不提交 |
| export_start | 独立导出目录、逐文件回读、可取消、清单内记录缺录/跳过 |
| recover / select_take | 接回已落盘且哈希通过的前缀、标记批次选用版本 |

采集与处理/导出互斥。长处理只启动后台并轮询，不把降噪计算放入桥请求。所有资源 ID 属于当前已授权工程，跨工程粘贴不开放。原始采样是应用收到的 float32，不代表关闭了驱动或系统 AGC。

## 只读显示投影

`spectrum.time_edges` 为相对当前窗口起点的秒数，长度为时间列数加 1，首尾严格为 0 和 `window_frames / sample_rate`。`times` 为实际 FFT 窗中心，`frequencies` 是未拉伸的实际频率。`max_frequency` 为 5000 与奈奎斯特频率的较小值，`nfft=1024`；`time_sampled` 表示时间单元间隔大于原显示 hop 256 帧。实时最多 128 列，录后最多 640 列。音频工程 schema 不变，新增字段是显示元数据，既有 `rows/frequencies/times` 保留。
