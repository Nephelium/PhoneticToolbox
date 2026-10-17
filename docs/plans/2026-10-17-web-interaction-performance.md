# Web interaction performance implementation plan

**Goal:** 完成已确认的六类性能优化，并在私测服务器验证，保留nephelium访问限制。

**Architecture:** 沿用现有一次一受限科学进程及共享计算槽，不引入常驻科学服务。采用有界内存结果缓存、轻重请求分离、前端项目查询合并及按任务状态退避。所有缓存保留归属/有效期校验，科研算法及输出字节不变。常驻进程可以减少首次不同视图的加载开销，但会改变资源生命周期，本轮不采用。

**Tech Stack:** Python 3.11/FastAPI、NumPy/Praat及原科学core、Vue/TypeScript、现有systemd资源隔离。

## 1. 有界预览缓存

- 新增backend worker内存结果缓存，明确owner scope，32 MB总上限、最多24项、60秒TTL，只缓存成功JSON结果且不共享可变对象。
- spectrogram_preview、recording_preview、parameter_preview、spec2wav_preview接同一适配器。API assets与spec2wav输入编排绑定owner/asset scope，缓存前后仍校验当前账号、资产归属/期限/摘要；Windows原生路径使用同一适配器，既有grant/fingerprint复核保持。
- 定向测试：相同请求复用、配置/声道/owner失效、TTL/字节淘汰、错误不缓存、返回值不可污染缓存，真实服务器重复结果与旧基准一致。

## 2. 录音轻重分离

- browser-recording.ts的status立即返回波形/电平/任务状态，独立有界异步请求最新实时谱，停止/关闭/换通道丢弃旧代次，不排积待处理窗口。
- RecordingPage.vue恢复180ms轻状态更新，谱保持独立节流，挂起谱不阻塞停止采集或电平刷新。测试以延迟谱请求证明轻状态/停止可以完成，迟到谱不污染新录音。

## 3. 任务查询

- 新增共用任务间隔策略，M06/M07/M08/M14按queued/running调整，不再120–150ms查询；取消不等待整个退避周期。
- serverFiles同一owner/project合并完整jobs查询，最多500ms缓存，创建/取消后失效；同时合并相同单任务的查询。保留最终完成与取消语义。
- 测试查询次数、并发合并、创建/取消失效及取消响应。

## 4. 声道监测

- core Runtime实例内只保留最近一次输出监测，key覆盖generation/end与全部显示参数。返回副本，代次或数据/参数变化重新调用同一analyze_output。
- 前端未变化输出不重复拉取监测，保持500ms轻心跳及播放确认；换窗/缩放主动刷新，播放期间照常更新。共用桌面路径同步验证。

## 5. 验证、私测部署与记录

- 定向Python/前端测试、类型/契约/UI数据/构建、架构与文档检查。
- 冻结单一候选，以实际服务器所有九类科学及预览、缓存失效/隔离、慢预览和请求频率验证；复用既有合成基准，不读取用户会话/资料。
- 验证通过后私测切换，核对当前官网和同期前端资产、入口/注册/找回关闭且仅nephelium有效。保留回退，不推送、不重打EXE。
- 更新CHANGELOG、台账、project-status与web-private-repair验收。测试输入驻RAM，清理仅处理本轮明确拥有且不再使用的对象，已被审批拒绝的本地旧目录保持。
