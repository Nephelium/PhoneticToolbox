# 后端与 worker 架构

后端分为 HTTP/身份层和任务执行层，二者消费同版 phonetic_core 与协议。总体进程见[总架构](../ARCHITECTURE.md)，修改边界见 [AGENTS.md](AGENTS.md)。

## API 装配

[ptb_api/main.py](src/ptb_api/main.py)的 create_app 根据 local/server 模式、账号存储、任务 store、认证设置和资源存储装配路由。主要接口包括账号、项目、任务、assets、preview，以及 health/capabilities。

| 位置 | 职责 |
| --- | --- |
| ptb_api/auth.py、account_boundary.py、account_store.py | 会话/账号边界与请求身份 |
| ptb_api/projects.py、jobs.py | 项目授权、任务创建/查询/取消/重试 |
| ptb_api/storage*.py | owner 下的资源、上传预留、额度/期限、读取和删除 |
| ptb_api/preview.py、各模块 *_models.py | 预览接口、请求/响应校验和模块协议 |
| ptb_worker/store.py（SQLiteJobStore / PostgresJobStore） | 持久任务状态、认领、租约和 fencing |
| ptb_worker/cli.py、executor.py | worker 生命周期、operation 分派、子进程收尾 |
| ptb_worker/*_executor.py、*_child.py、*_task.py | 模块计算执行、结果检查与任务提交 |
| ptb_worker/io/、native/、assets/、mfa/ | 文件/原生/资源/MFA 的外层适配 |
| migrations/ | 显式 schema 与迁移文件，存在文件不表示当前数据库已执行 |

capabilities 根据实际配置、资源、运行时及 store 能力计算。HTTP 模型校验、权限或科学运行条件失败应返回明确错误，不通过静默换算法或把客户端按钮隐藏当成后端检查。

## 本机模式和服务模式

本机 [ptb_api/cli.py](src/ptb_api/cli.py)由 Qt 所有者启动，绑定回环临时端口，校验 Host、Origin 和每次会话 token。SQLiteJobStore 读取指定既有任务库；LocalAcousticFiles 持有本机受管输入/结果，AcousticBatches 提供批次规则。CLI 启动一个自己拥有的 worker，父 stdin 结束即启动收尾。

服务模式经 create_app 装配账号、项目、数据库与 Storage，研究文件使用 owner/asset ID。SQLite 与 PostgreSQL 的部署和迁移分别核验，公开部署另见[平台部署说明](../docs/deployment/platform-release.md)。CLI 的回环开发入口不能替代生产服务部署入口。

网页额度/期限由 [storage_policy.py](src/ptb_api/storage_policy.py)定义，完整资源状态机见[账号与存储规格](../docs/specs/accounts-storage-jobs.md)。LocalRetention 是本机受管结果的独立维护机制，不将网页到期策略套到用户原始工程。

## 任务执行链

1. API 校验请求、owner、输入状态/摘要和预算，记录 operation、版本及不可变配置快照，幂等请求关联既有任务。
2. worker 从 store.claim 获取任务、worker_id 与 generation。租约心跳与 fencing 阻止失效执行者发布结果。
3. executor 按 operation 调用指定执行器，在独立子进程执行科学工作，取消/终止状态传递到所属进程。
4. 科学输出由模块适配解析，文件通过结果清单、摘要与预算检查。完成存储提交后才发布成功，失败保留可诊断状态。
5. 页面读任务和结果，用户导出走独立保存路径。重试保留前次 attempt 的证据，不把终态回退为旧运行状态。

[executor.py](src/ptb_worker/executor.py)的分派包括：

| operation | 执行器 |
| --- | --- |
| lip_analysis | m05_executor |
| mfa_alignment | m11_executor |
| speech_synthesis / phonation_synthesis | m06_executor / m07_executor |
| phonology_induction | m14_executor |
| acoustic_analysis / textgrid_segment / spectrogram_to_audio / egg_analysis / lpc_analysis / pitch_manipulation | acoustic_executor 及模块子路径 |
| pipeline_check | core_child 管线探针 |
| 其他受支持文件任务 | file_executor，继续按其准入校验 |

当前 run_worker 循环逐次认领并执行。部署槽数、store 的 max_running 与可启动 worker 数量另行配置；不能把某个默认字段当成已实测并发吞吐。

## 预览和资源 I/O

preview 接口提供受预算限制的音频、参数、TextGrid、语谱和交互 EGG 视图。InteractivePreview/SpectrogramSession 等会话在 API lifespan 结束时关闭，长音频按视野处理的显示链与正式计算/导出链分开。

服务文件写入遵循预留、临时写入、校验、结算和可见提交顺序。expired、deleting、delete_failed 属于资源生命周期，不能混作任务失败原因。磁盘或数据库出错时保留已有结果并停止不可靠的新写入。具体边界由存储规格、实现和相应测试共同核验。

process_entry 只分派固定科学模块；source_runtime 绑定当前源码。EGG/LPC、唇形及 MFA 的第三方兼容环境按模块隔离，业务代码不能从过期安装副本隐式回退。原生适配向 core 提供明确 ports，外部程序、文件系统和设备操作留在外层。

## 外部节点与验证

remote_scheduler、remote_files 和 remote/1 协议已有源码，但 [ptb_node](../packages/ptb_node/ARCHITECTURE.md)仍有未完成的绑定/准入路径。协议测试、科学任务接通和实际服务器容量须分别验收。

backend/tests 负责接口、store 和模块任务边界，tests/contracts 与 tests/security 检查协议/隔离，tests/parity 检查科学一致性。修改数据库须另有授权；文档整理只核对源码，不执行迁移或实际研究任务。
