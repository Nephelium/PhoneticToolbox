# 后端与 worker 架构

后端分为 HTTP/身份层和任务执行层，二者消费同版 phonetic_core 与协议。总体进程见[总架构](../ARCHITECTURE.md)，修改边界见 [AGENTS.md](AGENTS.md)。

## API 装配

`ptb_api.server` 复用账号、任务、清理与科学端口。没有 `public_origin` 时保留原回环开发入口；显式配置HTTPS源时启用Secure Cookie和Host/Origin检查，仅信任127.0.0.1代理传来的客户端地址。公开测试通过Nginx同源/app/接入，API和PostgreSQL均不监听公网。任务领取遇到暂时的storage_busy保留worker并稍后重试，其余存储身份错误仍抛出，避免后台线程退出后网页一直排队。

`preferences.py` 提供独立账号设置的读取与增量写入，`011_account_preferences.sql` 创建(owner,module)持久表。设置不绑定临时项目/工作区，不参与文件TTL；有效账号在工作区过期后仍可访问，且不会延长内容租约。复用会话、Origin/CSRF、账号头和有界请求中间件，响应no-store。字段/嵌套结构白名单、模块数量和字节预算限定设置范围，修订号及行锁防止旧整份快照覆盖其他设备。构造及启动不执行迁移，部署需显式应用相应脚本。

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

## 浏览器复用入口

`ptb_worker/browser_compute.py`只提供Worker内固定操作分派，调用已有读取器、录音处理、EGG Session及core，没有新增HTTP执行接口。浏览器EGG显式注入WASM运行时检查；服务器/桌面Session默认仍执行原fingerprint，用户请求不能传入Python回调。浏览器结果通过既有资产上传接口保存，身份/期限/额度仍在API事务中检查。浏览器任务快照是受管资产，服务器任务表、调度槽与数据库schema保持原约定。

固定分派亦复用m14_jobs的完整表格/导出、m06_child的元音生成/有界合成和lpc_child的完整三文件输出。M14使用原请求模型与限制；LPC的runtime_check仅由浏览器适配器注入，默认原生校验保持。数据表、DataFrame生成和平滑所需Pandas只在真正调用时导入，不更改读取器/数值语句。不可移植或超预算操作仍使用原服务器端口。

## 本机模式和服务模式

`ptb_worker/browser_vocal.py`是M10本地浏览器Worker的固定分派，直接复用原Engine/Runtime与vocal_web中的设备无关音频/构形库端口，不增加HTTP科学接口。恢复快照按已成功操作增量返回，输入校验错误保持原用户提示，内部异常留在Worker日志。VTL Wasm按独立来源锁装载，原服务器会话和MP4编码路径继续可用；当前只有本机候选验收，见[报告](../docs/testing/m10-wasm-report.md)。

本机 [ptb_api/cli.py](src/ptb_api/cli.py)由 Qt 所有者启动，绑定回环临时端口，校验 Host、Origin 和每次会话 token。SQLiteJobStore 读取指定既有任务库；LocalAcousticFiles 持有本机受管输入/结果，AcousticBatches 提供批次规则。CLI 启动一个自己拥有的 worker，父 stdin 结束即启动收尾。

卸载的专用缓存清理服务沿用相同回环身份和结果清理协议，只开放健康检查与清空受管结果，不启动计算worker。冻结入口跳过此模式不需要的科学运行时联接，保留Windows RedirectionGuard；正常任务服务仍初始化科学运行时并启动worker。

服务模式经 create_app 装配账号、项目、数据库与 Storage，研究文件使用 owner/asset ID。SQLite 与 PostgreSQL 的部署和迁移分别核验，公开部署另见[平台部署说明](../docs/deployment/platform-release.md)。CLI 的回环开发入口不能替代生产服务部署入口。

网页额度/期限由 [storage_policy.py](src/ptb_api/storage_policy.py)定义，完整资源状态机见[账号与存储规格](../docs/specs/accounts-storage-jobs.md)。LocalRetention 是本机受管结果的独立维护机制，不将网页到期策略套到用户原始工程。

实际2核部署以ptbscience.slice给P11科学任务一核合计上限和CPU 1亲和性；生理合成独立子进程在原生线程启动前读取管理员PTB_WEB_COMPUTE_CPUS，同样绑定CPU 1。API保持两核可调度和较高权重，物理计算与HTTP处理分离。配置模板位于deployment/p15-staging，操作核对见私测验收；不能将此2核配置盲用于其他主机。

## 网页注册与临时工作区

`registration.py/registration_store.py`复用既有账号表、Cookie、CSRF和SMTP适配器，仅在server模式显式注入。008增量表保存已验证邮箱、一次性令牌摘要和限流桶。邮件验证与密码重置不使用反馈系统身份。

`workspaces.py`与009增量表保存登录会话工作区、服务器last_seen和1800秒截止。项目映射继承到文件、批次和任务，API逐请求校验会话归属，worker写入/发布在数据库事务中再次检查。主动退出关闭工作区并取消任务，清理器按受控UUID路径删除实体后再释放账本和历史。删除失败保留占用和重试状态；API的EGG内存预览跟随资源失效释放。

工作区的default-project入口在现有存储写锁/事务内复用最早项目或创建默认项目，无新增schema；鉴权、CSRF和工作区到期检查保持。`web_status.py`在登录后的presence回应中只提供聚合运行/等待任务、调度器配置槽位和整机CPU/内存百分比，每API进程缓存5秒。排队数包含尚未物化的批次子任务，同一SQL快照避免双计；Linux通过/proc非阻塞采样，缺失/首个CPU样本返回null，不列出进程、路径或其他账号资料。

工作区鉴权与current读取使用短只读PG连接，不占文件写入锁；文件读取、心跳、关闭和发布仍按各自事务复查有效期。首期三个科学入口在既有调度事务内限制每账号8个、全站128个未结束任务，包含批次尚未创建的子任务，幂等重放不重复占额。额度满返回429与Retry-After，桌面和旧托管模式保持原规则。`server.py`预览可显式启用`enable_worker`，调用同一`run_worker`并随宿主停止；工作区模式单槽，Windows原生资源由`reaper_binary`与既有兼容环境指定。Linux正式运行仍由P15的API/worker/cleaner及服务器硬资源门负责。

该服务默认关闭，server.py/P15宿主需显式启用并预先完成获准迁移。旧storage_policy常量、桌面SQLite/LocalRetention及既有研究数据不自动转换。新协议使用workspace_policy_version=1区分会话寿命，旧policy_version仍表示兼容文件结构；验证边界见[首批验收](../docs/testing/web-launch-foundation.md)。

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
| recording_audio | recording_jobs → Linux有界recording_child，复用core recording的EDL/谱减/数字增益 |
| 其他受支持文件任务 | file_executor，继续按其准入校验 |

当前 run_worker 循环逐次认领并执行。部署槽数、store 的 max_running 与可启动 worker 数量另行配置；不能把某个默认字段当成已实测并发吞吐。

## 预览和资源 I/O

preview 接口提供受预算限制的音频、参数、TextGrid、语谱和交互 EGG 视图。InteractivePreview/SpectrogramSession 等会话在 API lifespan 结束时关闭，长音频按视野处理的显示链与正式计算/导出链分开。

Windows的SpectrogramSession在现有持久科学子进程之外，仅缓存最近一次成功的完整视图响应，预算512 KB。键包含每次实际输入的SHA-256及声道/起止/宽度；相同视图省去IPC与Praat计算，变更视图仍使用原子进程。进程退出、空闲到期、错误或显式关闭清除缓存，API会话身份与源文件授权仍逐请求核对。

EGG交互视野的科学谱值与显示dB映射分别缓存。只调整灰度范围不重复计算CQ/SQ与谱值；每会话只保留当前视野，原精度dB矩阵最多16,000,000字节，超限不保留矩阵。实际ROI、窗宽或科研参数改变仍使缓存失效，事件与音频复用沿用原会话边界。

M16的recording_preview复用PraatDisplaySpectrum，通过固定Linux子入口与同一预览槽执行。采集中二进制请求先鉴权再读取，限制8 MB；录后资源校验owner/project/摘要及期限。正式编辑、降噪、增益和WAV格式导出进入既有队列、fencing与ManagedScratch，计算前预留输出和临时空间。recording_project使用既有assets保存不可变工程快照，无schema变更；同项目旧快照ID作CAS，丢提交回执可幂等重放，成功发布后清旧内部清单，原始/历史音频按工作区规则保留。Linux能力仅在当前运行时摘要及真实科学回执通过后开放，桌面执行器保持原路径。

服务文件写入遵循预留、临时写入、校验、结算和可见提交顺序。expired、deleting、delete_failed 属于资源生命周期，不能混作任务失败原因。磁盘或数据库出错时保留已有结果并停止不可靠的新写入。具体边界由存储规格、实现和相应测试共同核验。

process_entry 只分派固定科学模块；source_runtime 绑定当前源码。EGG/LPC、唇形及 MFA 的第三方兼容环境按模块隔离，业务代码不能从过期安装副本隐式回退。原生适配向 core 提供明确 ports，外部程序、文件系统和设备操作留在外层。

M09 的可选 BigVGAN 沿用同一任务与 spec2wav_child，在 Griffin–Lim 后调用应用自有推理适配器，使用已检查的用户GPU Python或独立CPU组件Python。模型与 Python/Torch 环境经本机专用接口和共用 ComponentManager 下载、逐文件校验、自检后原子登记；仅接受内置版本身份，不接受客户端路径或下载 URL。默认任务不导入 Torch，也不启用该处理。父任务的 Windows Job Object 包含声码器子进程，取消和内存预算覆盖其后代；启用模型时预算为 3 GB。组件保存在本机 components/vocoder，主包不包含修复/声码器模型或独立环境。窗宽/动态范围小模型由 m09_parameters 在主应用内离线推理，资源为 core/spec2wav/data 的固定 float32 NumPy 数组，预览不依赖组件，重建只把预测后的实际参数发送给组件。安装和科学输出证据见[声码器验收](../docs/testing/m09-neural-vocoder-evaluation.md)。

## 外部节点与验证

remote_scheduler、remote_files 和 remote/1 协议已有源码，但 [ptb_node](../packages/ptb_node/ARCHITECTURE.md)仍有未完成的绑定/准入路径。协议测试、科学任务接通和实际服务器容量须分别验收。

backend/tests 负责接口、store 和模块任务边界，tests/contracts 与 tests/security 检查协议/隔离，tests/parity 检查科学一致性。修改数据库须另有授权；文档整理只核对源码，不执行迁移或实际研究任务。

M09 GPU环境由原生文件选择器授予临时ID，m09_runtime只登记兼容性检查成功的解释器身份。运行前重新核对Python及模型摘要，显式CUDA选择失败不静默降级。配置保存在用户组件目录gpu-python.json，测试可用PTB_M09_PYTHON_CONFIG隔离，不接受请求中的路径。GL在主应用core计算，可选组件只提供模型与运行库。

## 网页准入与管理桥接

临时工作区启用时安装WebAdmission：普通与恢复操作分别限制执行/等待数量，超时429；本机桌面不启用。workspaces/open用私有响应头标记新建工作区，供匿名Nginx累计使用次数，刷新不重复。

WebOperations复用容量事务及聚合状态，不增加研究库schema。内部`/_ops`须显式配置独立key并由Nginx拒绝公网；反馈admin/web复用管理员/可撤销设备认证，经WebOpsGateway仅访问固定回环端点，不接受客户端URL/密钥。缺少桥接时只使运维页不可用，反馈原功能保留。

## 声道交互会话

vocal API将身份/CSRF/Origin/项目/工作区检查与科学执行分开；VocalSessions以账号、登录会话、项目和不透明ID隔离原生状态。固定vocal_child进程注入BrowserAudio/MemoryProfile，调用唯一core vocal_tract/session；不导入桌面或设备库。最多两个会话、一个计算槽、每进程1 GiB；1800秒空闲、撤销、超时及关闭均回收自有进程和scratch。停止可在生成中生效，排队生成携带取消代数。服务器科学回退只接受33/65截面，禁用持续发声，仅0.5秒固定构形试听且播放期间拒绝改姿势，科学序列最多5秒；浏览器Wasm默认65、支持33/65/129/257、持续发声与12秒序列，视频编码单独允许12秒。自定义微扰遇旧服务器时保留文档并要求显式恢复默认再合成，不能静默改变声源。原生33能力与桌面菜单策略分开，当前证据见[浏览器能力验收](../docs/testing/web-vocal-browser-controls.md)。共享vocal_video/vocal_mp4保留桌面兼容入口，服务器视频只写会话临时目录并返回下载，客户端不提供路径。配置及验收见[网页版声道报告](../docs/testing/web-vocal-report.md)，本轮独立私有Linux/PG及宿主预算已定向验证，公开生产与长期容量另验。

## 论文文档适配

`ptb_api/papers.py` 在当前账号/项目鉴权、CSRF、配额和工作区期限内管理论文状态资产。每次请求在同卷私有临时目录还原快照，调用随 desktop 包提供的 `ptb_documents` 固定入口；Qt PDF 解析不在 API 进程执行。Windows 使用既有 Job Object，Linux 接既有 systemd 有界执行器。状态采用 CAS 发布，输入/推荐/回收/批注均引用既有受管资产，无额外数据库表。流式响应携带进度与明确结果，取消回收子进程和临时目录，崩溃遗留按实例和进程标记检查。具体限制和验证见[论文网页验收](../docs/testing/web-paper-reading.md)。

无头 Linux 可在宿主管理且摘要绑定的运行时配置中指定 `paper_libraries`。固定论文入口仅为本次子进程设置项目内图形库搜索目录；科学计算入口保持原命令及环境。文档 Python/Qt 依赖同样在项目运行时及摘要清单内管理，不要求安装系统图形桌面。

## 私测Linux与账号反馈

Linux科学入口由native/linux_runtime.py固定登记，并由当前运行时摘要和实际报告回执准入。M06/M07/M08/M09/M14与既有参数估计、LPC、EGG、录音共用受限进程和结果写入；EGG交互以有界单次子进程重算，原始输入仅保留在所属账号的临时会话，退出/超时释放。参数预览同样走固定profile，M02通过parent_result(table=true)按源SHA查最近成功表，显示的可读期限与计算预留期限分别检查。

运行库摘要只在文件身份或元数据改变后重算，进程内锁合并并发首次读取。每次准入仍逐文件检查设备、inode、大小、mtime、ctime和模式，并把缓存摘要与本次profile预期值比较；Linux同大小且恢复mtime的修改仍由ctime触发完整重验，原子替换由inode触发。摘要流式读取且缓存最多4096项，避免能力查询反复读取大型原生库。

web_feedback.py仅接受当前账号Cookie/CSRF和允许的反馈操作，服务器从已验证邮箱表读取联系邮箱；反馈侧只在已配置共享密钥的回环内部接口接受身份，复用已验证会话，绝不按相同邮箱认领未验证会话的内容。浏览器不持有反馈token；无新增schema或邮箱验证流程。独立反馈服务及API可分别恢复，内部入口在Nginx为404。

M10 MemoryProfile复用核心LibraryProfile，library操作加入现有会话白名单。BrowserVocalRuntime检查点携带全部套装及模型状态，重建先恢复library，旧检查点才使用presets/favorites字段；网页耐久保存由浏览器宿主负责，不新建服务器表。
