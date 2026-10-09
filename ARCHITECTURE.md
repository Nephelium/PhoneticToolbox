# PhoneticToolbox v3 架构

本页按当前源码说明系统怎样运行，详细实现分别维护在组件架构页。平台能力是否经过验收，查[任务台账](docs/plans/task-ledger.json)和对应报告。本页不保存逐次交付流水，也不把接口存在等同于生产部署完成。

## 系统与进程

PhoneticToolbox 使用一套 Vue 工作台，Windows 桌面由 Qt WebEngine 承载。科学计算主要运行于本机服务及其计算子进程，浏览器服务模式通过 HTTP 接入后端；感知实验等客户端能力直接在页面运行。

~~~mermaid
flowchart TD
    Entry["源码启动器 / 冻结 EXE"] --> Qt["Qt 宿主 host.py"]
    Qt --> View["Qt WebEngine：共用 Vue 工作台"]
    Browser["普通浏览器：同一前端"] --> WebAPI["服务模式 API / 账号与存储"]
    View --> Channel["QWebChannel：files / updates / papers"]
    Channel --> Local["自有回环 API：LocalService"]
    Local --> Worker["任务 worker"]
    WebAPI --> ServerWorker["服务模式 worker"]
    Worker --> Science["受控科学子进程 / phonetic_core"]
    ServerWorker --> Science
    Channel --> Native["M10 原生声道 / M16 设备与工程"]
    Channel --> Content["更新与论文服务"]
    View --> Client["纯客户端状态 / M15 IndexedDB"]
~~~

图中的服务模式是源码接线，远程部署、数据库版本及计算准入须按实际环境核验。桌面基础分析不要求公网登录。M10 原生状态、M16 设备和 M18 内容分发各有专用通道，不绕道普通批任务队列。

## 代码地图

| 目录 | 当前职责 | 详细结构 |
| --- | --- | --- |
| frontend/ | 公共工作台、模块页面、平台适配、只读说明书阅读器 | [前端架构](frontend/ARCHITECTURE.md) |
| desktop/ | Qt 宿主、本机文件授权、进程/设备生命周期、更新、论文、启动缓存 | [桌面架构](desktop/ARCHITECTURE.md) |
| backend/ | FastAPI、账号/资源/任务接口、SQLite/PostgreSQL store、科学任务执行 | [后端架构](backend/ARCHITECTURE.md) |
| packages/phonetic_core/ | 算法、科学数据模型、纯计算与原生能力 ports | [科学核心](packages/phonetic_core/ARCHITECTURE.md) |
| contracts/ | API/schema 审阅快照、生成 TypeScript、协议版本与资源身份 | [协议架构](contracts/ARCHITECTURE.md) |
| resources/、third_party/ | 运行资源、来源登记、许可及对应源码材料 | [资源](resources/ARCHITECTURE.md)、[来源](third_party/ARCHITECTURE.md) |
| manual/、tools/manual-studio/ | 结构化说明书源工程、独立作者编辑器 | [说明书](manual/README.md)、[作者工具](tools/manual-studio/README.md) |
| requirements/ | Python 模块依赖声明、版本锁及兼容环境清单 | [清单说明](requirements/README.md) |
| scripts/、release/ | 开发入口、生成与核验脚本、冻结和发行包装 | [开发](docs/development.md)、[发行](release/README.md) |
| packages/ptb_node/ | 外部可信节点客户端框架，绑定及科学准入尚有关闭的路径 | [节点架构](packages/ptb_node/ARCHITECTURE.md) |
| tests/、docs/ | 科学基准、架构/协议/安全等测试；当前规范与历史证据 | [测试](tests/ARCHITECTURE.md)、[文档](docs/README.md) |
| phonetic_toolbox/、run.py、run.spec | 继承的 v2 代码与资源，仍有实际资源引用 | 清理前查依赖，不能整目录视为无用 |

依赖方向为外层到科学核心。前端消费契约和平台接口；backend 与 desktop 分别适配 core，彼此通过受控进程及协议交互。core 不导入 Qt、FastAPI、账号或数据库。发行脚本只组合资源和运行时，不承担科学计算。

## 开发与发行入口

### 开发运行

[Start-Research-Workbench.ps1](scripts/Start-Research-Workbench.ps1)接受 home 或 M01–M18，使用已有 .venv/m14 主环境，经 [workbench_source.py](scripts/workbench_source.py)把当前 core/backend/desktop 源码放到导入路径前部并核查模块位置。各模块启动器是同一入口的快捷方式，不维护各自一份业务源码。

入口继续调用 [start_m01_workbench.py](scripts/start_m01_workbench.py)。该文件保留早期名称，实际承担共用工作台接线。它读取开发任务库、配置 LocalAcousticFiles 与 REAPER 后运行 Qt host。不会自动安装依赖、重建前端或迁移现存数据库。具体命令与环境见[源码入口说明](docs/development/source-entry.md)。

第三方依赖有兼容性隔离：主宿主使用 .venv/m14，EGG/LPC 使用 PTB_EGG_PYTHON 指向的 .venv/m03-compatible，唇形使用 PTB_M05_PYTHON 指向的 .venv/m05。环境中的第三方库与当前业务源码是两类对象，不能靠旧 site-packages 项目副本代替源码绑定。

### 冻结应用

[build_v3_local_preview.py](scripts/build_v3_local_preview.py)及 release/ 工具冻结同一份业务源码和静态资源。启动入口为 [v3_local_preview_entry.py](scripts/v3_local_preview_entry.py)，按发行清单恢复或复用运行时，再进入 [research_entry.py](scripts/research_entry.py)。

research_entry 在导入 GUI 前处理固定的 --ptb-worker、--local-service、--m10-worker 分派，未知开关拒绝运行。正常路径准备应用自己的本机工作区并打开同一个 Qt host。新建应用工作区与迁移现存研究数据库的授权边界分别处理。

持久启动缓存按 science、Qt、apps 等内容身份维护，校验后跨版本复用；共享文件优先硬链接，支持复制回退。详细限制、500,000,000 字节上限、MFA 排除及最终成品检查以[严格打包规则](release/PACKAGING_RULES.md)为准。

## 工作台与模块执行方式

模块 ID 与导航分组由 [registry.ts](frontend/src/app/registry.ts)维护。模块具体参数、格式和来源查[模块导航](docs/modules/module-migration.md)，总架构只列执行归属。

| 模块 | 主职责 | 主要执行链 |
| --- | --- | --- |
| M01 参数估计 | 参数批处理、联合输入、TextGrid 切分 | ResearchFiles / ResearchTasks → API → acoustic worker → core |
| M02 参数显示 | 参数表读取、选区与图表 | 文件预览/参数读取适配 → 页面显示 |
| M03 EGG 信号分析 | EGG 交互、F0、逆滤波及导出 | 预览会话或 egg_analysis 任务 → 隔离科学环境 |
| M04 LPC 谱图 | LPC、显示与导出 | lpc_analysis 任务及专用科学适配 |
| M05 唇形提取 | 视频/摄像头输入、唇形跟踪和媒体导出 | M05 port → lip_analysis → 独立唇形环境，设备授权在宿主 |
| M06 声学参数合成 | Klatt 参数合成、参数提取与连续统 | M06 port → speech_synthesis → core/原生适配 |
| M07 发声类型合成 | 源信号、F0 与发声类型合成 | M07 port → phonation_synthesis → 科学子进程 |
| M08 变速变调 | 音高/时长变换及批量导出 | M08 port → pitch_manipulation → core |
| M09 语谱图转音频 | 图片重建、原音语谱编辑和涂鸦 | 预览适配及 spectrogram_to_audio → core |
| M10 生理参数合成 | 声道几何、合成、关键帧与录制 | vocalRequest → VocalTractClient → 专用原生进程 |
| M11 MFA 自动标注 | 模型/词典检查、对齐与 TextGrid | M11 port → mfa_alignment → 登记的外部组件环境 |
| M12 TextGrid 标注 | 区间/点层编辑、关联和保存 | 页面编辑状态 → AnnotationPort → 本机授权文件或服务资源 |
| M13 汉字转国际音标 | 字表规则转换、排版和图片导出 | 客户端规则/资源，桌面字体能力补充 |
| M14 音系归纳 | 字表导入、符号归类、审阅和文档生成 | M14 port → phonology_induction → core 与文档 I/O |
| M15 感知实验 | 设计、刺激、计时、问卷及结果 | 纯客户端运行器、IndexedDB、导出，无 Python 科学任务 |
| M16 录音 | 同设备多道采集、处理、版本和导出 | recording 通道 → 本机设备/工程服务，独立后台处理 |
| M17 国际音标表Plus | 音标目录、输入、介绍与媒体 | 客户端目录/草稿；独立本机内容维护入口 |
| M18 语音学论文精读 | 双语 PDF、批注、引用和导出 | papers 通道 → 本机论文服务 → 校验后的内容分发 |

同一模块在不同平台可能暴露不同能力。没有桌面桥的浏览器可使用本地预览和已实现的纯客户端流程；账号项目通过 serverFiles 接后端。不能因菜单可见就认定原生设备或服务器计算可用。

## 三条典型数据流

### 科学批任务

1. 页面经 ResearchFiles 获得带 ID、类型、大小和摘要的输入。本机绝对目录由 Qt 授予 DirectoryGrant，服务端使用 owner 下的 asset_id。
2. 提交 operation、输入摘要、config_snapshot、幂等键。API 验证身份、文件状态、格式与预算，store 固定任务快照。
3. worker 认领任务，携带 worker_id 与 generation 更新租约/进度；executor 按 operation 选择科学执行器。
4. 子进程使用受控输入、明确版本与实际算法后端，产物经解析/摘要检查、存储结算和结果清单提交后才发布成功。
5. 页面读取任务/结果并展示。保存到用户目录是独立导出步骤，取消保存不等于取消已完成的计算。

### 交互预览

波形、参数表、语谱及 EGG 交互走预览接口或会话。视野、分辨率和显示降采样只影响展示；原始采样率、选区帧索引与最终导出语义不随窗口变化。会话关闭、切换输入和迟到响应均由相应页面/平台生命周期处理。长录音不能仅凭显示点数推断已经读取或计算完整原始数据。

### 客户端与原生专用能力

M15 刺激解码、预检、调度和计时留在客户端，网络往返不参与刺激时序。M10 可变原生状态由专用进程隔离，M16 原始采集和工程版本由设备服务持有。M18 内容下载和更新使用宿主服务，PDF 按内容摘要独立保存批注，原文与译文分别管理。

## 持久数据与可再生目录

| 位置 | 实际用途 | 清理边界 |
| --- | --- | --- |
| output/validation/p06/local-state.sqlite3 | 当前源码启动器读取的开发任务库 | 不能随验证目录整删，先确认当前入口和保留需求 |
| output/validation/m01/workbench-local.json 及所指 workbench-cache-* | 源码入口的本机文件/任务缓存位置 | 先查配置和活动任务，再清理可再生内容 |
| output/validation/m03-runtime 等 | 部分发行准备仍引用的运行时暂存 | 查 release/prepare_runtimes.py 的实际输入 |
| %LOCALAPPDATA%/PhoneticToolbox/v3/local-preview-20260927 | 当前冻结入口默认的任务/文件工作区 | 名称保留兼容性，不按日期认定过期 |
| %LOCALAPPDATA%/PhoneticToolbox-v3/workbench | Qt 网页持久存储、偏好与客户端草稿/数据库 | 与普通下载缓存区分，升级保持固定身份 |
| %LOCALAPPDATA%/PhoneticToolbox/v3/startup-cache | 经内容校验的运行时/应用启动缓存 | 通过应用缓存管理和活动租约处理 |
| %LOCALAPPDATA%/PhoneticToolbox/v3/updates、papers | 更新状态/下载；论文、首次日期和批注 | 不随启动缓存清理或源码测试清理删除 |
| 用户选择的录音、工程和导出目录 | 原始研究数据和正式结果 | 由用户管理，测试清理授权不涵盖 |
| manual/ | 正式可编辑说明书、素材和当前保留的作者历史 | 当前明确保留，生成阅读副本另行处理 |
| .venv/、node_modules/、dist/、output/build-* | 第三方环境、依赖、成品与构建展开 | 依赖按用途复用，旧包成功换版后按规则清理 |

操作系统路径由 [platform_paths.py](desktop/src/ptb_desktop/platform_paths.py)管理。上述 Windows 两个用户目录是当前兼容布局，未在文档整理中迁移。源码目录中的 papers/ 下载材料也与应用用户目录中的论文服务分开。

项目变大的主要来源是依赖、构建展开、正式媒体和测试副本。规则见[仓库内容管理](docs/development/repository-hygiene.md)：生成长音频和测试工程用后清理，必要摘要留报告，不随每次修改再复制整套工程。Git 忽略不删除磁盘文件，也不自动移除已跟踪内容。

## 科学契约、资源与生成链

- 音频保留真实采样率、声道角色、每道帧数与摘要，选区为整数帧半开区间 [start,end)。轨迹保留真实时间、单位、validity/reason 与实际后端，JSON 中的 null 不能被自动补零。
- 科研默认值、参数键、时间网格、NaN/无声/失败语义属于版本契约。显示降采样、配色和字体不改变科学输出。
- 后端源模型经 scripts/generate_contracts.py 生成 contracts 快照，再由 frontend/scripts/generate-contracts.mjs 生成 TypeScript。
- 参数目录与 third_party/source-registry.json 经 generate-ui-data.mjs 生成前端参数/致谢数据。运行资源还按各自 manifest/source lock 记录文件身份和来源。
- manual/project.json、chapters/ 与资产经 scripts/manual/ 校验和构建为 frontend/public/manual，再进入前端静态构建。作者编辑器复用阅读组件，普通应用不依赖作者服务。
- 固定运行资源在 contracts/resource-manifest.json 记录 SHA 与来源，自有交互资源显式标 origin=project。说明书通过单条 generated_resources 声明接入源工程、阅读索引和构建报告核验，不在公共清单反复登记上千个生成文件。scripts/check_architecture.py 核对生成内容、逐文件摘要、缺失/多余文件及联接；干净检出尚未生成的整棵阅读目录可缺省，已有目录必须完整且与源工程一致。
- 文献引用、代码对应、再分发许可及数值等价分别举证，禁止用一个结论替代其他项。

## 服务模式与当前边界

API 支持 local/server 两种装配，store 有 SQLite 和 PostgreSQL 实现。服务端资源策略中的额度为 1,000,000,000 字节，最长保留期为 259,200 秒，定义在 [storage_policy.py](backend/src/ptb_api/storage_policy.py)。owner、上传预留、删除失败计量、到期及结果发布规则见[存储规格](docs/specs/accounts-storage-jobs.md)，本机用户工程不直接套用网页到期政策。

已有队列、租约、重试、取消及远程协议代码不代表所有部署路径已经联调。ptb_node 仍含 protocol_unavailable/binding_pending 路径。约十位研究者并发交互是容量目标，实际重计算并发受部署配置和实测预算限制，不能据单 worker 或接口字段承诺吞吐。

Windows 源码、浏览器、Qt、冻结 EXE、实体设备和其他平台各自验收。静态源码核对能够说明职责与调用关系，不能证明干净机器运行、自然语料准确性或长期稳定性。验证入口查[策略](docs/testing/verification-plan.md)，取舍查[ADR 索引](docs/decisions/ADR.md)。
