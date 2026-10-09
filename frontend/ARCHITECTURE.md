# 前端架构

前端使用 Vue、TypeScript 和 Vite，浏览器与 Qt WebEngine 共用同一应用。规则见 [AGENTS.md](AGENTS.md)，整体进程和模块执行归属见[总架构](../ARCHITECTURE.md)。

## 启动和公共工作台

[src/main.ts](src/main.ts)先安装页面缩放并等待 initializePlatform，再挂载 App。桌面连接失败时显示错误，避免把未接通的桌面窗口当成完整能力环境。

[AppShell.vue](src/app/AppShell.vue)维护侧栏、标签、当前模块、帮助、设置及关闭保护；[registry.ts](src/app/registry.ts)维护 M01–M18 的稳定 ID 和分组。页面有同步与按需加载两种方式，不为各模块建立独立宿主。切换/关闭时协调未保存状态、播放、录音租约及更新退出确认。

| 目录 | 主要内容 |
| --- | --- |
| src/app/、src/layout/ | 工作台装配、模块容器、标签及可调整面板 |
| src/modules/ | 各模块页面、业务状态、控制器、领域 port 与模块 README |
| src/platform/ | 桌面/浏览器文件、任务、设备、导出、更新、论文等适配 |
| src/account/ | 账号与项目上下文，构建服务模式的研究文件入口 |
| src/components/ | 共用音频、波形、参数、对话框、设置与引用组件 |
| src/state/、src/design/ | 跨模块播放/工作区/偏好状态、主题、字体和公共样式 |
| src/manual/ | 结构化说明书类型、校验、只读渲染、搜索与媒体协调 |
| src/generated/、src/assets/、public/ | 生成目录与来源数据、字体/图标、静态阅读和声道资源 |

## 两层平台接口

基础 [types.ts](src/platform/types.ts)定义 HostCapabilities、FileProvider、ProjectStore 和 AudioAsset，主要供基础音频预览/工程状态使用。这里的 jobs/capture 标记不代表整套研究任务接口。

研究模块使用 [research.ts](src/platform/research.ts)中的 ResearchContext、ResearchFiles 和 ResearchTasks：
- ResearchContext 固定当前上下文 key、显示标签及可选 owner。
- ResearchFiles 负责列表、读取、TextGrid/参数/语谱预览、目录选择、会话释放，并提供可选 M05/M06/M07/M08/M11/M14 与 annotation ports。
- ResearchTasks 负责批次提交、查询、取消、重试与结果读取；输入通过文件 ID/摘要引用，配置来自版本化契约。
- DirectoryGrant 只传不透明 ID、标签和用途，不允许页面拼接本机绝对路径。

[desktop.ts](src/platform/desktop.ts)握手 QWebChannel 的 files/updates/papers，对请求分配身份、安装各模块 port，并把回调转为 Promise。[browser.ts](src/platform/browser.ts)提供基础浏览器能力；previewFiles 支持本地预览，serverFiles(owner, project, ...)通过 /api/v1/ 接服务端并处理失效账号和资源。可选 port 缺失时由模块呈现明确限制。

M10 使用 vocalRequest，M16 使用 recordingPort，M18 使用 papers channel。它们各有协议和生命周期，不能强行当成同一种批任务。具体代码和操作说明由[模块导航](../docs/modules/module-migration.md)进入。

## 状态、显示与保存

工作区与模块状态保留当前输入、选区、配置草稿、未保存状态和任务归属。输入切换后的旧异步结果不得覆盖新输入。公共 audio 状态与 capture lease 协调播放和设备占用，模块自己的播放器仍须遵守活动页/关闭边界。

文件读取、受管计算结果、草稿和用户导出是不同生命周期。取消原生保存后，页面保留可重试的结果；持久草稿恢复按模块显式处理。M15 的试次、恢复和提交使用 IndexedDB、事务及并发控制，不依赖后端研究任务库，详见[M15 决策](../docs/decisions/ADR-M15-client.md)。

波形与图表根据当前视野采样显示，原始时间、采样率、缺失值和后端信息沿数据传递。界面默认、尺寸与主题统一见[UI 规范](../docs/design/UI_SPEC.md)，模块 README 只记本模块例外。

## 资源和生成物

- contracts/generated/api.ts 由 OpenAPI 快照生成，业务代码消费类型，不直接改生成文件。
- scripts/generate-ui-data.mjs 从参数目录和统一来源登记生成 src/generated/parameters.json、sources.json。
- 说明书源工程位于 manual/，发布阅读资源生成到 public/manual；当前可编辑源稿和生成副本不能混用。
- public/vocal-tract 是 M10 现有嵌入界面及资源，由适配与 Vue 模块协作；不能误判为可删除的重复整套应用。
- 独立作者编辑器导入共用 ManualDocument/公式组件，前端产品不导入 tools/manual-studio。公式由受限 KaTeX 渲染，失败时显示源文本和提示。

## 验证入口

package.json 定义 typecheck、test、contracts:check、ui-data:check 和 build。按改动运行需要的检查，平台交互另外使用模块 E2E/Qt 脚本。构建成功只证明静态产物可生成，不能替代受管任务、保存、设备或最终 EXE 验收。
