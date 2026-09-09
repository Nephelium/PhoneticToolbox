# 本地测试、平台构建与发行规划

D0.3，未安装新环境、未构建、未部署。

## 1. 先本地，后服务器
先在 Windows 完成核心/前端/桌面启动原型；服务器功能在专用 WSL Linux 环境中运行与测试。当前发现的 NInfer 与 docker-desktop 不视为本项目专用环境，不能擅自升级或改配置。
WSL 后端、数据库、worker、存储用独立目录/端口/配置，默认仅本机访问；Linux 源码与数据尽量置于该发行版自身文件系统。实际创建发行版/全局工具与数据库迁移按会话授权执行。
服务器部署候选为阿里云 ECS 上的单机模块化部署；需求和负载验证之前不租机器。云供应商是运行环境，不进入算法/前端业务代码。

## 2. 环境可复现的前置问题
本机 phonetic_311 实测包元数据：NumPy 2.2.6、OpenCV-contrib 4.13.0.92；v2 pyproject.toml 仍约束 numpy<2、opencv<4.12。这是当前环境与声明不一致的事实。
P02 先验证并形成可靠锁定清单，不能直接按旧 requirements 重建就宣称同环境，也不能立即降级 v2。当前 exe 的打包依赖版本还需从产物/构建记录单独核对。
v3 建独立开发/构建环境。前端 Node/npm、Python wheel、原生 ABI 都锁版本；禁止 requirements 只写无限上界。

## 3. 各端交付共用版本清单
- 网页：frontend 静态资源、API/worker、数据库迁移、私有存储配置。
- Windows：单文件 EXE + 安装程序。安装程序可安装内部多文件布局，但用户点击一个桌面入口；单文件目标另行验证。
- macOS：.app 和磁盘映像/分发包；Apple Silicon 与 Intel 依赖分别核实。
- Linux 桌面：优先 x86_64，AppImage/适用安装包在原型后确定。不是用 Windows exe 包一层兼容工具冒充原生。
- 每个产物携带 app/core/api/assets 版本与来源清单，可追踪其构建输入。一个仓库可以分别构建发布各部分。

## 4. Windows
发布态只需要双击，不要求 Python/Node/浏览器服务配置。静态界面编译后打包，宿主自动启动本地服务。
继承 v2 的原样构建仍限 phonetic_311 和 python -m PyInstaller；v3 的 spec/build 脚本在 P12 专门建立并指向经过验收的 v3 环境。
单文件验证 WebEngine 资源/子进程、原生 DLL、启动提取时间、临时目录清理、杀毒误报观察、Unicode 路径、应用退出。
安装候选 Inno Setup，每用户安装优先；路径/快捷方式/卸载信息明确；卸载默认保留研究数据。安装版/便携版不要共享不兼容临时状态。
实际签名与公开分发在发布阶段决定；不能靠关闭系统安全功能作为正常使用说明。

## 5. macOS
需要 macOS 构建环境，PyInstaller 不提供从 Windows 直接生成 Mac 应用的交叉编译。[PyInstaller](https://pyinstaller.org/en/stable/)
可评估 GitHub macOS runner 做构建，但设备、声音、摄像头和窗口行为必须有相应验证。[GitHub runners](https://docs.github.com/en/actions/concepts/runners/github-hosted-runners)
检查 .dylib、架构、动态库加载路径、QtWebEngine helper、麦克风/摄像头权限说明、签名与公证、Gatekeeper 启动、应用退出。
暂无 Mac 实机，所以本轮不能承诺 Mac 功能已通过。后续通过测试者/设备获得证据，分别记录 arm64 与 x86_64。

## 6. Linux 与原生引擎
REAPER、VTL、FFmpeg/MFA 等分别核验来源/许可/平台可执行文件；不复制 Windows DLL 或整个 Conda 环境到服务器。
VTL 独立 C++ 源码与本项目 bridge 编译，ABI 函数签名/结构布局、线程/实例状态与声学结果测试；显示几何与计算管道一致性另测。
Linux 桌面另测 X11/Wayland、PulseAudio/PipeWire 适配、字体、沙箱支持与文件对话框。不能通过禁用 WebEngine 沙箱来掩盖依赖问题。

## 7. 阿里云部署前的计算与容量
10 账号各占 5 GB 只是用户额度上界 50 GB；还需系统、模型、数据库、临时预留及余量，不能据此购买正好 50 GB 的磁盘。
初期 2 worker 是压测起点，CPU/内存需求取决于音频时长、MFA 模型、声道和唇形任务。依据 P11 报告提供实例候选与预算，届时查验实时价格。
部署同源 HTTPS 入口；API 只在内部监听，私有存储不直接暴露；服务器备份策略不得违反用户音频最多 7 天。
保留账号/最少审计元数据与音频数据分开的恢复策略。部署前验证数据库迁移、worker 停止/恢复、资源清理与版本回退。

## 8. 本轮不做的操作与后续触发
不租服务器、不改 DNS、不购买签名、不改 CI/CD、不安装系统级依赖、不发布。P15 完成可审阅部署清单、成本与回滚验证后再进入实际发布。
旧 Vue 站和旧 API 目录当前保留：v3 不依赖它们，但旧前端有未提交修改，尚不能确认全部无用。清理前重新检查目录、数据、Git 状态与其他用途，并记录准确删除目标。
