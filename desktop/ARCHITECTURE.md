# 桌面宿主架构

桌面包负责 Qt、操作系统能力与自己启动的进程，不复制科学算法。[根架构](../ARCHITECTURE.md)描述跨组件关系，本目录约束见 [AGENTS.md](AGENTS.md)。

## 宿主与资源边界

[src/ptb_desktop/host.py](src/ptb_desktop/host.py)装配 Workbench、QWebEngineProfile、QWebEngineView、Page 和 QWebChannel：
- 静态前端经 ptbapp://app/ 读取，Assets 校验主机、方法、路径穿越和根目录归属。
- LocalOnly 允许应用资源、指定回环 API、WebChannel 脚本及 data/blob，页面不能任意请求公网。
- QWebChannel 注册 files、updates、papers 三个对象。files 内部分派预览、研究任务、录音、声道、文件选择等请求。
- 原生选择、剪贴板写入、全屏、下载保存、关闭及首次最大化呈现由宿主管理；网页不能直接获得任意文件系统访问。
- 非测试 profile 保持稳定持久目录，HTTP 缓存关闭以避免升级后仍读取旧静态文件。

此网络限制作用于嵌入网页。论文和更新的联网由各自宿主服务执行，并按自身来源、清单和内容摘要规则核验。

Assets、LocalOnly、URL校验与scheme注册共用 [web_assets.py](src/ptb_desktop/web_assets.py)，主宿主保留导入名称兼容。工作台另注册 feedback 原生桥。独立 [feedback_manager.py](src/ptb_desktop/feedback_manager.py)只装配管理静态页、反馈桥、DPAPI与托盘，不导入工作台或科学运行时；同一用户的数据目录对应一个本机实例，再次启动唤起已有窗口。文字、私有图片、设备配对及SSE请求均由原生网络层核验HTTPS。关闭隐藏到托盘，退出关闭本工具的连接与线程。最终EXE有独立入口及工程外验证，见[工具说明](../tools/feedback-manager/README.md)。

[window_chrome.py](src/ptb_desktop/window_chrome.py) 在 Windows 将标签与单一品牌区放入窗口顶边，由前端绘制主题一致的控制按钮。宿主保留 Windows 缩放/贴边样式，以原生边缘命中测试调整大小，最大化使用屏幕工作区。无边框客户区计算交给 Qt 的 WM_NCCALCSIZE 路径，避免宿主提前消费消息导致原生框架边距缓存失配；Windows 原生恢复保留最大化状态和页面，不重载编辑内容。files.windowAction 仅分派移动、最小化、最大化/还原和既有关闭协商，不扩大文件访问能力。状态信号同步还原图标，关闭仍经过未保存编辑保护。其他平台保留系统装饰；兼容的 files.setCaptionColors 仍只接受两个不透明六位十六进制颜色，不修改系统设置。

## 本地服务及工作进程

[local_service.py](src/ptb_desktop/local_service.py)只经受控协议访问后端，不 import API 实现。源码模式启动 python -m ptb_api.cli --mode local，冻结模式由同一 EXE 的 --local-service 分派。

父进程通过 stdin 传入每次会话 token、任务库与文件目录。API 先绑定 127.0.0.1 临时端口再回传地址，LocalService 验证回环身份和健康状态，HTTP 请求携带 Bearer/Origin。API 自己拥有任务 worker，后者再启动白名单内的科学子进程。

退出逐层关闭所属 stdin/进程并有界等待，超时仅终止自己拥有的子进程。不能用清理端口或进程名的方式关闭用户其他服务。

## 能力分工

| 代码位置 | 责任 |
| --- | --- |
| file_provider.py、task_bridge.py、各 mXX_bridge.py | 目录 grant、受控导入/导出、研究任务路由及模块文件协议 |
| vocal_tract/client.py、worker.py、runtime.py | M10 专用进程、原生引擎、状态隔离和明确资源位置 |
| vocal_tract/audio_output.py | 声道模块的本机音频输出 |
| recording/、m16_bridge.py | M16 设备枚举/租约、采集、工程、编辑/处理、后台保存与导出 |
| papers.py、papers_bridge.py、paper_library.py | 推荐目录/选择下载、启动及可调周期检查、个人文件夹库、离线读取、批注与导出 |
| updates_bridge.py、update_coordinator.py、update_apply.py | 更新检查/下载、关闭协商、独立换版助手 |
| startup_cache.py、compact_runtime.py、bundle_manifest.py | 持久缓存、压缩运行时展开、依赖身份与运行绑定 |
| startup_*.py、presentation.py | 准备进度、启动显示和窗口呈现 |
| platform_paths.py、cache_cleanup.py | 用户目录身份及受控缓存清理 |
| app_icon.py、icon_preferences.py | 共享图标目录、窗口/打包图标与原子保存的个人图标选择 |

M10 的可变原生引擎与 M16 的采集工程各有独立生命周期。M16 使用 recording(request_id, JSON)/recordingReady，长处理先启动后台再轮询；停止检测、停止录音、取消处理和取消导出不能互相混同。协议见[recording/1](../contracts/recording/README.md)。

M09 的已有Python入口由 `m09_bridge.py` 管理：原生文件选择产生绑定解释器身份的grant，前端仅传grant请求兼容性检查，不能提交任意路径。检查在后台任务队列运行，验证依赖、实际CUDA运算及可用组件模型，成功后由worker原子保存解释器身份；运行前复核文件摘要，不安装或修改用户环境。状态查询与研究任务共用串行队列，组件下载沿用独立队列。

## 数据位置和升级

platform_paths 以操作系统用户数据目录为基础，Windows 主数据位于 %LOCALAPPDATA%/PhoneticToolbox/v3。Qt WebEngine 沿用 %LOCALAPPDATA%/PhoneticToolbox-v3/workbench，以保持网页草稿、偏好和 IndexedDB 身份。声道用户资料由 host 单独指定。

应用图标选择与规范化的自定义 PNG 保存在主数据目录的 icons/preferences.json，升级后保留。files.applicationIcon 使用独立的有界请求，不扩大既有 invoke 限额。前端和桌面读取同一随包 catalog，科学任务与可再生缓存清理不修改这些偏好。

冻结入口默认任务目录 local-preview-20260927 保留兼容命名。源码启动器使用工程 output/validation 中既有数据库/缓存。两者不能相互当作无用副本清理，完整位置表见[总架构](../ARCHITECTURE.md)。

用户选择的正式录音、导出和工程由用户管理。startup-cache 是可再生运行缓存，papers 含推荐论文、偏好及按内容摘要绑定的批注，updates 含更新状态，不能因同在用户目录而一并删除。个人 PDF 另存用户选定的根文件夹，收藏夹为真实子文件夹，导入保留来源；papers 只记录库根路径和批注。网络与 Qt PDF 操作分开排队，文档对象始终归阅读线程。

## 打包与测试边界

release/ 决定资源收集、运行时共享、安装包装和大小限制；宿主按构建清单运行，不依赖开发机 PATH。缓存先验证身份、文件和活动租约，再决定复用、修复或清理。

源码测试、实际 Qt、冻结进程、工程外启动、安装换版和实体设备是不同证据。定向测试在 desktop/tests，交互验证入口位于 scripts/ 与 tests/e2e；保留各自报告的具体范围，不把合成设备当成实体录音验收。
