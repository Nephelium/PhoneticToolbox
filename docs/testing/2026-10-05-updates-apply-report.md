# Windows 更新换版接入与关闭保护

2026-10-05。状态：verified，限定 Windows 源码、安全纯核心、真实 Qt queued 线程与最大化隐藏工作台关闭接线。冻结成品实际安装/免安装换版、真实发布包下载、启动后科学运行及设置保留由主线继续验收。

## 实现

- 更新入口保持双确认与下载大小/SHA-256 校验。每个新进程首次启动必查两来源，重复组件挂载只复用本进程结果。同版本通知仍按实际展示后24小时抑制。两个来源均无可验证发布时返回 incomplete，同版本包摘要不一致返回 PACKAGE_CONFLICT 并阻止换版。
- host 注册 UpdateCoordinator。apply 的检查、hash和 helper 准备在工作线程，queued signal 进入 GUI 线程。活动/排队任务、原生处理中操作、M16 采集/检测/后台处理、M05 capture lease 和正在进行的感知实验拒绝更新。
- 前端沿用逐标签保存、放弃、取消关闭流程。取消或保存失败保留窗口与编辑。仅完成前端保护、原宿主 RequestClose/beforeunload 路径和原生录音收尾后，closeEvent 才交给 helper。source 模式和普通 test=True 宿主不提供 apply，测试只在独立脚本进程注入 dry prepare/launch adapter。
- 冻结 EXE 使用固定 --ptb-apply-update 分派。当前可信 EXE、_internal 顶层冻结运行文件和 desktop/src 复制到 updates/helpers/<uuid>，逐文件摘要登记并校验。helper 不绑定旧科学环境、不初始化工作台，在新的 _MEIPASS/cwd 下执行，避免锁住安装目录旧 EXE/DLL。
- helper 继承 SYNCHRONIZE/QUERY_LIMITED_INFORMATION Windows process handle，以 GetProcessId 校验当前进程身份，等待该 handle 完成。原程序未真实退出则不执行换版。包在交接前与退出后再次校验 size/SHA-256。
- 免安装 ZIP 唯一顶层 PhoneticToolbox/，严格拒绝绝对/逃逸/反斜杠/空路径/Windows设备名/非法字符/末尾空格或点/casefold重复/文件目录冲突/链接/加密与不支持成员。最多80000成员、20GiB展开、单成员4GiB、压缩比1000、相对路径1024字符、单路径成分255字符。展开到新的唯一兄弟版本目录，保留旧目录。校验 application.json 的 schema/version/entry，以及 _internal/desktop-bundle.json 的独立运行时路径与摘要。
- 安装版读取同级 .ptb-installed.json（ptb-install/1、installer），启动已校验的本软件安装器 /CURRENTUSER /VERYSILENT /SUPPRESSMSGBOXES /SP- /NORESTART /DIR=<当前应用目录>。等待退出码0并重查 application.json、EXE和运行时布局后再启动应用。没有卸载、旧版目录清理或用户数据目录操作。
- helper status.json 的 started 仅表示进程启动成功，未标记业务或窗口运行已验证。失败记录 failed，下载包与免安装旧目录保留；安装器内部失败的恢复能力仍需成品实测。

## 更新缓存

生产宿主首次启动及每小时执行独立维护，7天只作用于 updates/downloads、helpers、apply 的 UUID 子目录。cache-index.json（ptb-update-cache/1）按目录身份记录首次发现，历史无可靠时间则首次登记宽限。本进程下载、awaiting-close/waiting/launching 请求和其引用的 package/helper 保留。未能核实 pending 请求时全体缓存暂保留并报告 partial。删除前验证完整树无链接/Windows junction且位于owned根内，删除失败保留登记并报告 partial/failed。不扫描系统Temp、用户下载、程序目录、语料或托管科研结果。

## 实际验证

1. `.venv/v3-dev/Scripts/python.exe` 插入 desktop/src 后运行 `pytest.main(['desktop/tests/test_update_apply.py','desktop/tests/test_update_cache.py','desktop/tests/test_update_coordinator.py','desktop/tests/test_updates.py','desktop/tests/test_updates_bridge.py','-q','-o','addopts='])`：107/107，通过，无skip。包括真实owned Windows process handle等待/身份拒绝、ZIP边界、改包/helper/manifest拒绝、安装失败不启动、安装成功布局检查、实际Qt GUI线程排队、取消关闭不启动、采集/任务拒绝、缓存首次宽限/到期/保护/删除失败及真实Windows junction越界保护。
2. `npm --prefix frontend run typecheck`：通过。
3. `npm --prefix frontend test`：327/327，当时完整前端检查点。
4. `npm --prefix frontend run build`：通过，既有大chunk提示保留。
5. `.venv/v3-dev/Scripts/python.exe scripts/verify_updates_apply_qt.py`：5组通过。报告 `output/validation/updates-apply/qt-f3c58ebcadd446ad8a7ed08c31dc8982/report.json`。真实maximized隐藏Windows Qt，实际QWebChannel、worker/GUI线程、M13取消保文本与保存localStorage、真实合成感知会话拒绝更新、原生最终关闭后dry helper仅一次。owned本地服务正常退出0。无真实下载/安装/换版，此测试明确使用本地合成下载身份与dry helper。

初次Qt按钮定位未排除图标文本，修正测试定位；第二次实验结束自动导出触发保存对话框等待，保留失败目录并仅终止核实命令的owned测试Python。保存选择器改为测试专属输出后通过。隐藏Qt使用禁GPU旗标并有GLES fallback日志，不外推为实体DPI/DWM/默认GPU表现。

## 主线冻结后必验

- 新onedir冻结产物 stage_helper 最小运行文件是否足以执行 PyInstaller runtime hooks及固定helper分派，确认不绑定旧目录/科学环境。
- 发布实际ZIP逐成员安全准入和native完整下载/二次校验，免安装真实父进程退出、新兄弟目录启动、原用户目录设置与草稿回读。
- 实际当前用户安装器更新，确认在旧进程退出后替换成功、helper目录不锁旧EXE/DLL、安装器失败/退出码行为、应用只启动一次，以及相同userdata设置保留。
- 成品正常启动和科学运行由主线验收；started不能替代这些检查。无需GitHub发布，本轮没有push、上传、安装、生产发布或用户文件清理。
