# 反馈管理工具包验收

任务 FEEDBACK-R2，2026-10-11。这是前一版历史验收，本机当前 1.0.1 成品与 FEEDBACK-R3 变化见[反馈界面与邮件选择验收](feedback-r3-report.md)。本地交付 Windows 独立 EXE 与 Android 签名 APK，服务器更新已上线。范围为 Windows 11 最终 EXE、独立 Android 36.1 x86_64 模拟器最终 APK、公网 HTTPS 与 Ubuntu 隔离检查，未验证小米实体手机长期后台运行。用户端及既有真实 SMTP 收件证据见[反馈验收](feedback-report.md)。

## 实现与部署

管理端共用 Vue 页面和聊天组件，桌面双栏、手机列表/详情，支持搜索、未读、已解决、文字/图片回复、持久草稿和主题。Windows关闭进入托盘，再次双击唤起；设置可完整退出。电脑生成五分钟、一次使用的八位配对码，手机获得独立可撤销随机凭据，Keystore加密保存，重开及覆盖安装后保留登录。Android使用明确常驻通知的remoteMessaging前台服务，提供通知、小米自启动、电池和应用设置入口，厂商限制下不承诺即时提醒。

用户新留言与即时事件原子落库，幂等重试不重复。管理员回复产生会话更新事件，两管理端同步；09:00每日检查保留。设备授权以0600、文件锁和原子写入保护的JSON保存，只存HMAC摘要；撤销后既有SSE重新验权并断开。部署只更新授权源码，原数据库schema 2、SMTP、09:00日程、配置和原有会话未迁移。服务器健康回读 mailConfigured:true、allowUnverified:false、instantNotifications:true、adminPairing:true，服务active。

Windows包不包含科学模块、模型、研究服务、管理员凭据或签名秘密。Qt静态资源类抽至共用web_assets.py，主宿主保留兼容导入。完整Vue、Doulos OFL、PyQt6及WebEngine许可随包，来源沿用项目登记。Android原生桥仅接受固定HTTPS服务与明确操作，网页只读取本地资产，禁止外链、任意请求、文件访问、重定向与明文HTTP；设备凭据不传给JS，禁用云备份及设备迁移。

## 最终字节与实际检查

| 交付物 | 字节 | SHA-256 |
| --- | ---: | --- |
| 反馈收件箱.exe | 198553277 | 2f85afd98964fb185a576454652294d7a59128fc286dffa8afd4f43dba222bc1 |
| 反馈收件箱.apk | 583411 | b0e4d9b15532b1a83abd1c7c3dffc6137d238bfafbdc428cadbb322c7e90b9b3 |

产物在本机 `output/feedback-manager`，均低于500,000,000字节；未公开上传。APK为RSA3072签名、v2签名校验通过，后续更新使用工程外同一私有签名。

| 验证入口与环境 | 实际结果 |
| --- | --- |
| 当前源码绑定：backend反馈与设备测试、desktop反馈测试 | 29 passed，含配对一次性/过期/五次失败锁定/并发消费、撤销持久化、HMAC摘要、不泄露原始凭据、严格请求模型、即时事件与内容幂等 |
| frontend npm test、typecheck | 366 passed；类型检查通过 |
| frontend contracts:check、ui-data:check | 通过，无手改生成物 |
| Windows build_windows.py、verify_windows.py | 最终EXE独自复制到工程外目录，PATH仅Windows；实际Qt页面经私有TLS fixture连接，文字/图片回复、草稿与主题重新加载、即时SSE、配对/撤销、手机宽度布局、关闭托盘、再次启动唤起和完整退出通过，退出后无_MEI残留 |
| Android build_android.py --lint | 使用已有JDK/SDK/Gradle离线构建；assembleRelease、lintRelease通过，0错误、2提示：target35非最新平台；严格本地Vue页面启用JS需安全审阅。未添加lint抑制或降低检查标准 |
| 最终APK安装及公网HTTPS | 实际配对、读取会话、文字回复，Android系统PhotoPicker选图后文字与图片回复在用户API回读成功；force-stop后重新打开自动登录和草稿恢复，同签名覆盖安装保留登录 |
| 最终APK后台与系统通知 | 退到桌面，专用用户会话新增留言，系统展示新的用户反馈及常驻接收通知，实际前台服务运行；通知正文无邮箱或留言内容 |
| 最终APK实际模拟器重启 | 等待系统完成开机广播，在未打开应用时接收服务恢复到前台运行；重新打开保留登录。第一次过早读取sys.boot_completed时广播尚未完成，随后完整等待开机广播的检查通过，未修改APK或跳过断言 |
| 公网撤销设备与原生端恢复 | 电脑撤销专用模拟器设备后，原有SSE停止，手机重新验权并清除设备凭据，实际页面返回配对入口并提示授权失效 |
| verify_packages.py、apksigner verify | 最终包内无本机管理员凭据、私有配置/库/身份/签名文件，完整许可存在；EXE无外部ICU冲突库，APK签名有效 |
| Ubuntu部署隔离fixture及公网回读 | 即时消息、配对、撤销通过；源码哈希与本机一致，原schema及配置保留 |

Windows首轮最终包曾失败：Qt6Core需要Windows提供的未加版本号ICU函数，构建PATH误取其他工具的Poppler/ICU78同名库，其导出函数名不匹配。已限定打包PATH并重建，最终包不再包含这份外部icuuc.dll；相同工程外验收全部通过。失败记录保留。验收入口对本任务启动失败写独立日志并以非零退出，避免再次弹出PyInstaller回溯窗口；正常用户启动逻辑保留。Android首轮多选返回ClipData未处理，已正确读取单图/多图URI，并经真实系统相册验证。

## 证据与边界

本机忽略目录 `output/validation/feedback-manager` 保存 server-deployment.json、packages.json、android-final.json、android-signature.txt、android-build-lint.log、Android最终截图及Windows最终Qt报告/截图；Windows成功运行目录为 windows-final-26249ecb29884648979bf6fbc38ef594，失败目录单独保留，scratch清理回执均为deleted。

公网只使用独立、明显标注的 manager-package-selftest 合成会话验证，未修改原用户消息或向真实邮箱发测试邮件。自检结束后标为已读/已解决，撤销模拟器设备。cleanup_verification.ps1核对路径边界、owner记录、明确进程和联接后，停止本任务模拟器并清理AVD、工程外冻结运行副本、临时用户凭据、过期冲突构建及原始诊断dump，cleanup.json回执success:true。正式安装包、签名、本机管理员身份、已有用户AVD与最新必要构建缓存保留。

实体小米/红米HyperOS的省电菜单、自启动、重启和长期后台提醒尚未现场核验；Android 8至15其他设备未逐版本验收。Windows真实托盘已验证隐藏/唤起，系统弹出通知及其他电脑长期运行未专验。此次EXE只覆盖独立管理工具，没有重建PhoneticToolbox主应用或重复其全部用户反馈矩阵，主宿主共用资源导入兼容检查通过。邮箱投递复用已验证服务，本次未重复发信。可执行文件未添加Windows代码签名，安装包在本机交付，未上传发布渠道。后台启动约束参考[Android官方说明](https://developer.android.com/develop/background-work/services/fgs/restrictions-bg-start)，真实厂商行为以设备现场为准。
