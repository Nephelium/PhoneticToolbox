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

## 1.1.0 扩容与服务器管理（2026-10-16）

已构建本地EXE/APK，审批、监控、布局修复与最终Windows证据统一见[本批验收](web-launch-foundation.md#2026-10-16-管理端110http准入与官网四项统计)。默认进入用户反馈，监控单独页签，切换保留草稿。新APK仅构建/lint/内容检查通过，实际Android/HyperOS与生产管理桥接待完成；历史1.0.1设备证据不外推。

## 来源与版本显示1.1.1（2026-10-16）

任务FEEDBACK-PROVENANCE。本次范围为本地源码和收件箱1.1.1 EXE／APK，不继承上文历史部署或Android运行结论。工作树为main／origin/main，保留同期网页上线、反馈运维等已有改动。

### 行为与兼容

会话列表和头部显示最近一条用户留言的来源，每条用户留言旁保存当时的网页版，或3.0.1／3.0.2等主程序版本与安装版／免安装版。管理员回复不覆盖用户来源。同一用户升级或切换网页后，历史信息保持。历史数据缺字段时显示来源未记录、版本未记录或安装形式未记录；源码启动显示源码运行。

主程序host复用更新器版本和package_kind安装标记，通过反馈HTTP头上报；官网site.js发送web来源头，旧服务器忽略新增头仍能接收原消息正文。新服务验证source/version/packageKind的枚举、版本格式与长度，并兼容旧桌面User-Agent与网页Origin。来源为客户端自报的排查信息，不用于身份、权限或可靠设备识别，不采集IP/MAC/硬件信息。

元数据写入现有new_feedback事件JSON，与消息同事务提交；重复requestId保留首次信息。读取会话时按messageId关联，摘要取最后一条用户消息。反馈SQLite schema版本仍为2，不增表/列，不迁移现存库，旧记录不据正文猜测回填。

### 实际验证

- Windows 11，项目现有.venv/m14解释器，workbench_source.bind_sources绑定当前工程。pytest采用 `-o addopts= --import-mode=importlib -p no:cacheprovider -q backend/tests/test_feedback.py backend/tests/test_feedback_devices.py desktop/tests/test_feedback.py`，38项通过。覆盖持久化、逐条版本、升级后重试、来源切换、管理员回复、历史缺失、schema原样、请求头校验、两类包与源码上报、管理员不冒充用户来源。首次命令受根旧cov参数及同名测试导入影响，使用当前工作树绑定及importlib收集后完整通过，未删除断言或放宽阈值。
- frontend中 `npm run typecheck`、`npm run contracts:check`、`npm run ui-data:check`、`npm test`、`npm run build`均通过。独立管理页面使用Vite `--config vite.feedback.config.ts`构建，prepare_assets后打包。未新增依赖。
- `node scripts/verify_website_feedback.cjs`：真实独立Chrome与回环反馈服务验证邮箱、错误验证码、带图提交、丢回执重试只入库一次、client为web、回复、邮件偏好、重开、身份隔离和1440/390/320宽度通过。第一次在无关Release键盘断言失败，原测试假设只有一个真实版本；修正为按实际相邻页签断言，另验Home返回首项后完整通过。失败证据feedback-web-1792148955540保留，成功证据feedback-web-1792149000231。
- `python tools/feedback-manager/verify_windows.py`：当前最终EXE在工程外、独立DPAPI与WebEngine配置、Windows-only PATH、私有TLS、可见默认GPU运行。断言安装版／免安装版／网页版／来源未记录四标签；文字/图片回复、草稿/主题重启恢复、即时SSE、配对/撤销、托盘隐藏/重新唤起/单实例、1280×720/1000×650/390×844/390×600布局通过。人工回看浅色与390×600最终截图，文字可见、回复框完整。证据windows-final-d2730f5a82484e24b70e978f69a434b4，正常退出且无_MEI目录残留。未以窄屏Qt替代Android运行结论。
- `python tools/feedback-manager/build_android.py --lint`：沿用已有SDK/JDK/Gradle及原签名，离线assembleRelease/lintRelease通过，0错误、2项既有WebView警告。Android versionCode 4／versionName 1.1.1。adb devices无设备，本次未安装或运行Android/HyperOS。
- `python tools/feedback-manager/verify_packages.py`：两包私有凭据/文件缺席、完整许可、头像同字节、Windows无外来ICU检查通过。逐文件直接比对本轮EXE与APK内全部web资源通过，证据shared-assets.json。共用构建目录在打包后被同期任务重建，未用可变目录扩大最终字节结论。文档验证scripts/validate_docs.py通过（errors为空），受影响文件UTF-8无BOM、Python语法、台账唯一ID与git diff --check通过。

### 最终交付字节

目录 `output/feedback-manager/1.1.1`。

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| 反馈收件箱.exe | 199074359 | be80fc4447596d3beba4702c81e77a99036a40fc8d6384569afd509537d33105 |
| 反馈收件箱.apk | 1116246 | b0bed820dcbfe4e979df16cd69198ba71571ac91d30417a42a58e0021e726ce6 |

### 生效与清理边界

来源接收随后随下方回收站升级部署正式反馈服务，官网来源头尚未部署，主程序最终EXE未重建；只换收件箱不会让旧主程序上报版本。完整版本与安装形式要随新主程序启用，既有旧记录无法自动补齐。Android/HyperOS实际显示、后台通知、其他电脑与长期运行本轮未验。

两轮网页scratch及Windows内部scratch已清理，报告/摘要/截图保留。自动审批拒绝删除本次工程外Windows QA副本，返回blocked by policy，未尝试绕过或改用其他删除方式。该副本已正常退出，清理状态blocked，未声称释放其空间。旧正式收件箱及用户运行实例保持。

## 回收站与未读标题1.1.2（2026-10-16）

任务FEEDBACK-TRASH。范围为本地反馈服务、Windows／Android共用管理界面及最终1.1.2 EXE／APK。v3工作树main跟踪origin/main，保留同期其他任务改动，没有部署生产，也没有对真实会话执行删除、恢复或到期清理。

### 回收站语义

删除会话须确认，随后移入回收站。保留30×24小时，可查看完整文字、图片、当时的来源/版本和到期时间，恢复后继续回复。用户在回收站期间提交新留言会自动恢复；原消息的幂等重试不触发恢复。重复移入不延长原到期时间。删除确认绑定列表中的最后消息ID，期间有新消息即拒绝旧确认。

回收站状态使用现有events JSON中的conversation_trashed／conversation_restored，沿用SQLite事务与schema 2，无增表/增列或现存库迁移。回收站排除普通列表、未读、每日提醒；管理员只能查看/恢复，不能回复、改状态或重试邮件。正在投递邮件时暂拒绝移入；待投递/失败邮件取消，应用内回复和图片保留，恢复不补发旧邮件。

既有维护任务启动时、每分钟检查到期，相关会话读写前也检查。到期永久删除消息、图片、来源/状态事件及相关投递记录，保留无内容的删除同步事件和邮箱验证身份。用户以后仍能重新留言，旧管理员窗口到期后不能用迟到回复重新创建已清除会话。回收站内只读预览不自动标为已读。服务停机期间无法运行清理，重启后补做。

### 标题残留的原因与修复

旧FeedbackManager.notify仅在daily_feedback／new_feedback事件到达时setWindowTitle，没有依据已读状态清除，因此界面0未读时标题仍可残留“1个会话有3条新留言”。这与截图中10/15三条已显示留言一致；未据标题推断存在额外隐藏留言。

当前桥依据实时会话摘要发出inboxState，标题和托盘提示显示当前未读数量，0时恢复基础标题。历史SSE通知先核对当前真实未读，回收站排除，历史计数不能覆盖当前标题。批量重连事件合并摘要查询，避免为每条旧事件排队网络请求。mark_read仅在seen从0变1时发出conversation_changed，同步其他管理端且不造成刷新循环。

### 验证与最终字节

- 现有.venv/m14解释器与workbench_source绑定当前工程，pytest使用 `-o addopts= --import-mode=importlib -p no:cacheprovider -q backend/tests/test_feedback.py backend/tests/test_feedback_devices.py desktop/tests/test_feedback.py`，51项通过。覆盖回收站持久化/重启、图片/来源/状态保留、幂等截止时间、确认竞态、邮件取消/租约、普通用户/配对设备/撤销权限、精确30天边界、启动后台自动清理及原生标题/历史事件。之后补强到期旧管理员回复拒绝，受影响backend反馈27项再验通过。
- 前端401项测试、typecheck、UI-data、契约生成/匹配与完整build通过。OpenAPI和TS按生成器维护。草稿写入按键串行，永久删除同步清空对应管理员本机草稿，避免待写事务覆盖清空结果。
- Vite用独立 `output/build-feedback-manager/web-1.1.2`，资源准备及两平台脚本经进程级PTB_FEEDBACK_WEB_ROOT绑定同一目录，防止同期构建覆盖。Windows PyInstaller及Android离线assembleRelease/lintRelease通过，Android versionCode 5／versionName 1.1.2，lint 0错误、2项既有WebView警告。没有安装依赖或修改系统设置。
- 最终Windows EXE工程外、独立DPAPI/WebEngine身份、私有TLS、Windows-only PATH、可见默认GPU验收通过：此前来源标签/文字图片回复/持久草稿主题/即时SSE/配对/托盘/单实例/四尺寸；新增删除取消保留草稿、移入后列表消失、回收站只读历史与图片、恢复原会话和草稿、390×600布局、推进测试时钟满30天后彻底清除、新留言仅1条未读、读完原生标题归零、重放历史3条通知仍保持归零。成功证据windows-final-b7ff304cfc0844db81267c20ea974170。正常退出无_MEI残留。
- 初次最终EXE验证windows-final-92eb70a4c2e1438d83ef941fe947df67在回收站标题已出现但消息尚未异步加载时过早断言失败；修正测试等待实际消息/草稿就绪，同一最终EXE字节完整重跑通过，未删除断言或放宽阈值。人工查看最终宽/窄回收站和删除确认截图，控件、文字和图片可见。
- verify_packages.py的凭据/私有文件缺席、完整许可、头像和外来ICU审计通过；最终两包与冻结输入全部10项web资源逐字节匹配，证据frozen-assets.json。未连接Android设备，本轮不把Qt窄屏写成Android/HyperOS实机证据。

| 交付物 | 字节 | SHA-256 |
| --- | ---: | --- |
| output/feedback-manager/1.1.2/反馈收件箱.exe | 199078134 | 75c84f5563126ac03a4f98e2d749d68992c8726c87754419d7c7d994104ed9e4 |
| output/feedback-manager/1.1.2/反馈收件箱.apk | 1117794 | 3769d6aefeda9f1b97a498002aa3283b65363448280818ae5d03ccd706d6e6ba |

### 用户所见空列表的只读核查

用户随后报告记录全部不见。直接使用既有正式管理身份GET正式HTTPS服务，普通列表与includeTrash查询均为5会话、0未读；健康schema 2、邮件配置与即时通知正常。当前可见空窗口的进程脚本为同期另一网页上线任务的tools/feedback-manager/launch_isolated.py，其LOCALAPPDATA重定向到独立测试配置，固定连接127.0.0.1:18768测试服务，标题带隔离测试，与正式反馈身份不同。该脚本/测试窗口属于同期任务，本轮未修改或结束。

已另开本次最终1.1.2 EXE，使用真实Windows用户已有DPAPI管理身份连接正式HTTPS服务。只读检查实际Qt页面5张会话卡、0未读、回收站入口存在、未选择任何会话；保留此正式窗口供用户操作。证据output/validation/feedback-manager/live-inbox-recovery-result.json，只记录计数，不含邮箱、内容或凭据。没有向正式服务写入测试消息、修改记录或运行本地清理逻辑。

### 未完成范围与产物

正式反馈服务已按下方后续部署记录启用回收站/30天能力。来源/版本上报仍需官网及新版主程序配合，旧记录无法补齐未知字段。Android/HyperOS实机、跨电脑/长期后台与实际Windows弹出通知未验。

两轮内部scratch已由现有生命周期入口清理，报告/截图保留。工程外QA副本已停止但保留，前轮同类删除被自动审批策略拒绝，本轮没有重复尝试或绕过。正式收件箱正在运行，不能作为测试进程结束或清理。

### 后续修正：旧正式服务能力检查与1.1.3

用户截图中的“反馈服务暂时不可用”由真实服务缺少新路由造成。只读GET确认正式服务active、5会话/0未读，回收站路径404；未重试POST、没有修改真实会话。此前直接打开1.1.2客户端并未同步部署正式服务，缺少能力门控属于交付遗漏。

1.1.3增加严格能力检查：健康响应显式conversationTrash=true且trashRetentionDays=30才启用删除/恢复/回收站，旧健康字段缺失时停用并说明先更新反馈服务。普通查看/回复不受影响，刷新会重新检测。Windows／Android原生对缺路由404/405提供功能未启用提示，实际conversation_unavailable仍显示会话不存在，不清除身份。修正后没有将正式服务整体不可用作为结论。

56项当前Python反馈/设备/桌面检查、402项前端测试、typecheck/契约/UI-data/完整build通过。1.1.3最终EXE工程外默认GPU完整矩阵通过，新增旧健康模拟下删除/回收站停用、准确提示、点击不发POST、切回新服务刷新自动启用；此前回收站/30天/标题/回复/草稿/主题/配对/托盘/四尺寸仍通过。证据windows-final-05d1d8e446164e088869035df85d54f5，人工查看旧服务提示截图。两包共用web-1.1.3全部10项冻结资源相同，许可/私密文件审计通过。APK release/lint 0错误、2项既有警告，Android versionCode 6／versionName 1.1.3；本轮没有Android实机。

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| 1.1.3/反馈收件箱.exe | 199078983 | 83919e850eaaabd6666b16fe42cc2f3d832e1d14f4f450ca71a86c175b769c87 |
| 1.1.3/反馈收件箱.apk | 1118138 | d78c3e2fcb3a5598ca7e5f7b09528914bb725010b9c3e32b20f791f3c3a48abe |

生产升级准备仅涉及/opt/phonetictoolbox-feedback/backend/src/ptb_api/feedback.py，候选SHA eb4fbdd247974ff54710e24ee6c8082f28406a811767d43bedf734f3c58eef2b，前置生产SHA fe77faebdffec376edbbce8130b2dab9888b71b9f3ed3fd3f4a3f8312a72624b。已读取生产当前源码/依赖/入口，候选去掉尚未上线web_ops接口及其依赖；31项候选反馈/设备检查通过。通过SSH标准输入在真实Ubuntu既有venv运行候选，仅创建独立临时库，验证schema原样、图片和来源保存、恢复/新留言自动恢复、取消排队邮件、30天边界清除、其他会话保持与能力标志，未访问生产库或发送邮件，临时目录结束后清理。生产源码、服务配置、凭据及真实数据未写入。

完整差异、manifest、回退和备份步骤以及可复跑Linux脚本保留在output/build-feedback-manager/production-upgrade-1.1.3。本段准备完成时尚未部署，后续井井明确授权更新，正式部署及完整性结果见下一节。当前旧1.1.2托盘实例已不属于此前本任务启动的PID，未结束或覆盖，用户需从设置退出旧版再打开1.1.3。内部测试scratch已清，工程外副本仍保留，不重复尝试此前被策略拒绝的同类删除。

### 正式反馈服务单文件升级与运维缺数据核查

井井明确授权更新正式服务，并要求检查扩容与服务器数据。重新核对生产feedback.py、feedback_admin.py、包入口与server.py四件基线SHA后，仅替换既有反馈实现一件，沿用原服务、Python环境和配置。候选按已审阅内容转换LF换行，最终线上SHA为eb4fbdd247974ff54710e24ee6c8082f28406a811767d43bedf734f3c58eef2b。没有迁移schema、修改私有配置/密钥、Nginx或隔离网页实例。

停止服务后通过SQLite backup接口生成完整快照，并备份旧源码和配对/偏好文件，路径/权限受限于正式反馈数据目录。最终备份为/var/lib/phonetictoolbox-feedback/upgrade-backups/20261016T120136Z-113，服务器保留，不下载真实消息或凭据。原子替换后同一systemd服务恢复active，loopback及正式公网HTTPS均返回conversationTrash=true、trashRetentionDays=30，管理鉴权通过，GET删除/恢复路径为405，确认POST路由已注册而未实际执行删除。

升级前后逐表摘要核对：schema 2及12项schema对象、5个邮箱身份、16条消息、2个图片BLOB、6项用户会话凭据、5个会话全部一致；配对设备/邮件偏好文件和私有配置字节摘要一致。公网只读列表仍5会话、0未读、0回收站会话。未对已有真实会话试删除/恢复、修改已读/状态或发送测试消息，没有人工测试发信。30天到期与删除/恢复行为沿用前述精确候选Linux临时库及最终EXE隔离验证，不将能力检查写成真实记录删除验收。

最初上传保留Windows CRLF，与manifest的LF摘要不符，远端前置检查中止，正式代码未替换。修正为精确LF载荷后，首次停止受通知长连接阻挡，30秒命令超时；当时尚未安装代码，旧服务恢复。部署脚本改为观察指定服务状态、必要时仅对本次重启的主进程发送第二次SIGINT，随后重新核对基线、备份与正式部署成功。失败stderr及最终回执均保留，没有修改systemd超时或依赖配置。

扩容与服务器页面独立核查：正式/admin/web/status和/admin/web/requests均404；现有反馈入口未加载网关，私有配置没有webOperations。真实/proc/容量/账号监控目前运行于Linux隔离网页18765与隔离反馈18768，管理页只在该内测连接下能取得数据。正式网页版尚未上线，因此本次反馈单文件升级不会提供生产账号、容量申请及运维统计。未把正式管理端连到隔离库、未伪造0值，也未扩大为正式网页版上线或修改其凭据。生产运维桥接仍待后续明确范围与上线准备。

证据：output/build-feedback-manager/production-upgrade-1.1.3/{apply-preflight.json,deployment-result.json,public-verification.json,deploy.py,deploy-remote.py,verify-live.py}。安装包字节沿用已验证1.1.3，本轮只更新服务器，不重建EXE/APK。收尾validate_docs.py核查2108文件/127任务，errors为空，历史快照断链单列；UTF-8/JSON/部署脚本语法及git diff --check通过。
