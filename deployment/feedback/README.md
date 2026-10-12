# 反馈服务部署

批准方案见 [计划](../../docs/plans/2026-10-10-feedback.md) 与 [ADR](../../docs/decisions/ADR-feedback.md)。此目录不保存真实密钥。

当前服务器使用Ubuntu24.04/Python3.12.3的独立venv，桌面仍按项目Python3.11规则。反馈服务仅依赖FastAPI0.141.1、Pydantic2.13.5、Starlette1.6.0、uvicorn0.52.4与Pillow12.3.0及其传递依赖；安装使用TLS验证的包源，不修改系统Python或安装科学环境。`server.py` 与统一后端复用同一个反馈 API 实现，独立服务不导入科研计算、Qt 或现存科研数据库。正式服务只监听回环，由现有 HTTPS 站点代理 `/api/v1/feedback/`，静态首页与论文路由保持原配置。

私有 JSON 需要 `database`（独立新库路径）、至少 32 字符的 `secret` 与 `adminToken`、`port`（默认 18766）、`dailyHour`（默认北京时间 09:00）。当前部署含 `smtp` 对象及 `allowUnverified: false`，新用户首次验证邮箱；邮件暂停期间创建的未验证会话保留，由原设备凭据继续保护。已有用户可在反馈页验证联系邮箱以升级自己的会话，相同未验证邮箱不会合并或解锁其他会话。暂停期间的旧回复不排队或自动补发，未验证邮箱的新回复继续显示手动邮件，验证后的新回复才进入邮件队列。

发信配置二选一：`smtp` 对象含 host/port/user/sender/password/ssl，或 `smtpSource` 指向已授权本机脚本，后者只静态解析四个 SMTP 常量。`password` 填客户端授权码。服务器部署时只复制所需私有配置，不复制整个批量脚本。文件属服务用户、0600，目录0700；不把真实 JSON 加入 Git或分发包。当前用户脚本中的原腾讯服务器地址已过时，服务器私有 `smtp` 使用[校方公布](https://inc.sysu.edu.cn/cat/165)的 `smtphz.qiye.163.com:465` / TLS，未修改外部脚本。校方目前规定域外邮件每个账户每日最多200封，验证码与回复共用该额度。

```powershell
.\.venv\m14\Scripts\python.exe -B deployment/feedback/server.py --config <私有配置绝对路径> --check-smtp
```

上述只检查 TLS/SMTP 登录，不发送测试邮件。配置真实SMTP并启用后，验证码与已验证邮箱的新回复按持久队列投递；回复文本保存在会话，图片经校验和去元数据后同时作为邮件附件。SMTP 接受不能证明已经进入收件箱。失败指数退避，最多 12 次，管理界面显示失败并提供手动重试。

每日检查及发信维护都在服务器进程，systemd 确保常驻与重启。按 Asia/Shanghai 日历每天一次，9 点后服务恢复时补跑当天检查；长期停机后的未处理新增留言在当前检查归并通知，不逐天制造重复。SSE通知持久保存在库中，工具从本机最后通知 ID 补收，无需用户电脑入站端口。多次SMTP发送的不确定响应可能导致重复邮件，固定 Message-ID 协助收件端去重，不能承诺 SMTP 恰好一次投递。

上线前只读检查服务器 Python/依赖、站点与日志配置、磁盘、已有同名服务和库。为独立新库/服务准备路径与最小权限，核验配置后 `nginx -t`，使用既有服务管理流程启动和回读 HTTPS。真实站点配置、服务用户和凭据均由现场部署处理，不将模板存在标为上线。

Windows 管理入口：根目录 `打开反馈管理工具.vbs` 或 `scripts/Start-Feedback-Manager.ps1`。首次输入服务地址与管理员凭据，后续 DPAPI 自动恢复。关闭窗口隐藏到托盘，托盘“退出”才停止，电脑离线时提醒留在服务器。

当前现场部署：`ptb-feedback.service`已启用并运行，源程序及venv在 `/opt/phonetictoolbox-feedback`，私有配置在 `/etc/phonetictoolbox-feedback/private.json`，独立库在 `/var/lib/phonetictoolbox-feedback/feedback.sqlite`。服务用户为ptb-feedback，配置与数据目录0700、配置/数据库0600；服务只监听127.0.0.1:18766，systemd启用最小写入范围及重启。现有www HTTPS站点增加限定反馈路径的代理，nginx测试及reload成功，反馈access/error日志关闭，原站点配置保留在include目录之外。

Windows本机管理身份已用DPAPI配置，可直接启动工具。SSH密码按用户要求保存在Windows凭据管理器，仅本机辅助连接读取，不存入本工程和Codex记忆。公网保留此前标为功能自检的合成会话，已读并已解决。邮件接通时，生产HTTPS验证码与Linux隔离回复outbox各发送一封到用户自己的邮箱，已通过只读IMAP按两封特定Message-ID确认收到，回复附件摘要一致。隔离库已清理，生产原有身份、消息与附件数量未变，未向原会话追加邮件自检消息。详见[验收](../../docs/testing/feedback-report.md)。

管理工具包见[说明](../../tools/feedback-manager/README.md)。服务器已增加消息事务中的即时事件及独立可撤销手机配对接口。设备授权存入反馈目录 `feedback-admin-devices.json`，0600、文件锁、原子写入，只存凭据 HMAC，不改数据库 schema 2。SSE 每轮重验授权，撤销立即生效。现有 SMTP、09:00 日程和用户会话配置保持原值，部署与最终包证据见[管理工具验收](../../docs/testing/feedback-manager-report.md)。

## 用户邮件偏好

已认证用户通过 POST /api/v1/feedback/preferences 保存严格布尔 mailNotifications。服务端复用私有 feedback-admin-devices.json 的 mailPreferences 元数据（0600、锁与原子写入），部署和备份应保留此文件。默认开启，关闭取消未投递回复并阻止新回复入队；验证码不受影响。SQLite schema 2 与每日 09:00 检查不变。
