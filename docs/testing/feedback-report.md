# 用户反馈与独立管理工具验收

任务：FEEDBACK-R1。当前核验日期：2026-10-11。状态：verified，范围为当前Windows源码、源码启动的独立管理工具和已部署Ubuntu反馈服务。SMTP已接通并验证真实收件，冻结包完整反馈矩阵另列未验。

本报告保留R1已经执行的检查与当时限制。独立Windows管理EXE、Android签名APK、手机持久登录与即时通知现由[FEEDBACK-R2管理工具验收](feedback-manager-report.md)覆盖；主应用的冻结反馈完整矩阵仍未在R2重验。

## 实现范围

- 共用工作台关于页增加问题反馈。当前SMTP已接通，新用户首次验证联系邮箱，无注册页、用户名或用户密码。暂停期间建立的未验证会话可通过原设备凭据继续访问，或验证邮箱后升级并保留历史；有其他已验证身份冲突时拒绝静默替换历史。相同未验证邮箱不同设备不共享会话，凭据丢失时不能仅凭未验证邮箱恢复。
- 用户身份凭据由随机数生成，服务端保存 HMAC 摘要，Windows 本机由 DPAPI 保护。反馈与管理员凭据分离，普通用户端不含管理员或 SMTP 秘密。
- 文字、PNG/JPEG/WebP、多图预览、图片放大、草稿、失败后保留、内容幂等、未读提示与后续交流。图片每次最多4张，每张2 MB，解码重编码去元数据；每个会话总量上限100 MB。
- 独立 Qt 管理工具复用静态 Vue 资源和主题，支持邮箱搜索、未读/待处理/已解决筛选、回复、关闭到托盘、投递状态和手动重试。入口 `打开反馈管理工具.vbs`，也可 `scripts/Start-Feedback-Manager.ps1`。
- 独立反馈 SQLite 数据库存储与研究数据隔离。指向现存非反馈库时在只读检查阶段拒绝，不增加表或修改原库字节。
- 服务器按 Asia/Shanghai 每日09:00 检查一次。通知在事务中持久保存，Qt工具建立 SSE 长连接，离线重连补收；不创建 Codex 自动化或电脑入站端口。
- 回复消息正常进入应用。当前SMTP开启，已验证邮箱的新回复与邮件outbox原子落库，正文及图片附件立即投递并重试。旧回复不自动补发，邮箱仍未验证的会话继续显示可手动联系。固定Message-ID辅助去重，SMTP超时不确定结果可能造成重复邮件。
- 反馈库不保存 IP、硬件标识或设备指纹，不自动读取研究文件。新反向代理模板关闭反馈访问日志，真实站点/系统日志仍须部署现场核对。

## 已执行验证

环境：Windows，项目现有 `.venv/m14` Python 3.11、Qt WebEngine，Node 24，当前工作树源码绑定。未安装全局依赖。

| 验证 | 实际结果 |
| --- | --- |
| backend/tests/test_feedback.py、desktop/tests/test_feedback.py 及 tests/contracts，通过 workbench_source.bind_sources 与 pytest importlib 模式运行 | 112 passed，2条既有 Starlette/anyio 弃用警告 |
| frontend `node --test tests/feedback.test.ts` | 2 passed，迟到响应、断连和通知隔离 |
| frontend `npm run typecheck`、`npm run contracts:check`、`npm run ui-data:check`、`npm run build` | 通过，构建保留既有大 chunk 提示 |
| frontend 全量 `npm test` | 348/349 通过；source-coverage 的 M05 PKG-PYAV 模块来源登记与现存覆盖表不一致。本任务未修改这些来源清单，不删除断言或扩大本任务修复范围 |
| scripts/verify_feedback.py | 真实回环 HTTP、SSE 和本机 SMTP接收器通过，验证码/回复及图片附件实际收取，幂等及未读状态通过 |
| scripts/verify_feedback.py --qt（本机SMTP fixture模式） | 实际隐藏 Qt：关于入口、邮箱输入/验证、发送、独立管理工具收取/回复、用户同步、邮件状态、解决状态、图片放大、浅深主题通过 |
| scripts/verify_feedback.py --qt --no-mail | 关于页联系邮箱直接进入会话，Qt两端文字/图片/回复/未读/已解决/浅深主题通过，真实邮件发送0条 |
| Windows原生客户端 → 真实公网HTTPS | TLS证书核验、身份、私有图片提交、内容幂等、管理员回复、用户应用同步与未读、已解决通过；SSE通道认证并成功连接 |
| Ubuntu24.04/Python3.12.3 隔离检查 | 源码无包元数据导入、同邮箱设备隔离、09:00时区日程、每天仅一次、重启/持久回执、已读不提醒、邮件关闭、现存非反馈库字节不变通过，临时库已清 |
| scripts/Start-Feedback-Manager.ps1 -CheckOnly | 当前解释器、入口及构建资源存在 |
| 本次 backend/tests/test_feedback.py，当前源码绑定、pytest importlib模式 | 16 passed，2条既有Starlette/anyio弃用警告；根测试配置引用的旧覆盖插件在此环境不可用，按现有定向入口覆盖addopts后运行，未修改测试断言 |
| 本机及Ubuntu真实SMTP认证 | 脚本原腾讯地址535；按校方当前设置更正部署输入后，`smtphz.qiye.163.com:465` TLS/授权码认证均通过，认证检查本身不发信 |
| 服务器SMTP私有配置与服务回读 | 原子写入原路径，0600/服务用户保留，未改数据库schema；源文件摘要一致，systemd active/enabled，公网HTTPS回读mailConfigured:true、allowUnverified:false、每日北京时间09:00；未验证新身份创建返回403 |
| 公网HTTPS验证码 → 真实邮箱 | 一次请求，一封邮件；服务器队列sent、正文已清空，按精确Message-ID只读IMAP确认收件与验证码格式。首轮45秒未找到，稍后确认投递，未重复请求验证码 |
| Ubuntu隔离反馈库 → 真实SMTP回复及附件 | 源码FeedbackService持久outbox、SMTPMailer真实投递，应用会话可读且mailState:sent；邮箱收到一封回复，正文匹配，图片附件SHA与服务端去元数据后的图片一致；隔离库已清理 |
| 现有生产会话及管理工具 | 身份/消息/附件数量与启用前相同，未向原会话添加邮件自检消息；本机既有DPAPI管理员凭据通过公网HTTPS认证，未重写凭据或通知游标 |
| scripts/check_architecture.py | errors为空 |
| scripts/validate_docs.py | 当前文件errors为空，历史快照断链单列 |

Qt 首次收尾时，Chromium DawnGraphiteCache仍占用，清理保留明确回执。退出后通过原生命周期 finalizer 重新完成 scratch 摘要与专用清理，实际回执deleted；后续脚本增加 DeferredDelete收尾。保留报告与截图，不保留测试库、凭据或可重建缓存。

生产 SMTP 来源按用户指定 sendemail.py 静态读取，未执行批量脚本。用户将其中登录密码换成客户端授权码后，原腾讯服务器仍返回535。按[中大学生邮箱当前配置](https://inc.sysu.edu.cn/cat/165)核对，邮箱使用网易企业邮箱平台，部署输入改为smtphz.qiye.163.com:465后两端认证通过，外部批量脚本未修改。授权码仅通过已认证SSH输入流进入服务器私有配置，不出现在代码、命令参数、日志、报告或安装包中。真实验证码及隔离回复共发送两封，仅到用户自己的邮箱；IMAP只读取这两个精确Message-ID，INBOX只读选择、BODY.PEEK，不修改已读标记，也不读取其他邮件。SSH密码此前已按授权存入Windows凭据管理器，未写入工程或记忆。

## 部署与未验项

部署入口、服务常驻模板与 HTTPS 代理片段见 [deployment/feedback](../../deployment/feedback/README.md)。服务复用统一后端反馈组件，也可独立运行，不导入科研计算和 Qt。配置含服务器私有secret/adminToken及SMTP设置，allowUnverified:false；真实值不在源码、报告或客户端安装包中。

现场部署已完成：独立服务用户/源码/venv/config/data、systemd开机启动与自动重启、只监听回环18766、HTTPS反向代理及nginx配置测试、反馈路径关闭访问/错误日志、Windows管理工具DPAPI自动连接。服务器检查回执已有每天09:00之后的执行记录。本次启用前队列0条，启用后一条真实验证码已sent，生产会话数量未变。真实管理员API无凭据返回401。此前公网功能自检会话已读并已解决，不占未读提醒。源码启动首次暴露ptb-api包版本元数据依赖，已在包入口增加仅针对PackageNotFoundError的源码版本回退；现场启动及重启通过。依赖通过TLS验证的镜像安装，本次未变更依赖、站点代理、日程、schema或安装系统级依赖。

证据保存在本机忽略目录 output/validation/feedback：server-deployment.json、linux-server-checks.json，以及本次smtp-preflight.json、smtp-enable.json、smtp-delivery.json。smtp-delivery-failure.json保留最初的检查失败类别，成功回执独立保存，未擦除失败。Qt报告和截图在各integration运行目录，最新无邮件模式Qt报告成功。其专用清理入口仍因原测试父PID关联的搜狗SGTool进程保留scratch；已核对其路径位于搜狗输入法安装目录、命令不引用本测试目录。没有结束系统输入法进程，也没有绕过清理检查；run/scratch-manifest/cleanup回执完整保留。较早两次本机验收scratch已清，这一次Qt scratch明确列为保留。本次真实邮件隔离库已清，不新增Qt scratch。

未验项：真实OS托盘通知展示及电脑断网/关机恢复；独立浏览器额外布局；最终用户EXE完整反馈矩阵与独立工具冻结包；其他桌面平台、长期运行与公网滥用压力。公网SSE已验证连接和认证，通知正文/每日去重/离线补收由真实回环SSE与Linux隔离日程覆盖，没有改生产时钟或制造生产提醒。本次真实SMTP收件覆盖指定邮箱，回复使用Linux隔离会话验证，未向已有生产会话添加测试消息，未重跑生产验证码完成或冻结GUI矩阵。其他收件服务、真实用户长期双通道交付尚未专验；SMTP投递成功与收件箱可见均已有本次证据，仍无法承诺任何收件服务总能即时投递。

当前源码及在线服务已接通。当前Windows双版本EXE已由发行任务纳入反馈源码，身份以[当前发行说明](windows-current-release.md)为准，本次仅更新服务器私有配置与文档，未重建或重验冻结反馈GUI。用户可通过关于页及根目录独立反馈工具使用，已有未验证邮箱须补验证后才会对后续回复发送邮件。
