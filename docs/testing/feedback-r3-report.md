# 反馈界面与邮件选择验收

任务 FEEDBACK-R3，2026-10-11。当前源码、Windows 11 最终主应用 EXE、独立管理 EXE、Android 36.1 x86_64 模拟器最终签名 APK 与 Ubuntu 生产反馈服务已验证。本地包已生成，未公开上传。

## 行为

关于页的官方网站按钮使用默认浏览器打开 https://www.phonetictoolbox.com/。问题反馈和管理端共用井韶子的署名、原有官网头像及聊天组件，包含独立头像、左右消息气泡、时间、图片预览和整合输入区；跟随现有主题。头像逐字节复制，未改动位图，资源来源及 SHA 已登记。

回复邮件默认开启以保留已有行为，开启时每条新回复单独发一封邮件，标题相同、正文对应本条回复。用户可关闭回复时发邮件提醒，设置按会话 owner 保存在服务器，重新打开应用或验证邮箱找回同一会话均保留。关闭会取消尚未投递/失败的回复，并阻止新增回复入队；过期投递租约恢复后同样检查偏好。已经开始交给 SMTP 的邮件无法撤回。验证码不受该开关影响，重新开启不补发旧回复。应用会话、未读、管理员回复与通知均继续工作。

偏好保存在现有私有 feedback-admin-devices.json 的 mailPreferences 中，沿用文件锁、0600 和原子保存。严格布尔接口只允许用户修改自己的会话，管理员令牌和其他 owner 参数不能代改。既有 SQLite schema 2、SMTP 配置、每日 09:00 检查及真实用户消息不迁移。

## 最终成品

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| output/feedback-manager/1.0.1/PhoneticToolbox-windows-x64.exe | 496539816 | 28615fe0f7c39efe4a59f3621064f88a26ed2247641d69df7816c885e6bd9f1e |
| output/release-staging/release-feedback-20261011-r3/artifacts/PhoneticToolbox-3.0.0-preview.1-windows-x64-setup.exe | 498456559 | fbe3b3456d48d55c48e13b1709bcabb8b6bab600c8cddc5057dff8b147774a33 |
| output/release-staging/release-feedback-20261011-r3/artifacts/PhoneticToolbox-3.0.0-preview.1-windows-x64-portable.zip | 496540392 | 6ca070b04f37a8f375ae3f7f64d559f55077cc595be87c9ee74acedf71949c64 |
| output/feedback-manager/1.0.1/反馈收件箱.exe | 199159289 | d7f5b04eda4c184cc6f960f9887959f15fa6906024e9acbda6c010c58624eb23 |
| output/feedback-manager/1.0.1/反馈收件箱.apk | 1202012 | 2abb66a3e3cb5b24d1ea4fae3e90ed64001a974612da8a2776967cff0423b3a4 |

主应用版本为本次冻结时的 3.0.0-preview.1，源码快照 SHA 为 a5aef363858c3ce80a5dfbf0115e8564357c608efb55aebf675c4c84aac50610。构建后的同期版本号与运行时源码变动不回写已验证的冻结包；final-identity-r3.json 显式列出后续源码差异，不声称整个当前工作树与该快照相同，反馈相关源码仍一致。三类主应用包使用相同 EXE 字节，均小于 500,000,000 字节。APK 版本 code=2/name=1.0.1，沿用已有私有签名，v2 校验通过。

交付前原 R3 dist 目录已不存在，dist 另有同期 3.0.0-R1 构建。为保留本轮已验字节，从工程外验收 EXE 建立独立交付硬链接，SHA 与上表一致；不覆盖同期构建，不将其验证结论并入 R3。文档检查 errors=[]，架构检查 errors=[]。

## 实际验证

- 32 项反馈/设备/桌面 Python 检查通过，含用户隔离、严格参数、关闭队列、崩溃租约、找回/重启保留、重新开启只提醒新回复、验证码不受影响与 schema 2。启动/更新缓存另 23 项检查通过。
- 前端类型检查、367 项测试、契约和资源生成一致性通过。说明书严格检查 20 章、1044 素材、零警告/错误，生产阅读器解析与构建通过。头像已补入资源清单，架构检查 errors=[]。
- scripts/verify_feedback.py --qt 使用独立 HTTP/SSE/本地 SMTP 和实际 Qt：验证码、文字/图片、管理员收取回复、用户同步与关闭邮件通过。真实 Qt 点击官网按钮经 QDesktopServices 捕获目标 URL，未实际弹出用户浏览器。证据 integration-062f02a7e3184bac8eca4eeec805e03e。
- tools/feedback-manager/verify_user_windows.py：最终主 EXE 独自放在工程外，Windows-only PATH、独立 DPAPI 身份、私有 HTTPS fixture。官网入口、井韶子名字与头像、用户发送/收回复、邮件开关保存与重开、关闭后继续聊天不发邮件、浅深主题与正常退出通过。证据 user-final-r3-2164094128d5493282ccaeb557292c01。
- 最终独立管理 EXE：工程外单文件、Windows-only PATH、私有 TLS、文字/图片回复、署名和头像、持久草稿/主题、即时 SSE、配对/撤销、手机宽度、托盘隐藏/唤起/单实例与退出通过。证据 windows-final-463ba34982f14cd497bfc98efaeca2f8。
- 最终 APK：实际配对，原生本地 JPEG 头像显示，聊天署名及文字回复由专用用户 API 精确回读；已关闭的邮件提醒被遵守。force-stop 与同签名覆盖安装后保留登录，后台系统新留言通知、前台接收服务、撤销后返回配对入口通过。未重新验收本轮模拟器重启和相册回复，前轮证据见[管理工具验收](feedback-manager-report.md)。assembleRelease/lintRelease 通过，0 错误、2 项既有提示（target35、严格本地页启用 JS）。
- Windows 主应用最终 --verify-distribution：科学探针、MediaPipe、11 个托管任务、产物、帮助/说明书、声道及客户端工作流通过，恢复后的主原生文件逐一验 SHA。最终内容审计、Qt 身份、MFA 排除、ZIP CRC/同字节、OwnedQA 安装准备/复用/卸载及数据标记保留、真实中断/并发恢复、实际设置清缓存/取消/重开，以及两类真实辅助更新/重启/媒体解码通过。更新旧版本号为元数据夹具，来源为本次实际 EXE，未声称所有历史版本已验。打包/共享/启动机制未改变，未做启动性能对照；准备耗时只作诊断。
- Ubuntu 隔离邮件 fixture 验证后部署，公网用户 API 保存/回读开关、认证边界及健康检查通过。配置与 SQLite schema 摘要未变，09:00 日程保留。服务源码 SHA fe77faebdffec376edbbce8130b2dab9888b71b9f3ed3fd3f4a3f8312a72624b。真实用户未收到本轮自检邮件；专用合成会话标为已读/已解决，仅本轮模拟器设备被撤销，已有手机保留。
- 成品无本机管理员凭据、私有身份/库/配置/签名文件，完整许可文本齐全，两管理包头像与原素材逐字节一致。

## 修复与限制

首次管理 EXE 构建被正在运行的旧 EXE 拒绝覆盖，改为独立版本目录并保留原窗口。首次服务器更新等待旧通知连接结束超过检查超时，自动回退后按服务实际 90 秒停止上限延长检查等待，最终部署成功，未改私有配置。首次主 EXE 界面检查只等待 45 秒，冷准备尚未结束，按真实准备过程延长等待后通过；不据此声称启动性能。两项旧验收脚本分别还使用固定分发名和旧设置页位置，已按实际 build-info 白名单与设置标签修正，原断言保留后全部通过。

Windows 10、独立干净 Windows 电脑、小米/红米 HyperOS 实机、其他 Android 版本、物理音视频、Windows 弹出通知与长期压力未验。通知与邮件选择逻辑已验，不承诺所有厂商后台限制下即时推送。现有科研算法未因反馈功能而改变。证据保存在 output/validation/feedback-manager；模拟器已按 owner 记录停止，自检设备已撤销；删除测试目录的批量及缩小范围操作均被自动审批拒绝，仅返回 blocked by policy。大型运行副本、AVD、私有合成夹具暂留，cleanup-r3.json 记录 success=false，未绕过拒绝、未删除用户目录。
