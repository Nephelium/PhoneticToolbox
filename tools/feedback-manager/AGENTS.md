# 反馈管理工具

- 共用 `frontend/src/feedback-admin`、聊天组件和服务器反馈 API，不复制用户反馈逻辑。
- Windows 凭据用 DPAPI，Android 凭据用 Android Keystore。安装包不得携带管理员凭据、配对码、邮件或 SSH 密钥。
- APK 使用固定 HTTPS 服务，仅本地静态内容可访问原生桥。外链、重定向、文件访问和任意原生操作拒绝。
- 构建只使用已有项目解释器、JDK、SDK 和 Gradle，输出到 `output/build-feedback-manager` 与 `output/feedback-manager`。
- 本工具不包含科研运行时。验收覆盖最终 EXE、APK、登录持久化、配对撤销、图片回复和系统通知；模拟器证据与 HyperOS 实机证据分开记录。
