# 独立反馈管理工具

操作说明见[使用说明](使用说明.md)。共用 Vue 管理页面、FeedbackThread 与 HTTPS API，Windows 宿主为 PyQt6，Android 宿主为 Java/WebView。工具不导入科学模块，不启动科研后台或单独页面服务器。

构建使用已安装的项目 Python、Node、Android Studio JBR/SDK 与缓存 Gradle，无需全局安装。顺序为管理页面 Vite 构建、资源许可准备、两平台打包：

```powershell
& 'C:/Program Files/nodejs/node.exe' 'node_modules/vite/bin/vite.js' build --config vite.feedback.config.ts
& '.venv/m14/Scripts/python.exe' -B 'tools/feedback-manager/prepare_assets.py'
& '.venv/m14/Scripts/python.exe' -B 'tools/feedback-manager/build_windows.py'
& '.venv/m14/Scripts/python.exe' -B 'tools/feedback-manager/build_android.py' --lint
& '.venv/m14/Scripts/python.exe' -B 'tools/feedback-manager/verify_windows.py'
```

第一条在 `frontend` 目录执行，其他命令在工程根执行。新版产物位于 `output/feedback-manager/1.0.1`；运行中的旧 EXE 保留，先在旧工具设置中退出，再打开新版。Android 签名及其 DPAPI 密码保存在工程外 Windows 用户目录，不能加入源码或分发；后续更新沿用同一签名。EXE 构建主动限定 DLL 搜索 PATH，避免第三方 ICU 覆盖 Windows 的同名接口。

Windows 采用 DPAPI 与本机持久 WebEngine 数据目录。Android 随机设备凭据只进入 Keystore 加密存储，原生桥和 JS 返回值不包含凭据；禁用云备份和设备迁移。原生层只允许固定 HTTPS API、本地资源和明确操作，拒绝外部页面、重定向和任意脚本访问。后台通知使用 remoteMessaging 前台服务，可手动关闭；用户配对、退出及电脑撤销彼此独立。

原始图标由本项目几何绘制。Vue、Doulos 字体、PyQt6 与 Qt WebEngine 沿用工程登记和许可，完整许可文本随资源打包，见[来源登记](../../third_party/source-registry.json)。没有新增运行依赖。Android 构建工具版本为 AGP 8.10.1 / Gradle 8.14.3 / SDK 35 / Java 17；工具本身不进入应用。

最终字节、设备范围、服务器更新和清理见[本版验收](../../docs/testing/feedback-r3-report.md)，此前设备矩阵见[管理工具验收](../../docs/testing/feedback-manager-report.md)。
