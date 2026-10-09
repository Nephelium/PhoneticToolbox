# Windows 发行入口

构建参数、依赖边界、500,000,000 字节上限和最终成品验收唯一维护在 [PACKAGING_RULES.md](PACKAGING_RULES.md)。本页只说明入口与产物位置，历史报告不构成当前上传授权。

| 任务 | 入口 |
| --- | --- |
| 准备/核查独立运行时 | [prepare_lean_runtimes.py](prepare_lean_runtimes.py)、[verify_runtimes.py](verify_runtimes.py) |
| 冻结源码并构建单 EXE | [build_v3_local_preview.py](../scripts/build_v3_local_preview.py) |
| 校验应用身份与大小 | [finalize_compact.py](finalize_compact.py) |
| 从同一 EXE 生成安装包/更新 ZIP | [package_compact.py](package_compact.py) |
| 静态盘点真实成品 | [audit_release_contents.py](../scripts/audit_release_contents.py) |
| 工程外真实运行、安装与换版 | [verify_distribution.py](../scripts/verify_distribution.py)、[verify_update_handoff.py](../scripts/verify_update_handoff.py)及严格规则中的专项入口 |
| 成功构建后清理旧输出 | [cleanup_old_builds.ps1](cleanup_old_builds.ps1) |

- 免安装应用：`dist/<构建名>/PhoneticToolbox.exe`。安装版与更新 ZIP：`output/release-staging/<包装名>/artifacts/`。构建名唯一，按实际需要生成分发类型，不重复复制。
- 先生成本轮前端与说明书，再冻结源码。构建使用既有 `.venv/m14`，EGG/LPC 和唇形保留指定科学环境；缓存路径须来自实际盘点，不能照抄历史时间戳目录。
- 持久缓存按内容校验复用，安装预热、设置清理、卸载保护和真实进度均属于公共实现。首次准备与后续启动分别测量，详见[缓存决定](../docs/decisions/ADR-persistent-startup-cache.md)。
- 应用默认安装到 `%LOCALAPPDATA%\Programs\PhoneticToolbox`。核心任务/更新/缓存与 Qt 工作台设置分别沿用既有用户目录，升级卸载保留研究数据，不能把开发测试清理规则套到用户工程。
- 更新下载仍是完整包，须验证大小/SHA 并明确确认退出换版；进程已创建不能代替新界面及数据恢复验收。服务端清单最后原子发布，只有当前任务已授权时才上传/发布。
- 运行数据排除统一在 scripts/release_content_policy.py，包审计区分压缩与展开体积。作者编辑器、私有审计、未引用原媒体和 MFA 环境/模型/词典不进入普通应用包。
- 自然例音及处理结果仅允许软件分发，不进公开 GitHub；公开说明书使用新的独立输出目录和 `--distribution public`。来源通知、对应源码及未决许可分别审阅，不用论文引用替代代码许可。

验证报告写明实际应用字节、源码快照、依赖身份、命令、结果和未验范围。每类成品只保留最新版本，失败构建及活动目录按根规则处理，不在本页追加交付流水。
