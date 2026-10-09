# GitHub 源码同步

核对日期：2026-10-09。用户明确授权将当前项目代码更新至 GitHub。目标为 `Nephelium/PhoneticToolbox` 的既有 `codex/v3-rebuild` 分支，不合并主分支、不改写历史、不生成 Release 或上传安装包。

## 内容范围与来源

提交当前工作树积累的模块修订、桌面启动/更新/缓存、M18 阅读器、说明书及作者工具、依赖目录整理、构建测试脚本、来源登记与项目文档，并包含官网 ICP 页脚。已有两份论文 PDF 的停止跟踪变更一并纳入，保留本机文件与既有 Git 历史。

用户追加授权清理误传及无用内容后，对当前索引按目录、文件体积、忽略规则和实际引用进行核查。额外停止跟踪 `PT.png` 与 `image/ARCHITECTURE/1776325135506.png`，合计 5142770 字节，唯一引用均为历史基线清单，未发现当前运行、构建或文档使用。原图仍在本机，新增精确忽略项防止重复上传。已跟踪但命中通用音频/表格忽略规则的文件均为公开合成资源或测试基准，继续保留。继承的 v2 源码、说明书及相关图片仍被原生资源路径、回归捕获和迁移对照引用，保留其用途，不整目录移除。

采用明确文件清单暂存。检查根目录、共用 Git 目录、远程地址与分支，fetch 后本地与远程起点一致。共用 Git 目录位于相邻 v2，但操作只在 v3 工作树执行。

说明书索引中 953 项 `software-only` / `git:false` 素材均未进入索引，生成阅读副本、运行环境、缓存、安装包、下载论文与凭据由既有忽略规则排除。正文/索引无嵌入 data URI。明显私钥、访问令牌与密码赋值扫描发现的三处命中均为同一 `m16-task-...` 素材 ID 的引用，不是凭据。此扫描不作为全面安全审计。

已修改原生几何 DLL 的 SHA-256 为 `8f9ac397ab24efc25955b07b7159718822e77e2d62d46f1b864d0f1f620d7686`，与 M10-R14 既有验收及对应源码身份一致。必要原生资源按来源保留，不按扩展名一概排除。此次源码同步不改变最终软件发行中尚待核实的许可和对应源码交付范围。

## 实际验证

- `scripts/validate_docs.py`：最终 1804 个文件、397 条来源、55 项任务，零错误。旧快照链接单列，未更改历史快照。
- `scripts/check_architecture.py`：零错误。
- 前端 `npm run contracts:check`、`npm run ui-data:check`、`npm run typecheck` 通过，`npm test` 为 345 passed，`npm run build` 成功，保留既有大 chunk 提示。
- `tools/manual-studio` 的 `npm run typecheck` 通过，`npm test` 为 38 passed。
- `.venv/m14/Scripts/python.exe -B -m pytest -c tests/pytest.ini desktop/tests tests/test_source_entry.py tests/test_source_snapshot.py tests/test_release_content_policy.py -q`：在进程内设置当前工作树的 desktop/backend/core 源码搜索路径及 `QT_QPA_PLATFORM=offscreen` 后，363 passed、2 skipped。首次未设置源码路径时出现 40 项导入错误，修正调用后通过，未改断言或增加跳过。
- 官网授权上传后 HTTPS 返回 200，9955 字节与本地首页完全一致，SHA-256 `dc72fb49d10f46bd2ba5ba06b4377577f130362edccb56dd1b52928dc4259ac3`。此前本地浏览器 1440/390/320 宽度下备案号、链接、居中及无横向溢出通过。

本机审阅清单与测试日志位于忽略目录 `output/maintenance/github-sync/`，官网回读证据在 `output/playwright/icp-footer/deployment.json`。此次未重新执行完整科学回归、实体设备、跨平台或最终 EXE 验收。

`git diff --cached --check` 报告既有第三方许可原文、Markdown 换行和少量代码空行的尾部空白提示，原样保留来源许可。该项未报告为零告警，不影响已完成的类型、测试和构建结果。

## 同步状态

当前源码与上传边界检查已完成，待本次提交和远程推送后核对提交身份。
