# GitHub 源码同步

## 2026-10-17 当前源码与首页同步

井井明确要求上传项目代码并更新GitHub主页README。当前工作树为`main`，上游为`origin/main`，起点为`b596c3da`；GitHub默认分支和最新Release分别核对为`main`与`v3.0.3`。本轮采用明确候选清单，只同步源码与可公开资源/文档，保留v2、既有标签、Release附件与现有服务。

候选包含桌面与网页版适配、账号设置、录音/声道/论文、浏览器科学计算、预览缓存、契约、测试、打包及官网维护脚本，包含迁移脚本源码，未执行现存库迁移。运行环境、数据库、发行成品、下载论文、原始研究材料、模型权重、官网专用人像/截图/引用谱图与1059项受限说明书素材均不在清单。本机原件保留。新增站酷快乐体与既有来源登记SHA一致，附完整OFL；新官网图标核对生成来源、文件SHA与provenance，不生成或修改媒体。

语法及JSON检查零错误，新增行明显凭据/私钥扫描未发现真实凭据，15处密码字面量均逐项确认为隔离测试账号的合成样例，未输出值。精确候选路径、逐文件SHA、基线、检查日志及公开元数据回执保存在忽略的`output/maintenance/github-sync-20261017/`。

Windows当前检查：前端479项测试、typecheck、contracts:check、ui-data:check及build通过；Python定向211项在m14通过，EGG四项改用项目规定m03科学环境通过，合计215项通过、18项条件测试跳过。m14首轮四项因缺少Matplotlib失败，原日志保留，未安装依赖或放宽断言。文档2244文件/446来源/153任务零错误，历史快照断链单列；架构零错误，2026-10-16记录的五项摘要问题在当前源码检查中已不再出现。前端构建保留大chunk提示。

Python定向组覆盖web准入/容量/权限/公共源/状态/管理/反馈、论文接口与运行命令、预览缓存、Linux运行时摘要、worker争用、桌面资源后台读取/论文与tests/security、tests/contracts；命令使用`-B -X utf8 -m pytest -c tests/pytest.ini -q`，完整参数与原始日志保存在本轮本机证据。未重新验收实际PG、Linux服务、完整科研数值、Qt/实体设备、跨平台或最终成品。

README更新3.0.3与官网Mac r6入口，安装条件以实际HTTPS元数据核对，Mac归档仅HEAD核对476461931字节。独立Chrome本地1200/390宽度、21图、18模块、60本地引用及导航无溢出通过，详见[主页检查](github-readme.md)。源码提交`ebf22c65e2dcb01beb642f6aad88a53898cc38f2`已推送`origin/main`，483个候选Git对象与远端树逐项一致，完整当前树3650个文件未包含受限说明书目录、权重、原始论文、环境、数据库或成品。默认分支`main`、v2身份保持。README实际API内容与本地Git对象逐字节一致；实际GitHub1200/390宽度HTTP200，21图解码、四导航点击、全部锚点与无溢出通过。

直连GitHub443首次推送失败，使用本机现有7892代理以单次`git -c http.proxy=... push origin main`完成，未修改系统或Git持久配置。原始OFL与两份上游许可保留其行尾空格，排除这三份许可后的暂存diff检查通过。公开Mac维护文档去除内部文案范围讨论，原文片段保留本机忽略证据，原科学测试与设备验收不变。

旧GitHub检查脚本首轮把所有截图尺寸固定为2561×1601，而本次未修改的首页原图为2560×1600，图片均已加载。修正为按当前PNG头部逐图读取尺寸，并与远端Git对象身份共同核对；未编辑图片或改放宽页面断言，首轮失败回执保留，随后两宽度通过。

## 2026-10-16 当前源码同步

用户明确授权上传当前项目代码。起点为 `main` 的 `a36d913242ffda55aa2328d25efc358581300e62`，通过当前本机代理 fetch 后与 `origin/main` 一致。上传采用明确文件清单，保留既有 v2、标签与 Release。此次没有创建 GitHub Release、上传安装包或部署网站。

纳入模块/桌面/官网生成器、测试、契约、来源登记及文档等当前源码。50 项本机文件排除：未确认公开分发条件的参数预测权重、官网人像/截图/引用谱图及本机启动器。环境、缓存、原始音视频、数据库、软件专用说明书素材继续受原规则保护。精确清单和检查日志保存在忽略的 `output/maintenance/github-sync-20261016/`。必要字体的实际摘要与既有来源清单一致，保留 OFL 许可。

前端 397 项测试、类型、契约、UI 数据检查通过；候选 88 个 Python 和 27 个 JSON 语法通过；明显凭据/私钥模式扫描无命中。文档检查零错误，历史快照断链仍单列。完整科研/设备、Qt、最终 EXE、macOS 和生产负载未在本次重验。未随源码分发的模型与媒体需要单独获准取得，源码同步不代表完整分发包。

前端构建通过，保留现有大 chunk 提示。Python 定向组在 m14 环境为 111 passed / 3 failed，三项失败均为唇形专用 MediaPipe/PyAV 不在该环境；改用项目既有 m05 环境复验该文件 3 passed，合计 114 项均已通过。未安装依赖或改变测试断言。

架构检查仍有声道前端五项资源摘要不一致：app.js、index.html、presets.js、scene.js、floor-contact.mjs。这五份文件与本次起点 HEAD 相同，既有语谱图验收亦记录同一问题；此次源码同步保留该未闭合项，不修改摘要制造通过，也不据此发行成品。

暂存检查发现官网生成脚本末尾空行，已清理；ZCOOL 原始 OFL 文本第21行保留上游行尾空格，未改写许可证。除该原始许可行外，暂存差异空白检查通过。

源码提交 `37b4d4842514e8abc90aa3181d2dc2a89bdd67f7` 已推送 `origin/main`，随后 `git ls-remote` 回读相同身份；共243文件、17819行新增/892行删除。后续网页版需求及本记录的收口文档独立提交，不创建Release。

### 本轮检查命令与范围

- `npm --prefix frontend run typecheck`、`run contracts:check`、`run ui-data:check`、`test`、`run build`。
- Python以项目m14解释器、`PYTHONPATH=backend/src;packages/phonetic_core/src;desktop/src`、`-B -X utf8 -m pytest -c tests/pytest.ini -q`运行：backend/tests中的test_local_retention、test_m09_parameters、test_m09_pipeline、test_m09_runtime、test_m09_vocoder、test_m05_offline_exports，desktop/tests中的test_m09_bridge、test_paper_reader_updates，以及packages/phonetic_core/tests/test_m09_editing.py、tests/parity/test_spec2wav.py。唇形test_m05_offline_exports.py使用既有m05解释器复验。
- 项目m14解释器运行 `scripts/validate_docs.py` 和 `scripts/check_architecture.py`；按上文分别记录文档通过与架构失败，未混为全部通过。测试使用本机现有科学资源，未进行仅公开仓库的新克隆复现。
- `git diff --cached --check` 保留一处上游OFL空格；排除该许可证后源码检查通过。需求文档提交的完整diff检查须无错误。

当前同步目标：GitHub `Nephelium/PhoneticToolbox` 的 v3 主线为 `main`，旧版保留在 `v2`，旧标签与 Release 保留。本次 2026-10-11 的分支切换及图文主页上传已获用户明确授权，验收见下文。

原有源码同步证据对应 2026-10-09，当时目标是 `codex/v3-rebuild`。这些历史命令不再作为后续 v3 上传目标。

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

源码提交 `795c2dc` 已推送至 `origin/codex/v3-rebuild`，`git ls-remote` 与本地提交身份一致。提交涵盖 1098 个文件变更（包含目录移动和先前已发生的清理）。四项本轮仓库移除对象均未出现在提交树中，且仍保留在本机；工作树在源码提交后干净。本记录与台账的完成状态随后以单独文档提交同步。

## v3 主线与图文主页发布

任务 `GIT-V3-MAIN` 的2026-10-11主线切换已验证。远端原 main / 本机 v2 提交为 `4eeff893eb6cdcddb40456072571dd5d30c51a76`，原 v3 提交为 `ef764e2d4e32ba442b02350e1168d4e2b538e87d`。祖先关系检查通过，v3 已在原 main 基础上推进69个提交，因此本次通过普通快进推送保留完整历史。

远端 `v2` 已建立并回读为原 v2 提交。本地原 main 重命名为 v2，v2 工作树跟踪 origin/v2；v3 工作树重命名为 main，跟踪 origin/main。旧 codex/v3-rebuild 在本地及远端继续保留为历史参考。此次重命名只调整 Git 分支关系，没有检出或重写 v2 文件。

本次提交范围为根规则、图文 README、六张 PNG 与来源清单、公开文档导航、仓库分支约定、本轮相关验收和台账。共享的来源登记/任务台账/项目状态采用已提交基线加本轮条目的方式暂存，不把其他任务的未提交改动混入图文提交。配图不包含原始录音、TextGrid 或私人绝对路径。近期安装包报告仅作为主页已有链接的支持文档纳入，没有上传安装包。

main快进推送与默认分支核对、旧标签及7个旧Release/7个附件身份比较、GitHub README与六张图片回读已完成。基线与实际提交清单保存在忽略的 output/maintenance/github-main-transition/。

后续规则的唯一维护位置为根 [AGENTS](../../AGENTS.md) 与 [仓库内容管理](../development/repository-hygiene.md)：获准上传 v3 时直接使用 main，不再上传旧 codex/v3-rebuild。该目标约定不会自动授权上传无关改动。

## 当前公开源码与本机工作树

2026-10-12本轮修改前核对：默认分支为main，远端HEAD/main与本地HEAD均为8cf5c5396fcab63cf431c69b2cb4712af6a12430。10月11日主线切换与主页提交没有包含同期全部业务源码，最近源码同步为10月9日795c2dc。核查开始时本机另有279个已跟踪文件修改和263个未跟踪新文件，不能将已发布EXE或README更新等同于完整源码同步。

修改前main树共2,997个文件、147,945,364字节；953项受限素材文件均不在树中，无嵌入媒体data URI。没有发现运行环境、缓存、数据库或私钥文件，明显令牌模式命中为任务ID和证据路径。两个旧下载论文PDF已从main树移除，但仍在v2与保留历史中。

本轮经用户确认后，以主页提交 `2c93da2` 推送main。新增18模块、首页及设置共20张Absolutely浅色截图，两个根目录本机快捷启动CMD从main停止跟踪并加精确忽略，原文件保留本机。此次32项变更只包含主页、静态配图和相关文档；公开main树现有3,015个文件，GitHub API核对30项文件身份一致，两个CMD均不在树中。其余539项已修改或未跟踪工作文件的摘要保持一致，业务源码尚未随本轮同步。GitHub实际1200/390宽度下图片、按钮、锚点及无横向溢出检查通过，见[主页检查](github-readme.md)。
