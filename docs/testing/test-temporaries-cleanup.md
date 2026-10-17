# 测试临时文件整理

状态：verified，限定各批已核实的Windows本机与Ubuntu清理对象及文件保护检查；最新核对2026-10-17。历史批次按各自范围保留。

## 2026-10-17 大文件复查与剩余空间

本次按大目录/大文件盘点删除本机62项目标、31268文件、12914521444逻辑字节，可用空间实测增加12980211712字节（12.09 GiB）。包括7轮旧打包的14份前端输入、3份3.0.1旧独立前端、4份旧网页构建、20份旧官网构建、3个已消费传输包，以及6组已停止反馈测试的EXE/临时数据共18项；其日志、所有权记录、截图与清单保留。3110份最新成品/包装输入/保留证据SHA不变。服务器删除4个3.0.1过时安装包备份，共1987172000字节，可用空间增加1987186688字节（1.85 GiB）；保留29份当前Windows/macOS下载、频道、私测运行时元数据与最近3.0.3回退包，SHA保持。服务状态及PID保持，首页/网页版200、未登录401、内部运维404通过。

环境、模型、研究数据和正式素材未清理。当前包/构建输入、MFA与神经组件及其来源包、WASM/Praat工具链、当前/上一份回退、科研环境和混合用途验收目录保留。20份官网本地预览首页按既有publish.py的发布URL映射后，大小及完整SHA与历史发布清单匹配，其余资源直接逐文件核对，没有改写源首页。失败的审计脚本写入命令在执行删除前纠正，删除按最终固定清单执行；文件/联接/进程边界检查未放宽。 本机忽略目录output/maintenance中的large-audit-local-plan.json/results.json、large-audit-extra-plan.json/results.json、large-audit-summary.json与inventory.json保存实际清单/结果/盘点；完整名称均为large-audit前缀。远端审阅及回执为large-audit-remote-backups.result.json、large-audit-remote-remove.result.json，服务器回执为/var/lib/ptb-web/reports/large-redundancy-cleanup-20261017.json。 剩余可用空间最后按本机DriveInfo和服务器statvfs读取，保存在large-audit-space-final.json；大小/可用容量为快照，硬链接、分配单元及后台写入使逻辑长度与实际增量不同。未重建成品或复跑科研、GUI、设备及长期验收。

## 2026-10-17 内测副本、临时上传与浏览器配置

本次明确授权后，删除本地49项目标、25557文件、9152329417逻辑字节，含截图所指9份浏览器配置、此前另7份已结束浏览器配置、5份已拦截测试副本、重复前端构建和已完成传输包。所有目标不存在，D盘可用空间实测增加9212768256字节（8.58 GiB），1778份保留报告/截图/脚本/当前对照前端SHA不变。服务器删除13份失去引用的内测发布树和26个临时上传包，共39项、26985文件、9035673938逻辑字节，可用空间增加8632639488字节（8.04 GiB）；全部目标不存在。

保留当前web-private-20261017-c3、直接回退c2、web-integration-20261016共享Python/科学环境和vocal-candidate-20261016-f251d214f69a原生组件。保护依据为活动发布记录、当前/回退runtime与科学回执、当前systemd/代理配置及所有进程cmdline/cwd/exe/maps/open-fd引用；4167份当前/回退发布文件、科学环境/模型及配置SHA通过。3个证据传输包先保存内容，连同退役树manifest共32份证据写入私有reports并校验，之后才删外层归档。Nginx、反馈、API及PG状态/MainPID保持，首页/网页版200、未登录401及内部运维404符合预期。没有结束用户进程、重启服务、改配置/凭据/数据库、改科研算法或删除研究库/资产。 本机科学WASM/Praat构建工具链、上游源归档和当前对照frontend-current保留。原拦截回执保留在previous_attempt，当前删除绑定本次授权。

实际清单和删除/保护回执位于本机忽略的output/maintenance/private-cleanup-local-plan.json、private-cleanup-local-results.json、private-cleanup-local-verification.json、private-cleanup-remote-inspect.result.json、private-cleanup-remote-plan.result.json、private-cleanup-remote-remove.result.json与private-cleanup-summary.json。远端实际回执为/var/lib/ptb-web/reports/private-retirement-cleanup-20261017.json，保留证据为同处retired-transfer-evidence-20261017。 远端预检首次遇manifest的bytes/size字段差异，删除前修正读取适配后通过，失败诊断保留。完整旧树清单/大小/mtime及联接在删除前重核，当前/回退文件按SHA校验。空间使用删除前后快照，后台写入可能影响净变化。未重新执行科学、GUI、设备或长期稳定性验收。

## 2026-10-17 冗余版本与过时安装包

2026-10-17按本次明确授权继续清理冗余版本：本机5处退役构建的17个数据子目录、失败stage的egg/m05及2个明确被替代的MFA旧ZIP，共21项/22313文件/7064231636逻辑字节，D盘可用空间增加7102394368字节（6.61 GiB）。23份构建日志/生命周期/根诊断保留，8588个当前包、成功v2运行时、正式MFA组件和诊断文件SHA保持。服务器删除3.0.2旧EXE/安装器/更新ZIP及47份过时官网快照，共50项/4010文件/2527675318逻辑字节，可用空间增加2423615488字节（2.26 GiB）。官网保留当前r4840ddd775cdf7c2与上一份r208f040f9772b085，按当前入口/代理配置及保留快照的引用闭包保护。289个当前Windows/macOS下载、频道、官网/代理与回退文件SHA保持；Nginx/反馈/API/PG的状态和PID保持，首页/网页版/3.0.3三下载HEAD均200，鉴权401和内部管理404符合预期。

标准cleanup_old_builds.ps1 -PlanOnly将4处旧构建列为未完成。结合根日志、当前发布身份和进程路径确认这些已退役，旧owner PID之一现属于其它程序，仅按已核验路径清理数据子目录，没有按PID结束进程或改写历史失败状态。首轮失败运行时已由success=true的v2替代，删除其可再生egg/m05，selection-manifest.json和stage-report.json保留。MFA旧目录有勿上传标记，只清两旧ZIP，正式publish组件及对应源码包保留。Windows/macOS最新包、最新构建缓存、科学组件/权重、源码、研究数据、当前及内测服务目录和GitHub历史Release保持。

本机证据为output/maintenance/obsolete-versions-local-plan.json、obsolete-versions-local-results.json、obsolete-versions-local-verification.json及obsolete-versions-summary.json；服务器审阅/删除回执为obsolete-versions-remote-plan.result.json和obsolete-versions-remote-remove.result.json，原路径及逐文件SHA保留。远端回执为/var/lib/ptb-web/reports/obsolete-versions-cleanup-20261017.json。执行使用一次性固定计划的PowerShell Remove-Item -LiteralPath，远端复核绝对父路径、联接、文件清单/完整SHA与打开文件后删除。磁盘增量与逻辑长度分别记录，后台写入会影响时段空间。未重打EXE或执行产品科学/GUI/设备回归。

## 2026-10-17 拒删项清理

本次新增授权覆盖工具箱及独立训练工程的旧缓存。实际删除49项、176,909文件，所有目标删除后均不存在。工具箱40项、98,389文件，训练缓存9项目录、78,520文件。逻辑长度122,471,111,902字节，D/F盘可用空间分别增加9,886,416,896/104,430,481,408字节，合计106.47 GiB。这是同一清理时段的磁盘快照变化，活动程序及审计文件写入也会影响D盘，逐项变化另存回执。

9个此前拒删小文件均按原SHA复核后删除，已作废误截图、旧网站验收副本、停止的M18/反馈QA及4个隔离PG夹具完成清理。9条QA运行时联接先非递归移除，检查目标仍存在后再删除对应测试副本，未沿联接递归。首次检查发现Python返回的Windows扩展前缀与PowerShell显示不同，在任何删除前停止，核实并规范化前缀后通过，失败回执保留。

保留权重、清单、日志、指标与当前成品。4组旧数据集剩余118,166文件大小/mtime不变，选定权重/清单/结果SHA及当前免安装成品SHA通过；当前dataset-randomized-melfirst-20261014-v4的124,854条缓存引用零缺失。两处历史空缓存目录现在已有内容，且正式工具箱实例运行，本次保留；4处历史拒删目标已不存在，不计释放量。

本机证据统一为output/maintenance/retry-blocked-summary.json、retry-blocked-verification.json及三批逐项plan/results；命令为该目录的一次性audit-blocked-cleanup.py、audit-blocked-extras.py、remove-reviewed-blocked.ps1与verify-blocked-cleanup.py，删除使用PowerShell的Remove-Item -LiteralPath，完整路径先核验。保护检查覆盖实际路径、Git未跟踪且忽略、文件列表/大小、联接及当前进程。未重新运行产品功能、GUI、设备或跨平台检查，未构建发行包。 文档检查覆盖2184文件、442来源、136任务，errors为空；历史快照旧链接单列，定向diff检查通过。

## 2026-10-11 历史实际结果

27 个目标、45,934 个文件已删除，合计 16,087,103,776 字节逻辑长度。工作区盘点约 58.32 → 42.26 GB，盘点未沿目录联接遍历，零读取错误。文件长度包含硬链接的重复计数及稀疏虚拟磁盘，不能据此声称释放同等物理容量。

| 已删除类型 | 目标 | 文件 | 逻辑长度 |
| --- | ---: | ---: | ---: |
| 已停止的反馈测试模拟器磁盘和私有配置 | 2 | 25 | 3.47 GB |
| 已完成安装/换版验收的成品副本、隔离配置和启动缓存 | 8 | 45,796 | 12.31 GB |
| 可重建的合成长音频、逆滤波 WAV 与含音频的请求副本 | 11 | 11 | 222.85 MB |
| 已结束测试的 scratch，包括音频、缓存和测试工程副本 | 5 | 97 | 77.83 MB |
| 本次清理脚本的合成边界夹具 | 1 | 5 | 86 字节 |

所有已删除路径均复核不存在；21 份保留报告、所有权记录、收尾清单和生成脚本的 SHA-256 不变。M03 长文件请求的配置和输入 SHA 已另存，400 Hz/30 分钟及 48 kHz/120 秒的生成条件保留。最新 R7 免安装 EXE、安装包和更新 ZIP 三件套的大小与 SHA 均不变。

## 清理入口与边界

[批量入口](../../scripts/cleanup_test_temporaries.ps1)默认预检，`-Apply` 执行删除。它先运行[清单生成器](../../scripts/prepare_test_cleanup.py)，再调用[逐项删除入口](../../scripts/cleanup_workspace.ps1)。机制和命令见[产物生命周期](../development/artifact-lifecycle.md)。

本机清单限定审阅过的测试对象及 `ready` 且与收尾摘要一致的 scratch，实际删除再次核对文件列表、SHA、保留证据、Git 跟踪/忽略、联接与进程引用。清单中的保护路径同时约束实际执行，删除父目录也不能绕过保护。清理失败留回执，不结束用户或同期任务的进程。

保留同期 `release-300`、`release-review-20261011` 发行验收范围、最新构建缓存和实际运行时；保留当前开发数据库、M03 打包输入和仍有脚本引用的 `m08-wiring`。正式说明书、用户原件和研究数据不在清理范围。混合视频/原生设备数据及用途不明的历史目录继续逐项审阅，不能按文件扩展名或目录名称整删。

2026-10-11保留的9个拒删小文件已在2026-10-17本次授权下复核SHA并删除，原摘要留在本机清单，当前结果见本页最新批次。

## 验证和证据

7 项独立 Git/文件系统单元测试及 6 项既有生命周期回归通过，涵盖跟踪文件、外部路径、保护父子路径、证据条件、active/ready 状态、收尾清单变化及真实目录联接。6 项 PowerShell 预检探针符合预期：正常目标只预检、变更文件/证据及保护对象拒绝、完整验证根目录拒绝、旧格式缺少可选字段仍可识别已不存在目标。两份 PowerShell 脚本语法解析无错误，批量入口独立空清单预检成功，原实际清单与回执保留。微型测试目录本身已列入清单清理。

主要命令：

```powershell
.\.venv\m14\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -p test_test_cleanup.py -v
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/prepare_test_cleanup.py --rules output/maintenance/test-temporaries-rules.json
.\scripts\cleanup_workspace.ps1 -Manifest output/maintenance/test-temporaries-plan.json -ReportName test-temporaries-review.json
.\scripts\cleanup_workspace.ps1 -Manifest output/maintenance/test-temporaries-plan.json -Apply -ReportName test-temporaries-results.json
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/inventory_workspace.py
```

实际路径、删除理由、逐文件摘要和回执统一原位保存于 Git 忽略的 `output/maintenance/`：`test-temporaries-list.md`、`test-temporaries-rules.json`、`test-temporaries-plan.json`、`test-temporaries-review.json`、`test-temporaries-results.json`、`test-temporaries-verification.json`、`test-temporaries-audio-recipes.json`，以及合并后的 `manual-delete.md/.txt/.json`。必要记录保留，未来复用时更新同名文件。

本次未重打 EXE、复跑完整科研/GUI/设备或跨平台验证，未提交或上传。历史测试入口仍在复用或修改时迁入 scratch，清理入口不能代替各模块的功能验收。

文档检查覆盖 1,952 个文件，零错误；受保护历史快照的原有旧链接单独登记，不改写历史。当前状态、导航、生命周期说明与 WORKSPACE-MAINTENANCE 台账条目已同步更新，本机详细清单均受 Git 忽略保护。
