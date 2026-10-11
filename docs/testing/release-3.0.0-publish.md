# PhoneticToolbox 3.0.0 发行与官网发布验收

当前状态：`verified`，限定本报告的 Windows 本机与官网范围。**R7 已在官网公开，最终字节功能、公开 HTTPS 与真实浏览器验收均通过**。合并 QA 的 13 条正式命令全部退出 0 并绑定同一应用 SHA，独立 `publication-result.json` 确认6个上线步骤成功。用户明确将启动速度对照移至发布后，本轮发布前豁免，未形成完整性能结论。R6 因实际卸载失败未发布，历史局部成功不计为 R7 通过。

发行门槛唯一依据为[严格打包规则](../../release/PACKAGING_RULES.md)。本轮覆盖 Windows x64 本机工程外最终 EXE、安装/换版与官网 HTTPS；执行计划在本机 `docs/plans/2026-10-11-release-300.md`，原始证据保留于所列路径。

## R7 最终身份

| 交付文件 | 路径 | 字节数 | SHA-256 |
| --- | --- | ---: | --- |
| 免安装 EXE | `dist/PhoneticToolbox-300-20261011-R7/PhoneticToolbox-windows-x64.exe` | 496,815,155 | `162c8fa542b7b3b1203ce535e3021235e261f0dfaeb095659f9946baec055edb` |
| 安装 EXE | `output/release-staging/release-300-20261011-r7/artifacts/PhoneticToolbox-3.0.0-windows-x64-setup.exe` | 498,729,076 | `3c131b4897cd7d5b063530485c9cfc064c57440c3ff8f2420ab62c2fff3bcce9` |
| 更新 ZIP | `output/release-staging/release-300-20261011-r7/artifacts/PhoneticToolbox-3.0.0-windows-x64-portable.zip` | 496,815,721 | `43b5d3c618cf8705c57a197ffa30b69a84f767397ef642213ea4c0bc4a597c9d` |

三文件均低于 500,000,000 字节。冻结快照 `output/build-PhoneticToolbox-300-20261011-R7/snapshot/source-snapshot.json` 的 SHA-256 为 `3d1fd7bb865ca389df96ca0b90a3a39327267e90ca76df3fd01abc30e4ffa880`。`r7/final-identity.json` success，711 个冻结源码文件与 111 个前端资产匹配当前源码，差异为空，更新 ZIP 内 `PhoneticToolbox.exe` 与免安装 EXE 同 SHA。说明书阅读衍生资源按构建规则生成，不要求与原媒体逐字节相同。

`r7/audit-r7.json` 与正式 `package-report.json` 已通过。原科学/宿主运行库字节身份见本轮 `build-info.json`；本轮路径修复保留运行库字节。主包排除 MFA 环境、声学模型与词典，作者编辑器、未引用原媒体、私人语料和临时目录不进入主应用。

## 源码与修复边界

`source-build.json` 的说明书 strict 校验、software 阅读资源、类型检查、367 项前端测试、生成数据检查、前端构建及生产阅读器检查均退出 0。版本统一为 3.0.0，正式版默认 stable 并保留既有频道选择。80 项更新/绑定/原生路径定向检查及真实超长 EGG/MKL、M05、中文 M05 探针通过，相关原始日志和 `alias-probe.json` 保留。R7 guard 修复另有 12 项定向测试与 72 项入口回归通过。这些属于源码与真实运行库探针，最终包结论以本轮表格为准。

启动卡读取已有内置或自定义 PNG 偏好，在无 Qt 阶段保留图标色相并固定为浅色。默认 W1 背景为 `FFF9E8`；偏好不可读时使用默认。9 项定向测试及默认、另一内置、深紫自定义的实际 Win32 HWND 截图通过，证据在 `startup-card-f2762979/`。同一冻结素材生成的 32 款 64 px ICO 与色表共 208,448 字节，未增加运行依赖或额外源素材。R7 最终包仅验证默认卡，未逐款截图。

历史失败均保留：R1/R2 涉及 MediaPipe 扩展命名空间、长子进程入口和 MKL 加载；R4 中文深目录触发 PyInstaller `python311.dll` 启动失败；R5 私有 `_internal` 联接被 REAPER 原路径安全校验拒绝；R6 科学通过，但 Inno 子进程继承重解析点保护后卸载触发 WinError 448。R7 保留该保护，冻结运行文件使用无联接私有镜像回退，不复制用户任务目录；安装完成提示改为从开始菜单或桌面打开。未降低原路径校验或关闭系统保护。

## R7 最终字节验收

除启动卡专项目录外，当前报告集中于 `output/validation/release-300/r7/`。`qa-report.json` 已合并 science/rest，success=true，13 条正式命令全部退出 0，应用大小与 SHA 匹配上表。发布前启动性能对照按用户最新指示豁免，功能与身份门槛未减少。

| 必需门槛 | 实际证据与当前状态 |
| --- | --- |
| 封装、冻结源码/前端、更新 ZIP 同应用 | `final-identity.json` success，finalize/audit 完成，verified |
| 工程外单 EXE、Windows-only PATH、私有 profile/TEMP、直接 GUI 与自然退出 | direct-gui exit 0，`direct-relaunch.json` 和合并 QA 通过，verified |
| 普通/中文深目录科学、原生探针、托管任务、帮助/客户端流程及 TextGrid | `qa-report.json` 普通分发 201.875 秒、TextGrid 17.812 秒、160 字符中文 profile 与深任务目录分发 178.453 秒，均 exit 0，sciencePassed=true；精确范围见 `short-distribution.json`、`short-textgrid.json`、`long-distribution.json`，verified |
| 真实 payload 复制回退、损坏修复、逐次身份校验 | `copy-repair.json` success，9729 科学/418 宿主文件实际复制并完整核验，同大小同时间戳应用损坏后只重建 apps；qa-rest copy-repair 155.953 秒/exit 0，verified。受控禁用硬链接，未换另一物理文件系统 |
| 中断、并发、租约保护 | `cache-processes.json`、qa-rest cache-interruption 78.812 秒/exit 0，verified；主动强制终止的 `_MEI0000b9d02` 残留保留，性能结束后核对 |
| 设置取消/清理/重开 | `settings-cache.json` success，142.422 秒/exit 0；取消保留缓存/编辑，确认退出清理保留研究标记，重开恢复主题与真实模块草稿，verified |
| 缺空间、权限拒绝、保护路径冲突、不发布残缺缓存 | `final-failures.json` success，三项预期 exit 23，无 ready/last-launch，保护数据不变、ACL 恢复、自有 scratch 清理完成，verified |
| 最终默认卡图标、浅色背景、可读文字与正常退出 | `../startup-card-r7/report.json` success，实际 W1 黄蓝与 `FFF9E8` 截图已目视核对，两进程自然 exit 0、活动数 0、MEI 0、错误日志 0、缓存清理完成，verified |
| 独立安装、应用同 SHA、准备/复用、忙碌保护、真实卸载及数据保留 | `install-uninstall.json` success，安装 89.469 秒/exit 0，复用 exit 0，活动租约卸载 exit 1，释放后卸载 166.984 秒/exit 0；研究与设置标记摘要保留，安装文件与私有启动目录均无残留，verified |
| 两类真实换版、辅助进程/EXE 来源、重启、组件复用、设置/草稿/工程保留与媒体解码 | `upgrade-portable.json`、`upgrade-installer.json` success；两类 helper/重启正式命令均 exit 0，科学/宿主复用、apps 重新准备，主题/草稿/研究标记与9个播放器、16幅图片通过，verified。进程退出观测范围见下文 |
| 三轮交替旧/新首次 GUI、三轮正常退出缓存的再次 GUI、跨构建复用与性能门槛 | `startup-report.json` status=waived_by_user、success=false，发布后另验；3份部分测量保留，未形成完整性能通过结论 |
| 合并 QA 与报告身份 | `qa-report.json` success 且同最终 SHA，verified；启动性能取消回执仅记录本轮用户豁免 |

故障验收读取真实最终 science/host/apps payload，并逐字节核对冻结 `startup_cache` 与 `compact_runtime` 实现。低空间为明确的 `disk_usage free=0` 模拟，没有填满磁盘；权限使用本次自有目录的真实 DACL 拒绝写入，原 Access SDDL 精确恢复；保护文件冲突保持原文件。三项均失败且退出 23，检查覆盖缓存准备边界，不等同于冻结 GUI 错误弹窗验收。实际耗时 6.546 秒，输出同时保留为 `final-failures-r7.json`，复制后的 `r7/final-failures.json` 字节一致。

最终启动卡由仅属于本次 Job 的 offscreen HWND 通过 PrintWindow 捕获，未读取用户屏幕或其它窗口。图片为 930×468，RGB 255/249/232，SHA-256 `6e32573ec2d614a2bbfdb0c79edcbe87a7de6aaff5b01af3fbb88adf1027153a`。163.422 秒包含并行 QA 中的首次缓存准备与检查，不作为启动性能成绩。临时根清理限制见下节。

两类换版均从真实基线 EXE（SHA `28615fe0f7c39efe4a59f3621064f88a26ed2247641d69df7816c885e6bd9f1e`）的冻结辅助进程交接至 R7；旧版本号为元数据夹具，源码 Qt 仅初始化隔离设置和草稿，不操作用户数据。免安装正式 handoff/relaunch 为 9.094/83.406 秒，安装为 8.875/88.859 秒。原免安装失败是 QA source-mode 缺少 canonical entry，output-only 修正保留外部 origin 并建立基线同 inode 入口；第二次 D 盘空间不足后精确终止 Node（4294967295）。两次均保留在 previousAttempts，最终 retry2 在 C 盘独立 profile/TEMP 通过。额外免安装退出观察器因错误 origin 路径、进程已退出后 OpenProcess 87 失败，两份 observer-only JSON 保留。正式重启与实际 TEMP 无 MEI 检查通过，但未捕获免安装应用自然退出码；安装 `actual-process-exit.json` 有真实句柄退出 0。结论不扩展为所有历史版本或科学库升级。

启动速度对照原计划为同一 C 盘、交替旧/新三轮首次普通 GUI，再使用 new-3 正常退出后的缓存做三轮再次启动。用户明确要求发布后再测，本轮仅结束自有计时进程树，`timing-cancelled-processes.json` 保留精确身份与停止回执。`startup-report.json` 为 success=false、status=waived_by_user，3份部分测量保留，不计算完整中位数、不判定性能回归通过。后续测试仍采用普通 GUI 缓存、offscreen/软件渲染与 `max(1 秒, 旧包中位数的 5%)` 门槛，不清操作系统缓存。

## 官网资源与发布

| 对象 | 身份与 HTTP 范围 | 当前状态 |
| --- | --- | --- |
| MFA 3.3.8-r1 Windows x64 ZIP | 812,556,132 字节；SHA `610c6cacc7d6e8f6cb8610e8b6f55376925e56d88bfffafaa13fe620f40b7559`；服务器完整 SHA、HEAD 200、Range 206、完整 HTTP 下载 SHA 一致 | published / verified |
| MFA 对应源码 ZIP | 1,012,918,247 字节；SHA `6b7918c3bb05cd2e7a2e953dec753b92717e4a371b8f39ee53046eb508326522`；服务器完整 SHA、HEAD 200 与 Range 206，完整 HTTP 下载未执行 | published / verified，限定所列范围 |
| MFA release.json | 866 字节；SHA `23c60eb60b48070b75d4cde91a0e6d6ba907e2144f8dcff3969d4d2f5b6f3581`；服务器完整 SHA、HEAD 200、Range 206 | published / verified，限定所列范围 |
| R7 应用三件套 | `r7/app-upload-plan.receipt.json` 服务器完整 SHA 匹配；`public-verification.json` 三文件 HEAD 200、大小一致，首尾各1024字节 Range 206 与本地字节相同；未重复完整 HTTP 下载 | published / verified，限定所列范围 |
| 固定版本清单及 SHA256SUMS | 1123/324字节，公开 HTTP 完整摘要分别为 `2fb8bf897463a35a885988562d1311937aa2c6674977f98d60ccc19ea8ffa30c`、`56890cd24dfd91713a0433459765ff4afbee5f6f5f0a0432cd459001dd997cf9`，均匹配服务器回执 | published / verified |
| stable/preview 清单 | 两频道均3.0.0，与固定清单同 SHA，实际应用更新解析器接受，`manifest-upload-plan.receipt.json` 与 `public-verification.json` | published / verified |
| 官网两个按钮与既有路由 | 原子切换完成，两下载链接正确，34项网页资源实际 HTTP 检查通过，论文目录摘要不变、反馈服务 active、nginx配置未改；真实浏览器12项全部通过 | published / verified |

MFA 回执为 `mfa-upload-plan.receipt.json` 与 `mfa-public-http.json`。500,000,000 字节限制适用于主应用三件套。应用上传、更新清单与网页切换均在最终功能验收通过后执行；启动性能仅按用户最新指示豁免。公开检查在2026-10-11完成，网页发行身份为 `r2eb8b625d7798554`，回执包括 `app-upload-plan.receipt.json`、`manifest-upload-plan.receipt.json`、`website-publish.json` 和 `public-verification.json`。服务器完整大文件摘要复用上传回执，公开端点仅核验所列 HEAD/Range，未把分段验证写成完整下载。

真实浏览器报告 `output/validation/website/verify-1791731869286/report.json` success，errors=[]、requests=[]，12项包括两个3.0.0按钮的实际下载路由与附件名、1440/800/390/320宽度、18幅图片、交互/减弱动态和无JavaScript可用性。原 runner 最终 exit1发生在 finally 清理的自有进程退出竞态，页面结果通过；`cleanup-first-attempt.json` 保留。确认拥有进程全部退出后，原 helper 精确清理 scratch exit0/deleted。`r7/publication-result.json` success 严格复核6个上线步骤 exit0、页面结果及清理完成，原 runner exit1未隐藏。

## 未验与清理限制

工程外隔离测试仍在同一台 Windows 机器，未验证另一台干净电脑/虚拟机、Windows 10 实机、实体音视频设备、物理显示/听音、跨平台及长期运行。科学参数与媒体解码只按实际报告范围说明，换版只覆盖实际基线及协议。

六处已结束旧测试根已清理：R1/R2/R4/R5 的 CQA、R5 启动卡目录与 R5 DQA。`cleanup-receipt.json` 六项均 deleted，120 份必要 JSON/log/PNG 保存到 `cleanup-evidence/` 并核对 SHA。19,764,965,280 为包含硬链接重复的逻辑字节，未测实际释放空间。R7 旧包清理回执已删除旧 R6 staging，`output/build-PhoneticToolbox-300-20261011-R3` 因 incomplete/ownerPid 11476 保留，即使该 PID 当前已不存在也未顺手删除。活动 R7、基线、当前构建/交付、MFA 与未证实归属的目录均保留。

以下审批限制均未重试或绕过：

- `D:/PTB-Compact-QA-20261006/release300-r6/processes/temp/_MEI000013782` 为主动中断残留，清理返回 `blocked by policy`，见 `cache-interruption-r6-cleanup-limited.json`。R6 设置超时残留 `_MEI0000dad82` 亦保留，未猜测完整路径。
- `D:/PTB-QA-300-7cd42f16` 与旧源码测试 intentional junction 的历史清理被拒绝。精确旧联接条目未识别，未遍历或操作其目标。对象与原回执见 `cleanup-manual.json`。
- R7 卡片临时根 `%TEMP%/PTB-StartupCard-cfa867d7a2884e5ea1131b4dda98990c` 的精确清理命令被审批拒绝，进程未创建、无删除；`startup-card-r7/cleanup-limited.json` 记录实际绝对路径与保留。产品缓存已清且正常退出 MEI 为 0，该限制不计为产品退出失败。

R7 受控强制终止留下 `_MEI0000b9d02`，当前仅记录未清，不代表正常退出残留。剩余自有测试缓存与该残留按归属/进程单独核查，不因发布前计时取消而计为清理完成，也不阻塞已验证的公开入口。

`Remove-ReviewedReleaseTestTemps.ps1` 默认预览，`-Apply` 执行固定六项白名单，遇性能测试暂停，不能处理上述拒绝对象。另一个人工删除脚本的生成命令亦被审查以 `blocked by policy` 拒绝，进程未创建、脚本未写入、未执行删除。当前只交付六项脚本和不可自动处理清单，不宣称人工脚本已完成。
