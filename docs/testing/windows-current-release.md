# 当前 Windows 安装与免安装交付

当前交付为 **3.0.0 R7**，已在官网公开，本机最终字节功能验收、公开 HTTPS 与真实浏览器检查均通过，状态 `verified`，限定本报告范围。合并 QA 的 13 条正式命令全部退出 0 并绑定同一应用 SHA。用户明确将启动速度对照移至发布后，本轮发布前豁免，未形成完整性能结论。历史候选结果不计为 R7 通过。完整范围见[发行验收报告](release-3.0.0-publish.md)。

## 当前成品

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `PhoneticToolbox-windows-x64.exe` | 496,815,155 | `162c8fa542b7b3b1203ce535e3021235e261f0dfaeb095659f9946baec055edb` |
| `PhoneticToolbox-3.0.0-windows-x64-portable.zip` | 496,815,721 | `43b5d3c618cf8705c57a197ffa30b69a84f767397ef642213ea4c0bc4a597c9d` |
| `PhoneticToolbox-3.0.0-windows-x64-setup.exe` | 498,729,076 | `3c131b4897cd7d5b063530485c9cfc064c57440c3ff8f2420ab62c2fff3bcce9` |

免安装版在 `dist/PhoneticToolbox-300-20261011-R7/`，安装版及更新 ZIP 在 `output/release-staging/release-300-20261011-r7/artifacts/`。三种文件均小于 500,000,000 字节。冻结源码快照 SHA-256 为 `3d1fd7bb865ca389df96ca0b90a3a39327267e90ca76df3fd01abc30e4ffa880`。`output/validation/release-300/r7/final-identity.json` 核对 711 个源码文件、111 个前端资产，当前源码差异为空，更新 ZIP 内应用与免安装 EXE 同 SHA。

## 系统与安装行为

- 支持 Windows 10 1809 及以上、Windows 11 的 x64 系统。x86 32 位、ARM64、macOS 和 Linux 暂不支持。安装器使用 `x64os` 与 `MinVersion=10.0.17763`，支持声明不等同于所有系统与硬件均实测。
- 安装后及更新 ZIP 内入口保持 `PhoneticToolbox.exe`。关闭应用后可在原目录覆盖安装，无需先卸载。既有设置、草稿与研究资料保留。
- 安装完成后由开始菜单或桌面快捷方式打开。R7 保留系统重解析点保护，冻结运行文件使用无联接私有镜像回退，避免安装/卸载子进程继承该保护后访问联接失败；该回退不复制用户任务目录。
- 主应用不携带 MFA 环境、声学模型或词典，MFA 组件及对应源码单独提供。MFA 已上线检查不代表主应用已公开发布。

## 本轮实际结果

| 门槛 | R7 结果与证据 |
| --- | --- |
| 封装与身份 | `r7/final-identity.json`、`r7/audit-r7.json` 和正式 `package-report.json` 通过 |
| 安装与卸载 | `r7/install-uninstall.json` success：应用 SHA 一致、首次准备和复用 exit 0，活动租约阻止卸载，释放后卸载 exit 0；研究/设置标记摘要不变，安装文件与私有启动目录无残留。使用独立 OwnedQA 身份，未操作用户安装 |
| 工程外直接 GUI | direct-gui exit 0，`r7/direct-relaunch.json` 与合并 QA 通过 |
| 科学检查 | 普通分发、中文深目录分发与 TextGrid 均 exit 0，`r7/qa-report.json` success |
| 缓存恢复与设置 | 实际复制回退、损坏修复、准备中断/并发，以及清理取消/确认、数据保留和草稿重开通过。受控强制终止残留另列，不计为正常退出残留 |
| 缓存失败保护 | `r7/final-failures.json` success：受控模拟低空间、真实自有目录 DACL 拒绝、保护路径冲突均按预期 exit 23，无 ready/last-launch，保护数据未改，ACL 原样恢复 |
| 默认启动卡 | `startup-card-r7/report.json` success；真实最终 EXE 的 W1 黄蓝图标、暖黄 `FFF9E8` 背景与可读文字已截图目视确认，两个拥有进程自然 exit 0，MEI 为 0，缓存清理完成 |
| 两类换版 | 实际旧辅助进程至 R7 的免安装/安装交接与重启通过，组件复用、设置/草稿/研究标记、9 个媒体播放器与16幅图片回读通过。安装分支进程句柄退出 0；免安装分支正式命令和 TEMP 检查通过，额外退出观察器失败保留，不声称捕获到该应用的自然退出码 |
| 启动性能 | `waived_by_user`，移至发布后另验；保留3份已完成的部分测量，未完成完整对照，不声称性能通过 |
| 应用公开入口 | 官网三件套、固定清单及 SHA256SUMS 已公开，服务器完整 SHA 匹配；三个大文件 HEAD 200/大小及首尾各1024字节 Range 206 与本地相同，未重复完整 HTTP 下载。两频道更新解析及真实浏览器12项通过，含两按钮地址/附件名、4种宽度、18图、减弱动态与无JavaScript；最终 `r7/publication-result.json` success |

## 未验与清理边界

测试使用本机工程外隔离 profile、Windows-only PATH 和自有进程。另一台干净电脑/虚拟机、Windows 10 实机、实体音视频设备、跨平台及长期运行未验。换版采用真实旧/新字节与旧版本元数据夹具，源码 Qt 初始化隔离设置，不扩展为所有历史版本或科学库升级保证。免安装的原 QA 布局错误、D 盘不足及额外观察器失败均保留，成功重试在 C 盘隔离目录完成。

官网浏览器原 runner 退出1发生在 finally 清理的进程退出竞态，页面报告 success、errors/requests 为空。拥有进程退出后原清理 helper 精确清理 scratch 成功，原失败回执仍保留，独立发布汇总确认 UI 与清理均完成。

已清理六处结束的旧 R1/R2/R4/R5 测试根，保留 120 份必要报告、日志和截图。历史已被自动审批拒绝的目录、未知精确入口的旧源码联接和 R6 中断残留仍列清单，未重试或绕过。R7 启动卡的缓存已清，但删除其自有临时根的命令也被审批以 `blocked by policy` 拒绝，进程未创建，临时根保留。默认预览的六项清理脚本不处理这些拒绝对象，独立人工脚本生成亦被审批拒绝。详细对象与回执见[发行报告](release-3.0.0-publish.md)。
