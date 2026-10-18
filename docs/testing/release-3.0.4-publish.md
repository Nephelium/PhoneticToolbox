# Windows 3.0.4 发行验收

日期：2026-10-19。状态：`published / verified`。本轮为Windows x64最终字节验收，实体设备、第二台干净Windows/虚拟机、Windows 10实机和其他操作系统未验。

## 冻结范围与文案

近期生理参数合成、TextGrid标注、参数显示、LPC、录音与唇形、音频藏信息、显式免安装缓存及相关说明书纳入本次成品。最新正负VOT、前浊与总时长修复已重新冻结。官网仅同步Windows下载版本、体积及本版更新小记，采用线上原页面最小修改并复用原资源。其他网站独立改动保留在工作区，网页版不部署。

井井续行指定采用亲自编辑的更新说明，允许轻微调整，明确加入说明默认折叠、前浊/动作衔接/总时长及移除送气按钮两项，最后逐字指定结尾为标注不返工。此版本已获服务器/GitHub成品及源码发布授权，全部完成后关机。唯一文案为`release/notes/3.0.4.md`，SHA-256 `724bcdc0ede779ad94484c7490f43d1b3a8ef540d9bbbb3985ea99f9ae92fa65`。生产Markdown渲染器和UpdatePanel样式在独立Chrome浅深色、720/1100两宽度复验通过，6标题21条，无横向溢出，最终截图已核对，见`output/playwright/release304-notes/report.json`。

## 成品与身份

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `PhoneticToolbox-windows-x64.exe` | 496,270,906 | `f2818b137d20f6f6f9540ff1066e1d326d0cfbeccd50c130e312bd45aacd6af7` |
| `PhoneticToolbox-3.0.4-windows-x64-portable.zip` | 496,271,472 | `ca4b2610c5630caf0c3cd6a9be0541ed0035c6c43e720570f3f7495afc244699` |
| `PhoneticToolbox-3.0.4-windows-x64-setup.exe` | 498,186,108 | `7934caf23dc3fa2cb6f1168600940002d05cbfd7b44d13944dff3b35ca32e1c2` |

免安装成品：`dist/PhoneticToolbox-304-20261019-042107-9c1e34/`。安装与更新分发：`output/release-staging/release-304-20261019-042107/artifacts/`。三件套各小于500,000,000字节，同一应用SHA，ZIP CRC/结构和安装封装白名单通过。源码快照SHA `cd4cd7fbfc3659e3b841eb494fea193495b39546a4a041ebc4370798a1813701`，873份冻结源码与当前来源逐份一致，见`output/validation/release-304/final-frozen-closure.json`。科学与宿主归档分别为`14354853dabf5a6b81dffd3236e5ef531ce8b6a9699ac11f6d4869077cd821e9`、`d5cd940ba6093d2896ed90feb92ed37bb75baaf15b259d97aba231c1d76268d3`，复用已验证字节，Qt主原生文件身份、包内容/MFA排除审计通过。

## 构建与最终验收

- `scripts/package_local.py --version 3.0.4 --defer-cleanup`最终编排293.155秒。561项前端、类型/UI数据/独立前端构建，说明书21章1095素材严格校验/生成通过，370项派生媒体均复用已校验字节。另行契约、架构与文档检查通过。
- `scripts/verify_local_package.py --package dist/PhoneticToolbox-304-20261019-042107-9c1e34`通过15个阶段，最终报告`output/validation/local-package-cfa25fd3/report.json`绑定本次EXE。工程外单EXE、Windows-only PATH、科学探针/11托管任务、帮助/客户端流程、中文长路径OwnedQA真实安装准备/再次复用/卸载和受保护数据、缓存损坏/中断/并发、设置清理重开、免安装与安装换版/媒体/设置恢复全部通过。
- 可见窗口使用默认GPU和独立状态，主题/背景/材质/字号/尺寸/滚动及像素检查、重复工作检查通过。原生窗口截图`output/validation/local-package-cfa25fd3/native-owned.png`已目视复核，窗口正常退出0，无_MEI残留。
- `scripts/verify_trimmed_startup.py --before <3.0.3最终EXE> --after <本次EXE> --expect-portable-cache-selection`通过，3次交替冷启动中位数旧52.062秒、新52.691秒，增加0.628秒，小于2.603秒门槛。新缓存再次启动9.307/8.462/8.792秒。计时为offscreen软件绘制、未清系统缓存，与上述默认GPU可见验收分开。旧隐式缓存保持、新选空目录重新准备，同目录两类换版组件复用另验。证据`output/validation/package-trim/startup-4156671e333d4f8f8119ef6cecc8545b/report.json`。
- 最终免安装EXE原生缓存选择实测取消退出及中文空格目录选择、配置写入、三组件准备与退出清理通过，证据`output/validation/release-304/frozen-cache-choice-5a90fe3b3f84491697d7d52a0ae0df97/report.json`。

## 失败诊断与清理

首轮构建557/559前端通过后停止：旧喉位撤销夹具遗漏新增默认唇形，来源覆盖漏列已登记浏览器科学运行库与Pink Trombone。补齐完整期望/原输入未变断言和来源覆盖，不改科研实现或数值门槛。随后559项旧候选构建成功，因井井新增生理合成修复作废其最终验收身份并重新构建561项最终包。首轮/旧候选日志保留在`output/packaging/`。

缓存原生选择验证首次跨进程使用WM_USER指针导致测试进程失败，随后编辑框文字未改变树选择而选中默认目录。配方改为仅向自有测试进程分配选择字符串，读取选择显示并核对最终配置的完整路径，再释放字符串。产品实现保持，最终两流程真实运行通过。误选生成的`%USERPROFILE%/PhoneticToolbox-cache`删除在执行前被自动审批以`blocked by policy`拒绝，保留目录与`chooser-default-cache-cleanup.json`回执，未通过替代命令重试。

最终封装沿用`cleanup_old_builds.ps1`清理5处已识别旧成品/构建/安装输出，保留当前成品和必要缓存；4处历史不完整构建按脚本保留。回执`dist/PhoneticToolbox-304-20261019-042107-9c1e34/cleanup-report.json`。QA自有scratch与生成录音按既有生命周期清理并保留报告，不触及用户原录音/研究数据/设置/说明书正式素材。

## 发布

服务器三件套及校验、来源许可、构建证明、固定清单共7文件完整SHA核对通过，stable/preview两频道均切换至3.0.4。公网3份清单完整GET摘要及三件套HEAD/首1024字节Range通过。GitHub Release公开发布，7附件大小/服务端SHA-256与本机一致，正文和两处更新清单notes使用作者确认的同一文案。源码提交 `ff0bc8d1050bb075f948cb18f560ad276770b66f` 已在origin/main，GitHub Release绑定该提交。证据 `output/validation/release-304/published-verification.json`，上传回执为server-plan.receipt.json和channels-plan.receipt.json。未重复完整HTTP下载大文件。官网只更新3份版本文件并复用原线上资源，原目录/配置和反馈进程保持，4宽度真实公网浏览器及5版本切换通过，记录`output/validation/release-304/website/`。

[GitHub Release](https://github.com/Nephelium/PhoneticToolbox/releases/tag/v3.0.4) · [服务器安装包](https://www.phonetictoolbox.com/releases/windows-x64/3.0.4/PhoneticToolbox-3.0.4-windows-x64-setup.exe)
