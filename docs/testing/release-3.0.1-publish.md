# 3.0.1 构建与发行记录

当前状态：`built` / `verified` / `published`。2026-10-13首发、R5替换之后，两件EXE现已替换为R6。默认GPU可见窗口绘制、侧栏、滚动、n构型、最终工程外分发、安装/卸载和换版在本机最终字节范围通过。线上ZIP、Release正文和标签保持首发内容，不能把R6证据转用于原ZIP。详见[绘制回归](rendering-stability.md)和前轮[尺寸、致谢与卸载检查](m07-resize-stability.md)。

用户于 2026-10-12 恢复构建与发布，并明确要求不用验证。随后要求用井韶子写作风格技能整理完整更新说明并先交其审阅。该首发阶段未运行科研、GUI、设备、安装/换版或性能验收；随后R5/R6按追加要求完成各自验收，证据分开记录。

## 当前成品

当前两件EXE为2026-10-13 R6同一应用，主包不含MFA环境、模型或词典。用户限定仅替换两件应用，线上ZIP保留首发R2字节。

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `dist/PhoneticToolbox-301-render-20261013-R6/PhoneticToolbox-windows-x64.exe` | 495841235 | `27d0a664667dc9cb2c871aa1c29e4a4404564533d152c468cb130280659c5b08` |
| `output/release-staging/release-301-render-20261013-R6/artifacts/PhoneticToolbox-3.0.1-windows-x64-setup.exe` | 497757168 | `d65b1c4d72880ab6e87c89d681229746f049e8914f9d5383430dca48fe5c3910` |
| 线上既有 `PhoneticToolbox-3.0.1-windows-x64-portable.zip`（R2） | 495838322 | `41c100d88f3c2439517f0f8bc54c02b196d2e16de737b4a5d12c7e7c7d085736` |

三文件均小于500000000字节。R6源码快照SHA为 `319e20f0e1f89dfca3290d0447dfc3078101a88c129a6c19be9c42a031bcc894`。服务器六文件及GitHub四附件回执保存于 `output/validation/rendering-20261013/replace-final.json`，原文件有私有回滚备份。固定版本及两频道清单SHA为 `ab246f342c3631810cedf587a99503a7ed55892ef0c1d2949f538327a6a6b6d2`，仅同步两应用大小/摘要，notes保持首发原文。用户后续要求不再公网整包下载，先前校验已经结束，此后未再下载。本轮应用源码与打包规范经追加授权同步main，同期官网保持本地原状，标签不移动。前轮R5回执仍在 `output/validation/m07-resize-20261013/`，本轮以R6记录为准。

首发R2两EXE的大小/摘要分别为495837756/`dcf95ba9f69a18a787b05b2693e54b4b3a7bde4f28587a8e322d63e1805e73bc`、497754059/`c982dd7745ff7a444d40e45bb026a9d5be6f4c3a5fe199683829936251ddd013`。首发来源基准提交为 `3759c4320586f51ed9fdfb69251c117c457d6eab`，本机冻结清单另列；其源码快照SHA为 `29553b146d8adbcdd94fe075b4926d503a136bdbd0134a22e87ba41fbcd183f0`。下列执行/发布结果均是首发历史范围，不能替代R6回执。

## 执行范围

- `scripts/sync_versions.py` 同步 3.0.1 元数据，保留 API 1.1.0。
- `scripts/manual/build.py --project manual --output frontend/public/manual --distribution software` 生成本轮 20 章阅读资源，当前登记 1053 素材，数量不是打包门槛。
- `npm --prefix frontend run build -- --outDir ../output/frontend-301-20261012-R2` 生成独立前端，保留既有大 chunk 提示。
- 唯一 EXE 构建入口使用 `--name PhoneticToolbox-301-20261012-R2 --frontend-dist output/frontend-301-20261012-R2 --lean-qt --compact-onefile --persistent-cache`，科学及宿主缓存取自当时存在的 R8 构建。业务源码从当前工程冻结，前端从独立目录冻结。
- `release/finalize_compact.py` 完成构建内部版本、大小、嵌入压缩包、20 个主 Qt DLL 及共享文件身份记录；`release/package_compact.py` 完成同字节安装器/更新 ZIP 封装和内部大小、SHA/CRC/布局约束。这些是构建完整性检查，未执行应用功能验收。
- 更新说明按照井韶子技能重写，唯一正文为 [3.0.1 更新说明](../../release/notes/3.0.1.md)，应用更新清单和 GitHub Release 使用相同正文。通过应用现有 Markdown 解析器生成本地说明预览及官网候选，保留 3.0.0 历史栏目。

日志和发布计划集中于 `output/validation/release-301/`。首次 R1 构建因共享前端目录中说明书资源暂时不存在而失败，诊断保留；本轮可指定独立前端生成目录，未替换科学库或降低约束。

## 清理与待发布

构建成功后的既有清理入口删除旧 R8 `dist` 与构建工作目录，安装器成功后清理旧 R7 安装 staging，回执见当前 `dist` 的 `cleanup-report.json`。活动/失败 R3、R1 构建目录按 incomplete 保留，用户设置、研究数据、源码环境及必要科学组件不在此次清理范围。

用户已完成右侧更新说明修订，`output/validation/release-301/approved-notes.md` 保留其原文，`notes-review.json` 绑定最终正文摘要。服务器版本清单与 GitHub Release 使用相同正文。服务器版本文件、两个应用更新频道和 GitHub Release 属于当前发布范围；用户最新明确把官网交给其他人，未修改或部署官网页面，先前本地候选不采用。

服务器初次上传尚未连接时，构建环境缺少 Windows 凭据模块，退出1；改用服务器连接说明已有的 Miniconda 解释器，不安装新依赖、不修改凭据。初次诊断见 `server-upload-initial.json`。随后一次 SSH 传输被远端断开，日志 `server-upload-attempt-2.log` 保留；从任务专用分片续传成功。应用户要求先服务器、再 GitHub，精确停止本任务 GitHub 上传进程21972，保留草稿和已到位的两个小附件，服务器完成后继续上传三个应用包。

## 发布结果

- `app-upload-plan.receipt.json`：三个应用文件、固定版本清单及 SHA256SUMS 五项 success，服务器完整大小/SHA匹配本轮文件。
- `manifest-upload-plan.receipt.json`：stable/preview 两频道 `latest.json` 原子发布 success，与3.0.1固定清单同SHA `6bc235a75ac4915b0c79329dbdc719ea670f2a95548b488cb90e85d7b98e9c94`。
- [GitHub Release v3.0.1](https://github.com/Nephelium/PhoneticToolbox/releases/tag/v3.0.1) 已公开并设为最新版本。`github-release-receipt.json` 五附件均 uploaded，其大小及 GitHub SHA-256与上表/服务器一致；Release正文与用户最终稿一致。
- 源码明确审阅574个候选文件后提交 `9fd58a7fbaa2005d054a95b6d6891653e3fa9c74`，已推送 `origin/main`；Release标签指向该提交。运行环境、缓存、安装包、软件专用说明书原媒体、作者头像、下载论文与官网同期工程不进入提交。头像缺图的公开源码准备入口不进入已冻结应用；另清理9处空白格式问题，不替换已上传成品。冻结文件的原字节继续以上述构建清单为准。
- 官网仅因用户询问进行了只读状态查询，当时3.0.1更新小记已由其他任务出现、下载按钮仍为3.0.0。本任务没有部署官网、修改线上下载按钮或替代网站任务。

最终发布记录/台账文档另行同步到main，不改应用成品及已确认的用户文案。当前结果不等同于干净Windows、Windows10实机、实体设备或跨平台功能通过。
