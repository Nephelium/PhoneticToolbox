# 3.0.1 构建与发行记录

当前状态：`built`，用户已完成文案修订并确认采用最终稿，服务器上传和 GitHub 草稿附件上传正在执行，尚未切换应用更新频道。

用户于 2026-10-12 恢复构建与发布，并明确要求不用验证。随后要求用井韶子写作风格技能整理完整更新说明并先交其审阅。本轮不运行科研、GUI、设备、安装/换版或性能验收，不能把历史源码或其它 EXE 的结果转移到本轮成品。

## 当前成品

本轮冻结开始于 2026-10-12，构建完成于 2026-10-13。三种分发使用同一个应用 EXE，主包不含 MFA 环境、模型或词典。

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `dist/PhoneticToolbox-301-20261012-R2/PhoneticToolbox-windows-x64.exe` | 495837756 | `dcf95ba9f69a18a787b05b2693e54b4b3a7bde4f28587a8e322d63e1805e73bc` |
| `output/release-staging/release-301-20261013-r2/artifacts/PhoneticToolbox-3.0.1-windows-x64-setup.exe` | 497754059 | `c982dd7745ff7a444d40e45bb026a9d5be6f4c3a5fe199683829936251ddd013` |
| `output/release-staging/release-301-20261013-r2/artifacts/PhoneticToolbox-3.0.1-windows-x64-portable.zip` | 495838322 | `41c100d88f3c2439517f0f8bc54c02b196d2e16de737b4a5d12c7e7c7d085736` |

三文件均小于 500000000 字节。来源基准提交 `3759c4320586f51ed9fdfb69251c117c457d6eab`，其后的本机源码通过冻结清单单列，不能用基准提交代替冻结内容。`output/build-PhoneticToolbox-301-20261012-R2/snapshot/source-snapshot.json` 的 SHA-256 为 `29553b146d8adbcdd94fe075b4926d503a136bdbd0134a22e87ba41fbcd183f0`。

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

服务器初次上传尚未连接时，构建环境缺少 Windows 凭据模块，退出1；改用服务器连接说明已有的 Miniconda 解释器，不安装新依赖、不修改凭据。初次诊断见 `server-upload-initial.log`，本轮上传日志及回执原位记录实际结果。没有提交或推送未审阅的业务源码，也没有公开完整软件分发专用冻结快照。当前未标功能 `verified`。
