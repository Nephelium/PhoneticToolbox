# 语音学论文精读：发布论文

公共目录入口：<https://www.phonetictoolbox.com/papers/catalog.json>。服务器目录 `/var/www/phonetictoolbox-coming-soon/papers`，现有首页保持原文件。客户端读取目录中的栏目发布日期、作者、版本、许可、导读与 PDF 校验信息。

新增论文需准备原文 PDF、中文译文 PDF 和元信息 JSON，可另附独立解读 PDF，再发布完整目录。单独把 PDF 放进服务器目录不会自动上架，以免缺少许可和署名。应用不在用户电脑上自动翻译论文。

## 内容准备

先核对具体版本允许共享与翻译，并检查其中另有声明的第三方材料。当前目录协议接受 CC BY 4.0、CC BY-SA 4.0、CC0 1.0 的 arXiv 固定版本，其他授权形式需扩展审核规则。

元信息使用 `id`、`title`、`titleZh`、`authors`、`version`、`sourceUrl`、`submittedAt`、`publishedAt`、`license: {id,url}`、`translationNote`、`guide`。`publishedAt` 为栏目发布日期，格式 YYYY-MM-DD；`submittedAt` 为原稿投稿日期。稳定 `id` 可包含字母、数字、连字符和下划线。译文须注明改编者、许可、修改和是否经原作者审校。

可选字段 `abstract`、`abstractZh` 供往期窗口显示双语摘要，`analysis` 为独立分析正文，`analysisPdf` 为可离线阅读的解读文件。为已有论文补充内容时保留稳定 ID 与 `publishedAt`，填写 `updatedAt`，使较晚首次启动的用户也能获取更新。下载跳过已核验资产，原文与译文已获取时仍可独立阅读。

首篇内容制作项目在本机 `output/paper-reading/2610.00735v1`，包括原文、TeX/图表来源、174 个翻译单元、逐单元来源映射、译文和页面校验图。发布包在 `output/paper-reading/publish`。这些目录均不属于应用打包输入。

```powershell
# 先下载当前完整目录，后续新增时必须保留已有条目。
Invoke-WebRequest 'https://www.phonetictoolbox.com/papers/catalog.json' -OutFile current-catalog.json
.\.venv\m14\Scripts\python.exe scripts/prepare_paper_release.py `
  --metadata new-paper.json --original original.pdf --translation translation.pdf --analysis analysis.pdf `
  --catalog current-catalog.json --output output/paper-reading/next-publication
scp -r output/paper-reading/next-publication/. admin@8.134.196.183:/home/admin/papers-upload/
scp deployment/papers/publish.py admin@8.134.196.183:/home/admin/papers-upload/publish.py
ssh admin@8.134.196.183 'python3 /home/admin/papers-upload/publish.py /home/admin/papers-upload'
```

发布脚本先核对所有 PDF 大小和 SHA-256，写入以摘要命名的文件，最后原子替换目录。现有 PDF 不会被同名覆盖。目录更新后用 HTTPS 回读目录和全部资产，核对本地发布包摘要。未变的旧文件可以留在服务器，客户端跳过已校验内容。没有独立解读时省略 `--analysis`。

客户端启用自动获取后每次启动检查一次，持续运行默认每 6 小时检查，可设置 0.25–720 小时或关闭。定时器由 Qt 宿主持有，切换模块不重复检查。个人 PDF 仅保存在用户选定的本地文件夹，收藏夹是实际子文件夹，不上传分发服务。

栏目说明使用目录独立的可选 `notice` 字段，包含 `id/title/authors/publishedAt/updatedAt/pdf`，不混入科学论文 `papers`。`pdf` 同样包含安全相对路径、大小和 SHA。支持该能力的客户端在启动时取得说明，即使论文自动获取关闭或首次日期更晚；关闭时不下载科学论文。旧版程序忽略说明字段，须更新程序才能显示置顶入口。目录现为两篇科学论文的六份资产，另有一份栏目说明。

发布栏目说明可使用 `publish_revision.py --notice`，其余参数沿用基线、发布目录、仓库外 SSH 辅助文件和回执。只允许变更 `notice`，现有论文和其他目录字段逐项保持；与译文修订共用完整资产核验及目录最后发布流程。作者批准的制作项目、图片与回执留在本机忽略目录，不进入应用包或 GitHub。

2026-10-10 已在授权范围内补充 CC BY 4.0 的 2610.00735v1 独立解读，保留原文、译文和栏目日期，全部资产经 HTTPS 回读校验。其余三篇的 arXiv nonexclusive 许可尚无共享译文的明确授权，未上架。服务器凭据仅使用仓库外的本机加密存储，不进入代码、目录包或 Git。应用未公开发布，最新本地免安装EXE验证另见[R4报告](../testing/2026-10-10-m18-r4-report.md)。系统、SSH、HTTPS证书与nginx配置保留。

目录现另收录 CC BY 4.0 的 [2408.05641v1](https://arxiv.org/abs/2408.05641v1)：原稿为 2024-08-10，栏目日期为 2026-10-10，原文 5 页、全文中文译文现为 5 页双栏及独立解读 6 页。译文于2026-10-12按原稿格式修订，原文、解读与稳定ID保持。完整译读与核验来源位于本机 `output/paper-reading/2408.05641v1`，当前排版与发布配方在其 `layout-20261012/`，不作为应用资产。目录两篇及六个PDF均经HTTPS大小/SHA核验。内容更新不要求更换支持当前目录协议的EXE，收藏夹删除等应用功能修改须在更新EXE后使用。

## 修订已有译文

从当前HTTPS目录取得新基线，以原稳定ID与栏目日期准备修订元信息，更新`updatedAt`和译文内容摘要。[排版脚本](../../scripts/reformat_p2a_translation.py)复用固定原稿的模板、矢量图和既有译文，编译显式禁用包安装与shell escape。[修订发布脚本](../../deployment/papers/publish_revision.py)接受发布目录、基线、论文ID、仓库外SSH连接辅助文件和回执位置；默认只核对，`--apply`执行授权发布，必须保留其他论文、原文和解读，服务器目录基线变化时停止。

本次421173字节译文SHA-256为`93a2e1c5f22894136e170eb5576d3aad161cac61f6265470499938029016aaf6`。已下载旧译文的隔离客户端仅取得这一新资产，第二次检查零下载。用户可点检查新论文，然后切到原文再切回中文译文，重新打开五页新版。旧版页码批注按旧资产摘要保留，不能自动套用到新页码。服务器旧PDF不删除，个人文件不参与上传。
