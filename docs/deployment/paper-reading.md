# 语音学论文精读：发布论文

公共目录入口：<https://www.phonetictoolbox.com/papers/catalog.json>。服务器目录 `/var/www/phonetictoolbox-coming-soon/papers`，现有首页保持原文件。客户端读取目录中的栏目发布日期、作者、版本、许可、导读与 PDF 校验信息。

新增论文需准备原文 PDF、中文译文 PDF 和元信息 JSON，再发布完整目录。单独把 PDF 放进服务器目录不会自动上架，以免缺少许可和署名。应用不在用户电脑上自动翻译论文。

## 内容准备

先核对具体版本允许共享与翻译，并检查其中另有声明的第三方材料。当前目录协议接受 CC BY 4.0、CC BY-SA 4.0、CC0 1.0 的 arXiv 固定版本，其他授权形式需扩展审核规则。

元信息使用 `id`、`title`、`titleZh`、`authors`、`version`、`sourceUrl`、`submittedAt`、`publishedAt`、`license: {id,url}`、`translationNote`、`guide`。`publishedAt` 为栏目发布日期，格式 YYYY-MM-DD；`submittedAt` 为原稿投稿日期。稳定 `id` 可包含字母、数字、连字符和下划线。译文须注明改编者、许可、修改和是否经原作者审校。

首篇内容制作项目在本机 `output/paper-reading/2610.00735v1`，包括原文、TeX/图表来源、174 个翻译单元、逐单元来源映射、译文和页面校验图。发布包在 `output/paper-reading/publish`。这些目录均不属于应用打包输入。

```powershell
# 先下载当前完整目录，后续新增时必须保留已有条目。
Invoke-WebRequest 'https://www.phonetictoolbox.com/papers/catalog.json' -OutFile current-catalog.json
.\.venv\m14\Scripts\python.exe scripts/prepare_paper_release.py `
  --metadata new-paper.json --original original.pdf --translation translation.pdf `
  --catalog current-catalog.json --output output/paper-reading/next-publication
scp -r output/paper-reading/next-publication/. admin@8.134.196.183:/home/admin/papers-upload/
scp deployment/papers/publish.py admin@8.134.196.183:/home/admin/papers-upload/publish.py
ssh admin@8.134.196.183 'python3 /home/admin/papers-upload/publish.py /home/admin/papers-upload'
```

发布脚本先核对所有 PDF 大小和 SHA-256，写入以摘要命名的文件，最后原子替换目录。现有 PDF 不会被同名覆盖。目录更新后用 HTTPS 回读目录和两份 PDF，核对本地发布包摘要。未变的旧文件可以留在服务器，客户端跳过已校验内容。

本次仅发布论文内容，已清理旧发行暂存和测试部署，当前软件 EXE 尚未在公共渠道重新发布。系统、SSH、HTTPS 证书与 nginx 配置保留。
