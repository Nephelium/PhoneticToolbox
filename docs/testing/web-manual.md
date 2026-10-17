# 网页版说明书前置入门验收

2026-10-16，任务 `WEB-MANUAL-INTRODUCTION`。

## 范围

权威工程新增 `manual/chapters/web-introduction.json` 并在 `manual/project.json` 最前登记。内容按当前账号、文件空间、容量与设置源码编写，按用户追加要求插入6张浅色实际页面截图。其余20章与原素材继续从同一工程生成，18个工具的说明全部保留。

宿主在 `frontend/src/app/manual-project.ts` 选择阅读目录。浏览器保留网页入门，桌面过滤这一章及其搜索项。现有正文描述对象、文件路径和素材继续复用。全局帮助按宿主选择入门章，模块帮助保留原章节和小节目标。

## 验证

- `scripts/manual/validate.py --project manual --strict`：最终21章、1059项素材，错误和警告均为空。`scripts/manual/build.py --project manual --output frontend/public/manual --distribution software` 与 `npm --prefix frontend run build` 通过，构建保留既有的大分块提示。
- `node --test tests/manual-reader.test.ts tests/manual-reader-render.test.ts tests/manual-formula.test.ts tests/manual-platform.test.ts`：17项通过。检查网页版前置顺序、桌面顺序及搜索过滤、全18模块保留、模块小节目标，以及共用描述对象。首次新增检查误将含可选空锚点的作者正文当作阅读JSON校验，修正为读取源正文建立搜索索引，实际阅读JSON仍由正式构建和浏览器校验。
- `npm run typecheck`、`npm run contracts:check`、`npm run ui-data:check` 通过。文档校验2170文件、132任务零错误，历史快照链接警告单列。`git diff --check` 通过。
- `python -B -X utf8 scripts/verify_web_manual.py`：Windows独立可见Chrome 155、自有配置目录、实际构建后的Vue页面最终通过6组检查。全局使用说明打开网页版入门，目录含21章及全部18模块，新增内容可搜索，MFA章节和共同操作可打开，再次点全局帮助回到网页版入门，6张图在阅读器中解码且原始尺寸均为2560×1600。页面错误为空。账号HTTP使用合成夹具，未连接真实账号或服务器。
- `python -B -X utf8 scripts/verify_web_manual.py --capture`：采集登录、注册、文件管理、任务记录、容量申请、设置6张图，按对应说明插入正文。Windows读取AppliedDPI=144，独立浏览器DPR=1.5、页面100%，浅色模式逐图断言。原始采集为2561×1601，仅裁去右侧与底部各1像素得到2560×1600，没有缩放图像。登录和注册未填真实个人资料，文件与任务采用真实页面的演示空态，不制作虚构处理结果。首次采图因关闭按钮定位名与实际不符而失败，按实际无障碍名称修正后完整通过，失败诊断与清理回执保留。
- 6张正式素材及最终阅读器浅/深主题已实际查看，标题、正文、目录、图注和编号均可读。`output/validation/web-manual/closure.json` 确认原有20章源文件SHA保持，新生成清单与21章阅读JSON在 `frontend/public/manual` 和 `frontend/dist/manual` 字节一致，6张素材在源工程/public/dist的SHA均一致。没有复制第二套模块正文。正式图按software-only/git:false登记并受忽略规则保护。

最终浏览器证据：`output/validation/web-manual/browser-3c1bee0b32d6447b9579a1e81e658d2d/`；正式素材采集清单：`output/validation/web-manual/browser-e48ed30d486c40e4a03a571cce71bd78/captures.json`。自有浏览器与预览服务已关闭，各轮 `cleanup.json` 为 `deleted`，保留报告、必要截图与清理回执。

## 未验范围

本次仅本机源码、生成阅读资源和浏览器说明书检查，不包含生产部署、最终EXE、完整账号邮件投递、真实数据库及到期清理或科研计算。说明书内容描述当前实现，相关功能的验收范围沿用各自报告。

## 设置保存方式修订（2026-10-17）

网页入门已将同浏览器保留改为服务器按账号持久保存，并说明换浏览器/设备恢复、退出与到期保留设置、网络失败重试。按当前界面重新采集全部6张正式浅色图，采集证据为 `output/validation/web-manual/browser-8974befc3f0d41908c4de420c7985495/`。生成阅读资源后，`browser-13a4876077a44f41963428930efd7e7e` 实际确认新正文、21章/18模块、搜索与6图解码，零页面错误。严格校验21章/1059素材、错误和警告均为空，两轮scratch回执均为deleted。设置服务的真实PostgreSQL/Chrome/Edge证据及现有库尚未迁移的范围见[设置验收](web-settings-retention.md)，说明书演示HTTP不作为服务器功能验收。
