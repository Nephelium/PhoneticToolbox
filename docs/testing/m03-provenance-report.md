# M03-E3/A29 来源与方法说明复核

2026-09-13。直接源码来源与说明修订为 verified（限定本地静态核对及下列前端检查），最初文献对应与完整授权链未闭合，按井井本轮最新要求暂停继续核查。完整 M03 保持 in_progress。

## 本轮结论

井井明确 EGG 实现在 V2，本轮再次定位 `core/egg/analysis.py` 的事件检测和 CQ/SQ，以及 `core/egg/inverse_filtering.py` 的简化逆滤波。八份原文件 SHA-256 全部与 [迁移清单](../../third_party/egg-migration.json)一致。V2 是已确认的直接迁移来源，不能因更早的文献资料不足而把它写成未找到代码。

两个核心文件的 `git log --follow --format='%h %ad %s' --date=short` 均只返回 2026-03-20 整仓导入 `f6108fff90788db0d1d2315ecba48e90b82913ec`。文件、目录 README 和服务层的定向关键词检索没有发现作者/许可/DOI声明。该结果限定于这些文件及当前可见 Git 历史，不推断其他地方不存在材料。

V2 与 V3 兼容层均从 GCI 下一样本起固定取 3 ms、自相关/Toeplitz LPC 估计、至少三段系数平均；未使用 GOI 确认闭相，也未在下一 GCI 截断。已在 [方法核查](../references/m03-method-audit.md)、[使用说明](../manual/egg-analysis.md)、页面帮助和 IF 结果提示中明确。核心数值、默认值、文件输出及图窗布局均未修改。

## 来源登记与证据边界

`PENDING-EGG` 新增已核验 V2 快照说明、可见历史和实际 IF 步骤。保留该 ID，避免破坏旧结果来源关联；`license=not-established` 和 `observed_upstream_commit=null` 继续保留。不能用本地整仓导入提交冒充外部上游版本。登记生成的软件致谢现在能够显示 V2 快照信息。

现有尹基德馆藏和 Henrich（2004）条目的关系不变。本轮检索到其他逆滤波论文书目，但部分全文/出版社访问失败，未将它们补写为本代码的原始出处，也未新增论文登记或声称读过全文。来源总数仍 330，无新运行依赖，无论文全文复制。

## 验证

使用 `.venv/m09-ui/Scripts/python.exe` 的 `hashlib` 对照迁移清单：8/8 一致；`ast` 核对 V2 原函数位置。命令、源路径、函数名和完整哈希见方法核查。

`npm --prefix frontend run ui-data`、`run ui-data:check`、`run typecheck`、`run build` 已通过。帮助文案的独立 Chrome 显示检查随本轮[功能验证](m03-preview-switch-report.md)记录。不重跑完整科学基准、Qt、物理声卡或生产账号流程，说明修订不能扩展历史数值验收范围。

## 剩余工作

井井在本轮明确代码引用暂不处理，优先功能实现。A29 进一步核查暂停，不作为开发功能推进的前置条件。未决材料包括最初方法文献对应、SQ 定义出处与完整授权材料。已确认的 V2 代码迁移可继续使用现有证据，不需要重新向井井索取代码。后续获得材料再更新来源状态，不为补引用静默更换算法。

开发态功能覆盖见 [六功能组复核](m03-function-review.md)。实际声卡、原生多屏/DPI和生产证据仍单列；EXE、相关探针及打包按井井要求暂停，保留已有未完成草稿。本轮未执行 DDL、修改 V2/环境、push 或对外发送内容，不自动推进 M04。
