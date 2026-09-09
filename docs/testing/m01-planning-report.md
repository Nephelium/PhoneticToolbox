# P08 / M01 迁移前审阅报告

2026-09-09。交付状态：源码审阅和实施计划已完成；**M01/P08 实现仍 planned**。没有执行v3声学算法或新数据库迁移。

## 本轮交付

- [文件级实施计划](../plans/2026-09-09-m01-implementation.md)：按A–G拆分独立基准、科学核心、原生/格式适配、契约、共同页面、持久批次与联合验收；文件路径、先测行为、命令和退出条件逐项列出。
- [源码对照](../modules/evidence/M01-source-map.md)：六组功能的控件→函数→输出→新入口，并单列11项差异/缺口。
- [参数/设置机器清单](../modules/evidence/M01-parameter-settings.json)：80参数键/显示名/旧单位，14控件默认/范围/单位及3个非对话框字段；30个当前源码/资源SHA-256与函数起止行。
- [30项验收表](../modules/evidence/M01-acceptance.csv)：六组功能均有正常语义、边界案例、目标入口和实施步骤；所有实现验收行保持planned。
- 更新M01规格入口、总计划、任务账本、README和根工作规则；修正旧草案中不存在的 `npm run test:e2e` 入口，保留原功能要求。

## 重要发现及处理

1. 当前继承的30个相关文件与相邻v2逐字节一致；源码事实和旧基线可以建立明确对应。
2. GUI默认80键选择会筛掉两个额外SOE；P03默认服务79列不能直接当作GUI导出基准。77列是静态推导，尚需下一步独立捕获验证。
3. 旧设置仅进程内生效；min/max F0虽在REAPER分组，也用于Praat与WM；“ZCR判定”文案与最终F0并集mask不一致。计划保留数值，文案/隔离/元数据修正分别记录。
4. 旧批分析自动寻找同名TextGrid、逐文件导出；切分只处理选中项，当前切分层不影响批分析的所有标签层。M01拟采用逐文件完整产物加持久批次汇总，避免取消时丢弃前面已完成结果。
5. v3当前无科学计算依赖；P07也未提供REAPER原生输出目录的强制配额能力或桌面目录任务。计划显式安排环境审计、原生输出可行性门和真实桌面文件授权。
6. 四项唇形、GUI/子集配置、非默认设置和实际回退分支尚缺完整基准。安全格式/受限本地转换须独立验证，网页不能接收任意pickle执行。

## 实际验证

使用当前 `.venv/v3-dev/Scripts/python.exe`，Windows PowerShell，读取中文/JSON显式UTF-8。

```powershell
& '.venv/v3-dev/Scripts/python.exe' -X utf8 -m pytest -c tests/pytest.ini tests/parity/test_baseline.py -q
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/validate_docs.py
git diff --check
```

- 既有P03基线完整性测试：**11 passed**。这是冻结结果/比较器/合成配方检查，不是本轮执行v3数值parity。
- 聚焦只读AST/哈希/CSV核对：30文件与v2一致，80唯一参数键、14唯一设置、30唯一验收ID、6功能组、12来源ID均通过；所有新增实现标记保持planned。证据 `output/validation/m01/plan-audit.json`。
- 文档/编码/JSON/链接及任务检查：**211个文件、300条来源、32个任务，错误0**；结果写入 `output/validation/m01/docs-check.json`。原封保留历史文档的36个缺失链接单独报告，不把它们算作新文档通过。
- v2保存性检查前后记录于 `output/validation/m01/context-before-plan.json`、`context-after-plan.json`；核对427个基线文件、HEAD、index、Git状态、原环境位置与包版本元数据。最终检查见 `output/validation/m01/final-checks.json`。

本轮仅增加规划和审阅资料，没有科学依赖安装、全局配置变更、数据库DDL、浏览器操作、语料处理或对外上传。既有来源登记的许可/版本未决项保持；未新增第三方运行依赖或算法移植。

## 下一项

从 **M01-A 独立基准补齐与科学环境审计** 开始：先补GUI默认/服务None/参数子集及四项唇形合成时轴证据，再核对科学依赖具体版本。无需提前处理其他模块或重做P07。原生输出与批次数据库设计需要各自具体审阅；本轮未假定相应技术可行性或执行授权。

缺少专门确认的真实持续元音与唇形配套语料、完整原GUI对照、v3数值与双端页面验收，仍明确保留。Mac/Linux桌面、生产负载、完整发行与所有算法科学准确度不在本次证据范围内。
