# 开发与验证入口

业务代码在 frontend、backend、desktop 与 packages/phonetic_core。根 run.py/run.spec 与继承的 phonetic_toolbox 是迁移来源，不作为当前 v3 的日常入口。

## 启动与修改

先按[源码入口说明](development/source-entry.md)检查既有解释器和绑定：

```powershell
.\scripts\Start-Research-Workbench.ps1 -CheckOnly
.\scripts\Start-Research-Workbench.ps1
.\scripts\Start-Research-Workbench.ps1 -Module M10
```

第一条只读检查，不打开窗口。主环境与独立科学环境按原有兼容性保留；不照旧阶段文档重建环境、不回退到已安装的旧项目源码，也不修改 v2。已运行的 Python 进程需要重启，成品需要重新构建才会包含修改。

前端修改后按需运行（工程根目录）：

```powershell
npm --prefix frontend run contracts:check
npm --prefix frontend run ui-data:check
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
```

需要更新生成物时分别用 contracts、ui-data 命令，不手改生成副本。依赖已可用时不重复安装；依赖变化同时维护声明、锁和来源登记。模块依赖清单集中在 [requirements/](../requirements/README.md)，从仓库根目录引用时使用新路径；清单内部的相对引用保持不变。

## 检查与记录

- Python、原生、契约、UI 和发行检查按[验证策略](testing/verification-plan.md)及对应模块选取，使用该任务实际需要的解释器。
- 文档/来源检查：`.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/validate_docs.py`。失败须区分本次问题与已有历史问题，不能只略过报错。
- 架构检查入口是 scripts/check_architecture.py，源码绑定专项入口和副作用见[入口说明](development/source-entry.md)。文档变更无须运行 GUI 或全量科学计算。
- 账号、worker、文件与远程能力必须显式配置。服务入口不作为迁移授权，历史测试数据库可能已清理，复验按相关脚本重新建立独立测试对象，不能指向研究数据目录。
- UI 验证使用独立浏览器/Qt 和用户配置，退出只回收自有进程。生成输入、隔离数据库与导出副本按[产物生命周期](development/artifact-lifecycle.md)放入 scratch，用后清理，保留必要生成条件及结果。

## 发行与协作

Windows 打包只按 [release/PACKAGING_RULES.md](../release/PACKAGING_RULES.md)；旧 EXE 不能证明当前源码行为。提交前核查[仓库内容边界](development/repository-hygiene.md)，本地构建、提交、push 与发布分别按授权执行。

进度更新已有任务台账条目，命令和实际结果进入对应报告。本页只维护现行开发入口，不追加每次模块交付历史。
