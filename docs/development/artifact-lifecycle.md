# 测试与打包产物生命周期

## 测试目录

每次验证可使用独立运行 ID，但不保留完整运行副本。报告、日志、必要截图放运行目录根部；可重新生成的音频、复制工程、隔离数据库、缓存和大导出统一放 `scratch/`。

Python 使用 [verification_artifacts.py](../../scripts/verification_artifacts.py)：

```python
from verification_artifacts import verification_run

with verification_run('module-id', 'case', '生成脚本、采样率、时长、参数和随机种子') as (out, scratch):
    # 只向 scratch 写临时输入/输出，向 out 写必要证据。
    # 内层 try/finally 关闭并等待本次启动的服务、浏览器、科学子进程及文件句柄。
    pass
```

退出上下文会先记录 scratch 内文件的大小与 SHA-256，再调用 Windows 的 [清理入口](../../scripts/cleanup_test_scratch.ps1)。入口复核目录归属、各级联接和活动子进程，只清 `scratch/`。失败测试也执行收尾；原异常保留，无法清理的路径/原因写 `cleanup.json`，不得把 retained 当成已删除。强杀无法保证执行 finally，遗留的 active 标记需下次人工核对。

外部 `--source` 始终由调用方保管，不能因用作测试输入就删除。若要将前一步生成的长音频传给后一步，应在共同的外层上下文生成，完成所有读取与比较后清理，不依赖上一轮遗留 WAV。小型公开固定基准按 tests/fixtures 规则保留。

已接入的入口：M03-R7 原生长文件、任务保存、大 CSV，以及 M03 长文件 V2 parity。作者工具单元/浏览器测试使用 [artifacts.mjs](../../tools/manual-studio/tests/artifacts.mjs)，保存报告/截图与工程文件摘要，关闭服务/浏览器后由同一 PowerShell 入口删除独立测试工程和导出副本。正式 manual 工程的正文与素材保留，自动恢复稿只保留最后一份；成功保存后清历史/已提交事务，异常恢复证据和清理失败原因保留。详见[作者工具保存说明](../../tools/manual-studio/README.md)。

其他历史验证脚本尚未全部迁入此机制。重新使用或修改时先确认输出和硬编码历史输入，将临时对象移入 scratch。禁止仅按扩展名全盘删除或事后递归清空整个 validation。当前自动删除入口仅实现 Windows，其他平台写清 retained 回执，后续单独适配。

## 说明书生成资源

正式源工程和可重新生成的阅读目录分开。software 阅读版成功发布新索引/报告后，按上一版记录的 SHA 清除已不再引用的旧生成文件，失败不清旧版；未登记、已变化或含联接的文件报错保留。public 阅读版仍拒绝混有旧文件的目录，不自动清理，以防受限媒体进入公开输出。

架构检查直接根据源工程与已有构建报告核验当前阅读目录，不反复扩充公共静态资源清单。修改源工程后重新生成阅读版，运行 `python scripts/check_architecture.py` 检查内容、摘要及残留。

## 打包目录

- 构建：`output/build-<name>/`；免安装成品：`dist/<name>/`；安装与更新包装：`output/release-staging/<name>/`。不复制免安装 EXE 到每轮安装暂存。
- 两个主构建入口从首次写入起记录 `artifact-lifecycle.json`，含归属、PID、active/completed/failed 和结束时间。失败目录保留诊断；下次清理回执显式列出 incomplete，避免早期失败因无 spec/report 而消失在盘点之外。
- 新版成功后沿用 [cleanup_old_builds.ps1](../../release/cleanup_old_builds.ps1)清旧成品/旧工作目录。只打免安装版时保留最近正式安装版，最新构建缓存保留。活动构建引用的旧缓存、联接、运行中目录不删。
- 只盘点可使用 `-PlanOnly`，输出 `cleanup-plan.json`，不覆盖已执行的 `cleanup-report.json`。没有 completed 的目录需核对错误/进程再处理，构建失败不删除旧可用包。
- 每次实际发行仍完成[严格打包规则](../../release/PACKAGING_RULES.md)，生命周期检查不能代替最终 EXE 科研/安装验收。

## 本机整理和无法删除的文件

使用 `python -B -X utf8 scripts/inventory_workspace.py` 原位更新目录汇总及前 80 个大文件，不输出每个文件的全量路径。最新工作区盘点和清理清单统一放 `output/maintenance/`，原位更新 summary/plan/results，避免再生成数轮全量 inventory。需要保留的巨大旧 JSON 清单可压缩成一个可回读归档，核对 SHA 后删除重复原件。

[cleanup_workspace.ps1](../../scripts/cleanup_workspace.ps1)只接受已逐项审阅的 `ptb-reviewed-cleanup/1` 清单，默认只检查，`-Apply` 才删除。每次核对项目绝对路径、Git 跟踪/忽略、完整文件列表、SHA、联接与活动进程，变化或占用项保留并写原因。清单只允许明确测试范围，当前开发库、工作台配置及 M03 运行时暂存禁止进入自动清理范围。已授权的说明书自动副本清单还须显式标记 `manual_autosave_authorized`、指定保留恢复稿及摘要，并确认作者锁已释放；范围仅为 `.studio/history`、`recovery`、`transactions`，不包括正文与素材。

```powershell
.\scripts\cleanup_workspace.ps1 -Manifest .\output\maintenance\cleanup-plan.json
.\scripts\cleanup_workspace.ps1 -Manifest .\output\maintenance\cleanup-plan.json -Apply
```

删除回执和待人工处理列表应写实际结果，不把计划字节数当成释放量。未确认用途的目录另列审阅项，不能混进可批量删除清单。所有这些本机清单、临时输入和安装包均受 Git 忽略规则保护；提交源码时仍检查已跟踪内容。每次脚本修改和实际清理同步原位更新[任务台账](../plans/task-ledger.json)的范围、状态、验证与遗留项。
