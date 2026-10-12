# 测试临时文件整理

状态：verified，仅限当前 Windows 项目内已确认归属的测试临时对象与清理入口。核对日期：2026-10-11。

## 实际结果

27 个目标、45,934 个文件已删除，合计 16,087,103,776 字节逻辑长度。工作区盘点约 58.32 → 42.26 GB，盘点未沿目录联接遍历，零读取错误。文件长度包含硬链接的重复计数及稀疏虚拟磁盘，不能据此声称释放同等物理容量。

| 已删除类型 | 目标 | 文件 | 逻辑长度 |
| --- | ---: | ---: | ---: |
| 已停止的反馈测试模拟器磁盘和私有配置 | 2 | 25 | 3.47 GB |
| 已完成安装/换版验收的成品副本、隔离配置和启动缓存 | 8 | 45,796 | 12.31 GB |
| 可重建的合成长音频、逆滤波 WAV 与含音频的请求副本 | 11 | 11 | 222.85 MB |
| 已结束测试的 scratch，包括音频、缓存和测试工程副本 | 5 | 97 | 77.83 MB |
| 本次清理脚本的合成边界夹具 | 1 | 5 | 86 字节 |

所有已删除路径均复核不存在；21 份保留报告、所有权记录、收尾清单和生成脚本的 SHA-256 不变。M03 长文件请求的配置和输入 SHA 已另存，400 Hz/30 分钟及 48 kHz/120 秒的生成条件保留。最新 R7 免安装 EXE、安装包和更新 ZIP 三件套的大小与 SHA 均不变。

## 清理入口与边界

[批量入口](../../scripts/cleanup_test_temporaries.ps1)默认预检，`-Apply` 执行删除。它先运行[清单生成器](../../scripts/prepare_test_cleanup.py)，再调用[逐项删除入口](../../scripts/cleanup_workspace.ps1)。机制和命令见[产物生命周期](../development/artifact-lifecycle.md)。

本机清单限定审阅过的测试对象及 `ready` 且与收尾摘要一致的 scratch，实际删除再次核对文件列表、SHA、保留证据、Git 跟踪/忽略、联接与进程引用。清单中的保护路径同时约束实际执行，删除父目录也不能绕过保护。清理失败留回执，不结束用户或同期任务的进程。

保留同期 `release-300`、`release-review-20261011` 发行验收范围、最新构建缓存和实际运行时；保留当前开发数据库、M03 打包输入和仍有脚本引用的 `m08-wiring`。正式说明书、用户原件和研究数据不在清理范围。混合视频/原生设备数据及用途不明的历史目录继续逐项审阅，不能按文件扩展名或目录名称整删。

9 个此前遭自动审批拒绝的已核实小测试文件，合计 729,447 字节，当前仍存在且摘要吻合。8 个为 M18 成品科学 QA 的合成 WAV，另 1 个为清理边界夹具。已集中到本机手动删除清单，本轮未绕过此前 `blocked by policy` 拒绝，原回执没有给出详细原因。

## 验证和证据

7 项独立 Git/文件系统单元测试及 6 项既有生命周期回归通过，涵盖跟踪文件、外部路径、保护父子路径、证据条件、active/ready 状态、收尾清单变化及真实目录联接。6 项 PowerShell 预检探针符合预期：正常目标只预检、变更文件/证据及保护对象拒绝、完整验证根目录拒绝、旧格式缺少可选字段仍可识别已不存在目标。两份 PowerShell 脚本语法解析无错误，批量入口独立空清单预检成功，原实际清单与回执保留。微型测试目录本身已列入清单清理。

主要命令：

```powershell
.\.venv\m14\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -p test_test_cleanup.py -v
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/prepare_test_cleanup.py --rules output/maintenance/test-temporaries-rules.json
.\scripts\cleanup_workspace.ps1 -Manifest output/maintenance/test-temporaries-plan.json -ReportName test-temporaries-review.json
.\scripts\cleanup_workspace.ps1 -Manifest output/maintenance/test-temporaries-plan.json -Apply -ReportName test-temporaries-results.json
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/inventory_workspace.py
```

实际路径、删除理由、逐文件摘要和回执统一原位保存于 Git 忽略的 `output/maintenance/`：`test-temporaries-list.md`、`test-temporaries-rules.json`、`test-temporaries-plan.json`、`test-temporaries-review.json`、`test-temporaries-results.json`、`test-temporaries-verification.json`、`test-temporaries-audio-recipes.json`，以及合并后的 `manual-delete.md/.txt/.json`。必要记录保留，未来复用时更新同名文件。

本次未重打 EXE、复跑完整科研/GUI/设备或跨平台验证，未提交或上传。历史测试入口仍在复用或修改时迁入 scratch，清理入口不能代替各模块的功能验收。

文档检查覆盖 1,952 个文件，零错误；受保护历史快照的原有旧链接单独登记，不改写历史。当前状态、导航、生命周期说明与 WORKSPACE-MAINTENANCE 台账条目已同步更新，本机详细清单均受 Git 忽略保护。
