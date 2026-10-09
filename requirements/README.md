# Python 依赖清单

这里集中维护 v3 的模块依赖声明、版本锁和配套 Conda 清单。实际安装文件在项目 `.venv/` 中；移动清单不改变已安装环境。源码入口和环境绑定见[开发说明](../docs/development/source-entry.md)。

## 清单用途

| 文件组 | 用途与边界 |
| --- | --- |
| requirements-v3-dev.in / .lock | API、Qt 和工程测试依赖 |
| requirements-m01-science.in / .lock | M01 科学计算依赖 |
| requirements-m01-test.in / .lock | 在科学清单上增加测试与构建依赖 |
| requirements-m01-io.in / .lock | 参数导出及后端 I/O 依赖 |
| requirements-m01-ui.in / .lock | 在上述清单上增加 Qt 工作台依赖 |
| requirements-m03-compatible.in / .lock、requirements-m03-conda-explicit.txt | EGG/LPC 兼容环境；SciPy/MKL 原生构建另由 Conda 清单固定 |
| requirements-m03-exports.in / .lock | 在 M03 兼容清单上增加绘图和导出依赖 |
| requirements-m03-test.in / .lock | M03 纯科学测试环境，引用 M01 测试清单 |
| requirements-m05.lock | 唇形独立环境的固定版本清单，当前没有配套 .in |
| requirements-m10-ui.in / .lock、requirements-m09-ui.in / .lock | 声道录制和语谱图开发环境的递进依赖 |
| requirements-m14.in、requirements-m14-additions.lock | 音系归纳的依赖增量；additions.lock 仅包含新增项，不代表完整主环境 |
| requirements-m18.in | 论文批注导出的依赖增量，当前没有配套 .lock |

`.in` 表达依赖要求，`.lock` 固定具体版本，部分锁还包含传递依赖与下载文件 SHA。清单按目标环境选择，兼容环境和主宿主环境各自维护。根目录 [requirements.txt](../requirements.txt) 是继承的旧版通用清单，保留给旧入口；v3 环境按本目录及对应模块说明配置。

## 路径与复现

- 从仓库根目录调用安装/锁定工具时，清单路径使用 `requirements/requirements-*.in` 或 `.lock`。
- 清单内的 `-r requirements-*.in` 相对于包含它的文件解析，整组迁移后继续有效，保持原内容和依赖版本。
- 锁文件头部的生成命令与 `via` 注释保留生成时记录。旧报告中的根目录路径也属于历史记录；按当前目录复现时补上 `requirements/` 前缀，解释器仍使用实际目标环境。
- 依赖清单保留原文件名，目录变化不重算历史安装/科学验收结果。来源登记的当前使用路径与核验脚本同步维护。

例如，已有对应环境需要按 v3 开发锁安装时，从仓库根目录使用：

```powershell
uv pip install --python .venv/v3-dev/Scripts/python.exe --require-hashes -r requirements/requirements-v3-dev.lock
```

这是依赖安装命令，日常运行无需重复执行。首次建立环境、更新版本或重新生成锁文件须按实际任务处理；本目录整理不安装、升级或合并任何依赖。

## 维护方式

变更依赖时更新对应声明、锁和来源证据，使用原目标解释器及编译选项验证。生成文件不手工删减 hash，增量锁不冒充完整环境锁。新的依赖清单放在本目录，避免继续散落到工程根目录；第三方组件自带的 requirements 文件留在所属组件。

这些小型文本文件应进入版本控制。`.venv/`、下载缓存、构建展开和安装包按[仓库内容规则](../docs/development/repository-hygiene.md)管理。
