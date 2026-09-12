# M03-E3 导出字体预检

2026-09-12，verified，限定 Windows 开发态 EGG 导出字体预检及相关回归。完整 M03 和全平台 P04-FONT 仍 in_progress。实现范围见[计划](../plans/2026-09-12-m03-font-preflight.md)和 ADR-043。

## 行为与边界

- 单文件 CSV/三图和带图批次先冻结本次字体，再通过实际 EGG 兼容进程检查中文、英文/数字、固定 Doulos SIL。缺失显示角色和名称，检查失败不创建持久任务；批次保留选择、参数和原批次记录。
- 交互分析、纯 CSV 批次、IF 不依赖后台图中文字。科学计算前仍严格复核需要渲染的字体，防止预检后失效；重试保持旧快照，改字体需重新提交。
- 复用 jobs 本机令牌/Origin 和服务器会话/CSRF/账号边界及 16 KB 请求上限。检查只收有界字体快照，结果不含路径，不上传字体文件，不建立新数据库表或音频资产。
- 同一兼容 Python 和固定 bootstrap，单槽，启动握手后 Windows Job 限制 1 GB，处理超时 20 秒（另有启动/回收等待），父进程退出随 Job 回收。运行时缺失、繁忙、超时均明确失败，不修改系统字体或自动安装。桌面仅此请求等待上限 35 秒。
- 只调整后台渲染验证顺序与 UI 提交流程，未修改科学核心、数值规则、v2、已有语料或旧发行物。

## 实际验证

环境：API/Qt `.venv/m09-ui/Scripts/python.exe`；科学子进程 `.venv/m03-compatible/python.exe`，通过 `scripts/Invoke-M03-Python.ps1` 注入既定 MKL 路径。无新依赖。

| 命令 | 结果与证据 |
| --- | --- |
| `python -m pytest -o addopts='' backend/tests/test_m03_font_preflight.py backend/tests/test_m03_contract.py -q`，API 环境 | 20 passed。真实字体/两个角色同时缺失、忙槽、强制超时后再检查、缺失运行时恢复；本机鉴权、路径/IPA替换/字号拒绝；服务器会话/CSRF/账号切换及16KB拒绝。服务器账号查找使用内存 fixture，字体子进程是真实运行，未操作PG。 |
| `python -m pytest -o addopts='' backend/tests/test_fonts.py backend/tests/test_m03_exports.py backend/tests/test_m03_preview.py backend/tests/test_m03_ranges.py backend/tests/test_m03_export_names.py -q`，科学环境 | 原62项通过；追加“缺字体在分析前失败”后字体4项通过，共63个不同用例。已有CSV、数组/预览、PNG与导出命名回归保持通过。 |
| `npm --prefix frontend run test` | 54 passed，含纯数据跳过、缺失提示、快照不随偏好变动、预检不可用阻止导图。 |
| `npm --prefix frontend run typecheck` / `build` / `contracts:check` | 通过；共享 OpenAPI/TypeScript 同步。 |
| `python scripts/generate_contracts.py --check` / `validate_docs.py` / `check_architecture.py`，`git diff --check` | 通过；562文件、330来源、32任务，新增文档链接及架构检查无错误。未改历史快照中的既有缺失链接。 |
| `node tests/e2e/m03-fonts.cjs` | 独立 Chrome 和真实本机 API/兼容进程，3组交互检查通过。`output/validation/m03-ui/chrome-f9293a86531f4606bb15cf48dc8f175b/report.json`。注入仅存在于测试页面的缺失字体请求，真实后台给出缺失结果；批次选择保留，纯CSV成功；改回 Times New Roman 后真实三PNG成功，回读元数据的实际字体家族/哈希与固定IPA一致。 |
| `python -X utf8 scripts/verify_m03_qt.py`，API/Qt 环境并含desktop源码路径 | 9组实际Qt操作通过：单声道拒绝、四图、选区失效、三图保存、IF四图/双WAV保存、混合批次、批次保存、深色窄窗、关闭重开。`output/validation/m03-ui/qt-8b3293da239442d5b1cec84e711ee0c4/report.json`，未执行DDL。 |

Chrome最初的字体快照检查读取了折叠面板的不可见 `innerText`，断言失败。只读任务快照已确认字体正确；修正为展开参数快照后读取并额外回读实际渲染元数据，复验通过。缺失提示截图已人工查看。测试代码最初引用错误的 hash_token 模块，已改为现有 account_store 并通过。首次 pytest 未覆盖继承 v2 的 coverage 参数，因隔离环境没有该插件未启动；随后用项目既有定向命令 `-o addopts=''`，不安装或修改全局依赖。

已有两条 FastAPI/TestClient 弃用警告及 Qt 的 PNG 配置警告保留，未屏蔽。来源复用已登记 Doulos SIL 与 Matplotlib，无新字体、外部代码或来源条目。

## 剩余工作

本轮不代表生产服务器或两账号同时导出全链路、用户卸载字体的系统实测、多屏DPI/实际声卡、冻结EXE或其他平台通过。全局字体验收矩阵仍按原范围保留。下一项 M03 剩余连续交互与来源、F 候选范围审阅，不推进 M04。未 push、发布或部署。
