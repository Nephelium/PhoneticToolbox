# P02 工程骨架验收报告

2026-09-09 · **verified（Windows 独立环境、包边界与契约范围）**。

井井试用 P01 后反馈“目前没问题，可以继续”。依 ADR-013 收口 P01 Windows 原型与来源风险评审，并执行 P02。正式开发入口见 [开发说明](../development.md)，任务范围见 [P02 计划](../plans/2026-09-09-p02-scaffold.md)。

## 实际交付

- `.venv/v3-dev`：独立 CPython 3.11.14；31 个第三方 Python 包的固定版本与安装 hash；独立 npm lock 含 100 条传递/可选平台项，本机实际安装 75 项。
- 三个可构建/安装的包：phonetic-core、ptb-api、ptb-desktop，版本均为 3.0.0a1。核心暂只暴露发行元数据，未迁移任何科研算法；不能把包可安装视为科研计算通过。
- 正式 Vue/TypeScript 前端入口，目前明确显示工程空态，版本为 3.0.0-alpha.1；导航、播放器与完整主题属于 P04。P01 音频原型保留原样。
- API 1.0.0 的单一生成链：后端 Pydantic → OpenAPI/JSON Schema → TypeScript。采样帧、半开选区、真实秒数组与缺失原因可验证；前端不另手写一套模型。
- 本地与 server 开发模式共用 API，仅提供 health / capabilities；实际能力列表为空。local 的随机会话凭据通过 stdin 交付，URL 不含 token；Host、Origin 与 Authorization 校验已测。仅绑定 127.0.0.1；尚不是可部署的多用户服务器。
- 架构与资源检查、版本同步、文档检查、依赖审计、独立 wheel 安装验证脚本已建立。

## 实际命令与结果

以下 python 为 `.venv/v3-dev/Scripts/python.exe`，工作目录为项目根目录。

| 命令或验收 | 结果 |
| --- | --- |
| `python -m pytest -c tests/pytest.ini tests/contracts tests/architecture backend/tests desktop/tests -q` | 开发环境 25 passed；干净 wheel 环境同样 25 passed。涵盖选区 EOF/空选区/越界/类型、轨迹缺失/NaN/单调/长度、协议一致、禁止导入、资源登记及真实双实例 HTTP/退出 |
| `npm --prefix frontend run test` | 2 passed，半开帧区间换算与缺失原因 JSON 往返 |
| `npm --prefix frontend run typecheck` / `run build` | 均通过；Vue 3.5.42、TypeScript 5.9.3、Vite 8.2.2 |
| `npm --prefix frontend ci --ignore-scripts` | 从 lock 重装成功，随后契约检查、typecheck、test、build 再次通过 |
| `python scripts/generate_contracts.py --check` / `npm --prefix frontend run contracts:check` | Python 和 TypeScript 生成快照均无漂移 |
| `python scripts/sync_versions.py --check` | 三包、前端、API 版本来源一致 |
| `python scripts/check_architecture.py` | 正式源码无禁止依赖/旧路径/未登记资源；故意越界例子能触发失败 |
| `python scripts/verify_p02.py` | 构建三个 wheel，在新环境使用 `--require-hashes` 安装锁定依赖；逐包版本与开发环境一致，`uv pip check` 通过 |
| 核心独立安装 | 先只安装核心 wheel，从非项目目录以 `-I` 导入；确认该环境中不存在 FastAPI 或 Qt，版本正确 |
| 干净环境本地／server 入口 | 两种 mode 健康握手正确、退出码 0；本地双实例不同端口/凭据，关闭一个不影响另一个；错误凭据、Host、Origin 返回 403 |
| Qt 导入 | 独立 wheel 环境可导入 PyQt6 与 QtWebEngineCore；这是导入证据，不是重做全套 P01 音频或发行验收 |
| `npm --prefix frontend audit --json` | 修复后 0 vulnerabilities；这仅表示此次 npm 审计结果，不等于全面安全审查 |
| 文档与来源检查 | 现行文档链接/编码、结构化文件语法、282 个来源 ID 和 32 个任务通过；历史快照的失效引用另列，见下文 |
| v2 保持 | 427 个基线文件无变化；HEAD、index、status、安装依赖元数据摘要与本轮开始时相同 |

完整证据：`output/validation/p02/20260909-162915/`；摘要为 `output/validation/p02/latest.json`。干净环境 `.venv/p02-clean-20260909-162915`；三个 wheel 与 SHA-256 在该次证据目录和摘要中。测试用 wheel 不是完整软件安装包。

## 发现并处理的问题

1. **v2 声明与实际环境不同。** NumPy 声明 `<2`、实际 2.2.6；OpenCV 声明 `<4.12`、实际 4.13.0.92。逐项比较保存在 [P02 依赖清单](../../third_party/p02-dependency-inventory.json)。原环境未改，P03/P08 再按科研基线决定迁移依赖。
2. **契约生成工具传递依赖告警。** Redocly 1.34.19 精确依赖 js-yaml 4.3.1，触发 GHSA-2883-xcg3-v3hh。按官方公告仅覆写至 4.3.2，重新生成/检查契约并通过 npm 审计；没有自动升级其他主依赖。[上游公告](https://github.com/advisories/GHSA-2883-xcg3-v3hh)。
3. **开发安装的重复元数据。** setuptools editable 包在构建后会同时暴露源码 egg-info 与安装 dist-info，导致第一次按元数据行数比较失败。现在先拒绝同名多版本，再逐包比较版本；干净安装和开发环境实际版本一致。失败记录保存在先前时间戳目录。
4. **Node 测试与浏览器代码边界。** 初次架构检查发现 src 下的测试导入 node:test；测试已移至 frontend/tests，浏览器正式代码仍禁止 Node/进程依赖，未添加跳过检查的标记。

## 保留的限制

- Starlette 测试客户端对 httpx 和 AnyIO 的旧接口给出两条弃用告警；当前测试与真实 HTTP 均通过，未屏蔽告警。后续维护测试依赖时处理。
- 三份原样历史文档/上游摘录含 36 个失效相对链接，包括继承的乱码文件名。其原始字节 SHA-256 与仅归一化 CRLF/LF 的文本 hash 固定在 [历史文档登记](../baseline/archived-documents.json)，检查器报告原问题而不改写来源快照；不能称全部历史文档链接已修复。
- Python AST/前端导入扫描是架构检查，不能证明任意动态代码安全。JSON Schema/TypeScript 也不能替代 Pydantic 跨字段校验或 P03 科研验证。
- P02 没有正式窗口的视觉验收、算法迁移、账号、配额、任务队列、数据库迁移或服务器部署；没有新建完整 EXE。
- Qt/Chromium 完整原生许可组合、历史材料再分发、干净 Windows 机器和 Mac/Linux 原生/设备仍是对应后续门槛。

下一项为 **P03 科研行为与数据基线**：使用已经登记的本机只读样例，保存输入 hash、参数、真实后端、时间网格与缺失 mask，形成独立于 v3 的比较依据。P04 可在 P02 基础上建立正式公共界面，但不应在 P03 前迁移科研算法。
