# P16-REVIEW 逐项修复实施计划

> 执行方式：井井已明确授权逐条修复。秋叶在当前工作区串行实施、逐项验证并报告，不另行派发任务。现有未提交改动保留。

**Goal:** 修复 2026-10-01 审查确认的九项缺陷，并补齐可在当前设备实现的跨平台入口及资源选择边界。

**Architecture:** 共用科学核心与契约不复制。平台差异集中于 desktop/native adapters；科学行为修正单独记录为 acoustic/2，既有历史结果保持可读。未实测的 macOS 原生执行和完整发行保持关闭，不以平台模拟代替实机验收。

**Tech Stack:** Python 3.11 / NumPy / Parselmouth / FastAPI / PyQt6；Vue / TypeScript；现有项目虚拟环境。

## 顺序、文件与验收

每项均先加入针对真实缺陷的失败用例，执行确认后做最小修复，再复跑相关已有用例。源码不依赖 v2、不改现存数据库，不修改 CI/CD、系统依赖或生产配置，不生成或发布发行包。

| 项 | 预期行为 | 文件范围 | 定向验证 |
|---|---|---|---|
| 1 时间网格 | 每帧独立从真实时间换算采样点，误差不累积 | core/acoustic/common.py、energy.py；core tests | 44.1/22.05/48 kHz、5/非整数 ms、边界与能量解析结果 |
| 2 计算故障 | 计算异常不再伪装为成功的缺失值；正常无声保留 NaN | core/services/acoustic.py、ports/errors.py；worker errors、前端任务错误 | 注入各计算阶段异常、取消不被吞掉、无声正常结果 |
| 3 共振峰参数 | 3/4/5 对应实际 Burg 参数；旧槽位规则显式登记 | core/acoustic/formants_praat.py、科学元数据与说明 | 独立观测 Burg 调用参数、公开合成输入 |
| 4 REAPER 策略 | disabled/native_required/python_only/native_then_python 按约定执行 | worker/science_child.py、新后端选择适配；测试 | 原生缺失、失败、Python 回退、取消与后端元数据 |
| 5 MFA 导入 | 桥接按操作限制；64 MB 音频不被公共 8.1 M 字符上限误拒 | desktop/host.py、请求校验模块；desktop tests | 6.1 MB、角色限制、超限和未知操作 |
| 6 导出字体 | 屏幕偏好与可执行导出字体分开，实际解析字体可追溯 | worker/fonts.py、API 字体模型；frontend fonts/preferences | Windows 请求在缺字体主机的确定回退、全缺失报错、IPA 固定资源 |
| 7 能力准入 | capabilities 与资源档/部署允许表一致 | api/main.py、P15 host、能力测试 | 空允许表、server-small ZIP、科学白名单、旧测试断言范围 |
| 8 保存适配 | Windows 保持目录锁；POSIX 基于目录句柄完成保存 | desktop/task_bridge.py、annotation.py、平台保存适配 | 保存/重名/替换/符号链接/目录变化、Windows 回归，POSIX 实测若环境可用 |
| 9 快捷键 | 标注编辑接受 Ctrl 与 Command，防止输入区误触 | frontend annotation、共享快捷键与测试 | Ctrl/Meta 的复制剪切粘贴撤销保存及文本输入 |
| 10 跨平台准备 | 明确未知平台拒绝、用户目录跨平台、原生资源按 OS/架构选择 | native dispatcher、desktop paths、VTL resource loader、build preflight | 平台选择与资源清单单测；实际 Mac 原生/设备/签名待设备 |

## 执行命令

Python 统一使用 `.venv/m09-ui/Scripts/python.exe`，`PYTHONPATH` 指向本仓库 core/backend/desktop 的 src。pytest 使用 `-c tests/pytest.ini -p no:cacheprovider`，新用例文件名统一含 `review`，再运行受影响的原测试。

- 科学与协议：`python -m pytest packages/phonetic_core/tests tests/contracts backend/tests/test_m01_result.py backend/tests/test_m01_failure_codes.py -q`
- 桥接/字体/能力：`python -m pytest desktop/tests backend/tests/test_fonts.py backend/tests/test_p11_capabilities.py -q`（需要 GUI 的实际检查单独执行）。
- 前端：`npm --prefix frontend run typecheck`、`npm --prefix frontend test`、`npm --prefix frontend run contracts:check`、`npm --prefix frontend run build`。
- 契约变化由既有生成脚本更新快照与 TS，不手改生成文件。
- 结果记录：`docs/testing/2026-10-01-review-repairs-report.md`；逐项区分实现、Windows 实测、Linux/模拟、Mac 待验。

## 状态

本轮代码修复完成，限定 Windows 的定向验证完成。九项修复、额外接线问题和实际命令见[最终报告](../testing/2026-10-01-review-repairs-report.md)。平台选择、目录、资源清单与 POSIX 保存实现已补齐；Linux POSIX 实测、macOS 原生执行和完整发行仍为待验项，不将本轮结果扩大为完整三平台 verified。
