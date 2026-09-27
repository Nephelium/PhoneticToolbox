# M07 发声类型连续统实施

2026-09-27，in_progress。用户附件授权 A–E 连续推进。工作分支 codex/v3-rebuild，开始时已有其他模块与公共平台大量未提交修改，保留其内容。仅 M07 增量，不改 V2、现存库、全局环境、密钥、CI，不 push、部署或 EXE。

## 实施及文件边界

1. A：读取 V2 说明书14.1–14.4、模型、GUI、worker、service/core及测试。`tests/support/m07_baseline.py` 独立捕获原实现，两轮精确比较；公开输入和中间量存 `tests/fixtures/m07`，完整执行证据存忽略的 output。
2. B：`packages/phonetic_core/src/phonetic_core/manipulation/m07_{models,legacy,api}.py`。保留 LPC、残差与脉冲算法、1ms 网格、尾帧及幅度语义；F0 使用注入 port，核心无文件/Qt。验证与有限资源拒绝独立于迁入数值实现。
3. C：`backend/src/ptb_api/m07_models.py`、`ptb_worker/m07_{task,child,executor}.py`。复用正式 jobs、受管资产、配额与进程组。分析结果采用版本化有界 JSON 数组，禁止 pickle。分析/apply/生成都在受控进程执行。每组一个任务，全部六组共享批次快照身份，完整组原子发布并保留部分成果。
4. D：`frontend/src/modules/phonation-synthesis/`、`platform/m07.ts` 与 `desktop/src/ptb_desktop/m07_bridge.py`。统一 ModuleFrame/Toolbar、公共波形/试听、显式应用和关闭保护。应用前的编辑不能混入生成。角色/参数/分析/应用版本与请求代际校验。
5. E：Windows 原科学基准、受控进程/正式任务、本地HTTP、隔离测试PG、Chrome、Qt及已有WSL纯核心分别记录。Linux 服务和 remote/1 桥未接通，资源准入保持关闭，不更改系统。

公共接线最小增量：jobs 路由、JobView/manifest union、worker 分发/重试/发布、capability、AppShell 注册、Qt 文件桥和生成契约。修改前重新读现场内容。

## 科学与输出约定

- 实际 GUI 默认11025Hz、128/32采样点、20阶、9步、21控制点、Parselmouth、50–300Hz、1ms；默认当前类型 F0_ONLY。
- 两个 F0 阶段：原音频确定 LPC 区间，残差 F0 用于脉冲/连续统。编辑后只替换 F0，复用 LPC/残差/脉冲。
- 对齐模式控制编辑时间轴，原合成仍将目标有效 F0 重采样至源分析区间，保留此行为。
- energy_match 是残差周期绝对峰值匹配，normalize_to_source 是全段平均绝对振幅匹配，沿用 UI 名称但帮助解释实际计算，不宣称客观响度或知觉等距。
- 当前新增显式方向选择，补齐旧帮助承诺但旧 current 实际仅正向的差异。未应用控制点不自动应用，这是本轮明确要求。
- 初始准入：每输入≤10秒、≤480000帧、≤8MB；目标采样率固定11025Hz，2–50步。超限明确拒绝，不降低科学参数。预算在解析后复验，进程组1GB/120秒，输出及临时受P07额度约束。实测后记录可支持范围。
- 科学新生成成功时起最多3天，下载不续期；本地无TTL。公共政策代码已实现，目标旧库未迁移。M07不自行迁移或清理。

## 真实可用命令

```powershell
$env:PYTHONPATH='D:/PhoneticToolbox/PhoneticToolbox_v3/backend/src;D:/PhoneticToolbox/PhoneticToolbox_v3/packages/phonetic_core/src;D:/PhoneticToolbox/PhoneticToolbox_v3/desktop/src;D:/PhoneticToolbox/PhoneticToolbox_v3'
& '.venv/m09-ui/Scripts/python.exe' -X utf8 tests/support/m07_baseline.py
& '.venv/m09-ui/Scripts/python.exe' -m pytest -c tests/pytest.ini tests/parity/test_phonation_synthesis.py backend/tests/test_m07.py -q
& '.venv/v3-dev/Scripts/python.exe' scripts/generate_contracts.py
npm --prefix frontend run contracts
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
```

2026-09-27收口：A–E在现有依赖允许范围实施，Windows开发态闭环限定verified。实际源码44项、任务8项、新PG2项，浏览器/Qt/安装wheel/WSL/资源证据见 `docs/testing/m07-report.md`。Linux/远程准入仍关闭，完整模块保持in_progress。全部范围完成后停止，未执行项单列。
