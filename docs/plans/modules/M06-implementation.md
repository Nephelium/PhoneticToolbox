# M06 实施记录与验收计划

2026-09-27，井井明确授权核心、页面、正式接线与定向验证。状态 in_progress：Windows 开发功能完成，Linux 精确数值/任务资格失败并保持关闭。实际命令与完整结果见 [验收报告](../../testing/m06-report.md)。

## 范围和顺序

1. 相邻 V2 的 widget、klatt 六文件与说明书 4.1–4.5 只读捕获。独立基准脚本不 import V3；随机种子只在捕获进程固定，产品保留随机行为并记录实际种子。
2. 原样迁入 klatt 数值代码，移除设备播放、直接文件导出、演示 main 和 import 的 sys.exit。抽离 widget 的数组编排、参数曲线与提取，科学核心只接受配置/数组。
3. M06 专属契约、受限 child、任务 admission/executor。资源使用公共 writer/owner/project/租约/generation；不新增 schema、宿主、远程协议或依赖。
4. ModuleFrame/Toolbar/Section/Status、公共波形/播放/任务/关闭保护。23 参数和完整曲线，生成、合成、试听、导出分开。配置 revision 与异步 request generation 防迟到覆盖。
5. 接线前重读共享 diff。增量接 AppShell、main/jobs/store/executor、文件 manifest、平台 adapter、生成契约和政策操作表，保留 M08/M14。
6. Windows 独立基准与实际宿主、Linux 核心与任务资格门。短输入和拟支持上限分别测量。Linux 未过门保持关闭。

## 文件清单

- packages/phonetic_core/src/phonetic_core/synthesis/klatt/：数值、配置、曲线、输入验证、提取及来源许可。
- backend/src/ptb_api/m06_models.py；backend/src/ptb_worker/m06_{task,executor,child}.py。
- frontend/src/modules/speech-synthesis/{SpeechSynthesisPage.vue,state.ts,port.ts}；frontend/src/platform/m06.ts；desktop/src/ptb_desktop/m06_bridge.py。
- tests/support/m06_baseline.py、tests/fixtures/m06/、tests/parity/test_speech_synthesis.py、backend/tests/test_m06*.py、frontend/tests/m06.test.ts、tests/e2e/m06-host.cjs、scripts/verify_m06*.py。
- docs/modules/evidence/M06-source-map.md、docs/manual/speech-synthesis.md、docs/testing/m06-report.md 与现有计划/矩阵的 M06 行。

## 已发现差异与处理

- V2 辅音规则实际只提示已移除。保留说明入口，不虚构辅音算法。
- V2 解析器忽略非法 IPA。按用户要求新增位置明确的拒绝，合法序列的计算规则不变。
- V2 CSV 在 override 生效时不保存底层曲线，也不记录淡入淡出/平滑/采样率/静音区间。兼容读写旧四字段，同时提供版本化完整快照，避免声称旧 CSV 本身无损。
- V2 参数轨迹使用含终点 linspace；音频实际采样轴为 arange(N)/fs。均保留并区分。
- V2 默认输出 16 kHz，Klatt 内部 10 kHz；源音频加载后使用源采样率。Shimmer 内部为小数，显示为百分数。
- V2 说明书提示提取/复制合成效果有限，不能把数值迁移等价解释为感知或生理真实性。

## 实际候选命令（运行后才能登记通过）

```powershell
& .venv/m09-ui/Scripts/python.exe -X utf8 tests/support/m06_baseline.py
& .venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini tests/parity/test_speech_synthesis.py backend/tests/test_m06.py -q
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
node tests/e2e/m06-host.cjs
```

科学源路径通过显式 PYTHONPATH 指向本轮源码。Linux 复用既有 WSL 项目解释器，不安装依赖/修改系统。实际命令、构建版本、输入摘要、逐字段差异、峰值与未测项写验收报告。


## 正式资源协议收口

完整参数 CSV（现有 table 角色，2 MB）以资源 ID/hash 进入任务，音频为 audio 资源。DB 快照只记录这两类不可变引用及 action/seed，不存大曲线，不改 schema。窗口晚到保护覆盖原始输入文本编辑与异步解码。源音频与合成结果有独立预览，导出始终绑定实际合成快照。

Linux 专属环境为用户追加明确授权。最终 21 passed / 7 failed，全部逐字段记录，未扩大数值容差。公共 systemd 资源边界不可用。Windows 有界 10秒/480000样本实测、真实 Qt/Chrome/新 PG 验证已完成；更长核心配置不代表任务开放。
