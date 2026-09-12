# M01 最终逐项审阅

2026-09-11，M01 verified，范围为现有有界 Windows 桌面与 Windows 服务器/Chrome 工作流。P08 全体仍 in_progress。使用既有独立冻结基准及已落盘的真实操作报告，本轮补430项定向回归，不重复数据库迁移。

## 39项逐条核对

| ID | 核对内容 | 实现与通过证据 |
| --- | --- | --- |
| A01 | 目录、空目录、刷新、取消 | FileProvider 与 M01状态；desktop/tests/test_m01_files.py，m01-workspace-report |
| A02 | 输出授权和原文件保护 | TaskBridge.save；m01-persistent-report 本机幂等/重名/失败回收 |
| A03 | 勾选和全列表分析范围独立 | ParameterEstimationPage.startBatch；m01-layout-report，m01-persistent-report 17项 |
| A04 | 自动/手动TextGrid和重名 | M01 state.matchAssociation；m01-state.test.ts，test_m01_preview.py |
| A05 | 层切换、空层、区间试听 | M01页面与parse_textgrid；m01-workspace-report |
| A06 | 切分保留原样本、跳过空标签 | segment_child；test_m01_segments.py，m01-persistent-report |
| A07 | 文件名和非法区间、重名 | segmentation/TaskBridge；test_m01_segments.py，m01-report |
| A08 | PKL转安全JSON与关联 | legacy_pickle/TaskBridge.convert_lip；m01-legacy-report 15步Qt |
| A09 | 唇形时间、偏移、乱序和重复 | acoustic/lip；test_acoustic_associations.py，test_m01_legacy_inputs.py |
| A10 | 四项唇形、平滑 | M01-A独立捕获，M01-B科学对照，m01-legacy-report 实际转换与分析 |
| A11 | 80项选择、确认取消 | ParameterDrawer；m01-workspace-report 与 m01-state.test.ts |
| A12 | GUI全选和未筛选服务有别 | m01-baseline-report GUI-ALL，m01-persistent-report 冻结全列比较 |
| A13 | 单项、Energy映射、标注保留 | catalog/service；test_parameter_estimation.py，test_m01_result.py |
| A14 | 参数说明与来源 | MethodReferences；m01-workspace-report 与统一来源生成检查 |
| A15 | 10项常用设置 | SettingsDrawer，M01-A非默认捕获，test_acoustic_config.py 与契约回归 |
| A16 | 4项REAPER设置与交叉约束 | 同A15；M01-B原生参数/WM实际后端分别记录 |
| A17 | 草稿应用/关闭与任务隔离 | M01 store，test_m01_batch_policy.py，m01-workspace-report，m01-report 换账号 |
| A18 | 真实采样时间/缩放 | WaveformViewport；m01-layout-report，m01-display.test.ts |
| A19 | 试听与停止/切文件 | 公共audio状态；m01-workspace-report、m01-layout-report |
| A20 | 17项批次/中间失败继续 | AcousticBatches；test_m01_batch_policy.py，m01-persistent-report |
| A21 | 取消/所属子进程回收 | acoustic_executor/native；m01-persistent-report，test_m01_segments.py |
| A22 | 双格式完整发布/回读 | parameter_exports；test_m01_io.py，m01-report真实Chrome10份下载 |
| A23 | 部分成功、失败与单项重试 | summarize/AcousticBatches.retry；m01-persistent-report |
| A24 | 真REAPER/缺失/取消/预算 | native/reaper；test_m01_io.py，m01-core-report、m01-io-report |
| A25 | IRAPT与Praat回退 | acoustic后端；test_acoustic_backends.py，M01-A/B冻结对照 |
| A26 | 科学基准、mask、真实采样率 | test_parameter_estimation.py，test_acoustic_audio.py，m01-report四份自然录音 |
| A27 | 配额/截止/输入删除 | P07 FilePipeline；m01-persistent-report 真实PG及故障注入，额度上限沿用P07 |
| A28 | 租约/旧worker/重启 | M01-F2 PG/SQLite记录与test_m01_segments.py所属进程测试 |
| A29 | 本机无登录、网页owner隔离 | m01-persistent-report，m01-legacy-report换账号及越权拒绝 |
| A30 | 浅深/空态/窄窗/IPA | m01-report、m01-legacy-report真实Qt/Chrome |
| A31 | 默认单轨/可双轨/试听声道 | m01-workspace-report实际双声道 |
| A32 | 紧凑行/全选/取消 | m01-workspace-report、m01-layout-report 17项列表 |
| A33 | 真Praat语谱图与缩放 | test_m01_spectrogram.py，m01-workspace-report |
| A34 | 长音频预览与科学预算区分 | m01-workspace-report 600秒；m01-report超采样错误 |
| A35 | 三列独立/共同滚动 | WorkbenchColumns；m01-layout-report 多高度验证 |
| A36 | 时间刻度/锚定缩放 | m01-layout-report 指针0.6秒不漂移 |
| A37 | 进度定位/暂停/继续 | AudioTransport；m01-layout-report实际拖动 |
| A38 | 目录栏/Ctrl+A | m01-layout-report Qt和17项列表 |
| A39 | 同源与历史参数同步切分 | m01-persistent-report；m01-legacy-report原v2写入器生成旧表并逐值回读 |

引用的报告均在本目录。旧格式真实证据已再次读取：`output/validation/m01/task-window-9998fa477dce4ae09db763af7667c05e/report.json` success/normal_close为true，15步全部完成，4份历史切分表逐值一致，原输入不变。`persistent-7af298afbbbc4112b2117bcd1474829e/report.json` web_verified/postgres_stopped为true，schema_applied为空。

## 本轮回归

`.venv/m01-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini backend/tests desktop/tests tests/contracts tests/security packages/phonetic_core/tests tests/parity/test_parameter_estimation.py tests/parity/test_m01_conversion.py tests/parity/test_m01_capture_contract.py tests/parity/test_baseline.py -q`：430 passed，119.26秒，2条既有Starlette/AnyIO弃用提示。

首轮直接纳入整个parity目录时，M01旧wheel环境缺少新M10包而出现3个收集错误。随后按M01明确范围运行，未修改测试或跳过M01用例。此轮430项证明原已安装M01包回归，M02接入后的当前wheel和宿主另做联合验证。

同日收尾：新 m09-ui 环境安装当前 core/API/desktop wheel，包含 M01/M02/M09 及既有 M10 的完整定向目录回归为472 passed，见 [联合报告](m02-m09-report.md)。

限制：旧PKL支持明确结构，历史参数表支持有界XLSX/SQLite。唇形测试为合成时间对照，未新增自然唇形采集。网页额度故障注入不代表真实填满5GB或生产十人负载。现有单次分析200万采样值、240秒等预算保留。跨平台及完整发行保持后续任务。
