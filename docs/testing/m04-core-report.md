# M04-B LPC 纯核心验收

2026-09-13。**verified，限定 Windows 现有 Conda/MKL 科学环境及独立核心 wheel 安装目录。** 完整 M04 仍为 in_progress；任务、PNG导出与LPC页面尚未接入。A阶段已有PNG基准不代表V3导出已完成。

本轮先提交原样数值迁移 `38799f6`，随后单独增加公开入口验证和预算。`_legacy.py` 除模块说明外与原V2计算文件文本一致，未更换自相关方法、线程数或依赖构建。源码和已安装wheel两路的73项定向测试均通过。

## 核心行为与证据

| 项目 | 本次证据 |
| --- | --- |
| A03 WAV样本规则 | 5种冻结转换输入/输出逐字节一致，保留int16/int32/uint8缩放及多声道平均 |
| A05 标签/层级 | 6组标签、1组循环层级规则一致，包含IPA、跨区末尾排除和去重 |
| A06/A07/A12/A13 谱线 | 12个V2冻结场景，8个成功结果的频率/dB数组、动态纵轴精确一致，4个错误继续拒绝；无容差放宽 |
| A06/A11/A13 输入边界 | 49项新增检查，参数、类型、非有限值、过短、静音、溢出、超大整数、无效ROI、超限、样本切片与源数组不修改 |
| A14 预算与取消 | 10个独立进程探测；公开入口分析前/后协作取消通过，超限在自相关前拒绝。强制中断正在执行的自相关由C阶段进程适配实现 |
| 源码与依赖 | 7份V2来源文件SHA256未变；NumPy2.2.6/SciPy1.16.3，沿用 `third_party/m03-runtime-lock.json` 的兼容构建 |
| 安装包 | wheel安装到新的 `output/validation/m04/installed-core-final`，核对实际import路径、NOTICE和原样数值文件；未导入Qt/FastAPI/Parselmouth |

首轮新边界测试在入口尚未实现时收集失败，随后实现通过。补充超大整数回归曾暴露 `np.isfinite` 的TypeError，改为有界实数检查后通过。没有删除失败用例。原版Inf运算警告由公共入口提前拒绝，成功输入数值仍逐字节一致。

## 实测预算

独立子进程逐一运行，15秒为包括解释器启动的探测截止。超时只终止并等待该脚本持有的子进程。下面耗时为完成场景的函数内部测量，机器负载会影响结果。

| 样本数 | 阶数50 | 阶数200 |
| --- | --- | --- |
| 8,000 | 0.006 s | 0.007 s |
| 24,000 | 2.821 s | 2.743 s |
| 48,000 | 8.237 s | 8.088 s |
| 96,000 | 15 s进程超时 | 15 s进程超时 |
| 192,000 | 15 s进程超时 | 15 s进程超时 |

完成场景峰值工作集约87–89 MB，峰值提交量约841–843 MB，包含科学运行库开销。最终安装wheel用公开入口计算48,000样本/200阶耗时7.232秒，1024点有限，源输入哈希不变。

依ADR-047，单次分析上限为48,000样本，48kHz时1秒、16kHz时3秒。上限针对ROI，不能用来截断整份文件。公开入口支持1–768000 Hz整数采样率作为防异常数值的输入范围，本轮冻结对照只证明其覆盖的8–96kHz采样场景，不宣称所有设备采样率均已验证。C阶段拟实施30秒/2GB计算子进程预算，并验证超时/取消/失败回收；这些运行时保障当前为planned。完整音频解码内存与预览限制也由C明确。

原始报告保存在 `output/validation/m04/budget-source.json`、`wheel-verification-final.json`，最终wheel位于 `output/validation/m04/wheel-final/`。独立的是核心包安装目录，科学依赖仍使用现有项目兼容环境，未建立全新完整Conda环境。未替换现有已安装核心或污染V2环境。

## 实际命令

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'packages/phonetic_core/src')
& scripts/Invoke-M03-Python.ps1 -X utf8 -m pytest -c tests/pytest.ini tests/parity/test_lpc_spectrum.py tests/parity/test_lpc_boundaries.py -q
& scripts/Invoke-M03-Python.ps1 -X utf8 scripts/probe_m04_budget.py --output output/validation/m04/budget-source.json
& scripts/Invoke-M03-Python.ps1 -X utf8 -m build --wheel --no-isolation --outdir output/validation/m04/wheel-final packages/phonetic_core
uv pip install --python .venv/m03-compatible/python.exe --target output/validation/m04/installed-core-final --no-deps output/validation/m04/wheel-final/phonetic_core-3.0.0a1-py3-none-any.whl
$env:PYTHONPATH = (Join-Path (Get-Location) 'output/validation/m04/installed-core-final')
& scripts/Invoke-M03-Python.ps1 -X utf8 -m pytest -c tests/pytest.ini tests/parity/test_lpc_spectrum.py tests/parity/test_lpc_boundaries.py -q
& scripts/Invoke-M03-Python.ps1 -X utf8 scripts/verify_m04_core.py --installed-root output/validation/m04/installed-core-final --output output/validation/m04/wheel-verification-final.json
npm --prefix frontend run ui-data
npm --prefix frontend run ui-data:check
npm --prefix frontend run typecheck
npm --prefix frontend run build
& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/validate_docs.py
& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/check_architecture.py
```

核心两路73项通过，公共致谢数据333条、类型检查与构建通过；文档与架构检查无新增错误，既有历史快照坏链仍单列。仅更新公共引用数据，没有LPC页面视觉验收，也未运行无关模块全量科学测试。

## 引用与下一步

按井井追加要求，Makhoul(1975)已进入统一学术登记及BibTeX，NumPy/SciPy实际依赖另列。更早代码来源经本地源码/历史与公开精确组合检索未找到，保留unknown，不阻挡功能。方法与单位增益边界见 [方法说明](../references/m04-method-audit.md)。未打包论文、联系作者或声称许可闭合。

下一项M04-C：接既有持久任务与资产协议，实施硬预算和回收、真实300DPI/字体快照PNG，再接D页面。未执行DDL、push、EXE打包或M05；既有5份EXE探针草稿未修改或纳入提交。
