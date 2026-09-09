# M01-A 独立基准补齐与科学环境审计

2026-09-09。**M01-A verified，限定Windows旧实现、合成样例与控件逻辑基准**；M01/P08 为 in_progress，v3声学算法、原生输出适配和完整研究页尚未迁移。

## 完成内容和实际结果

已新增独立捕获调度器、原环境worker、合成用例定义、冻结清单与回归测试。原算法仅在原 `phonetic_311` 解释器内以 `-B -X utf8 -X faulthandler` 运行；原模块和原生程序未修改。子进程PATH补其原DLL目录，TEMP/TMP及最终GUI测试的Matplotlib缓存指向本次输出目录。

最终冻结 **28例，每例2次独立进程，共56份选定捕获结果**。27例来自 `output/validation/m01/20260909-215849-657259`，REAPER故障例在修正注入器后单独重跑两次；联合索引位于 `output/validation/m01/20260909-220406-224694/capture-summary.json`，没有伪称全部来自最后一个目录。公开黄金文件共2,003,548字节，仅含合成数据、软件版本和来源哈希，不包含研究语料、私人路径或访问凭据。

[公开冻结清单](../../tests/fixtures/m01/manifest.json) · [环境审计数据](m01-environment-audit.json) · [回归测试](../../tests/parity/test_m01_capture_contract.py) · [执行计划](../plans/2026-09-09-m01-implementation.md)

| 覆盖项 | 实际验证 |
| --- | --- |
| 服务默认、GUI全选 | 分别79列、77列，差异为 `SOE_pF0/SOE_rF0`；原服务空选择列表同样不筛选 |
| pF0/rF0/Intensity/Energy | 单项输出保留 `Time_s` 与目标列；Energy兼容为Intensity；数值与服务未筛选结果一致 |
| 原Qt参数控件 | 离屏实例化真实控件，验证80项全选/全不选；调用原事件处理函数验证空选拒绝、取消不覆盖。仅模态完成和消息弹窗用受控替身；不声称真实桌面点击或视觉验收 |
| 14设置 | 对10常用＋4 REAPER分组控件检查默认、上下界与非默认保存；每项独立捕获一次修改配置并双轮重复。13项还比较实际计算函数实参；only_voiced核对配置及原服务实际执行的读取行。没有声称每个设置都必然改变此合成音的数值 |
| 四项唇形 | 音频元数据锚点、伴随timestamp锚点、旧相对时间回零三路径；稳定排序、重复时间首项、手动offset只加一次、NaN源点插值、长度错误和不足时间点均验证 |
| TextGrid关联 | 合成中文/IPA两层均保留，标签按帧时间对齐；GUI80键＋4唇形＋2标签层时共83列 |
| 原生与回退 | 正常REAPER实际退出0；故障注入使同一二进制实际退出1，旧Python回退产生92个有限正F0；IRAPT受控抛错后旧Praat回退产生61个有效F0，实际调用已记录 |
| 导出 | 普通中文/IPA标签的XLSX和新建SQLite回读均通过，列顺序、缺失mask、标签精确，有限数值rtol=atol=1e-12；`params`表与 `idx_params_time` 索引存在 |
| 保护检查 | 修改时间、mask、后端状态或有意义的数值会使比较失败；冻结器拒绝不全案例、重复案例及覆盖已有基准 |

两轮科学字段使用既有P03比较器：有限值 `max(1e-10,1e-7*scale)`，时间/配置/形状/mask等精确；没有放宽容差。新增捕获未覆盖P03黄金文件。四项唇形采用解析线性值与二进制可精确表示的时间分数，期望数组独立计算，不调用待迁移的v3生成expected。

## 新确认的旧行为

1. **Excel公式解释：** 合成标签 `=literal` 在80个单元格中被旧Excel writer写成公式，按值回读不再等于原标签；同批SQLite文本保持一致。该测试预期记录这种差异，而非把错误回读称为等价。M01-C需单独修正文本导出策略并保留变更证据。
2. **空结果导出不是完整成功：** 空WAV的旧分析返回0行；XLSX已经生成，后续SQLite导出抛ValueError，留下前一文件。M01-C/F按每文件两产物原子发布处理，本轮未修改旧服务。
3. **唇形NaN可被插值跨过：** 与声学轨迹保留NaN缺口的平滑规则不同；现有行为已冻结，不能为了代码统一静默改变。
4. 设置文案/实际使用范围、采样率元数据旧问题继续沿用 [源码差异表](../modules/evidence/M01-source-map.md)；本轮不改变默认值、F0算法或科研意义。

## 捕获过程中的修正与证据边界

- 首次导出试验使用公式样标签，暴露了真实回读差异；随后拆成普通标签回读与独立公式案例，两者都保留验证。
- 初次重复比较捕获到随机临时路径，不属于科学变量；仅将观测日志中的绝对路径规范为 `<path>`，未改变数组、配置、返回码或mask。
- 原Qt整数SpinBox不能接float，修正的是测试驱动的控件赋值类型，原控件/默认值/范围未变。
- 早期REAPER输出路径故障注入造成**该测试二进制**异常退出，未作为最终正常失败基准；后续追加未知参数会被带完整参数的旧二进制忽略，所以不能算作触发回退。最终以无有效输入参数的受控调用取得实际退出1，再由原函数对已转换合成WAV执行Python回退。独立重跑该例；其余27例的输入、配置、旧源码和双轮结果重新校验后复用。两种recorder哈希随各结果保留，仅故障注入分支有差异。
- 没有操作Codex内置浏览器或复现其关闭触发路径。Qt测试是本项目拥有的离屏独立进程，没有声称修复Codex内部缺陷。

## 科学环境候选

原环境10个必要发行包版本与P03记录一致；候选不是已经安装的v3依赖锁。本轮没有下载、安装、升级包或修改系统环境。

| 层次 | 已核对版本 |
| --- | --- |
| 数值/分析 | NumPy 2.2.6；SciPy 1.16.3；praat-parselmouth 0.4.7 |
| 表格/导出 | pandas 2.3.3；openpyxl 3.1.5 |
| 当前必需传递依赖 | python-dateutil 2.9.0.post0；six 1.17.0；pytz 2025.2；tzdata 2025.2；et-xmlfile 2.0.0 |

xlsxwriter未安装，当前旧导出使用openpyxl，不为迁移额外引入另一writer。版本、有效requires、原包项目链接及许可元数据哈希见审计JSON；许可元数据不等于完整原生传递许可审查。

在项目 `.venv`、pip wheel缓存、uv wheels-v5缓存内定向检查：仅发现两个同名 `numpy-2.2.6-cp311-cp311-win_amd64.whl`，**均为0字节且不是ZIP**，不可复用。未遍历整个Users/C盘，未清理这些已有文件。没有由此推断所有其他缓存目录都没有包，也没有将文件名当有效wheel证据。

M01-B须在项目隔离环境准备上述确切版本的可校验发行物，审查hash/依赖闭包与包资源；不修改原 `phonetic_311`，不把Windows数值结果外推到其他平台。

## 实际命令与检查

```powershell
# 全量独立捕获（不安装、不更新已有golden）
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/capture_m01_baseline.py
# 修正单一故障注入后，明确复用未变案例
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/capture_m01_baseline.py --case REAPER-FAILURE --reuse-from 'output/validation/m01/20260909-215849-657259'
# 审阅通过后首次冻结；已有目录会拒绝重复冻结
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/capture_m01_baseline.py --freeze-from 'output/validation/m01/20260909-220406-224694'
& '.venv/v3-dev/Scripts/python.exe' -X utf8 -m pytest -c tests/pytest.ini tests/parity/test_baseline.py tests/parity/test_m01_capture_contract.py -q
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/check_architecture.py
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/validate_docs.py
git diff --check
```

基线与冻结保护测试 **23 passed**；架构检查错误0，文档检查216文件/300来源/32任务、错误0；36个未改历史链接问题单列。架构、文档检查和v2保存性最终结果写入 `output/validation/m01/a-final-checks.json`；前后context为 `context-before-a.json/context-after-a.json`。检查覆盖427基线文件、HEAD、暂存区、Git状态、原环境位置/包元数据，以及既有P03黄金哈希。新建SQLite仅为本次合成结果导出，未操作现存或服务数据库。

仍未验证：真实持续元音/唇形配套数据的专门确认、全部GUI视觉与真实设备操作、所有14设置边界值的完整科学行为、任意历史PKL的安全转换、受控原生文件输出、v3算法parity、Windows完整发行、Mac/Linux及生产负载。下一任务为 **M01-B 科学核心与数值对照**；M01-C原生文件门仍保留。
