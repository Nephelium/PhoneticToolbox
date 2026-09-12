# M03-A 独立行为基准与布局核实

2026-09-12。**M03-A verified，限定Windows原v2独立基准与环境审计。完整M03仍in_progress，v3核心、接口、页面和EXE均planned。** 井井已授权继续，并明确EGG尽量贴合v2布局、保留全部功能。

## 本轮产物

- `scripts/capture_m03_baseline.py`：原conda解释器、独立输出和临时目录、240秒子进程上限、输入/原源码哈希保护、双轮精确比较和拒绝覆盖的公开冻结。
- `scripts/m03_baseline_worker.py`：测试入口直接导入原v2，捕获配置、数值数组、dtype/shape/NaN/Inf、实际模块哈希、依赖版本、错误和取消。未复制实现生成expected。
- `tests/fixtures/m03/manifest.json`、9份公开合成JSON/NPZ与`tests/parity/test_m03_capture_contract.py`。NPZ以allow_pickle=False回读。两份自然录音及其完整结果只在忽略目录。
- [布局约束](../design/m03-v2-layout.md)：按原Qt实际控件建立四图、图下两行参数、最下方总览的关系；同步修正旧右侧设置方案。
- 井井随后强调统一风格，已重新核对UI_SPEC及实际v3组件，补齐具体复用与适配清单。约束明确涵盖页面内部的控件、图表和状态，第一版即统一，不只统一外壳。

完整证据：`output/validation/m03-baseline/20260912-141624-191903/`。每例含`1/2`两轮结果、数组、原环境清单和worker日志；源哈希清单及P03比较同目录。11例合计505个数组，其中公开421个，JSON另保留事件列表和配置。公开fixture约3.28MB。

## 实际验证与结果

| 命令/检查 | 结果 | 证明范围 |
| --- | --- | --- |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/capture_m03_baseline.py --freeze-public` | 11例×2轮，全部一致 | 原v2行为在当前Windows原环境可重复，包括预期错误；不代表v3算法已移植 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini tests/parity/test_m03_capture_contract.py -q` | 19 passed | 来源/哈希、数值解析规则、时间与mask、布局、实际CSV/PNG结果 |
| 原输入及源码前后SHA-256 | 11个输入、161份原v2 Python源码均未变 | 未改自然语料或原源码；六份关键EGG模块哈希还与P03基准一致 |
| 单文件/批处理原函数导出，Pillow实际解码 | 每轮各三PNG，尺寸、dpi、像素哈希一致 | 单文件1500×900/150dpi；批次1200×600/100dpi |
| 原EGGWidget离屏构造与渲染 | 中文可读，四图与下方参数/总览已视检 | 使用原控件及现有微软雅黑回退、浅色绘图主题；未通过加载按钮走完整应用流程，部分按钮仍为初始禁用态 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/validate_docs.py` | 无当前文档错误 | 旧归档未解决链接单列保留 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/check_architecture.py` | errors=[] | 测试专用旧导入边界未侵入产品 |
| `git checkout-index`导出到独立忽略目录后重算SHA-256 | 18份公开fixture哈希全部不变 | `.gitattributes`限定M03 JSON为LF，`.gitignore`仅放行EGG-SYN的NPZ，私有数组仍忽略 |

最初回归因尚无fixture而失败，捕获后通过。增加低F0解析检查时发现直接把IEEE浮点倒数写成精确整数50会产生7.1e-15差异；改用声明事件间隔的精确倒数作为期望，没有扩大容差或修改捕获数据。双轮比较始终为精确元数据和数组字节哈希，未使用近似容差。

## 样例与覆盖

| 样例 | 内容 | 结果 |
| --- | --- | --- |
| EGG-SYN-PCM16 | P03同一0.8秒/44100Hz双声道合成输入 | 8种GCI/GOI×自动/手动配置；4组ROI×raw/filtered；两F0、IF、GUI及实际导出 |
| EGG-SYN-FLOAT / SWAPPED | float32转换及声道反置后交换 | 正常返回，反置恢复后的两个归一化声道和处理EGG与PCM16精确一致 |
| EGG-SYN-SILENCE / SHORT | 双声道全零、仅8采样 | 静音无事件、Praat缺失；极短滤波警告返回未滤波输入，Praat失败分别记录 |
| EGG-SYN-MONO / MULTI / EMPTY / BROKEN | 单声道、三声道、空双声道、坏WAV | 原实现实际拒绝，错误类型/文本双轮一致 |
| EGG-01 / EGG-05 | 原授权私有录音，分别14.664376秒/22.831020秒，左EGG与反向配置 | 原P03输入哈希一致；实际事件、两F0、ROI与0.12秒IF输出双轮一致 |

以上合成输入只是可重复的解析波形，不宣称模拟真实生理EGG。私有IF仅捕获前0.12秒以限制原算法全相关计算成本，未宣称全录音IF通过。自然样例GCI分别1178/4700个；事件数量不能直接当生理正确性指标。

另外捕获：高/低通1/25/50Hz及越界参数行为；实际无效criterion_level/min_f0；CQ/SQ边界和多个峰；低F0/异常/重复GCI；F0变化启发式；5/20/50ms谱图数组与范围；IF自动/12阶、无GCI、超大阶数；批次两F0独立开关、静音mask、预取消、坏文件失败后继续。

## 核实与纠正

1. **默认值纠正：** EGGConfig为slope/slope，EGGWidget.init_ui把GOI改为scale。实际单文件和批次均slope/scale，与手册一致，上一轮遗漏的GUI覆盖已在所有当前计划纠正。
2. **布局保留：** 左上CQ/SQ、左下语谱与顶部色条；右上音频微观、右下EGG微观及上方滤波控制；下方参数两行与总览。EGG内部按此迁移，仍使用U2共同外壳和主题。
3. **Praat时间：** 本次0.8秒合成输入真实首帧0.020秒，旧服务首帧0.005秒，差15ms；F0值相同。差值与音频长度/底层网格有关，不推广为所有输入固定15ms。
4. **导出策略：** 单文件请求0.1–0.5秒，却有CQ padding行落在约0.000091–0.585850秒；单CSV159行、批次95行。批次静音阈值1.0实际把全部参数mask为NaN，正常阈值仍有有限值。
5. **缺失规则：** CQ边界0.05/0.95无效，但SQ可仍为有限值，二者mask独立（新增D10）。不在迁移时顺手统一。
6. **短输入：** 8采样的高低通分别提示长度不足并返回未滤波数据。此是旧行为，未来v3需明确处理失败，不能只显示滤波成功。

完整差异列表见[源码映射](../modules/evidence/M03-source-map.md)。此轮未修原算法，也未实施Praat时间、ROI或IF双WAV的产品行为修正。

## 环境和来源

| 依赖 | 原phonetic_311 | 现有m09-ui |
| --- | --- | --- |
| Python | 以每轮environment.json为准 | 3.11.14 |
| NumPy / SciPy | 2.2.6 / 1.16.3 | 相同 |
| Parselmouth / Pandas | 0.4.7 / 2.3.3 | 相同 |
| Matplotlib / Pillow | 3.10.8 / 12.0.0 | 均缺少 |
| PyQt6 | 6.6.1 | 6.11.0 |

未安装新依赖，未创建m03-ui，未改变现有Qt宿主或v2环境。B阶段可独立锁定相同科学包，Qt版本沿v3环境单列验证；不用原Qt版本覆盖现有宿主。当前继承filters还依赖PyWavelets导入，后续纯EGG核心只迁相关滤波，不机械引入无关算法。

来源沿既有SRC-PRAAT与PENDING-EGG登记，新增证据为本地来源哈希和行为记录，没有新增外部算法或下载依赖。PENDING-EGG的完整方法引用/许可仍未解决，未据本轮结果升级为可公开再分发。

## 限制与下一项

未测：v3新核心、完整加载/播放/拖动/长文件导航、声卡、浅深双端UI、持久任务、IF对比对话框及双WAV、数据库、冻结EXE、安装版及跨平台。预取消和批次文件间行为不代表文件内取消已实现。最终30项功能验收仍planned。

下一项M03-B：创建独立科学环境并迁移纯数组核心，以此次冻结基准验证数值、时间和mask；保持单文件/批次、raw/filtered旧规则。正常迁移与科学修正分别审阅。页面阶段严格保留上述v2布局，M03结束后停止，不自动推进M04。
