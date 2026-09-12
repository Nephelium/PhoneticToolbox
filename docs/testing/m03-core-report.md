# M03-B Windows科学核心与独立wheel验收

2026-09-12。**M03-B verified，限定Windows纯数组核心、指定Conda/MKL环境和安装wheel。完整M03仍in_progress，任务服务、页面、Qt整合与EXE尚未实现。** 井井在A及v3统一风格说明后明确授权继续。

## 完成内容

- `phonetic_core.egg`：双声道归一化/滤波、GCI/GOI四组合、自动/手动峰谷、全局与两种局部规则、CQ/SQ、两F0、F0变化启发式、简化CP逆滤波。
- 核心输入为数组和配置，未导入旧工程、Qt、HTTP、数据库、Matplotlib、PyWavelets或音频设备，也不通过临时WAV调用Praat。
- `EGGConfig`冻结参数，裸服务保留GOI slope，`for_workbench()`显式给出实际GUI的slope/scale。结果分别保留预处理和事件分析快照，不把后改配置伪装为旧结果配置。
- 明确`EggError.code`与`EggCancelled`。非法配置/声道/样本/采样率/ROI拒绝；滤波失败不返回伪滤波结果，IF不可用不冒充成功空数组。GCI/GOI局部窗口和IF逐闭合段增加取消检查。
- `praat_pitch.times`为真实帧时间，`legacy_times`保留旧轴；`sample_duration_s=N/fs`和`last_sample_time_s=(N-1)/fs`分开。历史`file_duration`仍明确代表最后采样时刻，避免偷偷改变ROI兼容结果。

正常数值迁移先提交`7666206`，随后补公共边界和明确元数据。来源函数、SHA-256及改动关系见`third_party/egg-migration.json`，包内含`egg/NOTICE.txt`。本地迁移不替代PENDING-EGG尚未完成的方法/许可核验。

## 实际环境差异与解决

原计划的`.venv/m03-ui`使用PyPI NumPy 2.2.6、SciPy 1.16.3、Parselmouth 0.4.7，与旧环境版本号一致，但SciPy实际构建不同：

| 项目 | 原v2与兼容环境 | PyPI候选 |
| --- | --- | --- |
| SciPy | conda-forge 1.16.3 py311hf127856_1，LLVM/Flang | PyPI 1.16.3，GCC/GFortran |
| SciPy BLAS/LAPACK | MKL 2025.3.0，libblas/liblapack 3.11.0 | scipy-openblas 0.3.29.dev |
| 同输入去趋势最大绝对差 | 原v2与兼容环境精确一致 | 对原v2约5.96e-8 |
| 去趋势后高低通结果最大差 | 精确一致 | 约3.23e-7 |

定位时同时运行相同输入和相同SciPy调用，保存于`output/validation/m03-core/probe.py`、`old.npz`、`new.npz`。最初怀疑数组视图布局，保留原交错布局后差异仍在；真正原因经构建配置及Conda元数据定位至计算库构建。没有放宽容差、改写基准或更换科学公式来通过。

原安装包均仍在本机缓存：32个包共约170.8MB，逐个SHA-256核验后离线安装到`.venv/m03-compatible`。Conda包缓存限定`.venv/m03-package-cache`，`CONDA_REGISTER_ENVS=false`、`CONDA_SHORTCUTS=false`，没有系统快捷方式或环境注册变更。未整目录克隆v2。

- Conda构建锁：`requirements-m03-conda-explicit.txt`及`third_party/m03-runtime-lock.json`（实际URL、build、MD5/SHA-256与包声明许可）。
- PyPI补充/测试锁：`requirements-m03-compatible.in/.lock`，17个包及传递依赖版本和wheel哈希；SciPy由Conda锁单独管理。
- 分发元数据清单：`third_party/m03-pip-audit.json`。本地元数据不视为公开发行许可审核通过。
- `.venv/m03-ui`保留为失败候选，使用`requirements-m03-test.in/.lock`，不能当已获数值验收的运行环境。
- `scripts/Invoke-M03-Python.ps1`仅为当前进程配置兼容环境DLL搜索路径，退出恢复原PATH，不启动GUI或修改系统配置。

## 实际验证命令

以下从v3根执行；安装后的回归实际将cwd设为独立的`output/validation/m03-core`，所有测试路径使用绝对路径，并清空进程PYTHONPATH。

```powershell
scripts/Invoke-M03-Python.ps1 -X utf8 -m build --wheel --no-isolation --outdir output/validation/m03-core/wheels-final packages/phonetic_core
uv pip install --python .venv/m03-compatible/python.exe --no-deps --reinstall output/validation/m03-core/wheels-final/phonetic_core-3.0.0a1-py3-none-any.whl
scripts/Invoke-M03-Python.ps1 -X utf8 -m pytest -c tests/pytest.ini tests/parity/test_egg_analysis.py packages/phonetic_core/tests/test_egg_config.py tests/parity/test_m03_capture_contract.py -q
scripts/Invoke-M03-Python.ps1 -X utf8 scripts/verify_m03_core.py --include-private
```

结果：**80 passed**。其中新B阶段61项、A冻结证据19项。重复GCI的旧缺失规则测试产生两条预期RuntimeWarning，未过滤或隐藏。`scripts/verify_m03_core.py`验证安装路径位于sys.prefix，随后核查没有导入旧包/GUI/服务层。

11个样例完成**31,761项精确比较**，该数字包含事件列表标量、数组与元数据比较，不能称为31,761个独立测试。正常样例保留数组dtype/shape/NaN以及精确值，额外比较数组字节（包括有符号零和NaN载荷）；原坏WAV由文件适配拒绝，单/多声道、空数组和短滤波通过明确错误路径。两个真实录音仍按原授权方向读取、哈希一致，未复制自然录音到仓库。

最终报告：`output/validation/m03-core/wheel-16540d26b2bd4de884fde0e8ee690db0/report.json`。此前开发对照和首轮wheel结果保留，最终以此报告为准。

最终wheel：`output/validation/m03-core/wheels-final/phonetic_core-3.0.0a1-py3-none-any.whl`，SHA-256：`5a0352c976f19da0f85948e99edd086e7012455ee23b410c3fcad6ecef4ea0b2`。它是本地科学包，不是可双击桌面应用。

额外核实：原v2 161份Python源码哈希与A阶段相同；11份输入前后哈希相同。`scripts/check_architecture.py`无错误，`npm --prefix frontend run ui-data`按共同登记生成80参数/327来源，未改EGG界面。

## 科学与行为边界

| 行为 | 本轮结果 |
| --- | --- |
| 数值、方法与缺失 | 正常事件/CQ/SQ/两F0/IF与冻结基准精确一致，SQ独立mask保留 |
| ROI滤波 | CQ仍为处理后信号±100ms再次滤波，事件仍为raw±50ms一次滤波，没有科学统一 |
| F0时间 | 实际帧轴与旧轴分别可取；合成0.8秒例旧轴早15ms，未把固定15ms推广到其他输入 |
| 取消 | Python窗口/事件/IF闭合段循环中可响应；Praat及一次原生相关/滤波调用只能在前后检查，C阶段仍需所属子进程超时/终止能力 |
| 逆滤波 | 自动和12阶值对照，测试ROI限前0.12秒；长ROI资源上限和双WAV文件写出留待C |
| CSV/PNG与谱图 | A捕获的旧导出规则仍为独立expected；B只迁数组核心，未实现新文件导出/谱图页面 |
| 平台 | 仅指定Windows构建通过，PyPI SciPy不具备本轮逐位兼容证据，跨平台仍planned |

下一项M03-C：共享任务/文件协议、受控科学worker、CSV/三PNG/IF双WAV结果。先具体审阅独立兼容环境接入与DLL边界，不能直接覆盖m09/m10宿主；若表约束需要DDL，另交具体迁移审阅。后续EGG页面从第一版使用v3公共规范/组件，并保留v2四图和参数位置关系。
