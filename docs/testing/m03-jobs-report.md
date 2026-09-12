# M03-C Windows任务、文件与导出验收

2026-09-12。**verified限定：Windows开发态、既有SQLite任务协议、独立Conda/MKL科学子进程、单文件及逐文件批次导出、逆滤波双WAV。完整M03仍in_progress。** 新增P04-FONT规范的公共字体偏好与导出字体快照尚未接通，字体专项保持planned。本轮没有实施EGG交互页面、整批目录调度、冻结EXE、实际PostgreSQL或跨平台验收。

## 完成行为

- 新`m03/1`请求通过`POST /api/v1/jobs/egg/create`进入P06通用任务，输入仅为同owner/project的资源ID和hash。单文件、批次单项及inverse三种模式使用独立参数快照。既有002/005不变，没有DDL。原批次表限定M01操作，EGG批处理在D阶段组合这些独立文件任务，不伪造整批完成状态。
- 复用输入校验、截止时间、心跳、取消、失效worker拦截、配额和原子发布。CSV/PNG/WAV先在子进程内形成完整字节集合，清单及每个hash核验后才能发布。桌面目录授权、逐块读取、同名冲突保护和重复保存复用共同桥接。
- 单文件输出CSV与三PNG，批次单项输出CSV及可选三PNG，inverse输出FLOAT64双WAV。每套增加`egg.ptb.json`，记录输入hash、声道、采样数、实际ROI、参数、方法、时间策略及运行库构建。
- IF的ORIG是峰值归一化后的分析音频片段，IF是简化闭合相逆滤波估计。两者同采样率、同样本数，保留原文件中的偏移，未误标为原文件原始字节或生理流量真值。
- `PTB_EGG_PYTHON`由可信宿主选择项目兼容解释器，`-I`排除环境PYTHONPATH和用户site，固定bootstrap只加载安装核心和本backend。DLL搜索只在所属子进程设置。SciPy/NumPy/Praat/Matplotlib/Pandas版本与Conda构建指纹不匹配则明确失败。旧m09/m10/v2环境和EXE未改。

## 科学兼容与明确差异

CSV导出策略为`sample-aligned/1`。单文件保留outer join原生指标时间网格，但改用Praat真实`pitch.xs()`，所有行裁到采样对齐的半开ROI。旧CQ的100ms padding和二次滤波仍用于计算，只裁输出行，不改CQ/SQ值。批次保持GCI网格插值、6位小数、20ms平均绝对音频幅度遮罩，遮罩仅用于CSV，图片仍不加该遮罩。无有效事件时返回有列名的空表，不生成假值。

波形横轴改为采样索引/fs，取代旧单文件含右端点的linspace。整体时长采用N/fs，末采样时刻仍单列。谱图保留Matplotlib PSD、75%重叠、5–50ms窗、旧灰阶和两类F0，未替换成M01的Praat谱。单文件保留旧ROI谱图extent，批次保留原PSD窗口边界。PNG尺寸仍为单文件1500×900/150dpi、批次1200×600/100dpi。

当前PNG字体沿用冻结V2的打印基准。**这仅是数值/文件验收，不是V3交互页面或P04-FONT字体专项通过。** 公共字体角色、可用性检查、任务字体快照、缺字回退需按[新字体计划](../plans/2026-09-12-global-fonts-design.md)接通后再验收；不能据此将M03整模块标为verified。

## 内存问题及修复证据

第一次60秒CSV通过，追加三PNG后触及1,500,000,000字节进程上限。诊断定位到Matplotlib一次FFT分配882×11969的complex128矩阵（约161MiB），并非任务库或输入损坏。保留失败记录`output/validation/m03-jobs/local-6bcd893d69a041b7bbe392fd9ec59a05/report.json`。

修复将独立FFT窗口每256帧计算，预分配实数PSD，窗口/重叠/归一化保持一致。三种窗长分别与原完整调用比较PSD、频率、时间的dtype及全部字节，均相同。渲染阶段仅去掉0–5000Hz范围之外的不可见像素，并保留原像素边界。没有增加内存上限、放宽容差或改动事件算法。

Win32实际进程内存诊断保存于`output/validation/m03-jobs/memory-dcc57b110f6a4baf9784619b333cba20.jsonl`与`memory-fb1ec46e94f145d0b42706a686ce23bb.jsonl`。修复后同输入三图完成，峰值private commit为1,344,184,320字节。此数值是这组Windows输入的测量，不代表任意设备/数据的固定内存占用。

本轮有界范围：输入≤64MB、≤2,880,000帧、≤60秒、双声道8–96kHz；ROI必须在文件内；IF另限≤1秒且≤48,000帧；进程≤1.5GB/120秒；结果包≤64MB；任务截止300秒。超出明确失败，无静默截短。长音频预览/分窗与全模块资源策略在D/E另验收，不把本轮预算当成完整长录音体验已实现。

## 实际验证

1. `.venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini backend/tests --ignore=backend/tests/test_m03_exports.py desktop/tests tests/contracts -q`：258项通过，含13项M03请求/权限/运行环境边界。保留原依赖的两项弃用提示，没有降低检查。
2. `scripts/Invoke-M03-Python.ps1 -X utf8 -m pytest -c tests/pytest.ini backend/tests/test_m03_exports.py -q`：23项通过。V2五种批次CSV逐字节相同，三种谱窗对冻结数组和分块前原调用精确一致，PNG解码/尺寸、Praat实际时间、两组IF双WAV逐样本回读及输入预算通过。
3. B核心回归80项通过，重复GCI保留的两项RuntimeWarning与原基准一致。C没有改动B的事件/滤波/IF数值函数。
4. `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m03_jobs.py --include-private`：真实本机HTTP、独立MKL子进程、SQLite、文件授权及导出验证通过。使用既有P06合成测试库的独立SQLite在线快照，不修改原库或执行DDL。报告`output/validation/m03-jobs/local-94608b6e70fb4ec4bf25e21c29ef837f/report.json`。
5. 实际验证包括同参幂等/异参冲突、三套产物hash与重复保存、单声道/无GCI错误、服务退出重开、错误owner/hash、排队与运行中取消、真实部分写盘故障后全部回收、未提交文件不可读、租约失效后拒绝迟到写入及成功重试。
6. 60秒合成音频含三PNG约18.39秒；48kHz一秒IF约11.55秒；已授权自然录音EGG-01/EGG-05完整批次单项含三图约7.20/6.75秒，输入hash与P03/A一致。原输入和旧任务行保持，最终临时文件为0。
7. 前端37项、TypeScript检查、契约生成/漂移、架构检查通过。已实际查看新导出的谱图。完整UI操作、跨屏/DPI、声卡、网页PG隔离/TTL和冻结EXE不包含在这些证据内。

Python后端检查需进程`PYTHONPATH=backend/src;desktop/src`的绝对路径。科学导出检查使用已安装核心wheel，仅添加backend/src，不能把m09-ui的PyPI SciPy当兼容环境。

本轮核心wheel为`output/validation/m03-jobs/wheels/phonetic_core-3.0.0a1-py3-none-any.whl`，SHA-256 `55ee692f962944c8701d55327609ab3eaf77fe91eab612015a379f4fce395fcc`。仅安装到m03-compatible。Matplotlib字体索引缓存位于该运行环境`var/cache/ptb-m03-matplotlib`，为共享渲染器索引，不写入用户资产目录，不保存用户文字/语料；公共字体配置/快照仍待P04-FONT。

来源：新增渲染依赖锁`requirements-m03-exports.in/.lock`与安装元数据`third_party/m03-export-runtime.json`，不替换Conda SciPy。V2 GUI数值/导出移入关系见`third_party/egg-migration.json`，PENDING-EGG学术与许可未决继续保留。后续先完成公共字体契约与M03-D页面，再做E逐项联合验收，M04不自动启动。
