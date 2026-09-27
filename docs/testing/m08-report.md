# M08 变速变调迁移验收

> 2026-09-27 更新：正式 Windows 接线与限定宿主验收见 [M08 接线报告](m08-wiring-report.md)。以下是 2026-09-26 独立模块历史证据，其中“尚未接线/Qt 未测”已由新报告的具体范围取代；网页 PG 与 Linux 正式链路仍未通过，完整 M08 保持 in_progress。

2026-09-26。**完整 M08 为 `in_progress`。** 模块纯核心、边界 handler、页面和真实计算的独立浏览器验证已交付。公共生产任务/存储/页面注册尚未接入，Linux 跨平台精确门仍失败，不能标记整模块 verified。

## 实施与边界

分支 `codex/v3-rebuild`，开始于工作树已有 M12、P11、P04 等未提交成果的状态。初始逐文件摘要在 `output/validation/m08/initial-files.json`。仅认领本报告下面27个文件。公共 agent 并行产生的变更不是本轮成果。未改 AppShell、tokens、公共组件、共享依赖、公共执行器、runtime/capability、后端入口、总台账或全局 ADR。

未迁移数据库、全局安装、改系统配置/凭据、push、公开发布、部署或生成 EXE。未修改 V2、用户语料及旧发行包。浏览器删除/重命名只针对每次运行新建 `output/validation/m08/browser-<uuid>/saved` 的公开合成结果，源码输入保留。

## changed_files

- `backend/src/ptb_api/m08_models.py`
- `backend/src/ptb_worker/m08_jobs.py`
- `backend/tests/test_m08_jobs.py`
- `docs/manual/pitch-manipulation.md`
- `docs/modules/evidence/M08-source-map.md`
- `docs/plans/modules/M08-pitch-manipulation.md`
- `docs/plans/modules/M08-wiring.md`
- `docs/testing/m08-report.md`
- `frontend/src/modules/pitch-manipulation/HistoryPlot.vue`
- `frontend/src/modules/pitch-manipulation/PitchCurve.vue`
- `frontend/src/modules/pitch-manipulation/PitchManipulationPage.vue`
- `frontend/src/modules/pitch-manipulation/port.ts`
- `frontend/src/modules/pitch-manipulation/state.ts`
- `frontend/tests/m08-live.html`
- `frontend/tests/m08-live.ts`
- `frontend/tests/m08.test.ts`
- `packages/phonetic_core/src/phonetic_core/manipulation/m08_batch.py`
- `packages/phonetic_core/src/phonetic_core/manipulation/m08_rules.py`
- `packages/phonetic_core/src/phonetic_core/manipulation/m08_synthesis.py`
- `packages/phonetic_core/src/phonetic_core/manipulation/m08_transform.py`
- `tests/e2e/m08.cjs`
- `tests/fixtures/m08/v2.json`
- `tests/fixtures/m08/v2.npz`
- `tests/parity/test_pitch_manipulation.py`
- `tests/support/m08_baseline.py`
- `tests/support/m08_browser_bridge.py`
- `tests/support/m08_linux_probe.py`


前三份直接迁移科学文件为 `m08_synthesis.py`、`m08_batch.py`、`m08_transform.py`。外层边界/命名/导入规则独立在 `m08_rules.py` 和 handler；Hz 单位兼容修正明确单列，不更改原 V2。没有碰 manipulation 目录下任何 M07 文件。

## 六组功能覆盖

完整逐项映射见 [M08-source-map](../modules/evidence/M08-source-map.md)，操作说明见 [manual](../manual/pitch-manipulation.md)。

| 组 | 本轮已实现/验证 | 未完成范围 |
| --- | --- | --- |
| F01 单文件 | 加载真实 WAV/Praat F0、速度、原音/合成音播放、视野/整段、保存编号；Windows 精确基准与 Chrome | 正式 host/FileProvider 的 WAV/MP3/FLAC、存储原子编号接线 |
| F02 曲线参照 | Shift 绘制、Ctrl恢复、视野联动、F0上下界/参考线、导入、PNG；数值/状态/Chrome及截图 | 所有DPI/设备实测，正式宿主联验 |
| F03 批次文件 | 本视野分组、明确ID列表、前缀替换/冲突检查、真实合成文件删除/重命名、历史F0 | 正式owner/配额/到期/跨账号/多进程冲突事务 |
| F04 文件夹批量 | UI入口、多选、先变速再乘比率再加Hz、逐文件状态、损坏文件独立失败 | 正式持久队列、native硬取消/故障恢复、输出目录适配 |
| F05 批量基频 | 四模式、16组起终组合+拐点、offset、实际WAV输出、256组合保护 | 正式配额writer/fenced发布，服务器预算 |
| F06 拐点与帮助 | 表格增删/不可删端点/排序/模式/保存、帮助、references emit | AppShell来源弹窗注册与真实标签关闭门 |

页面已从第一版使用公共 ModuleFrame、ModuleToolbar、ModuleSection、ModuleStatus、WaveformViewport、AudioTransport、TaskPanel、ModalDialog、公共字体和 SVG PNG painter。未自建产品宿主。测试专用 Vite middleware + stdio fixture 不进入生产。

## Windows：独立基准与核心

环境：`.venv/m09-ui` CPython 3.11.14；NumPy 2.2.6、Parselmouth 0.4.7、内嵌 Praat 6.1.38。公开双谐波输入16 kHz、0.6 s，未使用研究语料。

`tests/support/m08_baseline.py` 从原 V2 直接加载 synthesis/batch，AST提取原 service 的完整文件处理方法。没有调用 V3 生成 expected。两轮捕获66数组、571,595值，全部66数组hash及源码hash一致；输入样本2数组不计入64个实际输出/时间轴诊断。Windows V3对比64/64精确一致，最大绝对差0。固定基准包含16种起终点模式组合、order拐点、offset、当前视野和整段、0.8/1.5速度及阈值附近1.005。

两个非零Hz原始案例报 `Option argument Unit cannot have the value Hz`，保留为旧行为失败。原样函数默认仍复现该错误，handler显式传Hertz后以150 Hz合成音、1.2倍再+20 Hz验证结果约200 Hz。没有把修复后的输出冒充原成功基准。

首次未固定Praat随机状态时三个速度案例精确失败。用Praat官方可重复随机种子函数42在原捕获与V3测试每案例前重设后通过。生产默认没有增加固定种子或改变随机行为。方法依据见 [Praat](https://praat.org/manual/_random_initializeWithSeedUnsafelyButPredictably_.html)。

实际命令（项目根 PowerShell）：

```powershell
& .venv/m09-ui/Scripts/python.exe tests/support/m08_baseline.py
$env:PYTHONPATH='packages/phonetic_core/src;backend/src'
& .venv/m09-ui/Scripts/python.exe -m pytest -c tests/pytest.ini tests/parity/test_pitch_manipulation.py backend/tests/test_m08_jobs.py -q -p no:cacheprovider --junitxml=output/validation/m08/windows-final.xml
& .venv/m09-ui/Scripts/python.exe tests/support/m08_linux_probe.py . output/validation/m08/windows-numeric.json
& .venv/v3-dev/Scripts/python.exe -m build --wheel --no-isolation --outdir output/validation/m08/wheel packages/phonetic_core
python -m pip --python .venv/m09-ui/Scripts/python.exe install --no-deps --target output/validation/m08/installed-core output/validation/m08/wheel/phonetic_core-3.0.0a1-py3-none-any.whl
$env:PYTHONPATH='output/validation/m08/installed-core;backend/src'
& .venv/m09-ui/Scripts/python.exe -m pytest -c tests/pytest.ini tests/parity/test_pitch_manipulation.py backend/tests/test_m08_jobs.py -q -p no:cacheprovider --junitxml=output/validation/m08/windows-wheel.xml
```

结果：源码 **43 passed（1.00 s，无跳过）**；安装wheel **43 passed**。安装路径已打印确认来自独立target目录，没有覆盖开发环境的已装核心。后续最终wheel另构建到 `wheel-final`，供Linux安装，变动只为同一迁移文件的IO文档澄清。wheel与依赖hash见交付摘要。

原始环境失败也保留：m09-ui/v3-dev无pip，系统Python为3.13而本包要求3.11，首次安装被正确拒绝。随后用系统pip的 `--python` 指定3.11安装到独立target，未放宽Requires-Python或安装全局包。

## Windows：真实浏览器和前端

```powershell
node --test frontend/tests/m08.test.ts
npm --prefix frontend run typecheck
npm --prefix frontend run test
npm --prefix frontend run build
node tests/e2e/m08.cjs
```

- M08状态测试6项，前端全量137项通过（包括并行P04/M13当时版本）；vue-tsc通过。
- Vite生产构建通过，仍有既有大chunk提示。**M08未注册，所以这不能证明生产bundle已包含M08。** M08组件经真实Vite测试入口编译并在Chrome运行。
- 最终浏览器记录：`output/validation/m08/browser-3e1c4517/checks.json`，**15组，pageerror=0**。使用实际Chrome、真实Python/Praat、真实PCM16文件回读，不是返回固定成功曲线。
- 覆盖真实加载/默认、Shift绘制、导入插值、0.8倍实际1.250 s输出、两次保存递增、历史重提取、PNG、非法路径拒绝与真实rename/delete、视野0.5 s和整段1 s区别、AudioContext播放/停止、端点保护和实际4组合、两音频成功+损坏文件单独失败、dirty切换取消、hash绑定草稿恢复、1000/800/390宽无整页横向溢出、缺port时计算禁用且无伪F0。
- Chrome中修复：普通合成误带空拐点列表；取消切换后原生select显示新项；窄窗图字号过小；图形pointer使用SVG逆矩阵适配真实几何；PNG导出加入版本文件名。长错误使用机器码+可读说明，不输出后台绝对路径。
- 初次E2E有精确locator匹配错误，已修测试定位；一次随机端口1723被Chrome安全规则拒绝，测试入口改为5188起顺延可用端口，没有改浏览器安全设置。没有以这些失败截图作为最终视觉证据。

截图与图像已实际查看：

- [浅色](../../output/validation/m08/browser-3e1c4517/light.png)
- [深色](../../output/validation/m08/browser-3e1c4517/dark.png)
- [深色390宽](../../output/validation/m08/browser-3e1c4517/dark-390.png)
- [实际历史PNG](../../output/validation/m08/browser-3e1c4517/comparison.png)

主题切换后等待过渡完成再截图；不冻结计时取过渡帧。Chrome证据只证明独立模块和测试adapter，不是Qt、真实声卡、账号、数据库或正式host通过。共享workspace dirty和真实save返回值已提供，但AppShell标签关闭联验仍待接线。

## Linux WSL

在现有NInfer使用本任务独立 `/home/ninfer/ptb-m08-20260926` venv与project目录。解释器基于既有P11原生Linux CPython 3.11.14；未修改P11 venv、系统Python或WSL配置。WSL出网失败，因此Windows从官方PyPI下载Linux wheels，Linux用 `--no-index --find-links` 离线安装NumPy2.2.6、Parselmouth0.4.7和pytest。科学版本与Windows相同；没有因此宣称二进制构建相同。

最终测试使用安装的pure-Python核心wheel，不以Windows exe冒充Linux。核心import所需distribution metadata初次未装导致collection失败，安装wheel后完成实际测试，没有改包初始化去绕过。

```sh
cd /home/ninfer/ptb-m08-20260926/project
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  /home/ninfer/ptb-m08-20260926/bin/python -m pytest -c /dev/null  tests/parity/test_pitch_manipulation.py -q -p no:cacheprovider  --junitxml=/mnt/d/PhoneticToolbox/PhoneticToolbox_v3/output/validation/m08/linux-wheel.xml
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  /home/ninfer/ptb-m08-20260926/bin/python tests/support/m08_linux_probe.py  /home/ninfer/ptb-m08-20260926/project  /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/output/validation/m08/linux-numeric.json
```

原Windows冻结精确门 **25 passed / 5 failed（0.49 s）**，不降低判据。独立诊断遍历所有字段（不因早期assert中断而漏掉轴）：64数组中59精确一致，5不一致。

| 差异 | 最大绝对差 | 判断 |
| --- | --- | --- |
| Praat原F0（46/56帧不同） | 1.084364953385375e-8 Hz | 原精确门失败 |
| 4个float64合成波形 | 5.551115123125783e-17 | 原精确门失败 |
| 时间网格/输出时间轴、PCM16批量波形等其余59数组 | 0 | 本合成场景精确一致 |

追加Linux同环境V2独立捕获在单独 `same-platform-baseline` 目录，原Windows冻结fixtures与其Linux副本未替换。Linux原V2与迁移核心 **64/64精确一致**，见 `linux-same-platform.json`。这支持本轮纯迁移在该环境等价，但不关闭跨平台精确门，也不把差异全部归因于某一BLAS或CPU。

## 平台、资源、远程与UI状态

| 维度 | 状态 |
| --- | --- |
| Windows科学迁移 | verified，限定43项、公开短合成、指定wheel与环境 |
| Windows模块交互 | verified，限定独立Chrome与真实计算测试adapter的15组 |
| Linux同环境迁移 | verified，限定64数组原V2对照 |
| Linux对Windows冻结精确门 | in_progress，25/5，不开放科学capability |
| 资源 | 输入/输出/组合准入边界已测，进程组预算、长音频峰值、吞吐/故障清理未验证 |
| 服务器/远程节点 | 未测试，未占服务器槽、未部署 |
| 公共UI接线 | 组件复用已完成，正式AppShell注册和标签关闭待P04 owner |
| 正式网页任务/存储 | blocked_on_platform；owner/配额/到期/原子编号/持久任务/硬取消/重试尚未接 |
| EXE/Qt/生产 | 未测试，不在本轮授权范围 |

## 剩余接口与退出门

[精确接线合同](../plans/modules/M08-wiring.md)已列页面props/emits/save/dirty和M08Port方法、Python配置、计算handler、资源ID/哈希/owner/expiry/quota/fencing要求。当前ResearchTasks没有M08协议，公共执行器没有M08注册，FileProvider没有完整三格式适配，不能用test adapter注册生产页面。

需要平台owner完成真实M08任务/存储adapter、原子保存编号、结果管理与文件类型适配，公共UI owner再串行注册页面，随后做正式Windows宿主及网页双账号/配额/到期、标签关闭、取消恢复与Linux服务联合验收。跨平台精确差异须由平台/科学审阅决定兼容运行时或独立等价标准，当前不擅改算法和容差。

本轮在M08独占范围停止，未推进M07或其他模块。完整迁移不能因该范围内代码已写完而改成verified。

## 收尾核对

27个认领文件均UTF-8/二进制按预期存在，Python语法检查通过，定向diff空白检查通过。初始清单无文件缺失。继承来源与相邻V2的5份M08源码SHA-256均保持冻结值。并行P04/P11更新单列在 `preservation.json`，不归入本轮changed_files。

`python scripts/validate_docs.py` 检查723文件/333来源/41任务，仍只有两条既有M10-R5 EXE链接错误，未新增M08缺链。没有为消除旧缺链生成EXE。完整认领文件及wheel/Linux依赖摘要见 `output/validation/m08/delivery-manifest.json`。
