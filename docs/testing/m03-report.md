# M03-E 联合收口记录

2026-09-12。**E1 操作与导出对齐 verified，限定 Windows 开发态。完整 M03-E / M03 仍 in_progress。** 本轮为井井在 M03-D 后授权继续的第一批。冻结 EXE、生产网页账号联合路径、自然录音页面和未闭合来源不在此完成声明内。

最新进展：[E2 网页、自然录音与切换竞争](m03-e2-report.md)已完成限定验收。下文E1证据保留。E3当前范围修复和方法审阅见[追加报告](m03-e3-report.md)，完整E3仍in_progress。

2026-09-13 最新范围：EXE及相关探针按井井要求暂停。批次参数与提交错误恢复已限定Chrome验证，见[批次反馈报告](m03-batch-feedback-report.md)；完整M03仍in_progress，后续优先开发态结果窗口失败反馈及来源审阅。下文F安排为历史，不作为当前下一步。

2026-09-13 结果反馈追加：读取/重读、保存和下载反馈归属、历史预览恢复已限定Chrome验证，见[结果反馈报告](m03-result-feedback-report.md)。上条下一项结果窗口现已完成该限定范围，后续集中核对V2开发态功能清单及来源，完整M03仍in_progress。

2026-09-13 六功能组复核及总览单击/提交取消补齐见[功能复核报告](m03-function-review.md)。开发态入口已逐项映射，下一项集中处理方法/代码来源未决项；设备/生产和EXE范围不扩大，完整M03保持in_progress。

## 本轮修正

1. 单文件两类 F0 显示默认关闭，与原 EGGWidget 一致。旧草稿保留用户已选值。批次改为独立默认和独立草稿：GCI slope、GOI scale、高通 25 Hz、两类 F0 开启、图片关闭，不再继承单文件选区/显示/滤波设置。
2. 对照 `_save_csv_data` 发现旧单文件 CSV 始终含两类 F0。D 的显示开关会省略对应列，现修正为单文件先取得 Praat 轨迹并独立保存双 F0。PNG 仍按显示开关绘制，批次仍按自己的选项决定列。已有 CSV 不回写，重新导出才补齐此前省略的列；数值方法、时间网格与 mask 不变。
3. 受控结果名称仍为 `egg_DATA.csv` 等。JSON 新增 `input_name` / `export_names`，由后台使用任务输入名称和实际采样选区生成。桌面与网页下载共用保存名称，单文件/IF 如 `元音_0_10s_0_50s_DATA.csv`，批次如 `元音_DATA.csv`。旧结果无映射时仍用旧名。截短名称附摘要，禁止路径字符；已有不同内容另名保存，相同内容重复保存复用。
4. 四图补滚轮缩放、拖动平移、键盘左右与加减键，左侧红色中心定位线联动。手势结束后通过既有任务更新，处理中不重复提交，切文件/关闭时清除待提交手势。显示组件仅处理坐标变换。
5. 保存成功、取消、错误直接出现在结果窗口，保存期间禁用重复提交。取消目录选择不写任何文件，既有计算结果仍可保存。

原 V2 的八份来源文件 SHA-256 与 `third_party/egg-migration.json` 一致，未修改原源码。新增差异依据及未覆盖部分见[源码映射](../modules/evidence/M03-source-map.md)和 ADR-039。

## 实际验证

Python 测试仅在进程内设置绝对 `PYTHONPATH=backend/src;desktop/src`。科学计算使用安装过核心 wheel 的 `.venv/m03-compatible`，普通 API / Qt 使用 `.venv/m09-ui`。本轮没有安装依赖或重新打包科学 wheel。

| 命令 | 结果 |
| --- | --- |
| `node --test frontend/tests/m03.test.ts` | 新默认隔离用例先因缺失 `batchDefaults` 失败，实施后 7 passed |
| `npm --prefix frontend test` | 49 passed |
| `npm --prefix frontend run typecheck` / `run build` | 通过 |
| `scripts/Invoke-M03-Python.ps1 -PythonArguments @('-X','utf8','-m','pytest','-c','tests/pytest.ini','backend/tests/test_m03_exports.py','backend/tests/test_m03_preview.py','backend/tests/test_m03_export_names.py','-q')` | 42 passed；含原 V2 数组/CSV对照、双WAV、PNG、隐藏显示仍保留 CSV 双 F0、中文/IPA保存名称 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini desktop/tests/test_m03_save_names.py backend/tests/test_m03_contract.py -q` | 16 passed；真实目录写入、同名保护、重复保存、旧元数据回退、路径越界拒绝和本次新文件回滚，保留两条既有依赖弃用提示 |
| `node tests/e2e/m03.cjs` | 18 项通过，0 页面错误；真实手势触发科研任务，双默认、取消/重试、24 px 图字体、浅深/窄窗/模拟 DPI、命名与 CSV 列、批次、恢复 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m03_qt.py` | 9 组通过；包括真实选择目录取消不写文件、三路径实际保存与 IF 两 WAV 回读 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/validate_docs.py` / `scripts/check_architecture.py` | 538 文件、328 来源、32任务，errors=[]；历史快照失效链接单列保留；架构 errors=[] |
| `npm --prefix frontend run contracts:check` / `run ui-data:check` | 无漂移 |

UI 证据初轮：`output/validation/m03-ui/chrome-13b725225b584548950d0b4ce6187e89/report.json`、`qt-799a77e7762141b28080bc5aecde39cb/report.json`；命名/保存反馈截图 `named-save.png` 已查看。测试使用既有 P06 合成数据库的一致副本，`schema_applied=[]`。Chrome 接真实本机能力，不冒充生产网页登录验证。

补齐清除待提交手势和帮助文字后，最终前端检查、构建与 Chrome 18项再次通过：`output/validation/m03-ui/chrome-4e34923c3e7644f1a550d2c5615d462a/report.json`。

## 三十项覆盖与剩余验收

下表是当前证据映射。`verified` 限于所列范围，`in_progress` 表示联合项尚有缺口。B/C/D 的旧命令与产物位置分别保留在[核心](m03-core-report.md)、[任务](m03-jobs-report.md)、[页面](m03-ui-report.md)报告，未将历史测试说成本轮重跑。

| ID | 状态 / 已有证据 | 尚需补充 |
| --- | --- | --- |
| A01 | verified / B 输入形状，C 子进程预算，D/E1 实际单声道拒绝 | 限 Windows 已测数据 |
| A02 | verified / B 声道数值，E2 延迟历史结果后切文件与交换失效 | 限已复现竞争，不承诺任意并发操作 |
| A03 | verified / B 独立归一化、dtype、非有限拒绝、输入不变 | 限核心证据 |
| A04 | verified / B 滤波显式失败，D 原/滤波微观双轮数组 | 限已冻结样例 |
| A05 | verified / B 25/1000、参数/Nyquist/短输入 | 限核心与契约 |
| A06 | verified / B 两种 padding，D 原始微观边缘数组 | 科学统一不在兼容迁移范围 |
| A07 | verified / C/E1 5/20/50 ms PSD 逐字节与范围验证 | 限既有采样率/样例 |
| A08 | verified / B 四组合与缺事件基准 | 限原版数组对照 |
| A09 | verified / B 自动/手动阈值与重叠基准 | 限原版数组对照 |
| A10 | verified / B scale 固定 0.25，D 无假调节入口 | 不新增无效参数 |
| A11 | verified / B 解析 CQ/SQ、边界及独立 mask | 不代表生理效度 |
| A12 | verified / D 来源提示，E1 中心线与更新失效 | 不合并旧三种局部规则 |
| A13 | verified / B F0 变化启发式标记基准 | 不解释为器官位移测量 |
| A14 | verified / B 实际 Praat 帧，E1 隐藏显示仍导出原值 | 限已冻结输入 |
| A15 | verified / B 中点/低 F0/MAD 等原行为 | 限已冻结输入 |
| A16 | verified / B 逆滤波及不可用错误，C 一秒边界 | 限简化 CP 方法 |
| A17 | verified / C/E1 双 FLOAT64 WAV 回读、D 四图原数组、E1 取消保存 | 冻结程序另验 |
| A18 | verified / E3-B 120秒全段计算及导出，60秒总览视窗、末尾定位 | 限Windows开发态、576万帧及既定预算，见长文件报告 |
| A19 | verified / E3 5–5000 ms原版显示数组、完整事件及真实页面 | 限Windows开发态，原版显示抽点保留 |
| A20 | in_progress / E3连续试听、交换后实际Web Audio样本、标签/文件切换及关闭 | 实际声卡/听觉与多设备仍未验，见试听报告 |
| A21 | verified / C 时间/mask/部分失败，E1 源名称/时间戳/双F0/同名保护 | 限 Windows 开发态 |
| A22 | verified / C/E1 批次各 F0 列组合逐字节 | 限核心和导出 |
| A23 | verified / B/C 包络阈值及 CSV-only mask | 限冻结数据 |
| A24 | verified / E1 独立批次快照与实际默认，C 可选图片 | 网页联合路径另验 |
| A25 | verified / C 文件内取消/迟到 worker，D/E1 失败继续/取消重试 | 限本机任务 |
| A26 | verified / C/D/E1 恢复，E2 本机托管PG受控配额/到期拒绝和物理清理 | 不含生产部署、七天自然经过或调度可靠性扩大验收 |
| A27 | verified / E2 网页双账号读隔离与同浏览器切换 | 限两个测试账号、本机托管服务 |
| A28 | in_progress / D/E1浅深/窄窗/字体/键盘，E3取消关闭保留与焦点返回 | 原生多屏 DPI、其余焦点与未保存组合 |
| A29 | in_progress / E3官方馆藏、手册截图和Henrich原文，方法差异已说明 | 推荐论文完整原文页码、旧代码许可与简化IF方法链 |
| A30 | in_progress / D/E1 本机，E2 服务器三路径与两份自然录音页面 | 设备与生产环境；EXE按最新要求暂停 |

## 下一批与停止边界

E2 已完成上述限定验证，见追加报告；E3微观范围修复和方法审阅已完成本轮限定验证，E3-B长文件与字体预检已限定验证，当前优先剩余交互及来源。EXE及相关探针按2026-09-13最新要求暂停，不作为当前下一步。M04 不推进。

D 广域回归中 M01 Scratch 取消清理的一次 WinError32 继续保留，单项复跑通过不等于根因已修复，本轮未改该实现。未执行 DDL、未改 v2、旧 EXE、全局环境、CI/CD，也未 push 或发布。


2026-09-12 E3-B追加：120秒/576万帧有界长文件、原V2独立双轮对照、受限进程/Chrome/Qt末尾与全段导出已限定验证，见[长文件报告](m03-long-report.md)。上文60秒计算限制与长文件拒绝为历史证据；总览60秒视窗保持。完整E3/M03仍in_progress，下一项字体预检及剩余交互/来源，冻结EXE另行收口。

2026-09-12 字体预检追加：带字PNG提交前检查实际兼容进程，失败不创建任务，批次保留选择和旧记录；纯数据不受渲染字体阻断。限定证据见[字体预检报告](m03-font-preflight-report.md)。A28的多屏DPI/完整焦点与A29/A30仍待，不扩大为全模块verified。

2026-09-13 E3试听追加：修复IF两个试听标签按清单顺序误配、两播放器切换与过期设备请求失败打断新播放，见[试听报告](m03-playback-report.md)。历史双WAV保存/文件回读不能证明按钮角色，本轮补实际节点样本比对。F仅完成[候选范围审阅](../plans/2026-09-13-m03-f-candidate-review.md)，未实施冻结包。
