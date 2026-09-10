# M01-G 旧格式入口验收

2026-09-10，**本批次 verified（Windows 定向范围）**。完整 M01/G/P08 仍 in_progress，下一步是39项验收逐条审阅；M02没有因此标为已迁移。[计划](../plans/2026-09-10-m01-legacy-inputs.md)、[ADR-029](../decisions/ADR.md)、[操作说明](../manual/parameter-estimation.md)。

## 实际完成

- 桌面“关联说明与旧文件兼容”中选择 PKL，实际转换并独占保存同目录 `.lip.json`，关联当前音频。支持 v2 NumPy 1/2 数值序列化结构及 pickle 协议2–5；时间、手动偏移和四项唇形参数保留，元数据缺少音频起点时读取同名 `_timestamps.pkl`。不调用 pickle.load 或序列化构造器。只转换 M01 字段，landmarks 等保留在原 PKL。
- 切分区新增显式历史表来源及逐音频下拉框，支持 XLSX / `.ptb.sqlite` / `.ptb.sqlite3`。历史表与可信 parent_result 使用不同字段并互斥，不能进入声学分析请求，也不能冒充有原 WAV 证明的父结果。没有明确关联时阻止提交；输出清单记录文件名、SHA256 和 `user_associated_unverified`。
- 原音频切片、原表列/标签/非有限值及时间帧保持；`Time_s` 按实际首样本重基准，`Source_Time_s` 保留原时间，不重估。子进程/owner/project/hash/expiry/租约/配额/整组发布沿用 F2；没有数据库 DDL。
- 转换成功收起低频兼容区；历史参数选择紧邻切分操作，减少上下寻找。本机刷新保留仍有效的已选关联，文件身份或服务器 hash 变化则清除。颜色、播放器与滚动仍使用公共工作台。

## 验证命令及证据

以下文件均位于忽略目录 `output/validation/m01/`；无个人录音或数据库进入源码提交。

| 验证 | 结果 |
| --- | --- |
| `.venv/m01-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini backend/tests desktop/tests tests/contracts tests/security packages/phonetic_core/tests tests/parity -q` | **430 passed**，77.64秒；两条原有 Starlette/anyio 弃用提示；`legacy-regression.log` |
| `npm --prefix frontend test`、`run typecheck`、`run build` | **17 passed**，类型和生产静态构建通过 |
| `verify_m01_task_window.py` | `task-window-9998fa477dce4ae09db763af7667c05e/report.json`：15个真实Qt步骤；原生目录选择、PKL转换/关联、真实分析、同源切分、历史XLSX及SQLite切分、保存/刷新、正常退出全部通过；4份历史切分表逐值一致；原输入hash不变 |
| `run_m01_validation.py --approved-m01-schema-and-synthetic-tests --verify-web`（不加apply） | `persistent-7af298afbbbc4112b2117bcd1474829e/`：独立Chrome、真实PG/API/worker，10组操作通过；普通分析/同源切分10个下载，以及历史表切分14个下载均校验；4组历史双格式值与原样本一致；账号切换拒绝跨用户批次 |
| 浅/深色与布局 | Qt `qt-light.png`、`qt-dark.png`、`qt-narrow.png`；Chrome 1920/1280/1000/390 CSS px 的文档和每列无横向溢出。截图需完成主题切换动画，不能以中间帧判断最终对比度 |
| 保留检查 | `legacy-preservation.json`：v2 HEAD/index/status、427基线文件及环境元数据共7项一致 |

Qt/网页均使用测试脚本通过**相邻原 v2 `services/io/excel.py` 的实际写入函数**创建新合成表；源码 SHA256 为 `8e571e0cbfe970b7050833d9dca34f2e107cc8066f7fa25b463da92e5dd0494f`，只读并验证不变。PKL 样例按照实际 lip_gui 写出的 NumPy 时间列表、float32 landmarks、指标和元数据结构生成；没有把新读取器输出作为自己的期望值。应用没有动态导入 v2。

针对性错误覆盖：构造器载荷、object/complex dtype、循环/共享容器、尾随字节、数值预算、ZIP/XML体积与坐标预算、实体/外链、公式、重复列、非普通/生成列SQLite、时间越界、二次切片、无参数帧和失败不返回半套产物。转换接口在读请求体之前验证本机会话与Origin，网页模式不可用。

## 发现与修复

首轮 `task-window-71109c0891134651b3c639b058fe0ab1` / `persistent-7d14b77ff5da4723a674327aa07e67a0` 在普通分析处失败：新增可选 legacy_result 被带入 AcousticRequest。已在分析/切分边界移除该切分专用字段，随后普通分析和两种切分均真实重跑通过。失败记录保留，不降低契约或绕过校验。Qt失败轮仍正常关闭，网页API/worker退出码均0。

来源登记补充现有 Python pickle、openpyxl、SQLite 使用位置；解析器为项目实现，没有新库或新增第三方移植代码。架构/契约/统一来源生成与文档检查在交付前执行。

## 边界与下一步

- 本轮是有界旧格式兼容，不承诺任意 Python 对象 PKL、任意电子表格或数据库均可读。PKL16 MB/伴随2 MB，转换512 MB/20秒；表格16 MB、20万单元格、32 MB解压XML，切分沿用1 GB/60秒。公式、外链、生成列、时间越界及已有 Source_Time_s 明确拒绝，不静默修复。
- 旧表关联不能证明原始WAV或原分析后端。CSV、M02多轨显示、完整录像/landmarks迁移，以及新的自然唇形采集验证不在本批次范围。
- 下一步核对39项最终矩阵及已知范围差异，再进入M02。正式EXE、跨平台及生产负载仍属后续阶段。
- 没有push、全局依赖/系统配置变更或新建数据库表。Qt/Chrome/API/worker均为任务拥有的进程且已正常清理；避开Codex内置浏览器关闭，没有宣称修复Codex底层退出原因。
