# M01-F1 切分执行与持久批次准备验收

2026-09-10，井井在M01-E布局交付后回复“继续”。**F1 verified，限定Windows所属进程内的切分、参数派生、批次策略及迁移准备；M01-F和完整M01/P08仍in_progress。** 实际005建表、持久发布和页面保存按钮尚未接入。后续依赖是 [具体迁移审阅](m01-migration-review.md) 获明确授权，再实施F2/G。

## 本轮完成

- 纯核心切分计划保留v2 `int(t*fs)`、原声道/解码类型和静音标签跳过；缺失/重名层、非法/重叠区间、空样本片段明确失败。文件名增加原区间序号，避免三位小数时间相同导致覆盖；manifest保留去首尾空白后的标签，中文/IPA不受文件名字符过滤影响。
- 所属Windows Job生成WAV及可选XLSX/SQLite，受输入64 MB、3200万样本值、32声道、TextGrid 2 MB、输出总64 MB、1000片段、进程1 GB和60秒限制。命名管道逐块限额；完整结果经hash与格式回读后才返回。正常、取消、超时、心跳失败和预算拒绝均清理本次Scratch，终止所属进程。
- 对M01-MAN01补上参数派生规则和执行器：父结果必须匹配同一WAV的hash/采样率/帧数/声道/dtype；保留原参数帧与非有限值，不重新估计。派生Time_s相对实际WAV首样本，Source_Time_s保留原始时间。无父结果仅WAV；无参数帧明确no_frames。
- 17项有序批次快照、幂等子任务键、部分失败后继续、取消保留not_started、准确complete计数、1000任务容量预留规则已有定向验证。这是无数据库I/O的内部策略，尚未作为公开HTTP入口或持久调度器使用。
- 005 PG/SQLite增量SQL和执行工具已备妥，显式授权+精确SQL hash为入口前置。`show`与拒绝未授权/错误hash的行为实测通过；**没有执行任何005 DDL，不能据此声称SQL实际建表/隔离通过**。

## 实际合成文件回读

最终安装wheel后执行 `scripts/verify_m01_segments.py`，证据为忽略目录 `output/validation/m01/segments-b22f897b5dfc441b851a8a14bff8e515/report.json`。该目录保留输入、TextGrid、父结果、所有产物与manifest/hash；写出使用新UUID目录和`xb`，没有用户目录写入。

| 用例 | 验证结果 |
| --- | --- |
| 实际Praat参数父结果 | 自造0.8秒44.1 kHz信号，真实分析160帧，其中153帧pF0为有限值；按TextGrid切出两个有标签片段，静音段跳过 |
| 参数同步切片 | 两段分别41行、79行；XLSX与SQLite所有单元格与原计算表独立选帧结果一致；原时间/局部时间、NaN、中文/IPA与公式样文本保留 |
| 非网格起点 | 第二段首样本17684，首原始参数帧0.405秒，相对片段约0.0040022676秒；没有把首帧错误设成0 |
| float32双声道 | 原采样率、float32类型和双声道逐样本一致 |
| 600秒音频 | 8 kHz、9,600,044字节的int16合成输入；首尾两段与原始样本逐点一致，受同一预算约束 |

本次小信号父结果由验证脚本直接调用科学核心，REAPER显式disabled；没有把该测试路径接入用户任务。实际用户参数分析的全科学进程隔离、配额、TTL、独立续租、恢复与结果发布仍是F2。

## 实际验证命令

工作目录均为v3根。工程解释器`.venv/v3-dev`，最终测试解释器`.venv/m01-ui` / CPython3.11.14；核心与API均从新安装wheel导入，未增加外部依赖或修改v2环境。

```powershell
& 'C:/Users/13680/.local/bin/uv.exe' build packages/phonetic_core --wheel --no-build-isolation --python '.venv/v3-dev/Scripts/python.exe' --out-dir 'output/validation/m01/f1-wheels'
& 'C:/Users/13680/.local/bin/uv.exe' build backend --wheel --no-build-isolation --python '.venv/v3-dev/Scripts/python.exe' --out-dir 'output/validation/m01/f1-wheels'
& 'C:/Users/13680/.local/bin/uv.exe' pip install --python '.venv/m01-ui/Scripts/python.exe' --no-deps --reinstall --link-mode copy 'output/validation/m01/f1-wheels/phonetic_core-3.0.0a1-py3-none-any.whl' 'output/validation/m01/f1-wheels/ptb_api-3.0.0a1-py3-none-any.whl'
& '.venv/m01-ui/Scripts/python.exe' -X utf8 -m pytest -c tests/pytest.ini packages/phonetic_core/tests/test_segmentation.py backend/tests/test_m01_segments.py backend/tests/test_m01_batch_policy.py backend/tests/test_m01_migration_guard.py backend/tests/test_m01_io.py backend/tests/test_m01_result.py backend/tests/test_job_policy.py backend/tests/test_job_boundary.py backend/tests/test_storage_policy.py backend/tests/test_storage_boundary.py backend/tests/test_archive_policy.py tests/contracts tests/architecture tests/security/test_m01_formats.py tests/parity/test_baseline.py tests/parity/test_m01_capture_contract.py -q --junitxml=output/validation/m01/f1-final.xml
& '.venv/m01-ui/Scripts/python.exe' -X utf8 scripts/verify_m01_segments.py
```

**244 passed，41.79秒**；两个既有Starlette/httpx/anyio弃用提示，未更换已锁定依赖。初次测试缺模块为预期红灯；首轮双格式样例使用错误的pF0显示列名被已有协议拒绝，修正测试样例使用目录正式标签后通过，未放宽产品校验。最终还覆盖不合法manifest/路径、同名时间片段、无帧、父来源不符、取消/超时/模拟续租失败。

`scripts/check_architecture.py`、`scripts/generate_contracts.py --check`、`scripts/validate_docs.py`、前端`ui-data`/`ui-data:check`/`contracts:check`及`git diff --check`均通过最终一致性检查。文档检查errors为空，另报告未改历史快照的已知失效相对链接。UI源码未改，不重复浏览器/Qt全套截图验收。来源登记仍323条，更新已有SQLite、PG锁、SciPy、openpyxl、Win32和WAV记录的使用位置，未引入新第三方代码。代码/wheel hash和实际产物摘要见[机器证据](../modules/evidence/M01-execution-preparation.json)。

## 保存性与剩余限制

`output/validation/m01/f1-preservation.json`显示7项与E之后相同：v2 HEAD/index/status、427份基线文件、包元数据和环境路径。P03的8份与M01-A的28份冻结文件通过保护测试。Git根仍为 `D:/PhoneticToolbox/PhoneticToolbox_v3`；本轮不push，不上传Users目录，没有操作或关闭Codex内置浏览器。

现有按钮仍未启用；跨文件输出重名、目录刷新回读、参数父结果选择、任意旧版XLSX/SQLite导入兼容、PG配额/到期/并发/取消竞态和双端持久恢复均须F2/G实际验收。长音频案例只证明该600秒/8 kHz合成输入切分通过，不代表任意长度全参数分析或实时性能通过。未做跨平台/完整EXE/自然语料验收，不宣称修复Codex闪退根因或保证不再闪退。
