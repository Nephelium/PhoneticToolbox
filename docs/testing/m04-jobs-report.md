# M04-C 任务与导出验收

2026-09-13，**verified，限定Windows开发态的持久任务、兼容科学子进程、文件产物与保存适配。** 完整M04仍为in_progress，下一项D页面。服务器使用真实项目专用PostgreSQL、磁盘和认证路由，HTTP接口通过FastAPI TestClient的ASGI传输验证，未宣称本轮完成Chrome页面操作。

## 完成内容

- m04/1请求含WAV资产引用、可选TextGrid引用、明确时间ROI和独立参数/字体快照。沿用P06/P07幂等键、owner、输入哈希/到期、任务租约与原子产物发布，未增加表或执行DDL。
- 固定LPC子进程使用现有M03兼容Conda/MKL运行库，30秒/2GB、8MB产物上限。仅把上一阶段已验证的核心wheel重装进项目 `.venv/m03-compatible`，NumPy/SciPy等第三方包不变。API与桌面宿主不加载科学DLL执行LPC。
- 每次产出 `lpc.ptb.json`、`lpc_SPECTRUM.png`、`lpc_AUDIO.wav`。JSON包含完整1024点谱线、原参数、实际半开样本选区、原文件/标注哈希、来源与字体证据。选区WAV为FLOAT64单声道，保留原PCM转换比例和多声道均值，不归一化峰值。
- PNG为2400×1350、标记约299.9994 DPI、白底黑线，使用V3字体快照，标注标题固定Doulos SIL并提供中文回退。同名保存保留旧文件，重复保存同一结果复用已校验文件。PNG为导出产物，未把它称作LPC交互页面。
- TextGrid复用已有有界解析器与B阶段标签规则。标注超出音频超过1样本、无有效层、过长标签、非法输入和缺字体明确失败，整套结果不发布。

## 真实验证

| 范围 | 结果与证据 |
| --- | --- |
| 纯科学/导出 | 104项通过，含LPC既有73项、3个冻结V2谱线经实际JSON无损往返，PNG像素/DPI、1/8声道PCM、TextGrid错误与EGG原样回归；没有放宽数值容差 |
| 契约/共享任务 | 139项通过，覆盖LPC/M03入口、Origin/身份、字段拒绝、任务状态/批次和来源错误码等；2项既有EGG保存名测试通过 |
| 本地HTTP/保存 | 3种实际配置：普通选区、后半段动态纵轴、48000样本/200阶。耗时3.0、3.5、11.359秒；PNG/WAV/JSON逐文件哈希回读，同名旧文件保留和重复保存通过 |
| 重启与标注 | 本地服务关闭重开后恢复成功任务；实际TextGrid `ɑ̃˥` 标签、Doulos字体快照与导出通过。已目视普通谱图与带IPA标签图，无裁切的坐标文字；固定纵轴范围外的谱线按V2规则裁剪 |
| 取消/失败 | 排队取消、实际子进程启动后取消、PID退出、重试成功、部分产物写入故障回收、过期租约拒绝迟到封存；非成功任务无可发布结果，临时资产0 |
| 硬超时 | 将本轮拥有的子进程受控替换为停滞进程，复用实际executor的30秒限制，30.937秒任务结束，deadline_exceeded，PID已退出，无临时残留。此项是受控停滞，不冒充真实科学输入耗时 |
| 长文件 | 800万帧/8kHz/1000秒PCM16，实际16,000,044字节，计算999.9–1000秒尾部800样本，9.719秒完成；多1帧明确lpc_input_budget失败。未截断整份文件、未分析全部1000秒，临时文件均回收 |
| 服务器 | 两个真实账号和独立会话，跨owner输入、项目、任务读取/取消、产物下载均被拒绝；本人三文件回读哈希通过。用本轮配额预留触发真实失败并回收，无残留预留 |
| 到期 | 仅调整本轮结果的到期时间，认证下载410，显式删除后磁盘文件不存在。未声称自然等待七天 |
| 既有状态 | 本地测试基于原测试SQLite一致性副本，旧行及合成输入哈希未变。服务器旧任务行未变，项目专用PG仅由本轮启动/停止，未操作用户手动服务 |
| 公共前端 | 65项既有测试、类型检查、契约生成一致性和构建通过；只增加生成契约和来源数据，尚未接LPC页面 |

主要证据目录（均为忽略的本机测试输出）：

- `output/validation/m04-jobs/a4d85a8413824fa7a1bbcf6b6ec046a4/report.json`
- `output/validation/m04-server/774cefa1e9c74693b6f88d1681369557/report.json`
- `output/validation/m04-limits/43b517bd1dfd4d2f9d8f559455719e27/report.json`

## 实际命令

```powershell
uv pip install --python .venv/m03-compatible/python.exe --no-deps --reinstall output/validation/m04/wheel-final/phonetic_core-3.0.0a1-py3-none-any.whl
$env:PYTHONPATH = ((Join-Path (Get-Location) 'backend/src'),(Join-Path (Get-Location) 'desktop/src') -join ';')
& scripts/Invoke-M03-Python.ps1 -X utf8 -m pytest -c tests/pytest.ini backend/tests/test_m04_exports.py tests/parity/test_lpc_spectrum.py tests/parity/test_lpc_boundaries.py tests/parity/test_egg_analysis.py -q
& .venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini backend/tests/test_m04_contract.py backend/tests/test_m03_contract.py backend/tests/test_job_policy.py backend/tests/test_job_boundary.py backend/tests/test_m01_batch_policy.py backend/tests/test_m01_batch_boundary.py backend/tests/test_m01_failure_codes.py tests/contracts -q
& .venv/m09-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini desktop/tests/test_m03_save_names.py -q
& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m04_jobs.py
& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m04_server.py
& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m04_limits.py
& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/generate_contracts.py
& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/generate_contracts.py --check
npm --prefix frontend run contracts
npm --prefix frontend run contracts:check
npm --prefix frontend run ui-data
npm --prefix frontend run ui-data:check
npm --prefix frontend run typecheck
npm --prefix frontend run test
npm --prefix frontend run build
& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/validate_docs.py
& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/check_architecture.py
```

开发中先建立拒绝用例再接入口。一次测试把WAV字节放入默认参数化测试ID，导致Windows环境变量过长，改为短的显式场景ID后通过。最初本地验证脚本误用宿主没有的Pillow，改用标准库读取PNG头；科学导出测试仍在已有Pillow的兼容环境实测图片。没有安装新依赖。EGG重复时间基准保留既有数值警告，Starlette兼容弃用警告仍在，不以静默屏蔽处理。

来源登记已追加M04的Matplotlib使用位置和导出适配差异，学术引用沿用B。未修改原LPC数值核心、V2、旧EXE或5份暂停的探针草稿。2GB是已配置的Windows进程限制，本轮未故意分配到内存耗尽；设备、多屏、生产负载和其他平台仍未测。用户可见的选区/字体预检/错误提示/保存交互在D页面阶段验证，当前本地文件能力通过TaskBridge/API实际调用验证。
