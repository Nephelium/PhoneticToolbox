# M12 语音标注对齐开发态验收

2026-09-14。状态：**verified，限定 Windows 开发态功能**。井井明确提前迁移 M12，完成前不打包 EXE。已实现原 7 功能组、V2 说明书 13.1–13.6 对应行为，接入统一工作台。M04 保留 D 检查点，本轮未推进其他模块。

## 实现与原版对照

第一份本地提交 `7a90a8d` 记录 65 个原编辑函数的实例化迁移、原词典和基准；后续适配与修正单独提交。来源对应[映射](../modules/evidence/M12-source-map.md)、[实施计划](../plans/2026-09-14-m12-implementation.md)和 ADR-050。

前端原版函数直接运行对照，不用同一新函数生成预期值。原 RMS 包络、内收/外扩边界估计、四种参考窗口、拼音规则、Hann/FFT、900 点强度显示、唇形可见邻点及显示坐标保持对照结果。显示配色接入 V3 浅深主题，语谱图栅格最多 700 列/360 行，不改变标注或声音。主波形和试听使用公共组件。对M12显式限制8–96 kHz、8声道、800万帧，避免异常采样率头导致显示/强度窗口无界计算；仍受公共64 MB/3200万采样值约束。网页请求现有created排序，达到1000条资源列表预算时提示清理，避免误把被分页截断的集合当完整语料。

修正单列：完整 TextGrid 编解码保留点层、其他层和各层时域；错误和重叠显式拒绝；禁止拖动总时域端点及向过短空白插词；文件版本冲突拒绝覆盖；保存中出现新编辑阻止切换；文本框未按回车的输入纳入dirty和定时保存，输入法组字期间保留且等待完成；网页通过既有幂等上传流程续传，不读取尚未发布的资源；同名网页版本保留，扫描优先最新。自然录音验证发现原 TextGrid 六位小数与 WAV 帧时长差约 0.08 微秒，编辑层检查已采用原格式精度范围，与文件整体检查一致，原域保留。

## 功能验收映射

| 功能组 | 正常证据 | 错误或边界证据 |
| --- | --- | --- |
| F01 语料与层级 | 递归中文子目录、同名优先级、两份自然录音和自定义层名 | 重复/错误 TextGrid 拒绝；扫描/切换前保留编辑；其他层域不变 |
| F02 编辑与试听 | 真实 Chrome 拖边界及音素联动、整词/多词移动、插点/合并、直接输入/撤销、Ctrl/Shift 导航、真实 WebAudio | 端点与过短区间回归；取消关闭保留编辑；未测物理声卡音质 |
| F03 文本资源 | 原词典及自定义上传、独立 lab、搜索/替换确认/撤销、词表连续粘贴 | 损坏格式拒绝，替换可恢复，输入法文本与 IPA 保存回读 |
| F04 参考复用 | 四种模式各自真实点击及撤销；同名点层复用 | 无同名层、类型不匹配拒绝；其他层不被全局补空白改动 |
| F05 强度贴合 | 原版包络和 5 组内收/外扩结果精确对照，负值页面实测 | 静音/无可信边界返回提示；保持 −50 至 80 ms 范围 |
| F06 唇形 | 原时间轴、正负偏移，PKL 独立写回；Qt/网页安全 JSON | 3 种 pickle 协议 × C/F 数组保留；恶意全局不执行；唇偏保存不改 TextGrid |
| F07 保存 | 真实原子保存回读，空后缀确认，60 秒定时回调，切换前保存；下载哈希一致 | 来源/目标并发修改及注入磁盘错误；保存时继续编辑；关闭保存失败；网页真实配额拒绝 |

## 实际命令和结果

Windows，项目内 `.venv/m09-ui` Python 3.11；自身 core/API/desktop wheel 由 `.venv/v3-dev` 构建并以 `uv pip install --no-deps --reinstall` 安装到 m09-ui。只更新本项目的三个 wheel，未安装或升级第三方依赖。m09-ui 不含 pip，采用现有 uv；初次 `python -m pip` 失败未改变环境。开发入口 `start_m01_workbench.py --prepare-only` 返回 ready=true，复用现有库/缓存，未运行DDL。

| 命令 | 实际结果 |
| --- | --- |
| `npm --prefix frontend run typecheck` | 通过 |
| `npm --prefix frontend test` | 80 项前端检查，包括 11 项 M12 原版/格式/失败回归 |
| `npm --prefix frontend run build` | 共同前端构建通过，开发入口已更新 |
| `python -X utf8 -m pytest -c tests/pytest.ini desktop/tests backend/tests/test_m01_legacy_inputs.py backend/tests/test_m01_io.py backend/tests/test_m01_segments.py backend/tests/test_m02_display.py tests/contracts tests/architecture -q` | 176 passed，包含 12 项 M12 实际文件测试。设置 PYTHONPATH 为当前 core/backend/desktop src。两项现有 Starlette 弃用警告，未改变依赖 |
| `node tests/e2e/m12.cjs` | 18 组独立 Chrome 检查；实际原生 FileProvider/TaskBridge 与文件回读；2 份已授权自然录音的原文件哈希不变 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m12_qt.py` | 11 个阶段，已安装 wheel、真实 Qt 自定义协议/QWebChannel/本机文件选择与保存；两种 blob 下载经 Qt 文件对话框实际落盘并回读 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m12_web.py` | 真实 PostgreSQL 两账号、API 与 Chrome；上传/保存新版本/下载/退出重进/跨账号隔离；配额拒绝保留 dirty；受控到期 HTTP 410 与物理清理 |

Qt 测试窗口设置 WA_DontShowOnScreen，原生对话框由测试选择自己的输出目录。Chrome 为独立 headless 进程和独立配置，试听测试加 mute-audio。未使用 Codex 内置浏览器关闭接口。

## 证据位置

- `output/validation/m12-ui/6d61a6718e9b4e43a9f94994451243dd/report.json`、`natural-report.json`：18 组 Chrome 与自然录音回读，截图和每步 TextGrid 保存在同一忽略目录。
- `output/validation/m12-qt/970ca0c83bc440cb98fd0fc412720967/report.json`：11 阶段、原文件与下载双回读、浅深 Qt 截图。
- `output/validation/m12-web/8e5a99341c21403bbf33f75ab3e75a4d/report.json`、`web-report.json`、`limits-report.json`：真实网页联合结果，6 项常规路径和配额/到期路径。
- `output/validation/m12-wheel/` 与 `m12-*-build.log`：自身 wheel 构建。原文件来源哈希记录在 `third_party/m12-migration.json`，交付前复核 6 项一致。

首次网页测试暴露上传阶段资源不可读错误，已修复并通过真实服务复验及丢失 finalize 响应重试回归。后续两次测试脚本分别修正了跨账号列表应为空集合的预期、测试侧 Storage 实例的显式 recover 初始化；未为通过修改后端权限或配额规则。未按回车文本专项曾暴露raw文档对象身份不变导致dirty缓存未更新，已显式依赖编辑revision并通过真实输入法/切换回归；一次Chrome测试随机端口被浏览器拒绝，测试改为46212起选择空闲端口，未改变产品网络入口。

到期为测试调整时间并显式触发现有清理，不冒充自然经过 7 天。60 秒自动保存通过加速浏览器时钟触发原定时器，文件写入为真实能力。自然录音仅复制到本机忽略的验收目录，未上传或提交。

## 限制与交付

当前可使用[操作说明](../manual/annotation.md)中的现有开发入口。完整跨平台桌面、生产负载、异常硬断电和新采集设备同步未测。PKL 只支持受限数字/普通元数据结构；网页只读取安全 JSON。失效账号无法继续写入其项目资源，重新登录后的任务状态不跨账号复用。手动保存和下载均有明确结果。

未生成 EXE、未运行 PyInstaller、未执行 DDL、未 push、未改 CI/全局环境、未改相邻 V2 和既有发行物。完整早期源码/词典许可链仍 pending，沿已知证据登记，不为此延长本轮开发功能收口。井井后续明确要求后才准备临时 EXE。
