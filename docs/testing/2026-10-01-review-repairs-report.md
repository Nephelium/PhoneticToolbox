# P16-REVIEW 逐项修复报告

2026-10-01，分支 `codex/v3-rebuild`。依据：[实施计划](../plans/2026-10-01-review-repairs.md)、[ADR-P16](../decisions/ADR-P16-review-repairs.md)。

本轮九项缺陷的代码修复已完成，限定 Windows 的定向验证通过。跨平台基础适配已经补齐一部分，Linux 新 POSIX 保存与 macOS 原生执行/发行仍待目标平台验证，不能据此标记完整三平台项目 verified。

开始时工作区已有 53 个已跟踪文件差异及多项未跟踪工作，包括 P04、M03、M04、M13。此次保留这些工作，只在相关文件上追加审查修复。未修改 V2、用户语料、现存数据库或科学环境依赖，未 push、部署、生成 EXE、修改 CI/CD 或系统配置。测试使用项目自有进程及合成输入。

## 逐项结果

| 问题 | 修复与证据 | 本轮边界 |
|---|---|---|
| 1. 非整数采样步长累积偏移 | common.py、energy.py 每帧使用 round(k × fs × frameshift_ms / 1000)；44.1/22.05/48 kHz、非整数帧移、脉冲能量/RMS 独立断言通过。44.1 kHz、5 ms、10 s 的旧定位偏差为 1000 个采样点，约 22.7 ms | 会改变受影响旧数值，标记 acoustic/2 |
| 2. 算法异常被转成缺失值并成功完成 | 13 类服务阶段统一固定失败码，前端显示具体阶段；取消/资源中止向上传递。正常无声仍可成功返回 NaN，空数组核心 API 保持旧空结果行为，任务输入仍拒绝空音频 | 核心/接口错误注入通过；不承诺所有输入都能估计出有限值 |
| 3. 共振峰数量设置无效 | 取消强制至少 5 个，Burg 接收实际数量，帧读取异常也不再吞掉；3/4/5/7 参数观测通过 | 旧共振峰分槽阈值仍保留并写入手册，未重新设计科学规则 |
| 4. REAPER 策略与实际执行不符 | worker 新增延迟创建的 PolicyReaper；python_only 不接触原生文件，native_required 失败即失败，native_then_python 允许普通故障回退并记录 native_failed。取消/超时/预算不回退 | Windows 实际二进制、Python 实现、元数据 JSON 往返及故障注入通过；不宣称两种实现数值等价 |
| 5. MFA 大文件被公共桥接提前拒绝 | 按操作和角色分别限制 JSON/Base64/解码字节；音频 64 MB、词典 16 MB、文本 2 MB 与适配器一致 | 实际 Qt slot→后台线程→M11 解码→signal，6.1 MB 输入通过；未执行本轮全词典 MFA 对齐 |
| 6. Windows 字体偏好阻断其他主机导出 | 新请求显式 portable 策略，按确定候选解析并记录请求/实际字体/哈希；IPA 固定资源仍强制。旧请求默认 strict | Windows 实际字体和兼容子进程检查通过，缺字体回退为受控模拟；未在 Linux/Mac 安装字体 |
| 7. capabilities 宣告了部署禁止的能力 | 根据资源档关闭小服务器 ZIP 操作，并与 P15 可信允许表取交集；维护/排空时隐藏新任务、存储仅下载；另修 Linux 仅部分算法就绪时空 reasons 索引错误 | FastAPI/P15 主机规则测试通过，未改生产服务器配置 |
| 8. 保存逻辑仅有 Windows 目录锁 | 统一目录能力接口；Windows 保留 deny-delete 句柄，POSIX 从根逐段 no-follow 打开目录，基于 dir_fd 读写/发布/清理，普通新文件原子拒绝覆盖已有目标；标注保存保留版本检查 | Windows 保存/重名/失败回收与标注回归通过；2 项真实 POSIX 测试因当前宿主跳过 |
| 9. 标注快捷键不支持 Command | 编辑命令同时接受 Ctrl 和 Meta，保留输入区、IME、Alt/Shift 保护；播放不截获组合键 | Windows Chrome 真实 Control/Meta 两套剪切、复制、粘贴、撤销、保存、边界失败流程通过；Meta 事件测试不等于 Mac 实机验收 |

## 跨平台准备和联调修复

- 用户目录改为 Windows LocalAppData、Linux XDG 或 ~/.local/share、macOS Library/Application Support，不再依赖必然存在的 LOCALAPPDATA。
- VTL 资源按操作系统与 CPU 架构选择。Windows x86_64 兼容既有平铺目录，其他平台必须提供对应目录和库，分析与合成库内容需匹配。当前没有生成新的 Linux/macOS 原生库。
- 科学进程入口对未实现平台显式失败；语谱转音频复用统一调度，不再在通用分支无条件导入 Windows 管道。冻结进程流恢复也区分 Windows/POSIX。
- 新增可移动运行时清单及只读预检；本机预览清单必须明确 portable:false。通过预检不等于完整独立包成立，格式和待验项见[桌面清单说明](../specs/desktop-bundle.md)。
- 实际字体子进程发现源码后端混用旧安装核心。EGG/LPC 源码及冻结源码快照现在绑定同布局的配套核心，已安装 wheel 继续使用其依赖；未重装或升级科学环境。实际字体预检复跑通过。
- Chrome 标注流程复现侧栏边界落入波形区域。Vue 响应式 class 更新移除了控制器的定位类，现由控制器恢复，并仅在缺失时添加，避免自触发测量循环。原失败证据：`output/validation/m12-ui/8380bcc47538472894415debde22b02e/resize-diagnostics.json`。

## 已执行的验证

以下为分组结果，复跑项目可能重叠，不将各行相加作为独立测试总数。

| 分组 | 环境和结果 |
|---|---|
| 核心、契约、架构、能力、存储/标注定向综合 | .venv/m09-ui：394 passed，2 skipped，2 条依赖弃用警告 |
| 实际 REAPER、Python 策略、JSON 结果及原有 M01 I/O | .venv/m09-ui：25 passed |
| 字体解析与 LPC 导出 | .venv/m03-compatible：22 passed |
| 实际兼容子进程字体预检 | .venv/m09-ui 启动兼容子进程：6 passed |
| EGG 预览回归 | .venv/m03-compatible 按既有 DLL 启动方式运行：14 passed |
| Qt 大请求桥接、大小边界、源码绑定 | .venv/m09-ui：8 passed（含综合组中的 5 项请求边界） |
| 前端单元测试 | 185 passed |
| 前端检查 | typecheck、contracts:check、ui-data:check 均通过；UI 数据 80 项参数、335 项来源记录 |
| 前端生产构建 | Vite build 通过；仍有三个大 chunk 提示，未调高阈值压制警告 |
| Chrome 标注 Control | 6 组流程通过，证据 output/validation/m12-ui/c6014630a5a74e3d811b7bb4621f8fb8 |
| Chrome 标注 Meta 最终复跑 | 6 组流程通过，证据 output/validation/m12-ui/52e638f274ad42f0b476baef96afd5fd |
| Chrome 公共布局最终复跑 | 9 组检查，包括 40 个模块/视窗/缩放/主题组合，证据 output/validation/p04-resize/1790827496024 |
| 工作区差异检查 | git diff --check 通过，Git 另有现有 CRLF/LF 提示 |

### 主要复现命令

工作目录为仓库根目录，使用当前项目环境。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='D:\PhoneticToolbox\PhoneticToolbox_v3\packages\phonetic_core\src;D:\PhoneticToolbox\PhoneticToolbox_v3\backend\src;D:\PhoneticToolbox\PhoneticToolbox_v3\desktop\src'
& '.venv/m09-ui/Scripts/python.exe' -B -m pytest -c tests/pytest.ini -p no:cacheprovider packages/phonetic_core/tests tests/contracts tests/architecture backend/tests/test_review_reaper_policy.py backend/tests/test_review_platform_dispatch.py backend/tests/test_review_capabilities.py backend/tests/test_p11_capabilities.py backend/tests/test_m01_result.py backend/tests/test_m01_failure_codes.py backend/tests/test_m01_batch_boundary.py backend/tests/test_m03_contract.py backend/tests/test_m04_contract.py desktop/tests/test_review_bundle.py desktop/tests/test_review_directory.py desktop/tests/test_review_platforms.py desktop/tests/test_review_request_limits.py desktop/tests/test_m01_files.py desktop/tests/test_m12_annotation.py tests/staging/test_p15_staging.py -q --tb=short
& '.venv/m09-ui/Scripts/python.exe' -B -m pytest -c tests/pytest.ini -p no:cacheprovider backend/tests/test_review_scientific_integration.py backend/tests/test_m01_io.py -q --tb=short
& '.venv/m09-ui/Scripts/python.exe' -B -m pytest -c tests/pytest.ini -p no:cacheprovider backend/tests/test_review_source_runtime.py desktop/tests/test_review_request_limits.py desktop/tests/test_review_qt_request.py -q --tb=short
& '.venv/m03-compatible/python.exe' -B -m pytest -c tests/pytest.ini -p no:cacheprovider backend/tests/test_fonts.py backend/tests/test_review_fonts.py backend/tests/test_m04_exports.py -q --tb=short
& '.venv/m09-ui/Scripts/python.exe' -B -m pytest -c tests/pytest.ini -p no:cacheprovider backend/tests/test_m03_font_preflight.py -q --tb=short
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run contracts:check
npm --prefix frontend run ui-data:check
npm --prefix frontend run build
node tests/e2e/p04-resize.cjs
$env:PTB_TEST_SHORTCUT_MODIFIER='Control'
node tests/e2e/m12-r6.cjs
$env:PTB_TEST_SHORTCUT_MODIFIER='Meta'
node tests/e2e/m12-r6.cjs
```

EGG 预览测试需在同一测试进程、导入 NumPy/Praat 前按 egg_bootstrap.py 设置 PATH 的当前环境 Library/bin 并保留 os.add_dll_directory(...) 句柄，然后调用 pytest 执行 backend/tests/test_m03_preview.py。首次直接调用未设置该路径出现 0xc06d007f 原生加载崩溃，按实际启动方式复跑 14 项通过。仅修改测试进程环境，没有改系统 DLL 路径。

科学集成测试最初使用纯正弦并错误要求 REAPER 必须判有声，三个策略合法返回无声。改用含 15 个谐波、180 Hz 的周期激励后，原生和 Python 后端均得到有限 F0，序列化与原有 I/O 回归通过；未放宽算法容差或改动 REAPER。公共布局的 Escape 测试修正为等待原本通过 requestAnimationFrame 恢复的实际栏宽，仍断言恢复 520 px。中间一轮关闭浏览器时出现 ResizeObserver 通知提示，恢复定位类的操作改为幂等后，最终完整复跑未再出现该提示。

## 未验证和保留项

1. 当前 WSL 未找到 Python，未安装环境。新 POSIX 保存行为仍需 Linux/macOS 实际执行；此前已有的 Linux 科学准入证据不被本次 Windows 测试扩大。
2. 未访问实际服务器、未运行线上负载、未改现存数据库。源码更新后，Linux 对应包/运行时哈希和准入收据需在授权部署流程中重新确认。
3. macOS 原生沙箱、依赖构建、Qt helper、设备权限、签名/公证和完整离线包仍待 Mac；Windows 当前 EXE 也未重打，现有预览包继续依赖其明确声明的本机环境。
4. 共振峰固定槽位规则继续保留。新声学修订标记用于区分结果，不能将本轮修正描述成所有科学算法已重新验证或与旧版数值完全相同。
5. 构建大 chunk 提示仍在，涉及公共工作台、感知实验和普通话转 IPA 数据。本轮没有为压制提示改构建阈值，也不据构建成功宣称全部页面性能已验收。
