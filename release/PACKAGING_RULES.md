# Windows 严格打包规则

本文件是 Windows x64 发行包的默认验收规则。适用于免安装 EXE、当前用户安装 EXE 及更新 ZIP；公开上传、发布和 Git push 仍需单独授权。

## 1. 交付约束

- 按本次源码的实际模块与资源清单核对所需程序和依赖；新增模块分别记录验收范围，排除 M11 的 MFA 环境、声学模型与词典。MFA 的外部环境接入界面可以保留，不得因此声称 MFA 开箱可用。
- 使用者无需自行安装 Python、Conda、Node 或项目依赖。运行时绑定只能指向本包，不能依赖开发电脑的 `.venv`、PATH 或绝对源码路径。
- 免安装版必须是双击直接启动的 `PhoneticToolbox.exe`。安装版安装同一应用 EXE；更新 ZIP 必须包含同一个 SHA-256 的 EXE。
- 三种分发文件均不得超过 **500,000,000 字节**。报告使用十进制 MB，可另列 MiB。该限制是交付上限；本次约 448 MB 的结果不是所有未来版本的理论最低值。
- 交付时分别说明下载体积、首次展开后的持久占用、小型启动临时文件与清理行为。不能用展开重复量冒充 EXE 减少量。

## 2. 一套业务源码与冻结输入

- 开发入口统一经过 `Start-Research-Workbench.ps1`。构建使用现有 `.venv/m14`，EGG/LPC 与唇形继续使用已锁定的独立第三方运行环境。
- 唯一应用构建入口为 `scripts/build_v3_local_preview.py --lean-qt --compact-onefile --persistent-cache`。禁止为免安装版、安装版、脚本版另外维护业务实现。
- Python 入口、PyInstaller 分析与实际文件形式的工作进程必须来自同一次冻结快照。保留 `source-snapshot.json`、其 SHA 和构建前后验证结果，Git commit 不能替代未提交文件身份。
- 先生成本轮说明书阅读资源和前端，再开始 EXE 构建。禁止拿上次的 `frontend/dist` 冒充当前源码产物。
- 构建名每次唯一，不覆盖旧输出。已有用户修改和并行任务修改保留；只冻结明确的本次输入。

## 3. 依赖裁剪与跨包去重

- 只共享大小及 SHA-256 均相同的文件，不能按名称、版本号或相似用途替换。不同的 NumPy/SciPy、MKL、Qt、编解码库不得强行合并。
- 科学压缩包保持已验身份；主 Qt DLL 必须逐一匹配指定主环境原文件。禁止通过换科学库、降级精度、改线程实现或删硬件兼容分支来达到体积目标。
- `host-files/2` 保存完整路径清单及共享映射，绑定科学压缩包和清单 SHA。恢复优先使用硬链接，不支持时复制原字节并校验；路径冲突、重解析点、坏校验、预算超限必须失败。
- 主界面及科学子进程必须恢复各自原路径。全部恢复成功后才写完成标记，重复启动的子进程复用该目录。
- 缓存只有在输入及清单身份匹配时才能复用。v1 缓存不能直接充当 v2 去重缓存，失配应重新压缩并留下原因。
- 每次普通 GUI 启动均显示公共原生启动卡片。准备及逐文件内容校验显示真实阶段进度，界面初始化使用活动进度条，主窗口、字体和网页帧就绪后关闭。`--compact-onefile` 自动启用持久启动器；所有后续构建继承公共实现，不手工改成品。每次实际内容校验保持，仅在本次维护锁内去重硬链接读取，最多四个读取线程，不保存跨启动散列豁免。
- 持久缓存按科学组件、Qt 等宿主组件及应用内容分别寻址。项目自有 `resources/` 下的原生程序库放在应用组件，避免更新声道等模块时使整套 Qt 缓存失效。升级不以版本号猜测内容相同，也不强行复用发生变化的文件。
- 运行数据排除规则统一放在 `scripts/release_content_policy.py`。当前排除旧说明书 Markdown、契约开发说明与 TypeScript 声明；原源码文件保留。
- 不打包作者编辑器、未引用原媒体、完整开发环境、私人语料、密钥和临时构建目录。保留正在使用的模型、浏览器 WASM 回退、字体、版本元数据、许可、原生对应源码材料及实际验收依赖。新增排除项必须有实际用途核查和受影响行为验证，不能仅按扩展名删除。

## 4. 说明书只做基本可用性检查，允许持续追加

说明书按以下基本可用性要求检查。正文不冻结，不把内容完备程度作为打包门槛：未完成章节、预留位置、文字增删、章节/小节调整、图片或例音数量变化均允许。每次构建采用当时已有内容，只要求已发布的阅读资源能解析、已登记引用有对应资源且能打开；不要求达到固定字数、固定标题结构、固定章节或媒体数量，也不要求每次补齐说明书全部章节。

按顺序执行：

```powershell
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/manual/validate.py --project manual --strict
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/manual/build.py --project manual --output frontend/public/manual --distribution software
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run ui-data:check
npm --prefix frontend run build
node frontend/scripts/verify-manual-runtime.mjs frontend/dist/manual
```

作者工程不由构建回写。编辑器的可选 `id:null` 只在生成阅读副本时规范为缺省字段，真实标识、文字和媒体不变；非法非空标识继续拒绝。构建器已强制调用生产阅读器解析器，不能只依赖另一套 Python 校验。

媒体只选择正文实际引用的资源，压缩副本在构建快照内生成，原 PNG/WAV 等保持。已有模块帮助入口能打开对应章节或明确的预留页面并返回即可。涉及阅读器/资源打包机制时遍历当前已收录章节；仅修改正文或追加素材时，检查受影响章节及素材，无须每次重做全书人工审阅或截图。播放器检查按本次实际引用，不写死旧数量。预留与缺少尚未承诺提供的演示只记录，不阻止打包。

## 5. 构建、封装与身份核验

使用新的构建名，科学缓存和主原生缓存指向最近已验证的同类构建：

```powershell
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/build_v3_local_preview.py --name <新构建名> --lean-qt --compact-onefile --persistent-cache --runtime-bundle-cache output/build-<已验证构建>/snapshot --host-archive-cache output/build-<已验证构建>/host-archive
.\.venv\m14\Scripts\python.exe -B -X utf8 release/finalize_compact.py --package dist/<新构建名> --work output/build-<新构建名>
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/audit_release_contents.py dist/<新构建名>/PhoneticToolbox.exe --output output/validation/<本次验收>/package-audit.json
```

占位符须替换为本次已确认路径。没有可靠缓存时，从已盘点的运行时 stage 重新准备，不猜测其它目录。需要安装版时再执行：

```powershell
.\.venv\m14\Scripts\python.exe -B -X utf8 release/package_compact.py --package dist/<新构建名> --output output/release-staging/<新包装名>
```

`finalize_compact.py` 核对内嵌包大小、SHA、展开预算、共享关系、Qt 身份及 MFA 边界。`package_compact.py` 核对三种分发文件大小、应用 SHA、ZIP CRC 和目录结构。任一步失败，都不能标为验收通过。

紧凑安装包必须使用 `CompactOnefile` 文件白名单和 `PersistentCache`，只携带同一 `PhoneticToolbox.exe` 和 `application.json`。安装时同步准备缓存，准备失败不得悄悄当作安装完成。构建/清理日志保留在开发交付目录，不安装给普通用户。编译安装 QA 时使用同样两项开关，另加 `OwnedQA` 及专用 AppId。QA 生成真实卸载器但关闭产品卸载注册及快捷方式，须实际验卸载清理。

## 6. 成品验收门槛

以下条件必须针对**最终交付字节**完成，修改代码、资源或重新构建后不得直接沿用另一候选包的成功记录：

1. 拷贝单 EXE 到工程外新目录。清除 Python/Conda/开发环境绑定，PATH 只保留 Windows，使用独立的测试用户数据和 TEMP，验证直接启动及正常退出。
2. 运行实际成品 `--verify-distribution`：包内科学探针、MediaPipe 初始化、11 个托管计算任务、产物回读、已有模块帮助的基本打开/返回、声道引擎、录音软件流程、音标和其它客户端工作流通过。说明书内容完备程度按第 4 节宽松处理。MFA 只验证明确排除，不把未测实体设备写成已测。
3. 全部恢复后的主原生文件逐一验 SHA，核对实际共享路径身份。首次改变共享实现或共享映射时，用 `scripts/verify_shared_fallback.py` 对完整真实压缩包验证复制回退和原文件一致性，单元模拟不能代替这项检查。
4. 改动打包方式、依赖或启动恢复机制时，旧、新 EXE 各至少测三轮同条件首次启动，使用新的隔离缓存，交替顺序并记录每轮与中位数。新包另测同一缓存下至少三轮正常退出后的再次启动，以及真实跨构建的组件复用。明确是否隐藏窗口、软件渲染、是否清理操作系统缓存；后台大规模构建时的计时不能混入最终对照。不得用安装预热后的成绩冒充第一次免安装启动。
5. 启动中位数若增加超过 `max(1 秒, 旧版中位数的 5%)`，先复测和解释原因；未经明确接受，不将其标为启动性能合格。该阈值是本项目的回归门槛，不代表统计显著性或其它电脑的保证。
6. 安装到独立测试目录，回读应用 SHA，确认安装阶段完成准备、首次打开复用、卸载后应用及可清理缓存移除，原始数据/工程/设置的摘要保持。安装 QA 使用专用 AppId、`OwnedQA`，不写用户正式快捷方式和卸载注册；正式交付包使用正常产品身份。
7. 免安装和安装两种真实换版均验证辅助进程、跨启动器句柄传递、外部分发 EXE 来源、重新启动、未变组件复用、设置/草稿/工程标记保留、媒体解码及小型临时目录退出清理。较旧版本号若是测试元数据，要明确记录，不能称为所有历史版本升级已验。
8. 缓存必须每次校验实际文件内容，完成后才发布。覆盖损坏、中断、两个进程同时准备、活动实例禁止删除、清理取消、缺空间/权限错误、硬链接复制回退，以及实际设置按钮清理和重开。首次准备/修复/升级须有真实阶段提示。旧缓存不能仅凭版本号、文件长度或修改时间复用。
9. 所有失败保留诊断并查根因。自动验证失败须非零退出并留下日志，避免未处理异常弹窗阻塞 QA。检查脚本与当前协议不一致时修正其依据、重新验证；禁止删断言、放宽数值容差、用成功 mock 或跳过标记制造通过。

默认要求 Windows x64。只有另一台无开发环境电脑或全新 Windows 虚拟机实际运行后，才能声称已验证干净系统；本机工程外、仅 Windows PATH 的测试须单列，不能等同于所有 Windows 电脑保证。

## 7. 证据、清理与交付

- 报告至少包含 EXE/安装包/更新 ZIP 的路径、字节数、SHA-256、源码快照、科学压缩包与 Qt 身份、排除清单、共享与复制结果、实际成品测试、启动每轮数据、安装和换版结果、未验范围。
- 免安装版放 `dist/<构建名>/`，安装版和更新 ZIP 放 `output/release-staging/<包装名>/artifacts/`，验证证据放 `output/validation/` 或明确的工程外 QA 目录。仅需免安装版时不额外生成安装包。
- 旧包清理严格沿用根规则和 `cleanup_old_builds.ps1`：先成功生成新版，再逐目录核对并清理已识别旧输出；活动进程、最新构建缓存、运行时、验证记录、用户数据保留。不得扩展为清理用户其它文件。
- 最终状态区分 `built`、`verified` 和 `published`。构建成功不等于可用性验证成功，本地交付不等于公开发布。用户本次明确要求跳过某些检查时，记录具体豁免、来源和未验项，包只能按实际证据标注。

相关决定：[持久启动缓存](../docs/decisions/ADR-persistent-startup-cache.md)、[同字节依赖共享](../docs/decisions/ADR-cross-archive-dedup.md)、[统一源码与冻结快照](../docs/decisions/ADR-source-entry-and-snapshot.md)。
