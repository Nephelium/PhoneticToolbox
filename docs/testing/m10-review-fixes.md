# 生理参数合成审查修复验收

2026-10-22。状态：**verified，限当前Windows原生/Qt、独立Chrome真实Wasm及WSL原生核心**。macOS共用源码/构建输入同步，实机和最终包单列。

## 第三轮修复与验证（2026-10-22）

| 问题 | 修复 | 当前证据 |
| --- | --- | --- |
| 自动保存失败阻断导出 | 等待保存队列完成后，独立导出内存中完整草稿；保存失败不阻断导出，不修改原保存状态；导出本身的失败和取消仍如实处理 | Node成功/取消/导出失败三分支；Qt受控磁盘保存失败后实际写出文件；Chrome真实IndexedDB事务中止后下载完整草稿，原序列与持久化字节保持 |
| 舌侧缘选择仅改变滑块 | 前/中/后按钮复用selectSection，同时更新截面、上下轮廓、查看器与位置文字；声源快捷更新后编辑器和查看器持有不同对象时也同步 | Node分离状态对象；Qt三位置的滑块/文字/Canvas均不同；Chrome查看器截面、轮廓与选中截面的原数据逐字段一致 |
| 3秒统一时长生成非法浊塞音 | 闭塞上限统一为2999ms，释放至少1ms，三段合计不超过原单帧3秒；保留负VOT零送气及默认计算，不裁短时长 | JD2/W02 × b/d/ɡ/t × 0.05/0.2/2.008/2.1/3秒，文档往返不变；长闭塞、前浊起点、PCM长度与越界拒绝；Qt/Chrome实际IPA输入与2990ms编辑 |

本次扩展的是闭塞输入范围，原生ABI、文件版本、默认时长、负VOT公式和采样率不变。3秒浊塞音为2992ms闭塞、8ms释放、0ms送气，VOT仍为−30ms。旧输入保持原含义，新范围需使用本次更新后的读取器。

- 修复前：新增Node5项全部失败；原生长塞音和范围3项失败（40项其余参数组合未在红灯阶段运行）。日志：`output/validation/m10-review-20261022/third-red-d4453fcd73874a889cf07ff9f659c4a3/pytest.log`。
- Windows：新增核心43项与之前审查/时序合计109项先通过。随后完整 `tests/parity`、`backend/tests`、`desktop/tests` 下 `test_vocal*.py`/`test_m10*.py` **271项通过**，136.29秒，2条既有Starlette/AnyIO弃用警告。当前源码configure、`tests/pytest.ini`、禁用pytest缓存、basetemp置于自有scratch。日志：`output/validation/m10-review-20261022/third-all-native-76003d5aad9d4235b7261b3bcde91a9c/pytest.log`。
- 前端：M10/vocal **65项通过**，新增5项覆盖导出三分支、分离状态对象的截面选择及3秒闭塞边界。`typecheck`、`contracts:check`、`ui-data:check`、`build:workbench`通过，Vite保留既有大包提示。Node报告：`output/validation/m10-review-20261022/third-frontend-35f8951a8e9e4fcd96007a1c1c3ecb91/node.log`；最后的分离对象断言在收尾检查再次运行。
- Windows Qt：**11组通过**，独立profile和文件选择适配，未开放实体音频设备。报告：`output/validation/m10-shared-20261018/qt-96daaa8a24b249bb9f95fe3582cf964d/report.json`。
- 独立Chrome真实Wasm：**16组通过**，实际IndexedDB故障、文件下载、截面选择、3秒IPA姿态、6秒/288000采样合成及2990ms闭塞编辑通过，原有AudioContext和MP4回归保留。报告：`output/validation/m10-wasm-20261018/browser-07a99128cdd14a509cf7dc8643cdfaff/review-fixes.json`。浅深截图保留，深色长时序及Qt截面截图已查看。
- WSL NInfer：既有`/home/ninfer/ptb-m06-20260927/bin/python`/g++13.3，在自有scratch重建锁定R21，当前源码**43项通过**，8.58秒。日志/编译摘要：`output/validation/m10-review-20261022/third-linux-fc5037c269fe4808b906ab9eb076f876/`，自有候选库已清理。
- 同源Wasm Python/对应源码包和74项资源登记已刷新。Windows/macOS仍读共享前端与核心；没有本轮Mac实机/Safari、实体音频听感和最终EXE/Mac包证据，未公开部署或push。

Chrome首次检查发现source/update后snapshot与viewer.state分离，原selectSection仅同步section会保留旧轮廓，现同时同步upper/lower并加入Node回归。第二次中断来自测试选择器同时匹配爆破时序与嵌套说明summary，已限定直接子元素；第三次全16组通过。失败报告和各运行cleanup.json保留，没有放宽数值断言或跳过失败项。

收尾检查：最后一版65项前端测试再次通过，74项资源登记幂等、架构/文档/源码入口CheckOnly/定向diff/语法检查均通过；Wasm运行包与对应源码包逐字节匹配当前源。10份本轮运行及收尾自身共11份scratch回执均为deleted，当前Wasm构建输入保留。记录：`output/validation/m10-review-20261022/third-final-940b23b9b1dd4d0d8bd1457d027ed689/report.json`。

## 前两轮修改与验收依据

| 问题 | 实现 | 核验 |
| --- | --- | --- |
| 区段特征提前约20ms | 音频块末端取特征边界左值，边界处零长度更新右值；特征在最近采样点解析，试听补偿节点同时保留左右值 | JD2/W02，F0/气压/唇宽在0.1秒前的原始及试听PCM逐采样相同；非采样网格边界及重叠固定值按半开区间切换 |
| 手动舌根模式残留 | 临时生成在finally恢复原模式，持续发声入口同步Runtime当前模式，播放结束接收最后姿态的模式 | 双模型、原模式自动/手动、正常/取消/受控原生异常均恢复；真实Wasm ABI重复核对 |
| W02元音保留清声 | builtinPose统一完整起点解析，用于快捷加载、构型目录和IPA目录，保留作者默认姿态优先级 | 双模型从清声切回a时恢复周期声源和120/200Hz，返回数据与原资源不共享可变数组；Qt/Chrome实际按钮通过 |
| 声管变化后补偿过期 | pose及source/update按当前声管和声源开度取得补偿，复用既有Engine参考缓存 | 同240Hz从a到i得到当前声管参考值。JD2为0.658401023749307，W02为1.0614485536484648；Wasm分别为0.6584010237493088、1.0614485470669592 |
| 视频F0未含最终调整 | 核心最终采样器生成带事件左右值的timeline，动画缓存返回；视频曲线与文字使用最终轨迹/画面 | 原姿态200Hz、全局固定240Hz时，实际MP4六帧均绘制F0 240Hz，独立PyAV/FFmpeg解码1280×720六帧，保留首帧图像；区段280Hz/静音/清声也有回归 |

继续审查发现的三处已一并修复：

| 问题 | 实现 | 核验 |
| --- | --- | --- |
| 中途停止丢失当前姿态 | 停止与自然完成共用完整姿态接收，包含舌根、源、F0、声管和补偿；Web停止上报最终时钟，Wasm检查点接收新姿态，UI应用核心返回的姿态 | 双模型三种停止入口、结束检查点；实际Chrome停止后重新捕获的参数、舌根、唇宽/唇形、舌形、喉位及声源与核心逐字段相同；服务器原生子进程三入口通过 |
| 导入保存失败提前应用抖动 | 先提交序列，再应用抖动；保存失败不改源，后续应用失败恢复完整原序列和原设置。恢复失败明确提示重新核对，不声称已保留 | 前端覆盖配额、应用后失败、恢复失败和重试；Qt真实文件/原生IPC注入保存与应用失败；Chrome真实IndexedDB事务中止后持久化字节、当前序列和控件保持，重试成功 |
| keep_vowel预处理残留舌根模式 | session.prepare_sequence在进入预处理前记录原模式，整个准备在finally恢复；取消准备不接收旧动画缓存 | JD2/W02、两种原模式、正常/缓存/预处理异常/渲染异常/取消，Windows及真实Wasm/WSL同一状态检查通过 |

本次不修改原生C++、ABI、R21几何身份、已有科研默认值、文档版本和已保存参数。Windows/macOS读取同一Python包及静态前端，浏览器运行包封装同一份核心；Wasm二进制复用已锁定R21，仅刷新Python和对应源码包。

## 实际验证

- 双模型新增回归现共56项，包含前批26项和本轮30项。修复前keep_vowel有效用例16失败、4通过；修正停止用例重复frame_id后，用修复前Runtime运行6项均因缺少停止姿态失败。各失败原始日志保留，不把输入错误计为产品缺陷。
- 完整M10 parity/backend/desktop：225通过、2条既有Starlette/AnyIO弃用警告，110.25秒。之后追加服务器原生子进程三入口回归3通过、6.51秒，合计228项去重用例通过。使用workbench_source.configure和v3的tests/pytest.ini；日志：`output/validation/m10-review-20261022/followup-all-native-c9db54de3da142eea2e0ffce4ef351fc/pytest.log`、`output/validation/m10-review-20261022/followup-server-6483be0e9cef472dbf5fd78eb8766c84/pytest.log`。
- 全部M10/vocal前端：`node --test tests/m10*.test.ts tests/vocal*.test.ts`，60通过。typecheck、contracts:check、ui-data:check及build:workbench通过，保留既有Vite大包提示。
- 实际Windows Qt：8组通过，含前批7组及真实文件导入的保存失败、源应用失败恢复和成功重试。报告：`output/validation/m10-shared-20261018/qt-d62a6d76b67a4adeabb81291762fe67d/report.json`。Qt静音锁定，不宣称实体音频播放通过。
- 独立Chrome真实Wasm：13组通过，含双模型特征前4800采样不变、所有预处理退出恢复、最终时钟/停止、真实IndexedDB配额事务中止/重试、停止后重新捕获姿态和实际MP4六帧独立解码。报告：`output/validation/m10-wasm-20261018/browser-f6a626e08efb4bdda4f647afaeda4b1d/review-fixes.json`。最终深色界面及解码video-frame.png已查看，曲线/字幕为240Hz，无页面异常。
- Start-Research-Workbench.ps1 -Module M10 -CheckOnly通过，业务包指向当前工作树，没有重启用户已有源码窗口。
- WSL NInfer复用已有科学环境与g++13.3，在隔离scratch重建锁定R21，56项回归通过（16.47秒），同一双模型状态检查含keep_vowel全部退出及停止姿态通过。日志：`output/validation/m10-review-20261022/followup-linux-640f361acd0b4a7d9cb7f97c209def18/linux.log`，同目录build-report.json记录构建及SHA，输入/候选库已清理。不宣称不同编译器PCM逐采样相同，不代替Linux桌面/HTTP验收。
- 同源Wasm包已刷新Python及对应源码包，锁与74项M10资源登记同步并幂等。架构零错误，文档2410文件/448来源/176任务零错误，历史快照既有断链另列。7个Python、3个Node文件语法及定向diff通过，运行包31条记录和对应源码包逐字节匹配当前源。10份自有运行cleanup回执及最终检查自身回执均为deleted，共11份。报告：`output/validation/m10-review-20261022/followup-final-check-9c3e1111ad024b439d24fa76f82dfe7a/report.json`。

运行入口（PowerShell，工程根目录）：

```powershell
.venv/m14/Scripts/python.exe -E -s -B -X utf8 scripts/verify_m10_shared_qt.py
.venv/m14/Scripts/python.exe -E -s -B -X utf8 scripts/verify_m10_wasm.py --work output/validation/m10-smoothing-20261018 --science output/validation/browser-compute-20261017/browser-science --frontend frontend/dist --vocal-runtime output/validation/m10-review-20261022/wasm/vocal-wasm --review-fixes --port 18788
```

## 失败定位与收尾

完整测试最初未调用源码入口configure，子进程找不到当前业务包及M05编码运行时；跨测试目录又选到继承v2的pyproject配置。固定当前源码绑定和v3配置后195项全部通过，没有修改断言或跳过测试。

新Chrome脚本首轮在iframe未有body时取dataset；后续清声断言误把JD2作者保存t姿态当作清声，且声源快速更新时查看器几何缓存未刷新。改用明确的声音控件操作与当前控件/输出PCM核对，不改变作者姿态。脚本直接写pose引起的修订号冲突也已改为实际UI操作。最终无过期声源提示。

HTMLVideoElement的首帧Canvas读取得到黑图，不能作为视频图像通过证据。最终改用既有M05环境中的PyAV/FFmpeg独立解码同一下载MP4，断言6帧、尺寸及非黑像素，并保留首帧PNG供人工检查。

本轮Chrome新增停止测试最初误将连续F0与1Hz步长滑块原值比较，随后又将默认源null与界面展开后的voiced字典作原始对象比较。测试现按既有整数显示规则断言F0，并给输入显式完整声源，停止后重新捕获的其余字段保持精确相等断言。产品计算与容差未改。

全部本轮测试通过verification_run在自有进程停止后清理scratch，保留日志、摘要和必要截图，最终回执以各运行目录cleanup.json为准。当前Wasm运行包保存在`output/validation/m10-review-20261022/wasm/vocal-wasm`供后续构建使用，不列入临时删除。

## 平台与交付边界

本轮完成共用源码与本机网页候选，不公开部署或push，不重建最终EXE/macOS应用。当前没有可用Mac实机，macOS原生R21构建与Safari/实体音频设备未验；既有Mac R20库不能当作R21通过证据。Qt使用静音测试模式，Chrome实际通过AudioContext和WebCodecs链路，不宣称实体设备听感通过。
