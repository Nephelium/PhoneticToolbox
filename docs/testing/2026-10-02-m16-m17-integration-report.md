# M16 / M17 本地模块整合与验证

2026-10-02。井井明确授权两个 Astra / xhigh 子智能体分别实施，root 负责公共工作台和桌面桥接、审查与联合验证。源码阶段完成后，井井追加授权打包本机试用 EXE，成品证据另记。

## 行为与边界

- M16 默认同一声卡两路音频，EGG 必须手工指定；只支持一路输入的设备明确切为单音频。任务可选。原始 PCM、工程、派生处理、恢复和导出均在本机。
- M17 在一个页面切换 IPA / extIPA / VoQS，底部编辑器持续保留。500 个可点表项包含独立符号与明确标注的完整示例，其中 VoQS 基本表项为 56 个，不能将 65 个含示例按钮称为 65 种基本音质。
- VoQS 全部 56 个中文名称按井井提供的 UntPhesoca 译文及双语图核对，出处为 [VoQS：音质符号（2016 中英双语版）](https://zhuanlan.zhihu.com/p/203037479)，2020 unt 译。分类、查询与介绍使用同一数据。原表及 Ball 等（2018）论文出处分别保留。
- 不要求公网服务器、账号或云计算。M16 普通浏览器采集暂不开放。M17 网页取得静态页面和字体后可本地编辑，尚未提供关闭浏览器后离线重开的 PWA 保证。
- 原15个模块编号及分组保持不变；M16 加入分析与采集，M17 加入标注与实验。共享工程来源/草稿不与新本地工程混用。

## 公共整合

录音原生请求使用独立 QWebChannel 队列和锁，科学任务读取不会占用停止录音的通道。M05 和 M16 互斥占用输入；M16 采集期间阻止公共试听及 M15 开始播放。录音状态在顶栏持续显示，切到其他模块仍保留采集。关闭 M16 标签前停止并保存，失败继续保留标签；关闭桌面窗口也有原生保存检查。

音标页独立字体、草稿、撤销历史和键盘处理。桌面接纳本页面生成的 UTF-8 文本下载。非 BMP 字符、组合记号不经整段归一化或服务端传输。

## 已执行验证

所有 Qt 验证用独立测试窗口和临时目录，音频输入/输出明确为合成 PortAudio。没有打开实体麦克风或 EGG。

| 项目 | 实际结果和证据 |
| --- | --- |
| 前端 `npm run typecheck`、`npm test` | 226 项通过，0 失败/跳过。包含原模块回归、M16 默认/导入/快捷键、M17 500 表项精确插入与56中文译名、原15模块编号与共享录音路由。 |
| Qt M16 | 8 组通过，`output/validation/m16/qt-83a9be9cd0864269b627fc7d77d183cf/report.json`。实际桥接、任务快照、采集/停止、按帧编辑、空格停止、派生降噪/EGG 保留、WAV 导出、重录和浅深主题。上一轮稳定浅深截图已人工检查，最后输出/播放回收修复后复跑通过。 |
| Qt M17 | 三表在 1366×768 / 1920×1080 的完整目录、按钮边界、无内容滚动及底部编辑框通过；UTF-8 下载回读、原生文本/符号编辑、撤销与重载通过。最终术语/字体版本证据在下方收口记录。 |
| Qt 联合4组 | 最终 bundle 复跑 `output/validation/m16-m17-integration/qt-75689cf931e84ee4a99d9b626b3219a8/report.json`。默认双音频自由录制、切 M17 输入与空格、隐藏录音标签的取消/停止保存、重新打开同一 take。 |
| 工作台入口检查 | `scripts/start_m01_workbench.py --prepare-only` 返回 ready=true，复用已审阅本地任务库与既有缓存，无 DDL。新模块自身不依赖此任务库。 |
| M16 各层细项 | [M16报告](m16-report.md)：Windows Python、Chrome14组、WSL纯核心11项及60分钟等量加速流式读写。 |
| M17 内容与字体 | [M17报告](m17-report.md)：目录覆盖、中文名称、字体完整性、浏览器交互、失败恢复及许可文件。 |

完整命令（PowerShell，项目根目录）：

```powershell
$env:PYTHONPATH='D:\PhoneticToolbox\PhoneticToolbox_v3\desktop\src;D:\PhoneticToolbox\PhoneticToolbox_v3\backend\src;D:\PhoneticToolbox\PhoneticToolbox_v3\packages\phonetic_core\src'
.venv/m09-ui/Scripts/python.exe -B -X utf8 -m pytest -o addopts='' -p no:cacheprovider packages/phonetic_core/tests/test_recording_core.py desktop/tests/test_m16_recording.py desktop/tests/test_m16_host_integration.py -q
.venv/m09-ui/Scripts/python.exe -B -X utf8 scripts/verify_m16_qt.py
.venv/m09-ui/Scripts/python.exe -B -X utf8 scripts/verify_m17_qt.py --require-single-screen
.venv/m09-ui/Scripts/python.exe -B -X utf8 scripts/verify_m16_m17_integration.py
```

在 `frontend` 中执行 `npm run ui-data`、`npm run ui-data:check`、`npm run typecheck`、`npm test`、`npm run build`。Python根历史配置的 `--cov=phonetic_toolbox` 依赖当前独立环境未安装的 pytest-cov，首次原样调用被参数解析拒绝；定向命令仅去除此覆盖率插件参数，测试断言及用例均保留，没有修改项目配置或安装依赖。

## 不扩大验证结论

实体声卡、EGG接线、硬件增益/驱动处理、真实60分钟墙钟录制、实体DPI/多屏及 macOS/Linux 原生采集尚未验证。加速60分钟数据量逐字节回读通过不等于真实硬件持续录制通过。降噪不能称为 Adobe Audition 算法或自然语料保真已验。M16 原始文件不受剪辑与降噪覆盖。

M17 多字符圈围使用已标明的文本替代形式 `⟅…⟆`，没有伪造新 Unicode 编码。字体只确保模块内呈现；复制到其他应用后由目标应用字体决定。原始第三方 PDF / 图版不随产品复制分发。

Qt offscreen 日志含 GLES context fallback；界面与截图经实际渲染检验，不能据此声称实体 GPU / DPI 已验证。早期截图存在 compositor 前一帧滞留，验证器改为等待 CSS 状态并两次抓图。M17 首次重载检查误读导航前 DOM，修正为等待真实 `loadFinished` 后再验证，未放松文件回读断言。

源码入口：[Start-M16-M17-Workbench.ps1](../../scripts/Start-M16-M17-Workbench.ps1)。保留旧 EXE、V2、旧录音/原 PDF/CIN，不执行现存库迁移、push、公开发布、环境升级。保留同期 M05/M06/M10 等任务差异。后续新增独立命名的本机试用 EXE。

## 最终收口记录

- `npm run ui-data` / `ui-data:check` / `build` 全通过，345 条来源；M16新增2条、M17新增8条，复用依赖追加模块归属，保留既有335条及同期M06来源修订。Vite保留既有大chunk建议，未调低检查标准或隐藏警告。
- Windows Python 最终 **39 项通过**：M16核心/工程34项与共享宿主5项。播放初始化、abort/close失败保留设备引用并可重试；未完成WAV保留 `.wav.partial`，正式命名发生在回读核验后。
- M17 最终 **4组通过，包含6个单屏布局**：`output/validation/m17/qt-20261002-155652-b8b4c5/report.json`。已包含指定VoQS译名、最后字体与行高、IPA咽/声门半格阴影、存储quota异常捕获、文本导出和总来源面板的作者/原文链接。1366×768指定译名VoQS截图人工复核。
- 最终 M16 / 联合 Qt 再次通过，路径如表格。之后按井井反馈只精简长示例及长选区的码位显示，并将两份 OFL 许可原文嵌入帮助。定向5项检查通过：`output/playwright/m17/display-2026-10-02T08-04-42-249Z/report.json`；符号数据与实际插入不变。最终 `npm run typecheck` / `build` 通过，实际构建产物核对包含两份许可文本。
- M17 7000 UTF-16单元/1000行保留、追加与撤销已通过Chrome。105000单元/15000短行压力输入在原生裸textarea也超时，不能称为该规模通过；未借降低测试规模冒称修复该浏览器边界。

## 追加 EXE 授权后的修正

首次成品检查发现 M16 数字范围输入复用了拖动方向归一化：空选区先输入起点10再输入终点20会被变为0–20。已将数字输入改为保留当前输入端点，越过另一端时先折叠为该点；波形反向拖选继续沿用原排序。新增回归保护10–20精确删除，最终全套前端 **227项通过**，typecheck/build通过。

专项真实Qt复跑8组全部通过，包含删除后样本精确比对、原始SHA-256不变、实际spawn降噪与FLOAT WAV逐样本回读，以及两分辨率三表布局。证据：`output/validation/m16-m17-integration/pack-source-7ccfbfa9f78442748d92a4701f1d268f/report.json`。成品与其后续验证单列，不把此源码Qt证据替代成品验收。

后续先打开15个既有模块再测两项新增模块，进一步验证了公共字体加载完成后的布局。仅将音标页工具栏底部留白由6px收为2px，1366×768三表均为428px内容/视窗且全部入口在界内，1920×1080均为740px，无裁剪或提高溢出容差。录音关闭等待在途状态读取完成，并暂停下一轮读取后再停止保存，保留5秒超时失败保护。400ms状态读取延迟注入仍能一次停止保存并关闭。最终源码Qt8组含六布局全部通过：`output/validation/m16-m17-integration/pack-source-4751a0dfd50344a5a4fb0835fcfa6f44/report.json`。

井井截图中的成品测试弹窗经追溯来自M10 `native_platform_unavailable`：当前进程 `platform.machine()` 返回空值。已补Windows `GetSystemInfo`在空值时识别进程架构，实测x86_64且三DLL解析成功；显式传入的不支持架构仍拒绝。原生子进程初始化异常返回既有私有协议失败并写入stderr日志，避免落到PyInstaller未捕获异常弹窗。6项原生资源与失败协议回归通过，科学算法和系统环境未变。早期仅凭截图归为图形环境的解释已纠正。


最终追加成品已通过：[R2 EXE交付与验证](2026-10-02-m16-m17-exe-report.md)，原始及R1候选不得作为最终交付。
