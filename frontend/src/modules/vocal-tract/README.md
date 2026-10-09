# 生理参数合成 · M10

本目录是 Windows 本机声道工作台的公共界面适配层，模块名称为生理参数合成，持久模块 ID 为 `M10`，见[模块注册表](../../app/registry.ts)。**模块准备大改，操作正文按作者要求暂缓编写。** [说明书章节源文件](../../../../manual/chapters/m10.json)保留稳定 ID、标题和 `ptb-manual-chapter/1` schema，`body.content` 保持空数组。详细操作将在改版定稿后写入该章；本页记录当前工程与证据边界。

## 功能概览

当前模型加厚后卷舌尖并保留圆顺过渡，舌叶圆点绑定实际表面。前部气腔按组织剖面补足重建细缝，牙齿保持刚性，软腭背侧使用平顺连接。默认使用 257 个纵向截面、横向 96 点，几何精细化和采样变化可以改变声管面积与谱峰，持久文件格式保持兼容。

现有嵌入工作台连接 VocalTractLab 2.4 原生引擎，提供二维与三维构形观察、器官参数控制、浊声/清声/耳语近似声源、面积与传递函数显示、关键帧、F0 控制和同步视频保存。界面与完整操作流程需在后续改版时重新核对。

构形和交互包括：

- 声门高低控制点在二维、三维常驻，不要求先选软腭。原生牙齿保持刚性，下牙仅随下颌运动；重建舌腹和封盖避让牙齿体积，隐藏牙齿仍保留约束。软腭只描绘暴露轮廓，头部组织内部封闭线不再显示为突起。
- 舌叶可下压、抬高的舌尖可后缩，保留硬组织及口底限制。后卷采用后上方切线和前侧回接，避免整圆绕回及舌腹自交。参数模型可以形成后卷构形，具体语音类别仍需独立声学及听辨依据。

- 小舌本体沿原生表面保持固定形状和尺寸，接触舌面时整体止抵，软腭附着部分过渡。剖面融入头部背景，选中时保留细边界和拖动点。显示舌面保留原生三角形，口鼻连接固定前后对应关系，参考截面由口腔正中连续过渡到鼻腔旁中线 5 mm。原鼻腔和头壳资源不缩放重写。
- 舌尖可以前伸出口，舌叶放宽过严的构造限制，拖动从引擎实际位置开始。保留模型中的硬腭、门齿、口底和后咽壁约束。
- 独立声门高低默认为 0，可在 −1 至 1 cm 模型范围内调节。舌骨与会厌附着点固定，几何、声道长度、截面、传递函数、试听和动作合成使用同一偏移。
- 三维默认显示半侧组织及口鼻气腔，显式勾选显示完整模型后解除裁切。选择会随本机视图偏好保存。
- 面积图的实线为几何截面积，虚线为合成使用的声学修正面积。正中接触且有侧通路时说明通路仍开；全截面闭塞显示 0，声学求解器保留数值下限。

- 顶部恢复 `/a/` 按钮、来源弹窗中的布局标签与下拉框已移除。底部 `/a/` 预设、器官局部重置及撤销/重做保留。模型与来源按钮使用公共按钮尺寸。
- 布局仍读取既有 `two`、`three`、`auto` 偏好，默认自适应窗口；本文不提供已经移除的布局选择步骤。
- 保存/加载构形弹窗直接复用 M17 的完整扩展辅音目录，包含 14 列、13 类、186 个可点击辅音表项，另有 28 个元音按钮、20 个其他辅音按钮与自定义区。186 包含组合示例，不代表 186 个独立基本字母；总计为 218 个不同符号、234 个按钮位置。
- 音标表为当前构形提供名称。保存音标不自动生成对应标准发音，也不构成听辨通过的证据。

## 输入与输出

| 类型 | 当前工程边界 |
| --- | --- |
| 前端输入 | 桌面 `vocalRequest` 能力、公共主题/字体/按钮设置、M17 音标目录；不接收任意服务器文件路径 |
| 构形库 | 本机 profile 的 `presets.json`，`version: 1`；保存器官参数、声源、F0 及可选 `larynx_height`，缺失时按 0 读取，既有名称和记录继续兼容 |
| 关键帧缓存 | 本机 profile 的 `keyframes.json`，当前写入 `version: 2`，包含帧序列与 F0 曲线；由桌面 profile 适配器读取 |
| 关键帧交换文件 | `.ptb-vocal.json`，`format: phonetic-toolbox-vocal-tract`；旧范围且无独立声门偏移时保持 `version: 1 / VTL-2.4-JD2`，扩展舌位或非零声门偏移使用 `version: 2 / VTL-2.4-JD2-M10-3`，防止旧程序静默忽略扩展；模型与参数顺序必须匹配 |
| 视频与音频 | 同步动作保存为 `.webm`；合成与试听由桌面原生运行时及媒体适配器处理，Vue 层不实现科学算法 |

关键帧交换文件由[纯核心文档校验](../../../../packages/phonetic_core/src/phonetic_core/vocal_tract/document.py)检查，再由[桌面文件适配器](../../../../desktop/src/ptb_desktop/vocal_tract/files.py)执行选择与保存，当前文件容量上限为 8,000,000 字节。用户取消选择返回取消状态；导入模型、参数顺序或范围不符时明确拒绝，不能静默夹紧后替换原构形。构形库及缓存由[ProfileStore](../../../../desktop/src/ptb_desktop/vocal_tract/profile.py)维护，与只读引擎资源分开。

## 快速流程

详细教程暂缓，未来入口为[生理参数合成操作章节](../../../../manual/chapters/m10.json)。当前开发工作按下述源码分层与验证报告核对，不将旧手册的控件、默认值或流程直接填入空章。

## 格式与限制

- 当前页面要求 Windows 桌面 `vocalRequest` 能力；普通浏览器显示明确不可用提示。
- 当前序列文档绑定 VTL 2.4 的 JD2 模型。头壳为外观参考，平均鼻腔为独立研究数据的显示配准；二者均不能视为 JD2 同一个体的 MRI，鼻腔显示开关与 VTL 声学鼻腔管道需分别解释。
- 面积与传递函数是当前模型的计算结果。F0 曲线是合成控制轨迹，不能称为输出音频重新测量得到的 F0。清声/耳语近似无周期振动部分也不能据此赋予实测基频。
- VTL 2.3 手册仅作补充参考，具体参数与模型行为以当前 2.4 引擎、源码和对应证据为准。
- 持久构形键、既有文件和来源保留；近期界面检查不等于实体音频、跨平台或完整成品验收。

## 源码结构

| 文件 | 职责 |
| --- | --- |
| [VocalTractPage.vue](VocalTractPage.vue) | 装载同源 iframe，核对消息来源，转交桌面请求，同步主题/字体/按钮与音标表，连接未保存状态、引用和关闭事件；失活发送 `deactivate`，卸载发送 `shutdown` |
| [ipa-chart.ts](ipa-chart.ts)、[preset-chart.ts](preset-chart.ts) | 从 [M17 目录](../ipa-plus/catalog.ts)构造扩展辅音、元音、其他辅音和中文名称，保留 Unicode 原文 |
| [index.html](../../../public/vocal-tract/index.html)、[app.js](../../../public/vocal-tract/app.js) | 嵌入工作台布局、参数、视图、声源及关键帧的界面编排 |
| [platform.js](../../../public/vocal-tract/platform.js) | iframe 与父页面的 `m10-request` / `m10-response` 消息端口，不自行开 socket 或选择原生路径 |
| [presets.js](../../../public/vocal-tract/presets.js)、[layout.js](../../../public/vocal-tract/layout.js) | 构形命名/保存加载及既有布局偏好读取 |
| [桌面平台接口](../../platform/desktop.ts)、[Qt 宿主](../../../../desktop/src/ptb_desktop/host.py) | 经 QWebChannel 调用本机桥，原生文件选择与视频写入由宿主处理 |
| [client.py](../../../../desktop/src/ptb_desktop/vocal_tract/client.py)、[runtime.py](../../../../desktop/src/ptb_desktop/vocal_tract/runtime.py) | 每个拥有窗口的原生进程、请求归属、受限启动/关闭、profile、试听与动画缓存；只管理本应用拥有的进程 |
| [科学核心](../../../../packages/phonetic_core/src/phonetic_core/vocal_tract/) | VTL 原生接口、声源、轨迹、合成与文档校验；不依赖 Qt 文件对话框或数据库 |
| [原生资源](../../../../resources/vocal_tract/) | VTL 2.4 DLL、speaker、几何桥接、来源锁、许可和重建材料 |

调用方向为嵌入界面 → `VocalTractPage` → 桌面能力 → 原生运行时与 `phonetic_core`。M10 当前的窗口与原生音频进程路径没有自动成为网页远程计算能力。当前原生几何版本 `m10/3` 与运行时 API 握手 `m10/1` 是不同标识，不应合并为同一版本字段。整体职责见[前端架构](../../../ARCHITECTURE.md)和[总架构](../../../../ARCHITECTURE.md)。

## 开发与定向验证

从[统一源码入口](../../../../docs/development/source-entry.md)启动，[Start-M10-Workbench.ps1](../../../../scripts/Start-M10-Workbench.ps1)是直接打开 M10 的快捷入口。主环境绑定当前 core/backend/desktop 源码，不依赖早期同步到环境中的项目副本；启动器不自动构建、安装或迁移数据库。

```powershell
npm --prefix frontend run build
& './scripts/Start-M10-Workbench.ps1'
```

[m10_recording_entry.py](../../../../scripts/m10_recording_entry.py)保留录制/冻结程序的 M10 专用入口，在既有运行环境中装载 `frontend/dist` 和显式声道资源。它不代表另一套前端或科学算法，也不应绕过项目的环境配置直接复制到任意机器运行。

当前几何和交互的证据入口为 [R14 报告](../../../../docs/testing/2026-10-07-m10-r14-report.md)，较早行为按需查 [R13](../../../../docs/testing/2026-10-07-m10-r13-report.md)、[R12](../../../../docs/testing/2026-10-07-m10-r12-report.md)及台账。录制协议另见[录制功能报告](../../../../docs/testing/m10-recording-features-report.md)。不在 README 按轮次复制验收流水。

按改动选择类型检查、前端测试、构建、`ui-data:check` 与对应 Qt/原生检查。实体音频、物理 DPI/DWM、长期运行、Linux GUI 和当前冻结成品完整科研流程需要独立证据，模型构形也不等于医学真实性或听辨通过。

## 方法与来源

来源、版本、修改关系和发行状态以[统一来源登记](../../../../third_party/source-registry.json)为准，[来源映射](../../../../docs/modules/evidence/M10-source-map.md)保留历史功能对应，[随资源声明](../../../../resources/vocal_tract/THIRD_PARTY_NOTICES.md)记录原生材料与几何修改。下表按实际对象区分，不将一种许可套用于整个模块。

| 对象与来源 ID | 当前使用与许可范围 |
| --- | --- |
| `SRC-VTL`、`PROJECT-M10` | Peter Birkholz 与贡献者的 VTL API 2.4，原生代码 GPL-3.0-or-later；本项目从原生代码派生的几何桥接/适配继续按 GPL-3.0-or-later 处理，界面与平台集成单独说明。原版 API DLL 及同字节分析副本用于状态隔离。公开分发需另核对对应源码、修改声明和许可证，当前登记保留发行审阅状态 |
| `DOC-VTL24`、`DOC-VTL23`、`REF-VTL2006` | 2.4 官方手册、2.3 补充手册与 Birkholz、Jackèl、Kröger 的三维声道模型论文分别登记；仅提供书目/官方链接，原 PDF 再分发许可未因此确立 |
| `SRC-THREE` | Three.js / OrbitControls 0.180.0、r180，MIT；保留随本地脚本提供的 [THREE-LICENSE.txt](../../../public/vocal-tract/vendor/THREE-LICENSE.txt) |
| `ASSET-HEAD` | byzmod3d 的 FACE2 头壳，CC0-1.0；转换、三角化和显示配准后的外观参考，不参与声学边界计算 |
| `ASSET-NASAL` | Brüning 等人的平均健康鼻腔数据 v4，CC BY 4.0；登记已补官方 v4 元数据与原网格摘要对应证据，保留作者、DOI、许可与坐标变换说明。平均几何与 JD2 个体需明确区分 |
| `ASSET-DOULOS` | Doulos SIL 7.000，SIL Open Font License 1.1，公共 IPA 显示使用；字体条款与模型/代码条款分别保留 |
| `M17-IPA-CHART` | M17 共享交互音标表，登记为 2026 重印的 2015/2005 内容，CC BY-SA 4.0；自绘交互表保留归属和相同许可，不随包复制原 PDF |
| `M17-IPA-CHART-ZH-2007`、`M17-IPA-HANDBOOK-JIANG-2008` | 中文名称与短解释的参考书目，原图、原书、扫描页和私有路径不随包分发 |
| `SRC-VTLWRAPPER` | VocalTractLab-Python 仅作参考，未安装、未导入，也未声称捆绑其代码 |
| `REF-WEBCODECS`、`REF-WEBM`、`REF-FONT-RENDERING` | 编码、容器和字体 API/规范参考；实际编码来自既有 Qt WebEngine，规范引用不等于外部实现代码再分发许可 |
| `REF-M10-JORDAN-2017-MRI`、`REF-M10-OLIVEIRA-2012-MRI` | 腭咽闭合与欧洲葡萄牙语鼻元音 MRI 的书目/外链参考，论文版权归权利人；未携带原图、数据或 PDF，不能据此证明当前模型复现了论文或对应同一个体 |

Qt/PyQt 与实际运行时的其他代码、原生库及传递依赖另按统一登记和发行材料审计。本页不作整个组合发行物许可通过的声明。后续改版需要同步更新来源使用位置、本页、操作正文及对应验证记录。
