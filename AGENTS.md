# PhoneticToolbox v3 — Agent 工作规则

2026-10-04 M06-R4：井井授权按自然度核查实施。可选 WORLD/PSOLA、Harvest F0、原音高/编辑音高、同源哈希关联及四件套保存完成，见[报告](docs/testing/2026-10-04-m06-r4-report.md)与[ADR](docs/decisions/ADR-M06-R4.md)。限定Windows开发态verified：268前端、41核心、112任务/契约、Chrome15组/12新增布局、实际Qt5组/12新增布局及两次原生四文件回读通过，WSL41纯核心。Klatt保留；新计算m06-world/1、m06-psola/1，自然提取m06-natural-extract/1。仅项目m09-ui和既有WSL M06环境添加锁定PyWORLD0.3.5，WSL另加Cython3.1.5，无全局安装。合成输入限定，真实听辨/硬件/DPI/Linux正式任务GUI未验，GlottDNN仍planned。入口Start-M06-Workbench，旧EXE未打，无DDL/push/发布/删除用户文件，保留同期改动。

2026-10-04 M03-R5：井井授权 REAPER F0 和 Praat / REAPER 30–800Hz。限定Windows开发态verified，见[报告](docs/testing/2026-10-04-m03-r5-report.md)和[ADR](docs/decisions/ADR-M03-R5.md)。三开关、原生有界REAPER/缓存、批次与CSV/PNG、audio-f0/2来源元数据完成；历史legacy/1保留75–600Hz及旧幂等哈希，GCI/CQ/IF未改。266前端、56科学、21原宿主+3原生宿主、Chrome5组/6布局、实际Qt5组/2尺寸通过，WSL5纯核心含注入端口不等于Linux原生REAPER。原录音/引擎哈希不变。入口Start-M03-Workbench，旧EXE未打、硬件/DPI/Linux完整链未验，无DDL/push/发布/现存用户文件删除/环境安装，保留同期改动。

2026-10-04 M01-F0：井井要求 REAPER 默认30–800Hz，实际原值60–880。共享F0设置/生成契约已更新，旧草稿与显式历史值保留，原V2审计/core独立默认不重写。266前端、84后端/契约、Chrome三组、配置传递与WSL三静态哈希通过，见[报告](docs/testing/2026-10-04-m01-f0-default-report.md)。限定默认配置/Windows开发态，嘎裂自然语料检出率未验，未重打EXE/push/DDL/发布/删除/环境安装，保留同期差异。

2026-10-04 P19-R4：井井要求设置中允许主题/自定义波形线色。限定Windows开发态verified，见[报告](docs/testing/2026-10-04-p19-r4-waveform-color-report.md)。默认蓝色/主题色/色盘与HEX、本机即时记忆完成；公共/录音/EGG音频/M05偏移/M10监视接入，其他科研曲线色义保留。M02整幅/M03逆滤波前端PNG跟随显式选择，历史/后端结果不重写。266前端、Chrome5组含29配色×2模式/实际组件3×2/两PNG300dpi与精确色回读、实际Qt9记录通过；WSL4静态哈希一致，实体DPI/硬件/LinuxGUI未验。全库仍8条旧EXE缺链；未打EXE/DDL/push/发布/删除/环境安装，保留同期差异。

2026-10-04 M01-R3：井井要求默认同音频目录、独立TextGrid/唇形选择、去任务前缀与两格式保存。限定Windows开发态verified，见[报告](docs/testing/2026-10-04-m01-r3-report.md)。两目录默认跟随/独立覆盖/重置、当前目录批量关联完成；用户保存XLSX/SQLite，切分另含WAV，JSON来源与同源父结果保留于内部。冲突整组末尾(2)等数字且重复复用，旧文件不覆盖。266前端、25桌面、Chrome5组/5布局、实际Qt4组/3布局含400时间点两格式/内部JSON逐值与真实任务切分及M02绘图通过；WSL4静态哈希一致。合成输入限定，硬件/DPI/Linux任务GUI未验；源码入口Start-M01-M02-Workbench，旧EXE未打，无DDL/push/发布/删除/环境安装，保留同期差异。全库仍8条旧EXE缺链。

2026-10-04 M06-R3：井井要求窗长同排、完整底栏播放、AV/AH预设、元音按钮顺序及F0切换/自然度核查。限定Windows开发态verified，见[报告](docs/testing/2026-10-04-m06-r3-report.md)。CC/AC/原生REAPER、真实时间对齐、Shimmer百分数转比例完成，新提取记m06-extract/2，klatt/2与旧文件保留；M01/M06差异及WORLD/PSOLA候选见[核查](docs/references/m06-r3-resynthesis-audit.md)。264前端、Windows24核心/13任务、Chrome11功能组含129旧/8新布局、实际Qt4组含32旧/2新布局通过，WSL24纯核心。修复窄窗底栏与图表重叠；GUI后追加Shimmer纯核心修正已重新验证核心/任务，未重复GUI听辨。合成输入限定，自然语料听辨/硬件/DPI/LinuxGUI未验。入口Start-M06-Workbench，无EXE/DDL/push/发布/删除/环境安装，保留同期差异；全库仍8条旧EXE缺链。

2026-10-04 P19-R3：井井授权全局按钮轻微光效/阴影、重要操作纯色高亮及设置开关。限定Windows开发态verified，见[报告](docs/testing/2026-10-04-p19-r3-buttons-report.md)。17模块梳理，20文件补92个primary标记；默认按重要性高亮且特效开启，可选全部高亮/全部普通、独立取消特效，本机即时记忆。声道iframe共用CSS并同步偏好，普通模式保留选择/焦点/持续发声active提示。264前端、Chrome16页×12组合/29配色×6组合/三布局、实际Qt21记录通过，WSL4项仅静态哈希一致。未改科研算法或操作事件，未打EXE/DDL/push/发布/删除/环境安装，保留同期其他改动；实体DPI/硬件/LinuxGUI未验。

2026-10-04 M05-R3：井井授权三模式/选择保存/右栏视频/同步回放/偏移弹窗与布局修复。限定Windows开发态verified，见[报告](docs/testing/2026-10-04-m05-r3-report.md)。259前端、46科学与媒体、18桌面、Chrome四保存组合/三模式/离线任务及实际Qt三保存/三布局通过，最终8份录制34文件哈希与媒体回读通过；WSL仅同输出WAV/唇形/哈希。修复零点重复叠加，独立120帧实验仍见约12ms起点误差，默认0不代表物理校准，见[审计](docs/references/m05-r3-timing-audit.md)。异常PTS可保留原容器，未确定其产生根因。实时模型保持candidate、科学指标/滤波未改；R2停止自动保存改为结束后另存。入口Start-M05-Workbench，旧EXE未打，实体设备/长录制/DPI/LinuxGUI未验。无DDL/push/发布/环境安装，保留同期差异及原资料。

2026-10-04 M16-R3：井井要求录音布局/删除任务/导入说明/紧凑控件/完整语谱和5000Hz范围。限定Windows开发态verified，见[报告](docs/testing/2026-10-04-m16-r3-report.md)。259前端、50 Python、Chrome18组/6布局与窄窗滚动、实际Qt10组/3布局通过；WSL16项纯核心。任务删除即时保存并保留take，失败保留草稿；实时128/录后640列有界时间抽样显示，放大查看细节，不改原始/降噪算法。修复新任务旧图、检测停止残图及窄窗结果重叠；入口Start-M16-M17-Workbench。合成设备限定，实体音频/DPI/LinuxGUI未验；无EXE/DDL/push/发布/文件删除/环境安装，保留同期其他改动。全库文档检查当时8条旧EXE和1条同期M05报告缺链，M16本轮链接通过。

2026-10-04 M04-DISPLAY：井井要求动态纵轴按钮即时应用/固定纵轴、PNG沿用选定轴、修复语谱图改窗刷新及统一参数估计式播放栏。限定Windows开发态verified，见[报告](docs/testing/2026-10-04-m04-display-report.md)。254前端、Chrome10组/3尺寸及深色、5份PNG回读、实际Qt4类/2尺寸/3份原生保存PNG通过；WSL7项仅静态读取。独立PNG采用当前纵轴，完整结果目录保留原任务三件套；共享播放栏、历史片段试听保留。语谱图加载器随文件刷新、请求去重/计时器及迟到响应修复，未逐步复现截图持续等待的唯一根因。输入为合成短音频，未读截图WAV/论文目录，硬件/DPI/LinuxGUI未验。入口Start-M04-Workbench，未打EXE/DDL/push/发布/删除/环境安装，保留同期其他改动。

2026-10-04 M03-R4：井井要求选区与试听移到底栏、批量入口放刷新右侧并增加低通。限定 Windows 开发态 verified，见[报告](docs/testing/2026-10-04-m03-r4-report.md)。复用公共 AudioTransport，原始选区与归一化试听分开绑定，时长/微观留左栏；批量弹窗分组并保留独立参数。254 前端、Chrome6功能组/6布局/3弹窗、实际Qt5记录含真实批次1500Hz保存通过，WSL6项仅静态资源/参数读取。修复窄窗底栏离屏及参数区压缩重叠，原录音哈希不变。入口Start-M03-Workbench，旧EXE未打、科学核心未改、设备/DPI/Linux完整链未验，无DDL/push/发布/删除/环境安装，保留同期改动。

2026-10-04 M03-R3：井井授权紧凑侧栏、顶栏操作、逆滤波6/4/4/2图、全段EGG选显、导图居中精简，并追加动态F0轴和上下横轴对齐。限定Windows开发态verified，见[报告](docs/testing/2026-10-04-m03-r3-report.md)。251前端、48科学/14契约、Chrome10组/8侧栏、实际Qt12项含4种300dpiPNG及完整结果、公共图5/字体12/共享导图3组通过，WSL10项仅显示函数/契约/PNG。自动dB取当前PSD峰下50dB，保留手动；科学核心未改，6组原数据精确一致，原录音哈希不变。算法审查发现实际226起点有225个3ms窗跨下一GCI，阶数/阈值敏感性与未来方法planned见[核查](docs/references/m03-r3-algorithm-audit.md)，不宣称真实声门流。公共M02/M03/M08 PNG同步改进；旧EXE未打、设备/DPI/Linux完整链未验、无DDL/push/发布/删除/环境安装，保留同期改动。全库仍8条旧EXE缺失链接。

2026-10-03 M02-R2：井井授权参数显示按钮间距、2/3/4列单行勾选、缩短滚动列表、波形文字不拉伸、所有横轴对齐及直接拖选。限定Windows开发态verified，见[报告](docs/testing/2026-10-03-m02-r2-report.md)。246前端、Chrome6类/11组几何、PNG3类/5份300dpi回读、字体导出和实际Qt4类/5组通过，WSL仅4项静态读取。M02参数图改为左键选区/Shift平移，其他模块波形保持默认选项；窄窗和大字号共用水平滚动。旧M01/M02和M12脚本分别被改版选择器阻断，大矩阵原生读取失败另列限制，不算通过。入口Start-M01-M02-Workbench，未新打EXE/DDL/push/环境安装，保留同期其他改动与V2/原数据。

2026-10-03 M01-R2：井井要求左栏TextGrid切分、四列参数弹窗、新旧唇形自动加载及全列表关联。限定Windows开发态verified，见[报告](docs/testing/2026-10-03-m01-r2-report.md)。246前端、19桌面、Chrome5组/4弹窗尺寸/2侧栏高度、实际Qt混合PKL/JSON两任务/6结果/770时间点及四项唇形逐值一致通过，原文件哈希不变。唇形目录独立、JSON优先、歧义明确，单条改选/取消刷新保留；旧PKL提交时受限转换，不写原目录。WSL3哈希读取一致；硬件/DPI/LinuxGUI未验。源码入口Start-M01-M02-Workbench，旧EXE未重打，无DDL/push/发布/删除/环境安装，保留同期差异。

2026-10-03 P19 外观 R1：按井井反馈将代码字体改为完整下拉，JetBrains Mono 内置项置顶，保留旧值与自定义；应用时清除旧预览。移除 PTB 配色，默认/旧 ptb 采用 Everforest，其他选择保留，并提示尝试更多配色。限定 Windows 开发态 verified，见[报告](docs/testing/2026-10-03-p19-appearance-r1-report.md)。242 前端、Chrome 11组/58主题/15布局、字体12组、实际Qt4类/58主题/6布局通过，WSL4静态资源读取。未重打EXE，未验实体DPI/LinuxGUI，无DDL/push/发布/环境安装，保留同期改动。

2026-10-03 P20：井井授权最新本机EXE与旧包/备份/垃圾清理准备。最终 `dist/PhoneticToolbox-v3-Latest-20261003-R1/PhoneticToolbox-v3-Latest-20261003-R1.exe`，296.52MiB，见[成品报告](docs/testing/2026-10-03-p20-exe-report.md)。242前端、420包内文件、10任务/15页、M16/M17十一组及14布局、26子进程退出/三非法参数通过；首个Latest候选因字号验收选择器误取后台M13失败，R1仅修测试范围后重建通过，产品算法不改。仍portable:false、合成/Windows离屏限定，实体设备/DPI/跨平台未验。清理仅prepared，见[清单](docs/testing/2026-10-03-p20-cleanup-plan.md)：95目标18.48GiB，明确保留源码快照/日志/报告、实际MFA与科学环境、真实录制/数据/V2/Git；未删除，须井井明确授权具体范围后执行。不整删output或validation，无DDL/push/发布/环境安装，保留同期差异。

2026-10-03 P19：井井要求Codex配色/深浅联动、两栏紧凑设置、内置代码字体/默认宋体与Times New Roman，以及放大任务栏图标。限定Windows开发态verified，见[报告](docs/testing/2026-10-03-p19-appearance-report.md)。29同名适配方案+原主题共30套，缺失模式由PTB补齐，不能称逐像素官方复制。JetBrains Mono2.304/OFL内置、旧显式偏好保留、缺默认系统字体有回退提示；IPA固定Doulos。K2显示可见宽79%→约94%，原资源保留，QIcon/后续ICO同源。242项前端、Chrome60主题/15布局与12组字体导出、Qt60主题/6布局/M10配色联动、2项图标通过；WSL4项资源静态读取。旧EXE未更新，实体任务栏/DPI/LinuxGUI未验；无DDL/push/发布/环境安装，保留同期差异。

2026-10-03 M05-R2：井井要求本地录制有声、直接保存和放大面部。限定 Windows 开发态完成，见[报告](docs/testing/2026-10-03-m05-r2-report.md)。授权后麦克风优选/实际输入与电平、低信号提示、停止直接保存及另存、同名 WAV/唇形伴随与方法元数据、固定面部放大完成。239项前端、39项M05 Python、18项桌面、18项旧输入、Chrome7组及实际Qt5组通过，9份录制44哈希/媒体回读通过；WSL仅1项格式读取。实时仍为candidate，legacy重算可选，实体设备/同步/DPI/Linux媒体链未验。Qt离屏验证仅测试进程调整渲染参数，未改产品/系统。入口Start-M05-Workbench，旧EXE未更新，无DDL/push/发布/环境安装，保留同期其他修改。

2026-10-03 P18 EXE：井井追加新包与体积检查授权。最终 `dist/PhoneticToolbox-v3-P18-20261003/PhoneticToolbox-v3-P18-20261003.exe`，含P18/M17紧凑显示，见[成品审计](docs/testing/2026-10-03-p18-exe-report.md)。新增可选 `--lean-qt` 及快照hook，仅排除Qt调试/QML未使用资源/额外WebEngine界面语言，QtQuick/Qml底层依赖、科学库、中文英文、字体许可保留。实际342.77→296.38MiB，减13.53%；OpenBLAS等固定装载路径重复副本记录后保留。419文件哈希、自然/合成各10任务、14/15页、M10真实原生引擎、M16/M17八组和18分区、各25子进程退出通过，真实输入/旧R2哈希不变。仍portable:false及既有独立环境，实体音频/DPI/跨平台未验。本轮只改打包与验证脚本/文档，不改科学算法，无现存库DDL/push/公开发布/删除旧包/环境安装。

2026-10-03 P18/M17-R1：井井授权两名子代理统一15模块视觉与扩充音标，随后追加名称/符号紧凑多列、解释悬浮和上下滚动。限定Windows开发态完成，见[联合报告](docs/testing/2026-10-03-visual-and-ipa-report.md)。默认侧栏300px、栏框对齐、文件/草稿/引用/主要操作位置统一，M03操作栏移左且兼容旧宽度记忆；M10排除。IPA增加107、extIPA增加18输入入口，总625含组合/例示/旧版，不能称独立音标数；VoQS65入口/56指定译名与字体保留。231项前端、P18 Chrome92布局/5图表/8旧工作台、Qt60布局及指定音频真实预览、M17 Chrome36布局/17组625输入和实际Qt18分区/4组通过。原目录28顶层文件哈希不变，WSL显式CIN目录生成一致。原科学算法未改，实体DPI/硬件及LinuxGUI未验。全库文档检查仍有4条既有旧EXE缺失链接，未绕过。入口Start-M16-M17-Workbench，旧EXE未更新，无DDL/push/公开发布/环境安装。

2026-10-02 GitHub 同步授权：井井在 M16/M17 R2 本机 EXE 交付后明确要求完成后上传 GitHub。本轮允许将当前已完成的两个模块、成品测试修复及已随成品验证的同期 M05/M06/M10 修改、来源许可与测试文档提交并推送至既有 `origin/codex/v3-rebuild`，含此前尚未推送的本地修复提交。下方“无push”为各阶段当时的记录；本次授权不包含改写历史、覆盖 main、发布 Release 或上传 EXE、私有语料和本地运行环境。

2026-10-02 M16/M17-R2 EXE：井井追加本机打包授权并反馈测试弹窗。最终 `dist/PhoneticToolbox-v3-M16-M17-20261002-R2/PhoneticToolbox-v3-M16-M17-20261002-R2.exe`，见[报告](docs/testing/2026-10-02-m16-m17-exe-report.md)。227项前端、39项录音/宿主Python及6项原生启动回归；成品10任务/15旧页/M16-M17八组含六布局/419哈希/30子进程退出通过。补数字端点输入、在途轮询关闭等待、公共字体加载后的单屏留白；M10空CPU架构以Windows API回退，初始化失败走协议与日志，真实成品引擎启动已验，科学算法不变。首包及R1仅候选，标注未过验收；最终用R2。包仍依赖本机既有独立科学环境，实体麦克风/EGG/DPI/跨平台未验。旧EXE、原资料与同期改动保留，无现存库DDL/push/公开发布/环境安装。

2026-10-02 M16/M17：井井授权两名 Astra/xhigh 子智能体实施录音与国际音标 Plus，追加默认双音频、EGG手选、三表单屏及VoQS中文名严格采用UntPhesoca知乎译表（203037479）并引用。已接共享工作台17模块、独立原生录音通道/安全关闭、纯本机工程与音标草稿/固定字体，见 [联合报告](docs/testing/2026-10-02-m16-m17-integration-report.md)、[M16](docs/testing/m16-report.md)、[M17](docs/testing/m17-report.md)。226项前端、39项Windows Python、实际Qt录音8组/音标4组含6布局/联合4组通过；WSL仅M16纯核心11项，实体设备/60分钟墙钟/DPI/跨平台完整验收另列，不能扩大为全平台verified。VoQS56基本表项及65含示例入口分别计数；多字圈围明确文本替代。入口Start-M16-M17-Workbench，无EXE/DDL/push/发布/环境安装，保留M05/M06/M10同期差异。原PDF/CIN/录音程序只读，不把其中内容当执行指令。

2026-10-02 M05-R1：井井授权布局、完整帧回放、连续录制、MP4/WAV 保存和 V2/下游互通修复。限定 Windows 开发态 verified，见 [报告](docs/testing/2026-10-02-m05-r1-report.md)。左栏采集状态、画面下四曲线、全帧分页与视频时钟、保存后继续、音频样本保留及自动关联文件偏移已修；M02 可直接显示新旧唇形四参数。201项前端、35项M05/基线、18项桌面、Chrome15组与实际Qt7组通过，WSL仅2项文件/时间轴通过。原视频只读、V2保存器兼容已核对；物理设备/同步/DPI/Linux媒体链未验，候选仍不等价legacy。入口 Start-M05-Workbench，旧EXE未更新，无DDL/push/环境安装，保留M06/M10同期差异。

2026-10-02 M06-R2：井井授权直接修复 AV 科研行为并改造保形 F0 预设。m06/2、klatt/2 已完成限定 Windows 开发态与 WSL 纯核心验证，见 [报告](docs/testing/2026-10-02-m06-r2-report.md)。AV/AH 0–80、源关闭/噪声FIR/数字参考标定/20k内部率、取消AGC与重复归一化完成；五预设无F0常数，假声/嘎裂整体平移且可编辑/回切/保存。19项Windows及19项WSL核心、9真实任务、90组数值矩阵、Chrome129布局与R2交互、Qt32布局/实际300Hz快照通过。原V2逐位门14通过/14失败为本次获准科研变化，原断言保留；旧m06/1标尺文件明确拒绝，原文件/存储保留。Linux正式任务仍关闭，声卡/实体DPI未验；无EXE/DDL/push/环境安装，保留同期M05/M10等差异。

2026-10-02 M06-R1：井井授权语音合成布局/时间轴/控件/参考共振峰修复、代码审查及 AV 溯源。限定 Windows 192项前端、28项V2精确、Chrome129组和实际Qt32组通过，见 [报告](docs/testing/2026-10-02-m06-r1-report.md)。上下图固定高度/对齐和即时应用时长完成，旧音频不拉伸。AV200/190为历史自定义增益，噪声前级与AH/HNR映射存在偏差，见 [核查及方案](docs/references/m06-av-audit-2026-10-02.md)；本轮科研实现不变，新标准模式planned待独立授权。WSL仍21通过/7精确失败、能力关闭，实体DPI/声卡未验；未重打EXE/DDL/push/改环境，保留M10同期改动。

2026-10-01 M10 外观修复：井井仅授权统一背景、修复声学/实时图切换高度并移除两个旧试听按钮。源码/前端构建及实际 Windows Qt 30组主题/尺寸/布局切换已通过，见[报告](docs/testing/2026-10-01-m10-appearance-report.md)。统一蓝白/蓝灰背景、固定分析区域、160ms淡入及canvas尺寸反馈修复完成；1秒/持续发声保留。本轮未重打EXE，未验证音频硬件/视频/跨平台，无DDL/push/环境变更，保留同期其他模块差异。

2026-10-01 P17：井井授权先push当前代码、Astra/high子代理检验修复除M10外14模块并交付EXE。基线0779bee已push；后续ab07736/ea89397为本地修复。公共自适应布局/波形振幅与连续线/语谱预览、M08首显、M07整页三栏、M03两栏高图、M09原生选四角及空目录误报已修。最终入口`dist/PhoneticToolbox-v3-P17-R1-20261001/PhoneticToolbox-v3-P17-R1-20261001.exe`，见[总报告](docs/testing/2026-10-01-p17-report.md)和14份逐项规则。191项前端、最终成品10真实任务/14页/394文件哈希、26子进程退出通过。布局覆盖1920/2560/3840模拟视口，实体DPI另列。完整矩阵仍in_progress，MFA兼容模型、真实视频/唇形、M09预期截图背景、物理音频/多屏/跨平台有明确限制；摄像头麦克风由井井留待手验。包依赖既有独立科学环境，非可搬迁发行物。旧EXE/V2/原音频/现存库/环境保留，后续修复未push或公开发布。

2026-10-01 M03-R2：井井要求实时交互、四图手势/布局、总览显示和IF图窗，并追加暂时移除声门活动、高低通数值输入及默认低通2000 Hz。已完成限定Windows真实EGG录音、Chrome/实际Qt验证，见 [报告](docs/testing/2026-10-01-m03-r2-report.md) 与 [ADR](docs/decisions/ADR-M03-R2.md)。交互改为有界内存会话，正式导出保留任务；12组数值/字节对照、7项会话/HTTP、28项架构、15项前端状态、14组Chrome及Qt PNG保存通过。连续更新约0.13–0.22秒，首次仍需加载，未承诺固定延迟。Linux实时准入未开放，旧EXE未更新，未DDL/push/改V2或原音频。


2026-10-01 M01/M02-R1：井井授权修复音频列表撑高、批次轮询抖动、递归目录、TextGrid切分、图窗清空/删除、默认PNG及长WAV预览。源码/前端构建完成，真实48对WAV/TextGrid、274片段采样对照、实际Qt全48文件切分及PNG/图窗管理通过，范围见 `docs/testing/2026-10-01-m01-m02-repairs-report.md`。入口 `scripts/Start-M01-M02-Workbench.ps1`。井井追加要求后验证仅使用指定“男-范皓云-已标注”目录，禁止借此查找/改动博士论文目录；原文件只读、输出进验证目录。真实大文件未验，早期合成检查不能替代真实验收。无EXE更新/DDL/push/环境变更，保留同期其他差异。

2026-10-01 P12 EXE 更新：井井授权更新本机 EXE，随后自行检查并反馈 bug。已交付 `dist/PhoneticToolbox-v3-LocalPreview-20261001/PhoneticToolbox-v3-LocalPreview-20261001.exe`，见 `docs/testing/2026-10-01-exe-update-report.md`。真实冻结成品 10 项任务/15 页、M01 acoustic/2、标注侧栏三种状态定位、39 子进程退出和非法参数拒绝通过，限定 Windows/offscreen/合成输入。338 个包内源码/前端文件哈希已核对。原 local-preview-20260927 数据目录沿用，旧 R4 哈希不变；仍为依赖现有独立科学环境的本机包。未改现存库/V2/环境/CI，无 push/部署/公开发布，下一步等待人工试用反馈。

2026-10-01 P16-REVIEW：井井授权逐条修复审查问题。本轮九项代码修复及 Windows 定向验证完成，详见 `docs/testing/2026-10-01-review-repairs-report.md`。M01 新结果 computation_revision=acoustic/2，旧 JSON 缺失字段按 acoustic/1；勿将新科学行为重标成原样迁移。源码/冻结源码快照的 EGG/LPC bootstrap 绑定配套核心，第三方锁未改。新增 POSIX 目录保存、三平台用户目录/原生资源选择、运行时清单及 Ctrl/Meta 标注支持；Linux 新保存实测和 Mac 原生/完整发行待验，当前 WSL 未找到 Python。保留同期 P04/M03/M04/M13 未提交差异；额外修复 Vue class 更新丢失侧栏定位类。不改现存库、V2、系统环境、CI/CD，无 push/部署/EXE。原历史状态不得覆盖此轮限定结论。

2026-09-29 M03-RT：井井要求修复慢加载、完成空白、恢复自动更新，并追加代码审查。限定 Windows Chrome/实际 Qt 修复通过，见 `docs/testing/2026-09-29-m03-realtime-report.md`。加载/参数自动刷新、连续操作合并、取消/迟到结果、浮点历史恢复与试听复用已验。公共管道移除有数据时的等待，本机 EGG 按既有 1 MiB 上限读写，未请求 GCI F0 的预览省略无用全段事件检测；实际结果逐字节不变。77秒输入实际 Qt 首显约7.66秒、更新约5.72秒，仍非即时响应。入口 `scripts/Start-M03-Workbench.ps1`。WSL缺python3未验科学链，旧EXE未更新，无现存库DDL/push/发布，保留同期其他任务差异。

2026-09-29 P04-RESIZE / M02-DEFAULT：井井授权并已实施所有现有侧栏/导航栏统一边界拖动及本机记忆、M01 拓宽、M02 首次空图、删除剩余重复模块页首、设置/使用说明标签化。限定 Windows Chrome/实际 Qt 验证见 `docs/testing/p04-resize-report.md`，WSL 仅静态构建读取/哈希。保留 M13 同期独立右栏/横向滚动规则和其他任务未提交差异。旧 EXE 未打包，无 DDL、push、公开发行或全局依赖变更。

2026-09-27 M05：基线、浏览器候选、正式离线任务/文件和 Windows Chrome/实际 Qt 本机三模式录制已有分项证据。已按 V2 修复完整网格、同帧叠加、停止尺寸及 Qt 权限错误提示。浏览器新模型未通过等价门，正式结果保留 legacy；完整模块 in_progress，物理同步、Linux/远程准入及剩余设备矩阵待验。入口 `scripts/Start-M05-Workbench.ps1`，见 [M05 报告](docs/testing/m05-report.md) 与 [说明](docs/manual/lip-extraction.md)。未 push、DDL、部署或生成 EXE。

## 0. 当前阶段与授权

- 2026-09-27 M11：井井授权独立可选 MFA 与正式任务迁移。Windows 源码宿主／Qt／离线候选组件限定验证通过，见 `docs/testing/m11-report.md`、`docs/manual/mfa.md`。固定既有 MFA 3.3.8，运行时与模型独立，主 EXE 未重打；旧管线禁用 JIT 的失败及显式适配修正见 ADR-M11-001。完整 M11 仍 in_progress，Linux MFA／P11 委托与正式 remote/1 桥阻断，实际服务器与实验室未验；全词典实测约 1.76 GB 峰值，服务器回退保持禁用。原队列等待、账号隔离／新政策已用新建测试 PG 验证，无现存库 DDL、push、生产部署或 V2／旧包修改。并行公共接线已释放，勿把候选组件当公开发行。


- **2026-09-26 当前统筹要求：** 井井已购阿里云轻量应用服务器与域名。截图确认 Ubuntu 24.04.2 LTS/x86_64、2 vCPU、套餐 4 GiB/50 GiB ESSD；系统截图报告约 3.4 GiB 总内存、2.9 GiB available、无 Swap，资源预算以系统实测为准。本轮仅统筹规划、只读代码审查及一次明确授权的只读 SSH 连接尝试，认证被 publickey 拒绝，未执行远程命令。新入口为 [统筹计划](docs/plans/2026-09-26-server-coordination.md)、[远程计算设计](docs/specs/remote-compute.md)、[审查清单](docs/testing/2026-09-26-planning-audit.md)。后续模块由井井另行指派 agent 实施，本 chat 不自行派发或推进模块。
- **新政策与追加验收：** 用户明确将账号额度改为 **1,000,000,000 字节（1 GB）**、数据默认最多 **259,200 秒（3 天）**，下载不续期；当前运行时代码/数据库仍是旧规则，P07-POLICY 待实施。旧账号超额、既有到期时间及历史结果需兼容迁移，不能据此自动删除旧数据。每模块完成 Windows 验证后，还需 WSL 或授权实际 Linux 服务验证，单列平台/资源/远程计算/统一 UI 状态，不扩大旧 verified。云端重计算从禁用未验证能力开始，准入后最多一槽，初始计算进程组总预算约 1 GiB，数值待测；远程可信节点主动 HTTPS 领取为 proposed。删除模块内部大标题/说明行与重复关闭按钮，保留工作台标签栏和标签关闭保护，原操作迁移到统一工具栏/参数区。当前只改文档，未实施 UI、算法、DDL、环境、部署或 EXE。以下旧 5 GB/7 天/双 worker 记录是历史证据，现行要求以上述计划为准。

- 2026-09-19，井井追加 M12 图窗置顶、资源/搜索下移、四按钮移除，确认整段标注删除/剪贴语义，并明确本轮结束直接生成临时 EXE。R6 已 verified（限定 Windows 前端/Chrome 及冻结宿主基础链路），119 项前端、6 组 R6/8 组 R5 Chrome、类型/构建、冻结 EXE 11 步通过。入口 dist/m12-preview-r6/PhoneticToolbox-v3-M12-R6.exe，见 docs/testing/m12-r6-report.md。旧 R4 哈希不变，全局主题不改，原灰色截图是计时冻结造成的过渡中间态。首次沙箱内冻结检查超时，用户授权沙箱外复跑同一 EXE 通过；新剪贴按键端到端证据限定 Chrome。未推进其他模块/现存库 DDL/push，仍未封装 M03/M04 独立兼容运行环境。

- 2026-09-19，井井要求 M12 图窗精简、选区/强度同步、波形拖边界、毫秒语谱窗、三图双击新建、Ctrl 拉开、Backspace 删除及原始 TextGrid 优先。R5 已 verified（限定 Windows 开发前端、独立 Chrome 与本地文件桥接），见 docs/testing/m12-r5-report.md。113 项前端、8 组 R5 Chrome 与 3 组公共波形回归、类型与生产构建通过。另修缩放命中、音节音素越界不同步及反向框选 0 秒。旧 R4 EXE 未更新，未 Qt/长录音/DDL/push，不推进其他模块。

- 2026-09-15，井井要求修复 M12 大 WAV 读取超限、将秒数刻度移到标注层下方，并明确只修改和生成 EXE，不做验证。R4 实现与构建完成，入口 `dist/m12-preview-r4/PhoneticToolbox-v3-M12-R4.exe`；运行/功能未验证，不沿用 R3 verified 标签。见 `docs/plans/2026-09-15-m12-r4-long-audio.md`。本次授权不扩展其他模块，旧包/原音频/TextGrid 保留。
- 2026-09-14 井井追加 M12 边界精度/键盘操作/普通双击/中文自动保存及整个 v3 设置缩放，随后增加波形 Shift 平移/等高/连续曲线、语谱拖选和可选音素首边界。M12-R3 与 P04 本轮范围 verified（限定 Windows 及 docs/testing/m12-r3-report.md）。103项前端、134项Python、原18/R1六/R2十二/R3十三组Chrome、公共EGG三组、实际Qt及冻结EXE各50步通过。临时包 dist/m12-preview-r3/PhoneticToolbox-v3-M12-R3.exe；旧包/语料/V2保留，同源嵌入模块也禁止整页Ctrl滚轮。其他模块只接公共页面缩放，保持原迁移检查点；无现存库DDL或push，包仍不包含M03/M04独立兼容运行环境。
- 2026-09-14 井井追加六张截图的 M12 布局/层名/联动要求。M12-R2 现为 verified（限定 Windows 及 docs/testing/m12-r2-report.md 的真实 EXE 范围）。文件名颜色与不重叠、实际层名选择/空白显式创建、顶部总览/音量、选区空格试听、三图共选区和波形边界/自动振幅轴完成。94项前端、134项Python、原18组/顺序8组/R2新增12组Chrome、28步Qt与真实EXE、6步长录音Qt通过，公共EGG总览及M02整幅PNG回归通过。临时包 dist/m12-preview-r2/PhoneticToolbox-v3-M12-R2.exe；R1与原数据保留，V2六文件hash不变。R1自动建层及旧播放布局由R2取代，其他模块保持原检查点，无现存库DDL或push。
- 2026-09-14 井井已要求 M12 临时包并追加试用修复。M12-R1 现为 verified（限定 Windows 本机开发态及报告中的 EXE 路径），见 docs/testing/m12-r1-report.md。补无/空标注、顺序拼音起终点/内部声母韵母/连续终点、整体框选拖动与默认 1 ms 可调微步。实查并验证中文 312.57 秒 FLOAT32 录音，补唯一候选匹配，M12 沿用公共 64 MB/3200 万采样值预算。90 项前端、134 项 Python、原18组及新增8组Chrome、22步Qt/真实EXE、6步长录音Qt完整回读通过；原数据和V2六文件hash不变。临时包 dist/m12-preview-r1/PhoneticToolbox-v3-M12-R1.exe，原包保留；未包含M03/M04独立兼容运行环境。无现存库迁移或push，其他模块保持原检查点。
- 2026-09-14 井井明确提前实现M12，完成前不打包EXE。M12开发态功能现为verified（限定Windows独立Chrome、实际Qt/已安装wheel、本机托管PG网页），见docs/testing/m12-report.md。7功能组、原版编辑/强度/FFT/唇形显示对照，80项前端、176项Python、18组Chrome与11阶段Qt、网页双账号/配额/受控到期通过；两份原自然语料哈希不变。已更新开发入口的自身三个wheel，第三方依赖/V2/旧EXE不变，未DDL/push。临时EXE等待井井后续明确要求；M04保持D检查点，其他模块不自动推进。
- 2026-09-13 井井在C后继续授权。M04-D现为verified（限定Windows独立Chrome/真实本机任务与MKL子进程），见docs/testing/m04-ui-report.md。页面接入V3公共波形/标注/频谱/播放/任务与字体，24组Chrome、69项前端及13项Python通过；时间频率分离、Shift选区、历史/草稿/取消/迟到归属/保存下载和小窗滚动已验。公共刻度大字号边距修正并复验共享手势。完整M04仍in_progress，下一项E的托管网页/自然录音及20项收口。未Qt/EXE/DDL/push，不推进M05。
- 2026-09-13 井井在B后确认继续。M04-C现为verified（限定Windows任务/导出/本地HTTP与服务器真实PG认证ASGI），见docs/testing/m04-jobs-report.md。104项科学导出、139项共享契约/任务、2项保存与65项前端检查通过；PNG/JSON/选区WAV完整发布，取消/30秒超时/故障回收、800万帧尾部ROI、双账号/配额/受控到期通过。仅重装项目兼容环境核心wheel，第三方包/V2不变，未DDL/push/EXE。完整M04仍in_progress，下一项D的V3页面、字体预检与真实浏览器交互。
- 2026-09-13 井井授权继续M04-B并追加允许LPC学术引用、代码出处有界查找。B现为verified（限定Windows纯核心/独立wheel目标目录），73项源码与安装包检查通过，7份V2来源哈希不变，见docs/testing/m04-core-report.md。原样迁移与公开验证分开提交，保留原计算；实测单次ROI上限48,000样本，计算前后协作取消已实现，进程硬预算在C实施。Makhoul(1975)及实际依赖已登记，更早代码来源unknown不阻挡功能。完整M04仍in_progress，下一项C任务/PNG导出，页面未接入；EXE/DDL/push/M05不推进。
- 2026-09-13 井井在EGG开发态收口后明确继续，已授权推进M04 LPC谱图。M04-A现为verified（限定Windows原V2数值/服务文件/PNG基准），12场景、38数组44,497值双轮逐字节一致，见docs/testing/m04-baseline-report.md；7份原文件哈希不变。完整M04为in_progress，V3核心/任务/页面尚未实现，下一步按docs/plans/2026-09-13-m04-implementation.md进入B纯核心及预算。此前不推进M04为历史边界；M05不推进，引用/EXE继续暂停，未DDL/push。
- 2026-09-13 井井审阅最后功能收口后授权继续。M03开发态功能阶段现为verified（限定现有Windows环境及各报告的Chrome/Qt/本机托管服务范围），见docs/testing/m03-dev-closeout-report.md。补LP阶数草稿/未保存提示、关闭框内保存失败反馈，65项前端与23组Chrome通过；当前构建已更新。完整M03设备/生产/跨平台仍in_progress，A29与EXE按要求暂停，开发功能阶段不再因这些未测项延长。本轮未启动Qt/DDL/push；下一模块建议M04，尚未实施。
- 2026-09-13 最新要求：井井明确代码引用暂不处理，优先功能实现。暂停A29进一步文献/许可核查，不将此作为开发功能推进的前置条件；已核对的说明保留，未决项不扩大为已确认。E3文件切换旧预览错误反馈已限定Windows Chrome verified，65项前端与5项Chrome回归通过，见docs/testing/m03-preview-switch-report.md。EXE及相关探针继续暂停，不自动推进M04。
- 2026-09-13 井井继续E3并明确EGG源码位于V2。A29本轮已确认V2直接迁移来源，8份源码hash复核一致；澄清固定GCI后3ms、自相关LPC平均，未按GOI确认闭相。见docs/testing/m03-provenance-report.md与docs/references/m03-method-audit.md。本轮仅说明/登记修订，原算法未改；最初文献对应/完整授权链仍待材料，完整M03保持in_progress。EXE及探针暂停，不推进M04，未DDL或push。
- 2026-09-13 井井授权继续V2功能清单核对。本轮六功能组映射及两项补齐已verified（限定Windows独立Chrome），见docs/testing/m03-function-review.md：总览单击保留选区并更新、批次提交可取消且不误取消上一批。65项前端、5组新增/M01-M02默认选区、6组批次、3组总览通过；原V2八文件及手册哈希一致。完整M03仍in_progress，剩余方法/许可链与设备/生产证据单列，下一项来源未决项集中收口；EXE暂停，不推进M04，未DDL或push。
- 2026-09-13 井井继续授权E3。结果窗口读取/重读/迟到反馈已verified（限定Windows独立Chrome），见docs/testing/m03-result-feedback-report.md：错误在窗口内提示，旧读取/保存/下载反馈不能影响新窗口；64项前端、10组结果与4组试听Chrome回归通过，下载CSV实测哈希一致。故障/到期响应为受控注入，不冒充自然到期或生产认证验收。完整M03仍in_progress，下一项V2开发态功能清单集中核对及来源收口；EXE暂停，未DDL/push，不推进M04。
- 2026-09-13 井井继续授权开发态E3。批次参数和提交失败反馈已verified（限定Windows独立Chrome），见docs/testing/m03-batch-feedback-report.md：错误参数不发任务，全部拒绝保留选择/旧批次入口，部分成功逐文件说明；64项前端、6组批次与3组字体Chrome回归通过。完整M03仍in_progress，下一项结果窗口读取失败/到期及剩余交互；本轮未复跑Qt。EXE暂停，不推进M04，未DDL或push。
- 2026-09-13 最新范围调整：井井明确后续暂不考虑EXE，近期不需要打包。停止M03-F及其他EXE封装/构建/发行准备，不继续相关探针；后续推进开发态功能、交互与科学迁移。F1本轮仅生成筛选后的本地运行文件，首次迁移探针因缺numpy.testing失败，未构建任何EXE。新增探针脚本和output证据保留为未完成草稿，不称verified。以下F候选/探针下一步均为历史，恢复须用户另行要求。
- 2026-09-13 井井继续授权E3。连续试听与IF角色修复现为verified（限定Windows开发态），见docs/testing/m03-playback-report.md。修正结果清单排序导致ORIG/IF试听标签对调、公共播放器归属和过期resume失败竞争，62项前端、Chrome真实节点样本/连续操作、9组Qt回归通过。完整M03仍in_progress，实际声卡/多屏DPI/来源尚待；F候选范围审阅已形成planned方案，见docs/plans/2026-09-13-m03-f-candidate-review.md，独立MKL运行时与后台字体尚未封装。未打包、DDL或push，不推进M04。
- 2026-09-12 井井继续授权 E3。M03 导出字体预检现为 verified（限定 Windows 开发态）：实际兼容进程、认证接口、Chrome 缺失/恢复/纯CSV与Qt导出回归通过，见 docs/testing/m03-font-preflight-report.md。检查绑定本次字体快照，批次缺失保留选择与旧记录，执行前仍复核；原科学核心未改。完整 M03 仍 in_progress，下一项剩余交互、来源和 F 候选范围审阅；冻结 EXE 未打包，不推进 M04，不执行 DDL 或 push。
- 2026-09-12 井井追加EGG总览紧凑布局：波形置顶，双声道/缩放/适合窗口在图下同排。限定Chrome/Qt布局已通过，见docs/testing/m03-overview-report.md。仅EGG启用公共组件compactOverview，其他模块默认布局不变。120秒长文件成果已在9cbbf3c，完整M03仍in_progress，下一项字体预检/剩余收口；旧EXE未打包。
- 2026-09-12 井井在长文件下一步说明后授权继续。M03-E3-B 已限定 Windows 开发态验证120秒/576万帧完整处理、首尾查看与三路径导出，原V2双轮20数组23242942值精确一致，见 docs/testing/m03-long-report.md。按实测设3GB/240秒进程预算，60秒仅总览视窗；核心/V2/环境/旧EXE不变，未DDL/push。完整E3/M03仍in_progress，下一项字体预检/剩余交互及来源收口，再审阅冻结EXE范围。
- 2026-09-12 井井审阅滚动方案后授权继续。P04-SCROLL 现为 verified（限定 Windows 开发态公共内容/二维滚轮/Qt 尺寸），见 docs/testing/p04-scroll-report.md。普通滚轮滚动，EGG/M02 Ctrl＋滚轮缩放，弹窗正文独立滚动；M10 仅外层最小高度/滚动，模型手势与录制未改。旧 EXE 未重打包，完整 M03/E3 仍 in_progress，后续回到 E3-B。未 DDL、push 或改 v2。
- 2026-09-12 井井在E2后授权继续E3。本轮恢复微观5–5000ms、原版显示抽点与边界不重复提交，限定Windows开发态验证见docs/testing/m03-e3-report.md。新增官方书目/原手册截图/Henrich原文方法差异，代码许可未闭合。完整E3/E/M03仍in_progress，下一项E3-B长文件全段语义和预算、字体预检/剩余交互；66.4秒仅导航及拒绝已验，不代表全段分析完成。仅重装项目m03-compatible核心wheel，第三方库/v2/语料/旧EXE不变，未DDL或push。
- 2026-09-12 井井在E1后回复“继续”。M03-E2现为verified（限定Windows开发态Qt/本机托管PG与Chrome），见docs/testing/m03-e2-report.md。双账号读隔离/切换、服务器三路径10文件回读、受控配额与到期清理、两自然录音开头及较响ROI通过；修复历史预览回读迟到覆盖新文件选择。完整E/M03仍in_progress，下一项E3长文件/微观范围差异与来源/剩余边界；F冻结EXE未做。到期为测试调整时间并显式清理，不冒充七天自然经过。未DDL、未改v2/语料/环境/旧EXE、未push。
- 2026-09-12 井井在D后回复“继续”。M03-E1现为verified（限定Windows开发态默认/导出/手势对齐），见docs/testing/m03-report.md：独立单文件/批次默认，单文件CSV双F0不受显示开关影响，源文件名/时间保存及同名保护，四图滚轮/拖动/键盘/中心线与弹窗保存反馈。完整E/M03仍in_progress；E2下一项网页/自然录音页面及切换竞争。原V2微观滚轮5–5000ms与当前10–200ms差异已明确，长文件预算/来源/EXE仍待。未DDL、未改v2/旧EXE/环境、未push。
- 2026-09-12 井井继续授权M03-D，并反馈下方按钮散乱。D现为verified（限定Windows开发态Qt/独立Chrome四图、参数/试听、任务与逐文件批次保存），见docs/testing/m03-ui-report.md。首版遵守v3主题、公共字体与组件，四图贴合v2；控件改为两行分组，高低通集中EGG图上方。122项科学/导出/字体、14项M03契约、47项前端及真实UI通过。广域回归另保留M01 Scratch取消清理的一次WinError32，单项复跑通过但根因未确认；不称全套稳定全绿。完整M03仍in_progress，下一项E的30项/来源/自然语料页面/网页联合收口，F冻结EXE尚未实施。未DDL、未改v2/m09科学环境/旧EXE、未push。
- 2026-09-12 井井在M03-B后回复“好，继续”。M03-C现为verified（限定Windows开发态任务/文件/数值导出），见docs/testing/m03-jobs-report.md。m03/1接既有任务/资产协议，独立MKL子进程，单文件CSV+三PNG、逐文件批次导出及IF双WAV实际回读；取消/失效worker/故障回收/重开/两份授权自然录音通过。未执行DDL或修改m09/m10/v2环境。后续P04-FONT已接入字体快照并完成限定Windows图片专项，见docs/testing/p04-fonts-report.md；完整M03仍in_progress，下一项M03-D。EGG页面/整批目录交互/冻结EXE尚未实施，不能把本轮PNG导出图当V3交互界面。
- 2026-09-12 井井在M03-A和UI统一约束后回复“噢噢 那你继续吧”。M03-B现为verified（限定Windows纯核心及安装wheel），见docs/testing/m03-core-report.md：80项检查、11样例31761项精确比较、输入与原v2源码未变。新核心位于packages/phonetic_core/src/phonetic_core/egg，实际用户默认slope/scale，保留两种ROI旧规则与独立mask，真实Praat帧时间与N/fs元数据明确区分。SciPy同版不同构建产生差异，兼容环境为项目内.venv/m03-compatible（Conda/MKL锁）；.venv/m03-ui为未通过逐位基准的PyPI候选，不能误用。完整M03仍in_progress，下一项C任务/文件/导出。页面/Qt整合/EXE尚未实施，不重复DDL、不改m09/m10/v2环境，UI从第一版遵守既定v3风格。
- 2026-09-12 井井补充：EGG贴合v2仅指布局与功能，v3统一设计规范和公共组件必须从页面第一版落实。页面内控件/图表/状态也须统一，不能只换外壳或留到收尾调整。真实组件与适配边界见docs/design/m03-v2-layout.md；旧Qt截图仅是基准，展示时先标明，不能让用户误认为v3页面设计。
- 2026-09-12 井井在M03计划后回复“好，请继续”，明确EGG布局尽量贴合v2、功能囊括v2，其余可优化。M03-A已完成限定Windows原v2独立基准：11样例双轮一致，公开合成数值与私有语料分开，详见docs/testing/m03-baseline-report.md。完整M03仍in_progress，下一项M03-B纯数组核心；页面/EXE尚未实现。布局以docs/design/m03-v2-layout.md为准：左CQ/SQ与语谱、右音频与EGG微观、下方两行参数及总览，覆盖早期右侧集中设置方案。实际单文件与批处理均GCI slope/GOI scale，底层EGGConfig被GUI覆盖的事实已纠正。不得将基准verified扩大为模块verified，不重复DDL、不推进M04。

**2026-09-12 较早停止点（仅历史，当前状态以上方 M03-E3 为准）：** 井井在进度审阅后授权继续整理成果、补齐M02整幅PNG并细化M03计划。现有成果已形成本地检查点c800ce8。M02-F05整幅PNG已通过限定开发态Windows Qt/Chrome验收，见docs/testing/m02-png-closeout-report.md；旧Research-Fix1/M10-R5 EXE未重新打包。M03源码/说明书审阅及30项验收设计已完成，仍planned，下一步审阅docs/plans/2026-09-12-m03-implementation.md后进入M03-A。P08仍in_progress，P09整体planned，M10/R4仅历史限定范围verified，R5仍未检验。旧R4 EXE当前不在发行目录，保留原报告元数据。本轮未push、发布、执行DDL或修改v2。

### 历史授权与阶段证据（当时的下一步不作为当前执行指令）
- 2026-09-12，井井授权修复已迁移 M01/M02/M09 的实际桌面入口，并明确允许新建 v3 专用本地任务库。Research-Fix1 通过限定 Windows 合成数据与真实单文件双轮验收，见 docs/testing/desktop-repair-report.md。首次新库位于 LocalAppData/PhoneticToolbox/v3/research-v1，只初始化不存在的新目录并复用既有002/005；已有目录只校验，不运行DDL。修复产物为 dist/research-repair/PhoneticToolbox-v3-Research-Fix1.exe，原M10-R5及v2保留。已修冻结worker调度、截图隐藏、TextGrid时间轨并纳入主题修复，其他模块继续停止，网页部署/安装版/跨平台不在此轮。
- 2026-09-11 最新明确授权与停止点：井井要求继续完成M01收口，再迁移M02参数显示（默认同绘图区叠加曲线、可多图窗批量分配，按说明书2.2纠正并复验）和M09语谱图转音频，做完停止。本轮已完成M01最终39项审阅，以及M02/M09限定Windows本地/托管Chrome验收，见docs/testing/m01-final-review.md、docs/testing/m02-m09-report.md与docs/plans/2026-09-11-m01-m02-m09.md。当前三项标为verified的范围以报告为准，M09多屏/DPI截图设备未纳入。新项目内m09-ui与开发入口scripts/Start-Research-Workbench.ps1不修改M10运行环境/录制EXE；M10/R4保持冻结。下列“下一项M01-G/39项暂停/M02不启动”是历史检查点，当前停止点覆盖其执行顺序。不得自行推进其他模块、重复DDL或发布。
- 用户：井井；助手自称秋叶；默认中文。技术判断说明证据、限制和待验证项。
- 当前 **P04 统一工作台 verified（限定公共界面与 Windows 定向验证）**；2026-09-09 井井在 P03 后确认继续，授权范围见 docs/plans/2026-09-09-p04-workbench.md 与 docs/testing/p04-workbench-report.md。井井已反馈当前界面无明显问题；学术优先的分组致谢修订已落实。P04 提供共同界面、真实 WAV 预览和 Qt 演示，不代表 15 模块算法已迁移。井井已授权继续 P05，并对专属空库方案回复“允许”。P05 现为 verified（Windows 账号/会话/项目范围）：实际 PostgreSQL 建表、隔离、事务、并发与受控重启恢复已通过；井井随后回复“好，继续”，已授权 P06 实施；P06 现为 verified（限定 Windows 持久任务流程）：具体建表审阅后的继续指令已核实，实际 PG/SQLite、并发、取消/中断/重试、本机服务重启与网页账号隔离均通过；退出后的收尾复验与记录已完成（docs/plans/2026-09-09-p06-jobs.md、docs/testing/p06-jobs-report.md）；井井在 P06 验收后回复“好，请继续～”，已授权推进 P07；P07 现为 verified（限定 Windows 受控存储、工程生成与有界 ZIP 联合范围）；003 审阅后“好，继续”、004 审阅后“好，允许”的具体授权均已执行，旧行保留；真实 PG/磁盘、配额/TCP/到期、任务输入/结果/旧 worker fencing、进程中断重试与独立浏览器两标签页联合验收通过。证据见 docs/testing/p07-storage-report.md、docs/testing/p07-job-files-report.md，设计与边界见 docs/plans/2026-09-09-p07-job-files.md；未涵盖科学算法迁移、硬断电、生产负载、原生工具任意目录输出或跨平台发行。P08 已完成M01-A基准与M01-B限定Windows科学核心验收，M01-C已完成限定Windows受控原生/格式适配，M01-D已完成限定共享协议与结果语义验收，E目录、草稿、试听与Praat显示已限定验收；F2持久批次及双端真实操作现已接入，范围见docs/testing/m01-persistent-report.md；完整M01/P08仍in_progress；G当前联合验收见docs/testing/m01-report.md，旧PKL图形转换和历史参数导入仍待补齐；P09仍planned，按具体模块计划推进；见 docs/plans/2026-09-09-p05-accounts.md 与 docs/testing/p05-accounts-report.md。
- 2026-09-09 井井在下一步说明后回复“好，继续”，本轮已完成 P08/M01 迁移前源码审阅、文件级计划与细项验收设计，见 docs/plans/2026-09-09-m01-implementation.md 和 docs/testing/m01-planning-report.md。该审阅交付时M01/P08仍planned，随后M01-A的实施与当前进度见下一条。原生输出、持久批次和实际DDL的具体设计门见计划，不把本轮文档完成扩大为整模块verified或新的数据库迁移授权。
- 井井随后回复“好，请继续”，已授权并完成M01-A独立基准补齐及科学环境审计：28例双轮捕获、23项测试，范围与新发现见 docs/testing/m01-baseline-report.md。M01-A结束时M01/P08为in_progress，随后进入M01-B。该轮只对合成数据新建导出SQLite文件，不操作现存/服务数据库；M01-A时科学包仅审计，B轮独立安装/锁定见下一条。
- 井井在 M01-A 后回复“继续”，授权并完成 M01-B：独立科学锁与可安装核心 wheel、149 项 Windows 定向测试，数值/时间/mask 对照及转换字节通过；真实采样率与后端结果元数据单列修正。见 docs/testing/m01-core-report.md。后续C/D/E/F2已完成限定验收，下一项M01-G；B 的小型合成 native 测试适配不得接网页/用户任务，完整 M01/P08 仍 in_progress。
- 前阶段已完成 **P03 科研行为基线的 Windows 定向验收**。2026-09-09 井井在 P02 后回复“好，请继续”，授权执行 P03，并确认 EGG 样例的双声道方向；边界见 docs/plans/2026-09-09-p03-baseline.md 和 docs/testing/p03-baseline-report.md。该结果不代表 v3 算法、完整 EXE GUI 或跨平台验收。公开发布、服务器部署、全局环境变更和全面业务迁移仍不属于本轮范围。
- 用户明确不要额外备份；保留相邻 v2 和现有使用数据。不得擅自删除、移动或修改 v2。这里的 Git 源码基线不是额外整目录备份，也不是已验证的新版本。
- 旧网页目录的删除授权有条件：仅在证实没有必要保留的依赖、独有工作或来源资料后才可删除。当前检查发现旧前端存在未提交修改；本阶段保留两个旧目录。
- 本文件适用于 v3 全目录。进入 frontend、backend、desktop、packages/phonetic_core、contracts、resources、tests、docs、third_party 时，继续读取相应 AGENTS.md 与 ARCHITECTURE.md。

- 井井在M01-B后回复“继续”，已完成M01-C：有界原生Job/命名管道、WAV/TextGrid/安全唇形格式与XLSX/SQLite双产物准备，222项Windows wheel测试及两组160×83实际导出回读通过，见 docs/testing/m01-io-report.md。未接入PG配额/持久发布或正式UI；M01-D已继续完成，完整M01/P08仍in_progress。

- 井井在M01-C后回复“好，请继续”，已完成M01-D：API 1.1/m01/1共享协议、可信输入/TTL边界、实际结果无损JSON与单文件/批次清单，318项Windows wheel测试及三组冻结对照通过，见 docs/testing/m01-contract-report.md。D阶段未开放科学任务HTTP；F2现已接通，见docs/testing/m01-persistent-report.md，下一项M01-G。

## 1. 必读与事实来源
1. README.md：阶段、入口、不能误用的历史目录。
2. ARCHITECTURE.md：组件边界、依赖方向、任务/文件/平台协议。
3. docs/requirements.md、docs/plans/2026-09-09-v3-master-plan.md。
4. 对应模块计划、docs/modules/module-migration.md、docs/decisions/ADR.md。
5. third_party/source-registry.json、third_party/README.md 和 docs/references/source-audit.md。
冲突时以井井最新明确要求为先；其余文档冲突必须记录并修正，不能挑方便的版本执行。D0.1/D0.2 仅提供视觉/功能历史，已更新部分以 D0.3 为准。

## 2. 每次编码的固定流程
- 先给出当前任务 ID（Pxx 或 Mxx）、本轮要改的文件、预期行为及验收命令。
- 工作前查看当前分支、git status、已有差异；不得混入无关用户改动。
- 先迁移已验证的纯算法与数据规则；原样迁移和行为修正分开提交。默认值、单位、帧网格、NaN、有声判定、输出范围不得借重构静默改变。
- 对真实行为风险先建立回归用例，再作最小实现；不写只照抄实现的测试，不为了文档微调启动全套构建。
- 修改后检查 diff、边界、中文/IPA、来源、实际结果；运行计划规定的定向测试。失败必须定位，不注释问题、不扩大容差、不跳过检查来“通过”。
- 一次完成一个可审阅任务；记录命令、平台、环境、产物、未测项目。只有满足退出条件才标完成。
- 需要改变架构/交互语义/算法/依赖/发布目标时，先补 ADR 和受影响计划，再按用户已授权范围实施；重大新范围需井井确认。日常明确任务不反复询问。
- 实施授权后允许按计划做本地小提交；push、发布、生产部署、改 CI/CD/密钥/全局环境和数据库实际迁移仍遵守会话授权边界。

## 3. 依赖与文件边界
- frontend 只调用 contracts 与平台能力接口，不读取服务器路径、不导入 Python 算法、不拼 shell。
- backend 是接口、身份、配额、任务和存储编排；不能实现另一套声学算法。
- desktop 是窗口/启动/本地文件/设备/发行适配；不能复制业务算法或请求公网来完成离线基础功能。
- phonetic_core 不依赖 Qt、FastAPI、数据库、HTTP、用户账号或固定开发机目录。
- 本地服务与服务器服务使用同版核心包与契约；不同平台差异集中在 adapter。
- 不从 ../PhoneticToolbox_v2、旧 Vue 站点或旧 API 工程动态导入。继承的 phonetic_toolbox 是过渡来源；迁移完成后新入口不能依赖它。
- 禁止为某页面添加新的独立 http.server；全部模块挂统一宿主、统一页面注册表、统一任务生命周期。
- 资源有清单和版本；不要把虚拟环境、研究语料、临时输出或第三方整仓库不加筛选地塞进发行包。

## 4. UI 与科研正确性
- 2026-09-12井井授权全局字体实施及已完成模块同步调整，全部IPA固定Doulos SIL，不提供替换选项。P04-FONT已完成限定Windows开发态M01/M02/M09/M10及M03后台字体专项，报告见docs/testing/p04-fonts-report.md，全平台专项仍in_progress。中文、英文与数字、代码/等宽文字可调，图表与导出默认跟随且可独立配置，同级图中文字统一字号，总标题最多1.2倍。详见docs/design/UI_SPEC.md第2.1节及docs/plans/2026-09-12-global-fonts-design.md。所有新页面从第一版接公共字体，已有模块按专项证据标记；M10本轮仅字体为追加授权，旧EXE及原算法/录制流程不扩大范围。字体变更不得改变科研值，后台导出绑定字体快照并验证实际产物。
- 采用已选 U2 紧凑工作台、完整浅/深主题、侧栏导航与标签页；暂定 K2 波形团子图标。
- 以 docs/design/UI_SPEC.md 为视觉标准；禁止各模块自己造一套颜色、弹窗、播放条、加载状态。
- 不允许删功能来迁就设计。15 模块、83 功能组、80 参数、14 设置逐项核对；新功能追加验收项。
- 2026-09-10 井井明确：布局和设计风格允许与 v2 不同，但 v2 功能不能遗漏。每个模块实施前同时读对应 v2 说明书章节与实际源码，建立“说明书操作 → 源码行为 → v3 入口 → 正常/异常验收证据”映射。说明书与源码不一致时单列差异，不能忽略说明书承诺，也不能照抄过时参数。章节入口与 M01 已发现差异见 docs/modules/v2-manual-coverage.md；按钮存在或预览可用不代表切分/导出/计算完成。
- 真实音频时间、选区、单位、标签和曲线必须一致；无数据画空态，不生成假结果。
- 主题与截图仅证明视觉，启动仅证明启动；不能据此宣称算法、全平台、摄像头或实验时序通过。
- 桌面不要求登录；网页账号/项目/配额界面不侵入本地研究流程。

## 5. 网页运行约束
- 目标 10 名研究者同时使用，有登录和用户隔离；计算并行度单独配置并实测。
- 每账号文件空间 1,000,000,000 字节（1 GB），上传、结果、缓存和临时占用均计入；共享安装模型不算用户额度。此为 2026-09-26 新要求，运行时须经 P07-POLICY 迁移后生效。
- 用户数据默认最多保留 3 天；下载不要求自动删除、不延长有效期；用户可以直接删除，无需先下载。既有数据按专门迁移方案处理，不把新期限当即时删除授权。
- 原子预留额度、受控流式写入、失败回收、到期不可访问和实际物理删除都要实现。禁止仅前端判断额度或刷新页面重置配额。
- 每个任务有独立参数快照、owner、source/version；没有跨用户全局 Settings 单例。
- 长任务、设备和原生库进程由本应用明确持有；禁止按端口或泛化进程名杀掉其他服务。

## 6. 第三方与学术署名
- 凡引用外部代码、移植算法、参考论文、使用模型/字典/字体/数据，必须更新来源登记。
- 登记作者、题名/项目、URL/DOI、实际版本或 commit、使用位置、关系类别、修改说明、许可证、查验日期和未决项。
- 关系必须区分依赖、代码移植、论文方法、仅参考、数据素材与项目集成；登记表的 kind 使用对应细分类别，不能把方法参考写成原创实现或实际运行依赖。
- 上游今日 HEAD 不等于本项目最初使用版本；缺失证据标 unknown，不捏造 commit、作者或许可。
- 软件“关于 → 开源与学术致谢”、模块“方法与引用”、说明书与随包许可从同一登记生成；不得只在源码角落署名。
- PDF 优先链接作者/出版社/官方站点，标清论文/手册版本；不擅自镜像或打包原 PDF、研究数据。
- 声道说明必须包含 VTL 2.4 引擎、VTL 2.3 参考手册、几何资源各自来源及本项目适配边界。
- 未明确的再分发许可是对应发行物的验收阻断项；不把整套软件标为“完全原创”或不加区分地宣称全为 MIT。

## 7. 环境、测试与发行
- 继承 v2 基线的 Windows 打包仍仅使用 conda phonetic_311，并采用 python -m PyInstaller；P01 仅在隔离试验环境构建本地单文件探针，不打包或发布完整 v3。
- v3 开发/发行环境在 P02/P12 中单独创建并锁定，不能升级污染 v2 的 phonetic_311；具体环境名/Qt 宿主在原型验收后冻结。
- Windows 单文件直用版与安装版都保留；不能擅自以文件夹版替代已约定单文件目标。若单文件验证不通过，报告并修订 ADR。
- macOS、Linux 原生组件分别构建测试；WSL 的 Linux 服务测试不能冒充 Mac 或 Linux 桌面设备验收。
- 不改 .env、凭据、系统 CUDA/运行时、WSL 全局配置、CI/CD、生产数据库或对外发消息，除非该动作已在会话中明确授权。
- 操作 Windows 路径用绝对路径和 LiteralPath；递归删除/移动前验证最终路径位于明确目标内。文件编码 UTF-8 无 BOM。

## 8. Codex 测试稳定性暂行约定
- 2026-09-09 两次 Codex 退出都紧随最后一张内置浏览器测试页关闭。根因尚未确认，证据见 docs/testing/p06-recovery-and-codex-exit.md；暂不在本项目调用内置浏览器关闭/清理接口，也不为复现而重复该操作。
- 优先使用独立测试进程与已保存截图；需要新 UI 验证时用项目拥有的独立浏览器及独立用户配置目录，不操作用户正在使用的浏览器。服务清理由其父进程/EOF/停止信号负责，不依赖页面关闭成功。
- Git 操作明确限定 v3 根目录并核对索引；本地提交采用已审阅文件清单，不将用户目录、环境或测试数据库纳入。该措施是绕开触发路径，不宣称已修复 Codex。

## 9. 交付记录
每次交付写清：完成任务 ID、修改原因、真实验证命令和结果、来源更新、剩余限制、下一项依赖。实现状态用 planned / in_progress / verified / blocked；P00 可用 documented-baseline 表示仅规划/源码基线完成，不能冒充业务 verified。

- 井井在M01-D后回复“继续”，并追加单/双声道、波形高度、紧凑列表/全选、Praat语谱图和长音频显示要求。M01-E现为verified（限定Windows目录、草稿、预览与显示），见docs/testing/m01-workspace-report.md。新增m01-ui项目内环境、真实Praat受限预览；随后F2已接通持久页面，完整模块仍in_progress，下一项M01-G。
- 井井在M01-E布局修订后回复“继续”，已完成M01-F1执行准备：244项Windows定向测试、WAV与可选参数切片实际回读、批次策略和005增量SQL，见docs/testing/m01-execution-preparation-report.md。F1当时未执行005。井井随后对docs/testing/m01-migration-review.md回复“好，继续”，已授权并执行两库005和限定合成测试；F2持久提交/取消/恢复/发布及页面实际保存现已接通，见docs/testing/m01-persistent-report.md。下一项G，不重复索取005授权，不重复DDL。

- 井井在F2交付后回复“好，继续”，已执行M01-G本轮Windows联合审阅：真实上传/三组双格式下载回读、四份已授权自然录音对照、明确错误与小窗口布局及说明书更新；见docs/testing/m01-report.md。完整G/M01继续in_progress，优先补旧PKL图形转换和历史参数导入，不能把新文件供v2读取通过说成旧文件导入已实现；005不重复执行，测试继续用独立Chrome/Qt。

- 上条后的“继续”已执行M01-G旧格式批次：本机有界符号PKL转换/图形保存、历史XLSX/SQLite显式关联及原帧同步切分均已Windows定向验证，见docs/testing/m01-legacy-report.md。430项Python、17项前端、15步Qt及真实Chrome下载回读通过；原v2只读。完整M01仍in_progress，下一步39项最终逐条审阅，再迁M02；不重复DDL，不运行Codex内置浏览器关闭。

- 2026-09-11 井井将 M10 声道提前迁移用于 Windows EXE 录视频，之后明确追加 R4 录制增强与短帧/静音帧。M10/R4 现为 **verified / 已迁移（限定 Windows 本机声道录制）并暂时冻结**。历史证据见 docs/testing/m10-report.md，最新证据见 docs/testing/m10-recording-features-report.md，说明见 docs/manual/vocal-tract.md，入口为 dist/m10-recording/PhoneticToolbox-v3-M10-R4.exe。169 项 Python、22 项前端、12 组几何、两组真实单文件 Qt 与当前/六视图独立视频解码通过。最短帧 0.05 秒、新增默认 0.2 秒，静音帧不允许绘制 F0；包含本地文件/构形库、150 Hz 默认、缓存及同步视频。原生桥接 m10/2 保持不变，包含此前打包 ICU/UTF-8 修复。本任务只更新 M10，其他模块按各自计划和授权推进；其他平台声道仍 planned，不把本次标为全平台或正式发行。

- 2026-09-11 井井追加 M10/R5 起声与静音后渐入、拖动排序、一键清空，明确要求修改后直接生成 EXE、不检验。本次仅实施与打包，不运行测试或自检，也不将 R4 verified 扩展到 R5。最新入口 dist/m10-recording/PhoneticToolbox-v3-M10-R5.exe，说明见 docs/plans/2026-09-11-m10-onset.md。
