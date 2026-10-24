# TextGrid 三层、关联与浮动音标键盘验收

日期：2026-10-21。实施范围见[计划](../plans/2026-10-21-m12-three-tiers-keyboard.md)，使用说明见[模块 README](../../frontend/src/modules/annotation/README.md)和正式说明书 `manual/chapters/m12.json`。

## 行为与数据

- 三个可见全域区间层分别为音节/词、音素、自定义第三层。创建和选择层保留原文档全部其他区间层、点层及时间域，保存格式仍为原 TextGrid，未迁移现存数据库。
- 吸附默认关闭，按六个显示像素选择其他可见层的合法边界；取消后按实际指针时间编辑。三个关联开关组成可见层的关联组，默认词/音素关联，第三层独立。重合边界双向移动/删除，邻接限制取所有关联层的交集。
- 第一层整体移动、删除和剪贴携带关联层选区，音素与第三层的区间内容操作独立。关闭关联后文字词典、手动定界、词表顺序生成、移动/删除及强度贴合保留独立层。贴合继续使用原 RMS 检测方法，仅对关联的重合边界传播调整，无法保持正时长时整体放弃该词调整。
- 原 50 步撤销范围保留。键盘撤销仅保留弱关联身份与光标元数据，不额外持有已退出原历史栈的大份 TextGrid 快照。不可动端点、碰撞拖动和无结果贴合均保留撤销历史及 dirty 状态。第三层角色随原本机录音视图恢复，网页吸附/关联字段进入既有账号白名单。

## 键盘与性能

- `IpaKeyboard` 为独立按需块；精简 JSON 81,822 字节，相同权威 catalog 为 929,936 字节。内容投影只保留表格、字符与插入模式，共用 Plus 的 grapheme/连线/成对记号实现及原字体，不加载说明、例音与播放控制。
- 面板标题移动、右下角宽高调整、60–180% 字号、IPA/extIPA/VoQS 切换、Escape/关闭、文本选区替换和组合符号有效。输入法组字/保存/未选区间时键盘不插入文字。每个按键沿用原码位。
- Qt 原生 Home/Shift+Right 建立 0..1 选区后，将远处浮动键滚动到可见位置会把输入框选区折叠到 0..0，输入焦点仍保持。旧实现实测产生 `æab`。现在缓存最近主动选字/光标操作，按实际区间对象与文本校验，Qt 原生同流程产生 `æb`，文本框 Ctrl+Z 恢复 `ab`，再次点击回到 `æb`。相关失败报告保留，未降低断言。
- Chrome 性能探针：120 步真实面板拖动/调整大小及三次字号调整期间，主图 `drawImage/putImageData` 调用增量为 **0**，351 个 IPA 按键文字节点变更为 **0**。容器位置/大小每帧最多更新一次，不推进标注 revision。此结论针对指定交互，不等同于整机长期性能或所有系统的帧率保证。
- 全按钮着色、效果及磨砂壁纸同时开启时，按键最终 `background-image/backdrop-filter/filter/box-shadow` 均为 `none`，过渡时间为 `0s`。70/100/150% 页面缩放及 900×650 视窗中，键盘保持可达。浅深截图已核对，Qt 测试使用 offscreen WebEngine，无实体设备。

## 实际验证入口与结果

前端标注/Plus/设置/来源覆盖共 **126 项通过**，Python 原生/偏好 **20 项通过**。Chrome **12 组**真实页面流程和 Qt **5 组**原生流程通过，类型/契约/UI数据/工作台构建通过。说明书 strict 校验及 software 资源生成通过，21章1095项素材无错误。

最终证据：`output/validation/m12/three-tiers-browser-29e737431a684243b4b10f6e3fd17192/`、`three-tiers-qt-355cebac2a1b4eddb86d57e3fdc83cae/`；定向日志 `output/validation/m12/three-tiers-tests.log`。以上两次最终 scratch 均有 deleted 回执。文档检查2377文件、448来源、172任务零错误，历史快照断链单列；架构检查零错误，定向 git diff --check 通过。命令包括：

```powershell
node --test frontend/tests/annotation-*.test.ts frontend/tests/m17-*.test.ts frontend/tests/web-settings.test.ts frontend/tests/source-coverage.test.ts
npm --prefix frontend run typecheck
npm --prefix frontend run contracts:check
npm --prefix frontend run ui-data:check
npm --prefix frontend run build:workbench
.venv/m14/Scripts/python.exe -B -X utf8 scripts/verify_m12_three_tiers.py
.venv/m14/Scripts/python.exe -B -X utf8 scripts/verify_m12_three_tiers_qt.py
```

Python 原生/偏好测试通过 20 项。首次直接使用根 pytest 配置遇到旧 v2 coverage 插件选项，随后用当前源码绑定、显式 `addopts=` 运行相关 20 项，未更改测试断言、插件或全局依赖。源数据检查曾指出 M12 新来源未登记到致谢覆盖表，已补齐实际使用条目并原样通过检查。

临时 WAV、SQLite、浏览器配置与 Qt 缓存放在自有 scratch。Qt WebEngine helper 在解释器退出后才完全释放，验证包装器等待子解释器结束后重试同一受控清理入口；各次失败/成功输入摘要、报告和截图保留，scratch 清理单独记录。

## 验证边界

Windows/macOS/网页共用同一 Vue 编辑器、精简数据、Unicode 规则、字体与正式说明书，网页偏好白名单已同步修改。实际证据包括 Windows 独立 Chrome 和生产构建的 Qt/QWebChannel 原生写盘、重开、三层创建及帮助读取。网页已于同日上线并完成实际HTTPS复验，源码窗口已用统一入口打开到M12，详见下节。未push或重建最终EXE/Mac包，Mac实机、Safari、实体IME/触摸与长期负载未验证。


## 2026-10-21 网页上线与源码工作台

按井井明确授权，与自定义汉字/拼音音标字表一起上线 `m12-m13-20261021-77b3ae2db3cc`，入口 <https://www.phonetictoolbox.com/app/>。冻结417项源码，使用2026-10-20线上源码作控制，只覆盖M12、M13、键盘精简数据、共用来源致谢与M12偏好白名单。控制构建50个可达块与线上归一化一致。说明书只合并M12/M13正文、章节目录及搜索条目，其他章节保持原字节。

候选真实HTTPS账号通过14组，发布后独立Chrome通过15组，其中M12十组涵盖三层独立/传递关联拖动、可选吸附、IPA/extIPA/VoQS选区输入/撤销/移动/调整大小、实际四项偏好白名单与刷新恢复、自动保存/历史恢复/删除及本机下载。M13五组涵盖CSV/TSV/XLSX多工作表、命名/多读音/拼音匹配、PNG、完整浏览器重启恢复、独立草稿与帮助、浅深/窄窗。两模块研究上传和页面异常均为零，公网浅深截图已检查。

Linux九类科学任务、交互预览及0.5秒声道检查通过，新报告绑定当前773项运行清单。发布清单2203项和38项变化HTTPS资源逐项核对，Windows另独立回读入口、键盘块和两章正文共4项。旧发布2171文件再次完整比对保持，原生科学资源复用既有版本。切换前确认无排队/运行科学任务，只重启既有工作台API，原服务单元/运行配置/发布目录保留回退，官网、论文目录、Nginx和反馈保持。未修改凭据内容或数据库schema。

统一入口先运行`-CheckOnly`，再以`-Module M12`打开当前源码。实际窗口`PhoneticToolbox 3.0`，PID23396，句柄89794360，Responding/foreground均为true。没有结束其他手动启动的工具或浏览器。

证据 `output/validation/m12-m13-web-20261021/`：`deployment.json`、`final-summary.json`、`windows-https-readback.json`、`readback-remote.result.json`、`source-window.json`及各次报告/截图/冻结输入/部署脚本。6个自有QA身份全部停用，11项远端传输/科学临时副本及本轮scratch已按固定清单清理，报告与回退保留。首轮检查脚本错误来自生产构建无Vue开发属性、登录尚未呈现和恢复模块时仍等待首页，已改为读取真实下载TextGrid、等待登录/导航并从实际模块继续，未改产品代码或降低断言。

实际新增证据为Windows Chrome、Windows源码前台窗口和Linux服务。最终EXE/macOS包未重建，macOS/Safari、实体触摸/IME和长期负载仍未验证，未push或发布GitHub Release。


## 顶部工具弹窗与明确选择声道

日期：2026-10-21，按井井两次界面反馈追加。顶部增加吸附与关联、强度贴合两个按钮，原图下吸附/关联行和强度参数卡移入既有ModalDialog，关闭时卸载控件，不占标注区高度。关联立即生效，完成/Esc关闭；贴合显式执行，结果或错误在窗内显示，焦点回到启动按钮。

强度贴合初始为请选择声道，未选不可执行。计算声道作为明确参数传至原RMS和边界检测，未给或非法声道拒绝计算，保留撤销与数据。关闭重开同一录音的弹窗保留选择，切换录音后重置。波形、语谱、相对强度和试听同步使用选择，切换时暂停该录音原试听，语谱缓存键加入声道；未复制音频数组或增加后台持续计算。长文件若走原单声道轻量预览，只列该预览实际可用声道，原预览预算及原WAV保留策略未变。

93项标注测试、类型、协议/UI数据、保留旧哈希资源的工作台构建通过。原V2第一声道数值对照改为显式传0，25ms/5ms RMS、阈值和内收默认10ms原样保持。新增静音左/有声右声道数值测试：左声道无调整，右声道边界变化且不读取左声道，撤销复原；无/非法声道不读样本、不消耗撤销。说明书21章1095素材strict/生成零错误与警告，仅修订M12实际入口/声道描述、索引和模块README。

Windows独立Chrome15组通过：关闭弹窗无内嵌控件，选择自动带入，声道未选禁用、静音左不改/有声右改边界及撤销，波形与语谱声道一致，错误时间范围不编辑；浅深、1440×900和900×650/150%中弹窗首尾可见。三层/吸附/关联/键盘/保存回归通过，键盘拖动120步仍为主图绘制0及351键文字变更0。实际Qt六组通过，包含弹窗、明确选择第2声道后的实际原生TextGrid数值边界、IPA输入、保存/重开、三层创建、帮助和原文件摘要。Qt为offscreen，GLES3回退日志保留，不当作实体显卡验收。

同批网页先发布顶部控件，井井追加声道后再次发布`m12-intensity-channel-40b03d401c4743eeae32e31819236eef`。最新控制构建54块与当时线上归一化一致，冻结源码只覆盖AnnotationPage/Tracks/editor及声明四文件，M12正文/目录/搜索同步。最终候选12组/实际HTTPS13组通过，真实下载的TextGrid证明左右声道差异与撤销，实际AudioBuffer证明所选右声道非零发声且换声道停止原试听。零研究上传、零页面异常；17项变化HTTPS大小/SHA、2229项发布清单核对，773项运行文件/原科学回执/后端/各服务PID保持，未重启服务，1794份上一版网页文件完整保留可回退。M13自定义字表等其余源码及手册章节保持。

证据：`output/validation/m12-channel-web-20261021/`的最终候选/公网报告、`deployment.json`、`public-readback.json`、`final-summary.json`；首轮控件发布证据在`output/validation/m12-tools-web-20261021/`。最终本机Chrome为`output/validation/m12/three-tiers-browser-3f94dcfab1334bda82cd8dcf83e9f94b/`，Qt为`three-tiers-qt-27ac4b4dbabb481bb2eb325305577a91/`。旧候选包/结果保留，以最终入口SHA为准。两次网页共5个任务QA账号已停用，传输包由部署器核对后清理，两次网页scratch和本机各次Chrome/Qt scratch均有deleted回执，报告/冻结源码/截图与远端回退保留。

未push、重建最终EXE/Mac包或迁移数据库。当前本地构建已更新，仍打开的旧源码窗口需重载后读取新资源；未在用户编辑录音时强制刷新。Mac实机/Safari、实体IME/触摸、长期负载及长文件其他原声道的重新解码未新增验证。

本次收尾：正式工程manual/project.json的M12目录与159条搜索记录已同步当前正文。文档2378文件/448来源/172任务零错误，历史快照断链另列；架构零错误，定向差异检查通过。

## 搜索与键盘顶部入口及说明书截图

日期：2026-10-21。按井井后续要求，搜索替换使用顶部按钮打开既有ModalDialog，关闭后卸载输入与导航行。音标键盘唯一入口移到同一顶部工具栏。查询与替换内容在关闭和取消全部替换确认后保留，取消/Esc确认返回搜索窗，重新打开按当前TextGrid刷新匹配。真实交互发现原始searchIndex变化未触发计数文本更新，已用revision驱动的计算位置修复。搜索/整段替换与关联音素的既有语义不变。

93项标注测试、typecheck、contracts:check、ui-data:check及保留旧哈希资源的build:workbench通过。独立Chrome16组通过，覆盖顶部唯一入口、关闭时零内嵌控件、Enter/Shift＋Enter和可见序号、单项/全部替换、取消/撤销/重开刷新、三层关联/吸附、明确声道及键盘/保存恢复。120步移动/调整大小仍为主图绘制0、351键文字变更0，按键磨砂/背景/滤镜/阴影/动画均关闭。

新增scripts/manual/capture_m12_current.py和register_m12_current.py，实际Windows Qt、原生DPR1.5、页面100%、浅色最大化、收起导航，用自有合成立体声和TextGrid重拍12个旧图位并新增5个工具图。15张全图2560×1600，原始2561×1601仅裁掉最右/下1像素；两张局部876×382与876×624保留父图和裁剪范围。截图及原生保存/原文件摘要等3组检查通过，全部17图已检查。正式手册旧素材完整保留，新素材git:false/software-only，稳定旧图节点原位更新。M12正文、工具表、目录、搜索与模块README同步，strict/生成21章1112素材无错误警告。

从原线上417项冻结源码只覆盖AnnotationPage，54个控制可达块与原线上归一化相同。发布仅更新M12页面、正文/目录/搜索和17张图，其他模块与章节保留。网页已原子发布`m12-search-keyboard-b733d0e814124be2ab576dfb7a6c575e`，候选14组/公网15组通过；真实帮助逐图加载17张并核对像素，零页面异常及研究文件上传。实际下载的TextGrid验证替换、取消、左右声道差异和撤销字节一致，服务器偏好、保存/历史/删除回归通过。服务不重启，后端、科学运行文件/回执、官网/配置/目录和进程保持，旧网页保留回退。

证据：output/validation/m12-search-web-20261021/中的candidate-r2/report.json、live/report.json、deployment.json、public-readback.json、各检查日志、冻结源码与部署脚本；本机Chrome为output/validation/m12/three-tiers-browser-e5db1d76f6a3472a95ac749c4107d052/，最终Qt截图为output/validation/m12-manual/screenshots-20261021-896d5caa22764e18ad8e5c696b752a69/。初轮SFC多余闭合标签、真实搜索序号问题已修复；截图脚本按隐藏Qt窗口实际DOM坐标派发双击、使用布尔等待和现有按钮名称，公开候选帮助标题断言适配章节编号，未移除断言或改变容差。失败诊断和截图保留。

未重启或强制刷新井井当前首页源码窗口。共用源码覆盖Windows/macOS/网页，实际新增证据为Windows Chrome/Qt与公开HTTPS；最终EXE/Mac包、Mac实机/Safari、实体IME/触摸和长期负载未新增验证，未push。

公网收尾核对：34项变化HTTPS资源大小/SHA、2259项完整发布清单、773项运行文件与原科学回执、1807份回退网页文件均通过。三个本轮QA身份全部停用，远端传输包已由部署器核对后清除。文档2380文件/448来源/172任务零错误，既有历史快照断链另列；架构零错误，定向diff检查和两个新手册脚本语法检查通过。

本轮网页scratch的5211个文件已按摘要、归属、联接和进程检查清理，逻辑字节1416009630，不作为实际释放磁盘容量。六次Qt截图运行及本机Chrome各次scratch均有deleted回执。报告、失败诊断、截图、冻结源码/最小增量、部署脚本和服务器回退保留，未删除任何旧正式手册媒体。最终回执为同目录final-summary.json和cleanup.json。

## 2026-10-22 键盘文字与布局同步缩放

按井井反馈，IPA/extIPA/VoQS 的表格标题、行列文字和元音方位文字与音标统一默认 19px。缩放范围仍为 60%–180%，文字、按键尺寸、表格列宽和元音图随比例变化，键盘窗宽按相同比例变化。手动拉宽后的宽度作为新的比例基准；可见窗口受屏幕约束，目标宽度保留，扩大可用视窗后恢复。标题栏操作控件保持可操作尺寸，内容超出时沿用原滚动区域。

实现仅修改共用的 IpaKeyboard.vue 与 IpaKeyboardChart.vue，并同步 README、CHANGELOG、当前状态及台账。继续复用既有 Plus 字符数据、插入规则和原浮窗生命周期。

验证命令：

- `npm run typecheck`、`node --test tests/annotation*.test.ts`、`npm run contracts:check`、`npm run ui-data:check`、`npm run build:workbench`：通过，标注测试 93 项。
- `.venv/m14/Scripts/python.exe -B -X utf8 scripts/verify_m12_three_tiers.py`：Windows 独立 Chrome 17 组，三套表、浅深主题及 60/100/180% 均核对实际计算字号；默认 100% 文字与音标均 19px，窗宽分别 456/760/1368 CSS px。IPA 普通列宽约 27.83/47.09/85.64px。窄窗限制后扩大视窗恢复目标宽度，手动调整后继续按比例缩放。插入、撤销、保存、切换与原三层行为通过，120 步拖动/调整大小与缩放的主图绘制、按键文字重建均为 0。
- `.venv/m14/Scripts/python.exe -B -X utf8 scripts/verify_m12_three_tiers_qt.py`：实际 offscreen Qt 7 组通过，三套表、浅深主题、60/100/180% 实际字号和窗宽通过，原生选区输入、TextGrid 保存/重开、原始输入摘要保持通过。GLES3 回退日志保留，不作为实体显卡验证。

浏览器证据：`output/validation/m12/three-tiers-browser-c5a5e7e0da4c4a38b9dac19b9f30032e/`，Qt 证据：`output/validation/m12/three-tiers-qt-6c455e833678481e8a4a510c515f7372/`。保留报告、日志和浅深截图；脚本按既有生命周期处理独立测试 scratch。首轮浏览器新增窄窗检查使面板停在左边缘，后续旧测试向左拖动无空间，修正测试起点后原拖动断言通过，失败证据保留。

仅更新本地共用源码与工作台构建。已打开窗口需重开载入；最终 EXE/Mac 包、公开网页、Mac 实机/Safari 未更新或新增验证。
收尾：本轮三次 Chrome 与一次 Qt scratch 均有 deleted 回执，报告、截图和失败诊断保留。`scripts/validate_docs.py` 检查 2430 文件、456 来源、178 任务，错误为 0，未改历史快照的已有断链另列。定向 `git diff --check` 通过。