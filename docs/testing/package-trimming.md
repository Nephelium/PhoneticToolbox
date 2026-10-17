# 3.0.3依赖精确裁剪验收

2026-10-17。井井授权应用前轮候选、写成规范并重新打包，随后明确保留其它语言。规范见[严格打包规则](../../release/PACKAGING_RULES.md#31-已核实的运行时裁剪边界)，只改新stage，开发环境不修改。构建与最终字节本机验收均已完成，服务器与GitHub发布按本轮明确授权接续。

## 范围与输出对照

- 排除非必需NumPy测试源码/数据、OpenCV Haar级联XML、Python IDLE及交互帮助主题。完整保留公共numpy.testing、_natype及其包入口、原生辅助、全部原科学库/MKL变体、字体、版本和许可。
- 32项依赖/应用资源/共享边界及验证后篡改拒绝检查通过。真实科学/媒体输出须在相同冻结源码和隔离Windows PATH下对照，不能用导入通过替代数值和编码结果。
- 首轮runtime-bb0b3aa67a3e438a93ceb9d624ba4f14的EGG/LPC/REAPER指纹和输出一致。M05两侧均在动画导出缺少metric-spec.json处失败，记录保留；原探针只复制legacy-runtime.json，已修为复制实际resources/m05输入，完整重跑，不删除断言。首轮scratch已deleted。
- 重跑`output/validation/package-trim/runtime-0b8329e7576c4e059eb7285b5baa3783/`两侧EGG/M05全部退出0，科学/图像与媒体签名和环境指纹一致。覆盖EGG四种模式、REAPER、LPC、字体、公共NumPy测试API及线性求解、两种FaceMesh图、可变PTS视频分析、CSV、原录音MP4/AAC/WAV保存与回读、动画MP4/GIF编解码。测试媒体scratch已deleted，新stage保留作构建输入。
- 新旧清单9729→8550文件，排除1179文件/26,008,692逻辑字节，其余大小/SHA完全一致，无新增运行时文件。分组：998份NumPy测试13,051,382字节、17份Haar XML9,763,649字节、158份IDLE1,640,669字节、6份帮助主题1,552,992字节。保留公共numpy.testing下四份测试辅助，因此比原候选少裁144,770字节。见`output/validation/package-trim/runtime-diff.json`。
- 修复从已有归档恢复时，许可补充清单与已有相同目标重复计数304,692字节的问题。实际文件与科学结果不变，首轮运行工具已载入旧计数实现，现有stage只订正计数字段并保留前后清单SHA与`count-correction.json`。构建器仍逐文件核对已比较的选择清单，源环境不修改。
- 本轮前端442测试、类型、UI数据及隔离构建通过。公共源码入口`Start-Research-Workbench.ps1 -Module home`用独立配置实际显示首页/字体，正常退出0，指定自有窗口截图可读，scratch已deleted。证据`source-preview-32bd94743a324d7698f4db52c56c8319`。冻结科学源码333文件与比较输入一致，仅正常sync_versions重建的egg-info/SOURCES.txt不同，初次全文件诊断保留，见scientific-code-closure.json。
- 另修打包时宿主重复压缩：完整宿主文件和共享路径保持时，先校验旧归档及旧清单，再更新当前科学绑定；实际输入变化仍压缩。32项包含schema、文件摘要/路径、共享映射及未知字段变化拒绝复用。本轮真实命中，119,863,056字节宿主归档SHA保持`d5cd940ba6093d2896ed90feb92ed37bb75baaf15b259d97aba231c1d76268d3`，回执在新work的host-cache-rebinding.json。

## 冻结与分发身份

构建`PhoneticToolbox-303-20261017-015527-89c0e7`，完整流水`output/packaging/<构建名>/report.json`，864.918秒。当前科学归档重新压缩为235,146,080字节，比旧238,184,372少3,038,292字节，SHA`14354853dabf5a6b81dffd3236e5ef531ce8b6a9699ac11f6d4869077cd821e9`。科学清单8550文件/1,146,241,887展开字节，SHA`f5a9c1f795ddec735f1b4433ba9f7eca255af4341dc80108b4b63d6d0e03385b`。主Qt原生身份保持，73条共享路径/235,559,072字节按原映射复用。

新免安装EXE为494,149,024字节，SHA`3a4158ff9f2126d1287b5b6eeeac689614c0d28da9a13d9c223869a3673ae0cf`，比上一份497,341,875字节减少3,192,851字节。总减幅包含本轮同步当前源码/资源的变化，不能将其全部归为科学依赖裁剪。源码快照SHA`b685c0fb96a43966eb36353a6e081febf284f2c04b0b1eeaed301bd47e168593`，Git基线b596c3da，仅Git身份不能替代未提交输入清单。冻结业务源码及前端在本轮生成，打包/验收工具后续修订不改变已生成产品字节。

成品静态审计与finalize通过，最终归档无上述应排除文件，NumPy必需支持齐全。Qt157份语言文件与旧包逐文件大小/SHA一致，全部宿主文件及共享目标也一致，见`output/validation/package-trim/qt-retention.json`。正式安装器496,063,939字节，SHA `553cdbc2411c2150149b2db66659ab5def673e1ac845f8cba78f175d856621b2`；更新ZIP494,149,590字节，SHA `2e4cb1982007847a44ef190b006c357a09cfa69622784d32378df72615219ab0`。ZIP为同一EXE，CRC/布局通过。两个EXE的Windows文件/产品版本均为3.0.3，三个文件均低于500,000,000字节。

## 历史故障适用性

本轮依赖与归档身份变化，严格规则第0节及第6节适用。最终EXE执行科学分发、默认GPU可见音标页/侧栏/键盘焦点、三字体七字号/浅深/普通与最大化/背景与玻璃/滚动168帧、楷体15号临界宽度288采样，复查PDF切换额外请求和同章DOM/媒体身份。长路径OwnedQA安装/预热/卸载、448无联接回退、租约及研究设置保留、缓存损坏/中断/并发/取消/失败、两种真实换版、组件共享与复制回退按最终字节执行。新旧各三轮交替冷启动、新包三轮复用及跨构建复用单独计时，无本任务构建同时运行。

沿用已修的像素边界求交与自有PID/标题PrintWindow取图，失败保留且不放宽断言。用户随后明确授权服务器/GitHub上传，公开部分在发布记录中单独核对；没有声道科研语义/UI设计变更，既有全仓资源摘要问题不扩大为本轮修复。实体设备、其它显卡、第二台干净电脑及跨平台未验。

## Qt语言资源

原包157份`.qm`、9,414,985字节，覆盖以下36种语言/地区标识。本轮全部保留，最终逐文件大小/SHA复核。各Qt组件收录语言的覆盖不同，保留资源不代表工具箱自身已提供这些语言切换。现有WebEngine的en-US、en-GB、zh-CN、zh-TW四份locale资源沿用已有选择。

| 标识 | 语言 | 标识 | 语言 |
| --- | --- | --- | --- |
| ar | 阿拉伯语 | bg | 保加利亚语 |
| ca | 加泰罗尼亚语 | cs | 捷克语 |
| da | 丹麦语 | de | 德语 |
| en | 英语 | es | 西班牙语 |
| fa | 波斯语 | fi | 芬兰语 |
| fr | 法语 | gd | 苏格兰盖尔语 |
| gl | 加利西亚语 | he | 希伯来语 |
| hr | 克罗地亚语 | hu | 匈牙利语 |
| it | 意大利语 | ja | 日语 |
| ka | 格鲁吉亚语 | ko | 韩语 |
| lg | 卢干达语 | lt | 立陶宛语 |
| lv | 拉脱维亚语 | nl | 荷兰语 |
| nn | 挪威尼诺斯克语 | pl | 波兰语 |
| pt_BR | 葡萄牙语（巴西） | pt_PT | 葡萄牙语（葡萄牙） |
| ru | 俄语 | sk | 斯洛伐克语 |
| sl | 斯洛文尼亚语 | sv | 瑞典语 |
| tr | 土耳其语 | uk | 乌克兰语 |
| zh_CN | 简体中文 | zh_TW | 繁体中文 |

以下示例由QTranslator实际读取现有qtbase语言文件，均来自Qt上游，秋叶没有新增或改写翻译。它们用于Qt标准按钮、文件对话框等。QTranslator读取结果在`output/validation/package-trim/qt-translation-examples.json`。

| 界面文字 | 日语 | 韩语 | 法语 | 德语 | 西班牙语 |
| --- | --- | --- | --- | --- | --- |
| 打开 | 開く | 열기 | Ouvrir | Öffnen | Abrir |
| 保存 | 保存 | 저장 | Enregistrer | Speichern | Guardar |
| 取消 | キャンセル | 취소 | Annuler | Abbrechen | Cancelar |
| 新建文件夹 | 新しいフォルダ | 새 폴더 | Nouveau dossier | Neues Verzeichnis | Nueva carpeta |

## 最终字节验收结果

`output/validation/local-package-0684fd47/report.json` 为success，15阶段均退出0：工程外科学分发、OwnedQA安装器、长路径安装/卸载、进程/缓存/设置、免安装和安装两种真实换版、默认GPU可见绘制、重复工作、像素边界及清理预览/执行。工程外Windows PATH和清空环境下11项真实科学任务、18项帮助以及M16/M17通过。实际安装路径227字符、LocalAppData140字符、恢复350字符，安装57.141秒、再次准备6.078秒、卸载210.547秒；忙碌卸载拒绝、研究资料/设置SHA保持及缓存清理通过。正式安装包同SHA绑定在阶段流水，测试OwnedQA使用独立身份。

可见默认GPU的168帧字体/侧栏/音标页和楷体15号288临界宽度采样通过；PDF切回额外请求0，同章节DOM/媒体保留，目录无焦点框。实际自有PID/标题PrintWindow的`native-owned.png`已人工检查，页面、菜单、字号和中文/音标可读，无缺失文字或黑屏；未使用桌面截图回退。该次正常退出0，工程外QA副本已清理。

`startup-342a9139d1ac4324a6754a922af97012/report.json`：旧/新各三轮交替冷启动中位数55.24493/54.16318秒，允许退化阈值2.762秒，检查通过；新包三轮复用9.559/9.365/8.633秒，科学/宿主/应用三组件均复用，旧→新宿主跨构建复用通过。这里采用离屏软件绘制，只测就绪时间，未清OS缓存，不外推实体机或可见交互速度。测试期间未并行本任务构建，结束后scratch已清。

`copy-fallback-4919fd60218d42b99c80e4048f1cf4ca/report.json`：强制不支持硬链接，73项共享目标实际复制235,559,072字节，418项宿主大小/SHA全部通过，40.234秒，scratch已清。新成品成功后标准清理识别并删除3份旧成品/工作目录，保留4处未完成或同期目录，回执保留。失败阶段诊断保留，不删除失败记录或降低断言。

被替代的首轮失败stage `output/release-staging/runtime-trim-303-20261017`拟在校验归属、绝对边界和无联接后清理，但整条命令被自动审批返回`blocked by policy`，未执行也未重试。当时失败stage与诊断保留，回执`output/validation/release-303/failed-stage-cleanup-blocked.json`。2026-10-17按新的明确清理授权复核后，已删除其中可再生的egg/m05两目录，selection-manifest.json与stage-report.json继续保留；成功v2 stage全部字节SHA不变。最新回执为output/maintenance/obsolete-versions-summary.json，详见[版本清理](test-temporaries-cleanup.md#2026-10-17-冗余版本与过时安装包)。

实体设备、其它显卡、第二台干净电脑、长期完整负载及跨平台未验。当前Git工作区含同期源码变化，未把Git基线等同于冻结源码；公开成品身份以本节EXE/安装器/ZIP、冻结快照摘要与最终字节报告为准。

## 发布记录

井井已编辑并指定使用`release/notes/3.0.3.md`。完整更新文案与应用更新说明使用同一文件，不混入本机验收记录。实际更新弹窗用该文件复核浅深、窄屏、大字及滚动，最终文案摘要和回执在`output/validation/release-303/`。

参数预测模型来源MODEL-M09-PARAMETERS在冻结后获井井明确许可随软件公开分发，权重303,856字节/SHA `e6edbf5c450fc08f4ff4c39d2828a15e24c4dd7286c619308c671ed4df7dffdc`与本次EXE绑定，见来源登记和发行附件`source-permissions.json`。冻结内元数据仍记录授权前历史状态，后续授权针对原字节生效，不以改动权重或重建应用代替许可核对。授权不包括原始训练语料、独立神经组件或权重加入源码Git。

已正式发布[GitHub v3.0.3](https://github.com/Nephelium/PhoneticToolbox/releases/tag/v3.0.3)，为Latest、非草稿/非预发行，共7件附件：三个成品、对应GitHub URL的ptb-release.json、SHA256SUMS、来源许可及构建来源回执。全部大小与GitHub服务端SHA-256一致，正文与井井确认文件逐字一致。

服务器三个不可变成品全量SHA回读通过，ZIP由同字节EXE在服务器推导并核对精确SHA。版本目录4份小清单及stable/preview两个latest.json均已上传，发布前确认两频道仍为原3.0.2摘要，发布后9条路径全量SHA回读通过。公网HTTPS的版本清单/两频道逐字核对、三种下载的HEAD长度与1024字节Range及本地首段相同通过。最终冻结包的实际更新传输/列表/解析器分别从服务器与GitHub发现3.0.3，notes与两个kind的大小/SHA相同，未下载或应用更新。回执`server-published.json`、`github-published.json`、`https-published.json`和`completion.json`在`output/validation/release-303/`。

初次发布文案SHA为`c445ce927c1433e5690758d01e1638d30ecaed30fec507d84c3065de541c81c9`。井井随后明确确认仅补齐最后一条结尾的`**`，本次严格比较仅新增两个字节，正文、标点、换行与文件结尾其余字节保持。当前文案SHA为`85f400df8d6a4ace387a2ac35c74edb425d3d74dbf953ab9648db0aa635631cf`，实际更新弹窗`check-d5ffecd0-b3b8-420c-88b0-137643db356e`7组浅深/窄屏/大字/滚动检查通过，末条正常加粗，截图已查看。

仅修订服务器版本目录的说明清单、构建来源回执和校验表，以及stable/preview两份说明清单，共5个指定元数据路径。逐项核对旧SHA、检查JSON除notes外字段保持后CAS原子替换，留存私有回退元数据，常规发行器的不可变成品策略不变。同步GitHub Release正文及对应3份小元数据附件，原三份成品及许可附件的大小/SHA和GitHub附件ID保持，tag/目标提交/Latest及7附件数量保持，没有重打包或重传成品。公网三份服务器说明清单与最终冻结更新解析器的服务器/GitHub两来源读取当前全文一致，包大小和SHA仍为本报告已验身份。修订前回执及定点修订证据位于`output/validation/release-303/notes-format-r1/`，当前主回执同步新摘要。

2190文件/443来源/138任务文档校验零错误，旧历史快照缺链单列；版本无漂移、UI数据检查和定向diff检查通过。

本轮上传对象为发行附件与自动更新清单。同期工作区未整体提交或推送，v3.0.3标签对应构建基线b596c3da21c9a94590bc1357daa1cf0a7deb7f5e，实际冻结快照包含基线后的本机修改，二者边界已在公开`build-provenance.json`明示。完整冻结包含软件专用资源，未作为源码附件公开，未上传训练语料或另行开放权重。官网发行信息后续按截图反馈完成单独同步，保留其它内容与同期样式，详见[官网验收](website-redesign.md)。本次未修改正式网页版服务。
