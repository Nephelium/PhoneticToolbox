# M17 国际音标 Plus 验证报告

日期：2026-10-02。状态：**verified，限定本报告的 Windows 开发网页与实际 Qt 集成范围**。500 个输入入口、56 个指定 VoQS 中文名称、随包固定字体和双尺寸默认单屏验收完成。Linux GUI、实体 DPI、物理输入法、冻结发行物与极多短行吞吐不在此结论内。

## 实现与入口

侧栏“国际音标 Plus”，主页面 `/#M17`。同页 IPA/extIPA/VoQS 切换，保留真实矩阵与元音图；下方120px多行编辑框可调高、字号可调、手输/点击共用撤销历史、选区/非BMP/组合附加记号安全插入、全选复制、UTF-8下载、清空撤销、本机自动草稿、跨窗口冲突保护、可开关悬停与完整详情。数据无任务/API/上传依赖。

实现清单：`frontend/src/modules/ipa-plus/` 下的 `IpaPlusPage.vue`、`ChartView.vue`、`SymbolButton.vue`、`SymbolDetails.vue`、`catalog.ts`、`types.ts`、`unicode.ts`、`editor.ts`、`storage.ts`、`fonts.css`、`data/{catalog,sources,voqs-zh-names}.json`。字体仅 `frontend/src/assets/ipa-plus/PTBIPAPlus-Regular.ttf` 和两份 OFL 文本。共享接线、来源登记、统一构建由主代理完成。

## 目录与名称

`python scripts/verify_m17_catalog.py` 通过。IPA244、extIPA191、VoQS65，共500出现项/操作入口；VoQS包括原2016表全部56项和9个程度/范围/例示入口。264个目录所用字符不能与500入口混作“独立符号数”。完整覆盖逐行见 [`m17-symbol-coverage.csv`](../references/m17-symbol-coverage.csv)，含来源定位、显示/插入区分、Unicode序列和对应点击测试。

VoQS中文名严格对齐 [UntPhesoca 指定文章](https://zhuanlan.zhihu.com/p/203037479)，例如常态浊声、糙声、室襞性发声、咽门化声。56条独立名称断言覆盖界面、检索与详情用的统一数据；各项 sourceRefs 同时标论文和中文译名来源。英文科学语义和符号仍以原表为准。图表2016、论文2017在线/2018卷期、中文2020译年分列。

## 实际执行与证据

| 检查 | 实际结果 |
| --- | --- |
| `node --test frontend/tests/m17-ipa-plus.test.ts` | 11/11，包括全目录、每项插入/撤销、字素边界、桥接、56术语、圈围降级标识 |
| `npm --prefix frontend run typecheck` | 通过；主代理最终统一全前端226项及构建另见联合报告 |
| `python scripts/verify_m17_catalog.py` | 500项结构、清单和码位校验通过 |
| `python scripts/verify_m17_fonts.py` | 生成字节一致；原3944字形字节不变；264字符无缺码位 |
| `node tests/e2e/m17-ipa-plus.cjs` | 最终17组实际Chrome检查通过，空 pageerrors，输入流程0外部HTTP请求 |
| `scripts/verify_m17_qt.py --require-single-screen`（主代理运行） | 实际Qt开发宿主6组默认布局、输入/撤销/草稿刷新/真实原生TXT下载/来源面板通过 |

最终 Chrome 152.0.7977.83 证据目录：`output/playwright/m17/2026-10-02T07-55-57-780Z/`。`report.json`记录17组、15份布局测量，每个可见按钮 bbox、编辑框 bbox、chart 和 module 的 scrollHeight/clientHeight。`font-platform-evidence.json`逐500节点验证所有符号实际来自自定义 PTB IPA Plus，无系统字体补字。`font-shaping-{0,40,80}.png`为复杂组合形态表，另有真实整页浅/深色截图。

Qt 已读证据：`output/validation/m17/qt-20261002-155652-b8b4c5/report.json`。此轮已包含最终异常处理/两个半阴影格。长例句隐藏逐码位展示后的轻量复核路径由[联合报告](2026-10-02-m16-m17-integration-report.md)登记。实际Qt限定offscreen开发宿主，未冒称冻结EXE或物理DPI。

Chrome实际验证：逐500按钮清空、点击、码位回读；非BMP选区替换；中文与组合输入；切表保留光标；范围模板包选区；统一撤销/重做；IME事件事务；真实剪贴板回读；TXT逐字节读取；IndexedDB刷新；两窗口revision冲突；断网编辑；关闭悬停；剪贴板拒绝反馈；7000 UTF-16单元/1000行长文；注入quota同步异常保留文字并另存；未知版本恢复可读文本而停止覆盖；字体请求失败反馈与文字保留。

Windows剪贴板回读只发生LF→CRLF换行转换，完整记录在 `clipboard-codepoints.json`；UTF-8 TXT 回读包含原LF且字符字节正确。IME覆盖合成浏览器composition事件及Qt输入链，未冒称实际拼音候选窗人工操作。

## 单屏布局

测试均在实际 AppShell，包含224px左导航、标签栏和底部状态，不是裸模块。100%默认界面缩放，三表所有入口均在 chart bbox 内且无纵/横溢出。

| 窗口 | 三表 chart 可见高/scrollHeight | 编辑框位置与尺寸 | 表项超出 |
| --- | --- | --- | --- |
| 1366×768 | 427/427 | x234，y553，1112×120，底673 | 各0 |
| 1920×1080 | 739/739 | x234，y865，1666×120，底985 | 各0 |

深色100/125/150%另有9组截图和测量。125/150%按设计允许表内滚动，底部编辑框与操作仍可见，不声明放大后整表无需滚动。原图PDF纵排已改为横向区块，未删符号或示例凑单屏。extIPA网格26px行高/22px符号，给下记号留白；大视口30px行高。原图 IPA 两个塞音格的右半灰区也保留。

形态审阅覆盖所有VoQS组合、13个圈字母、六类组合括号、扩展非BMP字母和实际低行高表；详情可放大查看码位与完整例示。两项多字圈围明确为 `⟅…⟆`文本降级，**不宣称它们是原图圆框的官方Unicode表示**。

## 发现并修正的实际问题

- 初稿受全局按钮padding和行高影响，1366单屏未通过。重排横向分区并调整模块按钮空间后，保留全部500入口且默认六组通过。
- 初版组合括号紧贴小圈。调整新增轮廓留白后单侧/双侧可辨，保留码位；字体接触表重新审阅。
- 草稿恢复的异步watch曾把打开第二窗口视作修改，导致无编辑写入。恢复后先flush，再允许自动保存，跨窗口真实场景复验通过。
- 同步store.put异常需要abort事务，已补捕获；注入quota后保留文字和原记录，独立另存可恢复。
- 后期人工复核补齐IPA咽/声门塞音的右半阴影；符号数量不变。
- E2E未知版本测试最初错写独立组件默认key，已改用实际AppShell的local:M17；这是测试夹具修正。
- 主代理早期Qt下载复跑碰到reload旧DOM就绪判断，修复等待loadFinished后通过；未归因产品缺陷。

## 明确边界

105000 UTF-16单元、15000极短行一次原生灌入未在30秒内完成。同样输入在无项目代码/无自定义字体的裸textarea也超过12秒，见 `output/validation/m17/long-text-boundary.json`。这支持原生输入/排版路径存在成本，不排除模块也有额外成本，不宣称该压力规模通过。7000单元/1000行实测通过。当前文字不主动截断；极长文本建议按段编辑并及时导出。

网页仅承诺资源加载后的离线编辑，无PWA冷启动保证。本模块未执行Linux GUI字体/剪贴板验证；不外推到跨平台发行。物理输入法、实体DPI/多屏及外部编辑器各自的字体塑形未验。未生成新EXE、未push、未执行DDL、未改用户资料或原Doulos。

配套：[手册](../manual/ipa-plus.md)、[ADR](../decisions/ADR-M17-client.md)、[资料审计](../references/m17-source-audit.md)、[字体审计](../references/m17-font-audit.md)。

最后展示修订：按井井要求，长例句/完整示例详情与proof不再铺开逐字码位，选区超过8码点隐藏页脚码位，内部数据和输入不变。`node tests/e2e/m17-ipa-plus-display.cjs`五项定向检查通过，证据 `output/playwright/m17/display-2026-10-02T08-04-42-249Z/`，typecheck通过。完整500项功能矩阵沿用上述最近通过记录，未因纯展示调整重复执行。

字体许可打包复核：两份OFL以 `?raw` 导入，帮助内可折叠查看，不依赖未引用的源树文本被Vite自动复制。主代理最终 `IpaPlusPage-2om9CVdp.js` 已检查包含两份版权首行，TTF仍为888480字节，最终typecheck/build通过。


追加交付：井井后续授权打包，最终R2实际EXE检查通过，范围与修复见[成品报告](2026-10-02-m16-m17-exe-report.md)。本文件之前的无EXE说明为开发态阶段记录。
