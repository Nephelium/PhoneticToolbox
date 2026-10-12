# M13 结果排版、统一字音修复与说明书验收

当前状态：verified，范围为 Windows 源码工作台、独立 Chrome、实际 Qt 和本章说明书。最终 EXE 未重建。

## 当前行为

- 默认输入与结果两栏等宽，设置栏独立；用户拖动或键盘调整后的宽度仍可保存。
- 等宽排版默认关闭，开启后每个字音单元按当前结果最大的自然宽度对齐。字号、字体、声调、读音、显示模式和儿化组合改变时重新测量，标点同宽，空行保留。旧 v1 草稿兼容。
- 行距改变实际行间空隙，页面与 PNG 使用共同排版规则。PNG 不绘制多音小三角，页面仍保留选择读音入口。
- 字符数位于汉字输入标题右侧，输入与结果框底边对齐。儿化提示采用自动组合后请核对独立的儿字。儿化操作只在 Unt 两项显示。
- 引用卡删除各单字词条入口，保留修訂本和異體字字典两个总入口。查证凭据留在内部修复登记，不在界面逐字列出。

## 字音与来源

历史 HTML 冻结不变，生成表保留 21,572 行的汉字和顺序。修复配置为 `frontend/src/modules/mandarin-ipa/reading-repairs.json`，版本 `m13-reading-repairs/2026-10-11`。

已查证修复包括丢/丟、铥/銩、呗/唄、数、葛、额、距，共10条记录；另恢复29条被缺失值处理误删的 nan 拼音。拼音字段共改变39处，声调字段改变4处。其余八列IPA各纠正10个错误单元，其他单元不变。相同音节采用本列既有值；diu 按本列 di 声母和 liu 韵母组合，属于原标准内部补齐。额与距原IPA分别与错误的 ge、shu 同音节行一致，已按 e、ju 修复。没有进行全表逐字词典审查。

查证来源为教育部[重編國語辭典修訂本](https://dict.revised.moe.edu.tw/)及[異體字字典](https://dict.variants.moe.edu.tw/)。具体词条URL、原字段及修复依据见修复配置。Unt 两列继续依据[作者网站正文](https://phesoca.com/aws/268/)2026-10-11访问快照，更新记录至2024-10-11，采用中派、自由变体第一项。

当前生成表 3,095,722 字节，SHA-256 `0213de35105d0251a4776f60461fee73b52a415620b711be1a69c1ac9343773f`。全行比较见 `output/validation/m13-result-layout/mapping-audit.json`。Unt 宽/严两列相对冻结旧表改变6860/9614个单元，包含作者正文修订与本次纠错。

## 说明书

先完成模块后修改正式说明书。删除声调显示和拼音 ü 独立小节，不再使用女略示例。正文及行内代码中的音标使用内置 PTB-Doulos（Doulos SIL）；汉字字体按正文设置。

本章全部17张在用工作台截图均由实际 Qt 重拍，2560×1600、实际DPR1.5（150%DPI）、页面100%、最大化、浅色、侧栏折叠。原生截图2561×1601仅裁去右下边缘各1像素，没有拉伸。图14.1及其他工作台图均按100%图宽展示。PNG示例另由实际Qt保存，3600×906，共18张图片。

实际 Qt 输入与结果宽度均649.333374 CSS px，底边差0，字符数位于标题。等宽开启时本例单元均65 CSS px。严格说明书校验零错误零警告；本章18图在实际阅读页加载并解码，Doulos字体已加载，删除小节及示例不再出现。未扩大到全书各章视觉验收。

## 已执行验证

- `node --experimental-strip-types --test frontend/tests/m13.test.ts frontend/tests/m13-unt.test.ts frontend/tests/source-coverage.test.ts`：20项通过。
- `node --experimental-strip-types --test frontend/tests/manual-reader.test.ts frontend/tests/manual-reader-render.test.ts`：13项通过，覆盖普通文本与行内代码的音标字体。
- `node frontend/src/modules/mandarin-ipa/generate-data.mjs --check`、前端 `ui-data:check`、`typecheck`、`build`通过。构建保留现有大块体积提示。
- 结果排版独立Chrome 8组（浅深×1440/900×页面1/1.5）通过：最大宽度、字号重测、真实行距、空行、儿化、草稿、默认等栏及手调保存、离线PNG。行距0.8到3.0时示例PNG由1260×1086变1260×1602，行进距由67变124.2，未绘制三角。
- Unt独立Chrome 4组通过，其他9项隐藏儿化操作、切回保留决定，组合/拆分、选音、草稿与PNG通过。
- 来源独立Chrome 4组通过，字典卡仅两总链接，方法页、关于页和复制引用正常。
- 实际Qt17图及PNG通过，实际说明书阅读18图通过。测试自有数据库副本及浏览器profile已在进程退出后清理，保留报告与最终截图。

最终内容完整性检查确认本章18张图片的源文件、public及dist字节一致，17张工作台图的尺寸与DPI登记一致，来源卡仅两个总链接。文档扫描1918个文件，当前错误为空，历史快照链接单独保留。证据为 `output/validation/m13-result-layout/final-integrity.json` 和 `docs-validation.json`。

最终证据：

| 范围 | 本机报告 |
| --- | --- |
| 结果排版8组 | `output/validation/m13-result-layout/chrome-4bb55b9cd4c3417787e31784e175c5ea/report.json` |
| Unt回归4组 | `output/validation/m13-unt/chrome-deea8a5439c448f3aa08da79e9f09d99/report.json` |
| 来源4组 | `output/validation/m13-sources/chrome-62fc7bdcadbf47a591c23e31fe631ca0/report.json` |
| Qt17图及PNG | `output/validation/m13-result-layout/manual-qt-026ce15269b2471c80a85723f36a6ce0/report.json` |
| 说明书18图 | `output/validation/m13-unt/chrome-5a231fe6da3847b6a23420363f6cde53/report.json` |
| 说明书严格校验 | `output/validation/m13-result-layout/manual-validation.json` |

检查期间暴露的截图脚本引号、脏草稿按钮匹配和过时字体错误文案定位均已修正；Qt默认两栏曾有1.333px差，改用实际分数宽度后两栏完全相等。阅读渲染断言按实际Vue片段结构修正，13项重新通过，无跳过断言。失败运行的自有scratch也已清理。

未验范围：最终EXE、实体输入法、多屏和其他物理DPI、跨平台、全表逐字词典核验，以及其他转换标准原书版次/页码逐字对应。萡/be和作者未列出的特殊儿化继续保留缺值，不推测补入。
