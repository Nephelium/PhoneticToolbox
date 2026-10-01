# M13 P17 逐控件规则

状态：in_progress。本规则依据实际 V2 手册 10.1–10.2、V2 源码与当前 V3 页面。历史通过不算本轮通过。

V2：`phonetic_toolbox/gui/resources/ipa_trans/ipa_converter.html`，入口/事件 `convert / variant / export`。V3：`frontend/src/modules/mandarin-ipa/MandarinIpaPage.vue`。V2 手册原文和源码只读，私有审阅摘录在 `output/validation/p17/source-review/`。

## 控件清单

每个动态 v-for 行还需逐一遍历所有选项/角色/参数。操作前记录当前值，按下表动作更改，检查可见值、dirty、数据归属，再恢复。数值控件另测上下界、空值、越界和键盘输入。按钮另测重复点击、禁用态与失败重试。文件选择器另测取消/无效格式/同名文件。下表逐行记录本轮实际范围。partial 表示所述路径通过，但整行通用边界或组合尚未全覆盖；not_run 未执行；blocked 有明确前置阻断；not_applicable 当前产品无该可用能力。不得把 partial 汇总为全交互通过。证据相对 output/validation/p17；报告相对本目录。

| ID | 手册 | V2 源 | V3 控件/事件 | 前置 | 操作与判据 | 证据 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M13-C001 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 收起提示 · `error=''` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：字体/编码失败后关闭错误并恢复。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C002 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 待转换汉字文本 · `draft.text` | 空态/已载入/结果态按显示条件 | 填入有效值并触发提交；当前值准确更新，空/越界被拒或给出可理解反馈，原始文件不变<br>本轮：已通过路径：空/普通/多音/换行/1200字实际输入。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C003 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 1" type="button" class="m13-token m13-mapped m13-ambiguous" :data-index="token.index" :data-value="token.value" :aria-label="`${token.char}：${token.variants.length} 个读音，当前 ${token.value}`" aria-haspop · `原生控件输入` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：逐位置多音字打开。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C004 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 关闭读音选择 · `closeVariants(true)` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：关闭按钮/外部点击/Esc关闭。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C005 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | {{option.pinyin}}{{option.toneLabel==='轻声'?'（轻声）':option.toneLabel}}{{option.value}} · `chooseVariant(selectedToken,option.index)` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：逐位置选择读音且不影响同字其他位置。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C006 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | {{exporting?'正在生成…':'保存为 PNG'}} · `exportImage` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：实际离线 PNG 栅格和字体调用；字体/编码失败恢复。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C007 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 保存本机草稿 * · `save` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：保存草稿及关闭保护恢复。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C008 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 帮助与来源 · `emit('references')` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：实际AppShell方法来源对话框打开、Esc关闭；链接外站未执行 | `layout-c/1790855035363/report.json` | partial |
| M13-C009 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 转换标准 · `draft.standard` | 空态/已载入/结果态按显示条件 | 依次选择每个选项，检查对应视图/模型/字段更新；禁用项不生效，切回保留合法状态<br>本轮：已通过路径：全部11标准对照明确预期。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C010 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | draft.display · `draft.display` | 空态/已载入/结果态按显示条件 | 填入有效值并触发提交；当前值准确更新，空/越界被拒或给出可理解反馈，原始文件不变<br>本轮：已通过路径：汉字IPA模式。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C011 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | draft.display · `draft.display` | 空态/已载入/结果态按显示条件 | 填入有效值并触发提交；当前值准确更新，空/越界被拒或给出可理解反馈，原始文件不变<br>本轮：已通过路径：IPA单独模式。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C012 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | draft.layout · `draft.layout` | 空态/已载入/结果态按显示条件 | 填入有效值并触发提交；当前值准确更新，空/越界被拒或给出可理解反馈，原始文件不变<br>本轮：已通过路径：左右布局。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C013 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | draft.layout · `draft.layout` | 空态/已载入/结果态按显示条件 | 填入有效值并触发提交；当前值准确更新，空/越界被拒或给出可理解反馈，原始文件不变<br>本轮：已通过路径：上下布局。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C014 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 汉字字号 · `draft.hanziSize` | 空态/已载入/结果态按显示条件 | 填入有效值并触发提交；当前值准确更新，空/越界被拒或给出可理解反馈，原始文件不变<br>本轮：已通过路径：汉字字号独立调整。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C015 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 音标字号 · `draft.ipaSizeUserSet=true` | 空态/已载入/结果态按显示条件 | 填入有效值并触发提交；当前值准确更新，空/越界被拒或给出可理解反馈，原始文件不变<br>本轮：已通过路径：IPA字号独立调整。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C016 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 字音间距 · `draft.gap` | 空态/已载入/结果态按显示条件 | 填入有效值并触发提交；当前值准确更新，空/越界被拒或给出可理解反馈，原始文件不变<br>本轮：已通过路径：音节间距调整。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C017 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 行距 · `draft.lineHeight=Number(($event.target as HTMLInputElement).value)/10` | 空态/已载入/结果态按显示条件 | 填入有效值并触发提交；当前值准确更新，空/越界被拒或给出可理解反馈，原始文件不变<br>本轮：已通过路径：行距调整。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C018 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | B 粗体 · `draft.bold=!draft.bold` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：汉字粗体开关。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C019 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | I 斜体 · `draft.italic=!draft.italic` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：汉字斜体开关。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-C020 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | U 下划线 · `draft.underline=!draft.underline` | 空态/已载入/结果态按显示条件 | 点击并等待可见反馈，检查事件结果和保存/下载回读；重复点击不产生重复结果，取消不丢数据<br>本轮：已通过路径：汉字下划线开关。其余通用边界/重复/取消未逐项覆盖 | `M13-full/1790852653036/report.json` | partial |
| M13-I01 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 11个转换标准逐一与V2数据映射一致 | 对应功能就绪 | 逐项执行上述操作，对照V2与当前授权差异；记录原始样本/产物及失败<br>本轮：11标准与多音字逐位置选择通过 | `M13-full/1790852653036/report.json` | partial |
| M13-I02 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 多音字弹层选择/点外部/Escape焦点恢复 | 对应功能就绪 | 逐项执行上述操作，对照V2与当前授权差异；记录原始样本/产物及失败<br>本轮：字体/PNG编码失败和空输入已验 | `M13-full/1790852653036/report.json` | partial |
| M13-I03 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 左右/上下排布，右侧参数区保持，长文本内部滚动 | 对应功能就绪 | 逐项执行上述操作，对照V2与当前授权差异；记录原始样本/产物及失败<br>本轮：两布局、两显示、四数值与三字体开关已验 | `M13-full/1790852653036/report.json` | partial |
| M13-I04 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | PNG实际回读且包含长文本全部内容 | 对应功能就绪 | 逐项执行上述操作，对照V2与当前授权差异；记录原始样本/产物及失败<br>本轮：原交互组合未完整逐项覆盖，具体已验路径见本轮报告 | `M13-full/1790852653036/report.json` | partial |
| M13-I05 | 10.1–10.2 | `gui/resources/ipa_trans/ipa_converter.html` | 字符输入20次测median/p95/max，含中文、非BMP、换行、标点 | 对应功能就绪 | 逐项执行上述操作，对照V2与当前授权差异；记录原始样本/产物及失败<br>本轮：原交互组合未完整逐项覆盖，具体已验路径见本轮报告 | `M13-full/1790852653036/report.json` | partial |
| M13-G-L01 | 10.1–10.2 | 同上 | 1920×1000 CSS及实际Qt最大化，记录DPR | 对应状态 | 按P17总规则执行并保存DOM几何/截图/原始计时<br>本轮：最新源码五页三栏、右栏收起增宽及重载记忆，四档CSS窗口/两缩放；最终production dist隐藏Qt已验，隐藏窗口inner1440×900，真实可见最大化在此前批次1707×996 | `layout-c/1790855035363/report.json；qt-c/3a44621f31624425bfdcad3fe4944e9b/report.json` | partial |
| M13-G-L02 | 10.1–10.2 | 同上 | 空态/有数据/结果/错误的scrollHeight，长列表只在独立容器滚动 | 对应状态 | 按P17总规则执行并保存DOM几何/截图/原始计时<br>本轮：见报告实际尺寸与截图；未覆盖的尺寸/状态不能外推 | `layout-c/1790852346905/report.json；qt-c/a2d31781aeee478e8969d0fe22e3dea6/report.json` | partial |
| M13-G-L03 | 10.1–10.2 | 同上 | 1366×768/1280×720及125/150%时末尾动作可达 | 对应状态 | 按P17总规则执行并保存DOM几何/截图/原始计时<br>本轮：见报告实际尺寸与截图；未覆盖的尺寸/状态不能外推 | `layout-c/1790852346905/report.json；qt-c/a2d31781aeee478e8969d0fe22e3dea6/report.json` | partial |
| M13-G-L04 | 10.1–10.2 | 同上 | 浅深色/IPA/侧栏拖宽，无裁切和重叠 | 对应状态 | 按P17总规则执行并保存DOM几何/截图/原始计时<br>本轮：见报告实际尺寸与截图；未覆盖的尺寸/状态不能外推 | `layout-c/1790852346905/report.json；qt-c/a2d31781aeee478e8969d0fe22e3dea6/report.json` | partial |
| M13-G-P01 | 10.1–10.2 | 同上 | 20次交互计时及冷/热加载分开 | 对应状态 | 按P17总规则执行并保存DOM几何/截图/原始计时<br>本轮：真实汉字输入各20次到两帧结束，min/median/P95/max=12/17/22/25ms；含自动化开销，不是科学任务耗时；未完成冷/热各20次加载 | `M13/1790855668622/interaction-timing.json` | partial |
| M13-G-STATE | 10.1–10.2 | 同上 | 取消/失败/迟到/切页/关闭未保存保护 | 对应状态 | 按P17总规则执行并保存DOM几何/截图/原始计时<br>本轮：报告列明已验草稿/关闭/失败路径；取消/迟到矩阵未全覆盖 | `M13-full/1790852653036/report.json` | partial |

| M13-CNEW01 | P17追加 | V3公共布局 | 右辅助栏收起/恢复并记忆 | 默认三栏 | 点击收起，中间宽度增加；重新加载仍收起；恢复后全部原控件可达 | `layout-c/1790855035363/report.json` | verified |

## 当前授权覆盖旧行为

M12原始TextGrid优先、三图双击与整段剪贴按R5/R6现行约定，不退回旧手册。M11组件固定版本并独立管理，保存到新目录。所有模块不安装环境，不改现存库、不改V2/原录音。M15试音确认由自动化操作只证明流程门禁，真实听感/设备/物理时延仍需人工。
