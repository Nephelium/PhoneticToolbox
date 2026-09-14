# M12 V2 源码与说明书映射

2026-09-14，Windows 开发态功能 verified；逐组证据见 [验收报告](../../testing/m12-report.md)。来源是相邻 V2 的 `phonetic_toolbox/gui/resources/web_praat_editor/app.js`、`index.html`、`default.dict`、`services/web_praat_server.py`、`services/io/lip.py` 与 `Phonetic_Export/index.html` 第 13.1–13.6 节。散列清单由迁移脚本生成于 `third_party/m12-migration.json`。

| 功能 | V2 函数/行为 | V3 落点与验收重点 |
| --- | --- | --- |
| F01 语料/层 | find_items/preferred_textgrid_for_wav/applyTierNamesFromInputs；递归 WAV，_webedit > _post > _auto > 原名；words/phones 记忆 | annotation 文件能力、页面层设置；扫描不丢编辑，重名目录不误配 |
| F02 编辑 | onGridMouseDown/Move/Up、moveBoundary、dragWord、splitPhoneAt、deleteSelectedBoundary；20 ms 边界间隔，15 ms 插点限制，50 步撤销；Ctrl 多选/复制/粘贴，Backspace 清字，Alt+Backspace 合并 | 实例化 editor、共享波形/播放、层编辑器；拖动、多选、联动、Unicode 输入、撤销、导航 |
| F03 文本 | parseDictText/pinyinToPhonesFallback、labIndexForWordSelection/nextCopiedWordText/pasteCopiedWord、doSearch/replaceCurrent/replaceAll | 词典和词表独立状态；全部替换显示数量，可撤销 |
| F04 参考 | replacementWindows/complementWindows/clipIntervals/spliceTier/applyReferenceSplice；所有同名层拼接 | 4 模式、无同名层/无参考/反向区间提示，其他层保留 |
| F05 强度 | localIntensityEnvelope/detectIntensityBoundsForWord/fitIntensityRange；第一声道、25 ms RMS、5 ms hop、平滑半径2、−50–80 ms | 保留原数值步骤，内收/外扩/静音与原版直接对照 |
| F06 唇偏 | read_lip_data_json/resolve_lip_time_axis/save_lip_offset；实际时间轴，open/outer_width，单一 metadata.lip_manual_offset | 同一轴，PKL 安全读取和保留元数据写回，安全 JSON 下载，两曲线独立可见性 |
| F07 保存 | serializeTextGrid/saveTextGrid/savePendingChanges；默认_webedit、手动后缀、切换前/60秒自动保存 | 版本冲突保护、原子写、失败不清 dirty；网页版本另存受既有配额管理 |

迁移修正与替代边界见实施记录。旧 JS 的长格式解析会忽略点层，V3 必须保留点层或明确拒绝，不能静默丢弃。旧空后缀实际服务目标由 WAV stem 派生，界面必须展示真实目标，不能误称覆盖当前带后缀源。
