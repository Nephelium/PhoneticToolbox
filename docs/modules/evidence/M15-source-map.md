# M15 说明书、源码与客户端入口映射

2026-09-27。V2 只读来源：相邻 `Phonetic_Export/index.html` 6.1–6.2，全文逐项核对。HTML SHA-256 `acdf62260ac1bd70c9f636cc0e702280ba0baa44c6744e9dd3da3846967364e5`，v3 继承副本相同。说明书 SHA-256 `a46c984929bcd8073ff1daf4e6b382a6685d4ca19e0fee0e29de3ecf0fb39ad5`。

Python `perception_service.py` 仅定位 HTML 并用 os.startfile/webbrowser 打开，没有实验算法或结果存储；`perception_models.py` 只有 PerceptionLaunchResult。新模块不迁移这个独立浏览器启动器，不创建空 core/API。

## 功能与证据入口

新版根目录为 `frontend/src/modules/perception/`。统一正式入口为 AppShell → 标注与实验 → 感知实验。测试标识与结果见 [报告](../../testing/m15-report.md)，下面的源码核查不冒充设备精度验收。

| 功能 | 说明书 / 原 HTML 符号 | V3 入口 | 正常 / 异常证据 |
| --- | --- | --- | --- |
| F01 范式 | 6.1 范式选择；TrialRunner.startActualTrial 266–323 行 | model.roles/generate；Runner.begin；页首范式 | baseline 四顺序；m15.test 与 Chrome 四范式；缺 A/B、复杂媒体不兼容阻止运行 |
| F02 素材 | 6.1 分组上传/拖入/试听/清空；handleFileUpload 586–704，复杂列表 effect 707–758 | 素材页签、分组导入/拖入/清空、逐项移除、带控制条试听、MediaBank；id/hash/path | 同名不同 hash 实际不同内容呈现；缺 Blob、损坏音频错误；保留原文件内容 |
| F03 序列 | 6.1 X 数量决定试次、按组顺序配对、升降序、拖拽；handleSort 761–797、shuffleInPlace 146–154 | 序列页签、上下移动/拖动/复制/移除、model.shuffle | 含首尾范围之外不变；mulberry32-v1 固定种子、before/after 及最终实际列表；越界拒绝 |
| F04 参数/按键 | 6.1；config 554–575、getKeyRangeForTrial/getEffectiveKeys 159–181、handleKeyDown 355–418 | 参数页签、Runner 阶段状态机 | 默认 1000 ms ITI / 500 ms ISI / 提示音开启；0/负数/非法、首匹配重叠、长按/组合键/IME、跨阶段拒绝 |
| F05 问卷 | 6.1 文本/单选/多选/必填；renderDesigner 问卷、submitQuestionnaire | 问卷页签→被试表单；newSession 快照→结果 | 配置往返、必填与选项验证、独立 participantId/sessionId、实际答案导出 |
| F06 资源/配置 | 6.1 路径前缀、目录模板、XLSX 列头；processFilesForXLSX、handleHelperDrop、handlePlaylistXLSX、saveProjectConfig/loadProjectConfig | 资源与恢复页签；formats.ts、drop.ts、LocalStore | XLSX/JSON 实际往返；缺文件配置仍可恢复但不能呈现；同名歧义不按首个名称误绑定；目录预算 |
| F07 运行/结果 | 6.2 问卷→指导语→试次→结束、强制结束；TrialRunner、finishExperiment、generateXLSX | 正式专注视图；Runner；Session/Attempt；JSON/XLSX/CSV | Chrome A、Qt C、局部导出、刷新恢复、异常 attempt、真实 IDB 写失败与多标签；物理端到端时延未测 |

## 原样语义与修复分开登记

| ID | 旧行为证据 | 处理 / 属性 |
| --- | --- | --- |
| S01 | audio sequence 完成后 setStatus('responding')，再 Date.now()；playing/preparing 忽略回答 | 保留音频 RT 定义，换单调 performance.now；诊断记录多时钟映射 |
| S02 | AX A→X，ABX A→B→X，AXB A→X→B，无内建正确答案，只记有效反应键 | 原样保留；不将 Is_Valid_Key 解释为作答正确率 |
| S03 | 分段 ranges.find 首匹配；空 keys 回退 global；阶段提示仅在生效段 start | 保留首匹配与回退，UI 明示；禁止非法/越界范围 |
| S04 | 洗牌 Fisher–Yates，1-based 闭区间，重叠顺序执行 | 保留范围行为，种子与 mulberry32-v1 为明确增强，同时保存最终顺序 |
| D01 | config.isi || 500 导致合法 0 变 500 | 修复，0 原样调度，非法输入拒绝 |
| D02 | play()/error catch resolve；缺 step.id wait(100) 继续 | 修复，阻断正常完成，保存 invalid；不自动补播 |
| D03 | React 状态异步更新窗口内可能多次接受；无 repeat/modifier/IME 过滤 | 同步进入 saving，加按键按下集合和事件时间边界；每 attempt 至多一次响应 |
| D04 | 文本/图片在 preparing 已渲染，文本异步 fetch，图片未等待 decode | 用户本轮明确同意：预加载、实际解码、两次 rAF 后开放；记录可观测回调，非物理显示测量 |
| D05 | 上传/资源助手排除 .txt，运行器仍有 text 分支；图片支持 jpg/jpeg/png/gif/bmp/webp/svg | 恢复 X 的 UTF-8 TXT 能力，图片保留；复杂范式拒绝非音频，避免偷偷发明混合呈现规则 |
| D06 | autoPlay/allowReplay 在 config 存在，运行器未读取；说明书承诺手动点击播放 | autoPlay=false 增加明确手动开始入口；不实现无记录重播，显式再呈现保留新 attempt |
| D07 | saveProjectConfig 只按 item.fileId 写 playlistConfig，复杂 trial 只有 stimuli，被 filter 丢掉 | 新 JSON 保存全部角色、hash、最终序列与范式；已丢失的旧复杂配置不伪造恢复 |
| D08 | 说明书说 CSV，当前源码实际 generateXLSX；结果只存 X | 保留旧 XLSX/CSV 前置列，追加角色、hash、attempt/status 等；JSON 是完整溯源主文件 |
| D09 | HTML dangerouslySetInnerHTML，可能带远程资源/脚本 | 保留静态格式白名单，移除远程资源/脚本/样式，不自动联网 |
| D10 | 没有落盘事务、会话锁或恢复边界 | IndexedDB 事务、CAS revision、Web Locks；刷新只恢复已提交边界；结果可独立于刺激导出 |

## 独立基准

`node tests/e2e/m15-baseline.cjs` 从相邻 V2 原 HTML 提取原函数体，在 VM 中用可控 HTMLAudio/时钟适配执行，不 import V3。`output/validation/m15-baseline/baseline.json` 保留四序列、计时状态、0→500、错误继续、100 ms 缺失等待、首匹配和闭区间结果。属于独立源码函数执行，未声称 V2 实际声卡/完整 React UI 通过。源码与用户语料未修改。

## 依赖与来源

旧 HTML：React 18.2.0、lucide-react 0.263.1、未锁定 Tailwind/Babel CDN、SheetJS 0.19.3。新页面只复用现有 Vue 3.5.42、公共 CSS/Doulos SIL；SheetJS CE 固定 0.20.3 官方 ESM 随按需模块构建，许可/hash 在 vendor/README.md 与 LICENSE；旧依赖在来源登记保留历史归属但不列为本页当前依赖。
