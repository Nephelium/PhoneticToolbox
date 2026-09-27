# M12-R6 图窗优先与标注剪贴

状态 verified（限定 Windows 前端/Chrome 及冻结宿主基础链路，详见 [R6 报告](../testing/m12-r6-report.md)）。2026-09-19 用户明确要求图窗置顶、工具下移、删除冗余按钮、标注删除/剪切/粘贴。保持全局配色。

图窗卡片为中央首项，顶部放文件列表及保存 TextGrid。强度、词典词表、搜索替换依次放到图窗下方。移除音素自动填充、撤销、复制词、连续粘贴四个按钮，Ctrl+Z/C/X/V 与既有双击音素编辑保留。

用户已确认的语义：选中区间时 Backspace 删除该段标注，原位置留空，音节对应音素一同处理。Ctrl+X 剪切保留原时长、文字、内部音素边界，Ctrl+C 复制同样的完整片段，Ctrl+V 从所点击空白时间起粘贴，不改变拼音声调。非空重叠/越界/层角色不符显式拒绝，失败不改文档；每次操作一步撤销。选中边界时 Backspace 保持 R5 合并规则。文本输入框继续使用原生文字快捷键。

范围：AnnotationPage.vue、editor.mjs/editor.d.mts、独立剪贴纯函数、前端和 Chrome 定向测试、说明文档。沿用本次会话已读取的 brainstorming / playwright 流程。

夜间颜色只调查：tokens.css 为全局 120 ms 按钮背景过渡，R5 截图冻结时钟造成灰色中间态。真实浏览器恢复计时后背景为 rgb(24,36,50)，即 #182432。证据 output/validation/m12-ui/1c6c063e634d4e6a866c4855f0bc7942/theme-colors.json。本轮不修改主题源码，只等待实际稳定后截图。

验收命令：node --test frontend/tests/annotation-r6.test.ts；npm --prefix frontend run typecheck；npm --prefix frontend test；npm --prefix frontend run build；node tests/e2e/m12-r6.cjs；R5 核心交互回归按新删除语义检查。只用项目独立合成语料与浏览器，不修改 V2/原语料，不执行 DDL/push。用户后续明确授权本轮结束直接生成临时 EXE，新增 R6 输出目录，保留旧包。使用项目 m09-ui 环境及已有单文件配方，构建后执行冻结 EXE 的 M12 加载/编辑/保存/退出检查。
