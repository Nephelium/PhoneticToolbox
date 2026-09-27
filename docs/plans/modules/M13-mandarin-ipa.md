# M13 · 普通话转 IPA迁移计划

状态：**verified（限定 2026-09-26 Windows Chrome、Windows Qt 宿主和 WSL2 Ubuntu 静态托管）**。M13 是本地前端逐字转换，不依赖服务器科学计算链路；来源/许可 `PENDING-IPA`、生产部署及 Linux 真实浏览器交互仍单列未完成。证据见 [验收报告](../../testing/m13-report.md) 和 [源码映射](../../modules/evidence/M13-source-map.md)。共用 [架构](../../../ARCHITECTURE.md)、[测试规范](../../testing/verification-plan.md) 和 [UI 规范](../../design/UI_SPEC.md)。

## 现有代码与目标文件
现有路径均已确认存在；目录内逐函数对应由实施第一步记录，避免把旧类名机械套给新实现。

- [phonetic_toolbox/services/ipa_trans_service.py](../../../phonetic_toolbox/services/ipa_trans_service.py)
- [phonetic_toolbox/gui/resources/ipa_trans](../../../phonetic_toolbox/gui/resources/ipa_trans)

实际模块路径：

- `frontend/src/modules/mandarin-ipa/MandarinIpaPage.vue`
- `frontend/src/modules/mandarin-ipa/state.ts`
- `frontend/src/modules/mandarin-ipa/export.ts`
- `frontend/src/modules/mandarin-ipa/ipa-data.json`
- `frontend/src/modules/mandarin-ipa/generate-data.mjs`
- `frontend/tests/m13.test.ts`
- `frontend/tests/m13-live.html`
- `tests/e2e/m13.cjs`
- `scripts/verify_m13_qt.py`
- `scripts/verify_m13_linux_static.pl`

本轮核对后确认无需专属 core/API，也没有创建空后端层或任务契约。公共 AppShell 接线由公共 UI agent 维护；M13 页面通过 `dirty` 事件与 `save()` 接口接入。

## 布局与全部原功能分组
左侧汉字编辑；中央转换结果；右侧转换标准和排版；顶部保存图片和帮助。

| 编号 | 原功能组 | 必须保留的功能 | 新位置 | 行为约束 |
| --- | --- | --- | --- | --- |
| M13-F01 | 转换内容 | 汉字输入、音标结果、11 种既有标准 | 左侧输入＋右侧标准 | 保留原全部标准及多音等说明；本轮不更换转换规则。 |
| M13-F02 | 文字排版 | 汉字字号、音标字号、字音距离、行距、粗体、斜体、下划线 | 右侧排版组 | 汉字与 IPA 字号独立；组合附加符号不截断。 |
| M13-F03 | 显示方式 | 仅音标/字音同显、横向/上下排布 | 结果上方显示工具 | 切换布局不清空原文，结果可重新布局。 |
| M13-F04 | 输出与帮助 | 保存图片、帮助与提示 | 页顶动作栏 | 导出字体和符号完整；把在线 html2canvas 依赖本地化后验证断网导出。 |

## 状态、重用与双端差异
空输入、多音提示、字体缺字、图片保存失败分别显示；不自动把有歧义读音标为唯一正确答案。

标准列表来自源码：Standard Chinese (Beijing)、Standard Chinese (Beijing)严、胡裕树、黄伯荣/廖序东、钱乃荣、吴宗济、赵元任、《汉语方音字汇》、UntPhesoca宽、UntPhesoca严、汉语拼音。

迁移重点：冻结 11 种标准和映射数据；标准名称≠规范来源已经核验；明确多音字和上下文限制，排版逻辑可重用。

平台边界：规则转换可在共享前端完成，无需每次上传文本；导出所需字体与图片库本地构建后离线验证。

## 实施记录

1. 已完成说明书 10.1–10.2 与实际 `ipa_converter.html` 的四组映射、默认值、歧义与错误边界记录。
2. 固定旧 HTML 源哈希后提取 21,572 行映射；10 个 IPA 列原值保留，汉语拼音沿用旧声调规则。只把旧非标准 JSON 的 3 个裸 `NaN` 规范化为 `null`。
3. 转换、状态和排版保持纯前端。M13 资源按需分包，未创建 core/API、数据库结构或服务器任务。
4. 页面使用公共 `ModuleFrame/Toolbar/Section/Status`、公共浅深主题和 Doulos SIL；AppShell 负责标签、关闭保护与帮助入口。模块内没有重复大标题、关闭按钮或音频条。
5. 图片改为内置 Canvas 导出，不再调用旧 CDN html2canvas；实际离线下载、字体绘制调用及像素内容均已检查。
6. 已运行单位/静态检查、真实 Chrome、Windows Qt 宿主与 WSL2 Linux 静态托管；范围和未测项见验收报告。
7. 说明书与来源记录已更新；全局 task ledger、ADR 和 `module-migration.md` 留给统筹 agent，未越权覆盖其并行改动。

## 专项验收
11 标准逐一、组合音标/上下标、字音间距、横/竖排、格式切换、断网图片导出、空文本/歧义。

每个分组至少一个正常路径和一个相关错误/边界路径；录制、原生时序和数值算法必须在真实目标环境验证。

实际执行入口：

```powershell
npm --prefix frontend test
npm --prefix frontend run typecheck
npm --prefix frontend run build
node tests/e2e/m13.cjs
$env:PYTHONPATH='desktop/src;backend/src;packages/phonetic_core/src;scripts'
& '.venv/m09-ui/Scripts/python.exe' 'scripts/verify_m13_qt.py'
wsl.exe -d NInfer --exec perl /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/scripts/verify_m13_linux_static.pl
```

仓库没有 `npm test:e2e`，因此没有报告该不存在的命令通过。Chrome 脚本直接使用项目现有 Vite 与 Playwright 运行时。

## 完成条件
- 本页全部 4 组功能以及原矩阵相关参数/设置有映射，没有把折叠项当作删除项。
- 数值/文件/时间轴差异均解释并审阅；已有功能不得静默改语义。
- 浅深色、未保存保护、错误恢复与真实操作可用；Web owner/配额检查适用的路径已覆盖。
- 软件/说明书的来源一致；尚缺权限/设备证据时状态仍为待处理，不能以隐藏控件绕过。

上述功能条件已在限定平台满足。来源许可、生产服务部署、Linux 真实浏览器、macOS 与 EXE 未纳入本轮，不扩大 `verified`。
