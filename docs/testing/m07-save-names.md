# 发声类型合成保存命名

2026-10-21，任务 `M07-SAVE-NAMES`。

## 实现

完整组保存复用历史结果描述，由已生成结果的快照确定方向、连续统类型和步数。Windows/macOS共用桌面适配，网页ZIP采用相同名称，例如 `源到目标 · 仅发声类型变化 · 9 步`。桌面新建独立目录，同名时追加 `（2）`、`（3）`，不合并或覆盖原组。组内WAV步骤编号、CSV与JSON格式保留。

## 已验证

- `node --test tests/m07.test.ts tests/m07-r1.test.ts tests/m07-r3.test.ts`：9项通过。
- 前端 `npm run typecheck`、`npm run contracts:check`、`npm run ui-data:check`、`npm run build:workbench` 通过，构建保留活动窗口仍可能读取的旧资源。
- `scripts/Start-Research-Workbench.ps1 -CheckOnly` 确认当前v3源码绑定；Python桥语法通过。
- `node tests/e2e/m07-save-names.cjs`：独立Chrome读取生产页面和 `desktop.ts/M07Port`，经真实FileProvider、TaskBridge、LocalService及Praat，从公开固定合成输入生成六组九步结果并实际写盘。合成后将当前步数改为4，六个历史组仍按各自的9步快照命名。重复保存另建第2组，原组中已修改的步骤文件完整保留；非法路径名拒绝。
- 同一批真实结果通过生产浏览器ZIP保存入口下载，核对六个中文文件名、各12份文件的完整清单、ZIP校验及结果SHA。输入摘要保持，页面错误为空，浅/深主题截图保留。

最终证据位于 `output/validation/m07-save-names/chrome-ba6088d19bf947dab581534e4c31f600/` 的 `report.json`、`exports.json` 和两张截图。首次验证脚本的源音频定位器匹配了下拉框与同名图窗，收紧为combobox后两轮上述验证通过，失败报告另存。架构检查及文档2396文件/448来源/174任务零错误，未变历史快照缺链另列。

## 未验与产物

本轮修改Windows、macOS和网页共用源码。未进行原生Qt窗口、Mac实机、Safari、最终EXE/Mac包及线上网页验证，未公开部署。没有改变科学算法或宣称新增数值一致性结论。

三轮运行的scratch包含自建输入、测试任务库/缓存、导出与下载，现均已通过项目受保护清理入口删除。宿主finalizer的清理子进程曾退出1，未提供具体错误文本；所有自有服务和浏览器退出后，由同一入口重新检查归属、联接与进程再清理成功。验收驱动已增加宿主退出后检查回执和受保护重试，最后一轮完整导出及自动收尾通过。各运行的报告、截图、SHA清单和deleted回执保留。
