# 公共 Web 界面 · 组件架构

D0.3 / 2026-09-09 / 实现提案。上级约束见 [总架构](../ARCHITECTURE.md)。

## 职责与依赖
公共 Web 界面。允许依赖：contracts 生成的类型和 frontend 内部公共组件；平台能力通过 FileProvider、JobClient、AudioController、CaptureProvider、ProjectStore 接入。禁止依赖或行为：Python 算法、任意本地/服务器路径、桌面进程、数据库、旧站点路由/登录/配置。

## 目标代码位置
src/app/{AppShell,Sidebar,WorkspaceTabs}.vue；src/design/tokens.css；src/components/{AudioTransport,WaveformViewport,TaskPanel,MethodReferences}.vue；src/modules/<module>/；src/platform/{browser,desktop}.ts

上述为完整目标结构。P02 已实现的最小包、入口与生成契约以 P02 验收报告为准，其余业务目录仍待对应任务实施；不以空实现冒充业务通过。

## 输入、输出和边界
输入来自版本化 contracts 或本组件明确定义的配置；输出为同一协议可理解的结果、错误、状态和可归属的资源。跨边界错误需含机器可读 code 与用户可理解 message，不暴露完整服务器路径或个人语料。

主路由只登记页面；波形/选区/播放集中在共同状态模型；组件尺寸使用主题令牌；未保存编辑切换标签不丢失。图表只负责显示，导出科研数值走明确数据契约。

## 实施与验证
npm --prefix frontend run typecheck；npm --prefix frontend run test -- --run；npm --prefix frontend run build；对应页面 E2E 与浅深色截图。

测试状态与实现状态分别记录。涉及科研参数时附源版本、参数和数值差异；涉及界面时附浅深色/空态/错误态；涉及文件写入时覆盖失败、取消、额度和清理。组件未通过自身验收不得只靠上层 UI 掩盖问题。

## P02 实施记录

2026-09-09：已完成 Windows 独立环境、包/入口与契约的定向验收，具体文件、命令和边界见 [P02 报告](../docs/testing/p02-scaffold-report.md)。上文完整功能结构仍按后续任务实施，不代表算法、正式 UI、账号/任务或全平台发行已通过。
