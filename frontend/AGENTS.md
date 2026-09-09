# frontend — 工作规则

继承 [根 AGENTS.md](../AGENTS.md)，先读本目录 [ARCHITECTURE.md](ARCHITECTURE.md)。2026-09-09 已获 P01 隔离原型实施授权，具体边界见根规则与 P01 计划；本文件不扩大为全面业务迁移或部署授权。

- 负责：公共 Web 界面。
- 允许依赖：contracts 生成的类型和 frontend 内部公共组件；平台能力通过 FileProvider、JobClient、AudioController、CaptureProvider、ProjectStore 接入。
- 禁止：Python 算法、任意本地/服务器路径、桌面进程、数据库、旧站点路由/登录/配置。
- 每次开始定位总计划任务 ID、相关文件、来源和验收项；已有可用代码优先迁移并保留证据。
- 主路由只登记页面；波形/选区/播放集中在共同状态模型；组件尺寸使用主题令牌；未保存编辑切换标签不丢失。图表只负责显示，导出科研数值走明确数据契约。
- 预定验证：npm --prefix frontend run typecheck；npm --prefix frontend run test -- --run；npm --prefix frontend run build；对应页面 E2E 与浅深色截图。这些命令需要相应计划中的脚手架先实现，当前不声称可运行或通过。
- 改动边界/算法/平台承诺前同步 ADR 和任务；不通过放宽检查来获得“完成”。
