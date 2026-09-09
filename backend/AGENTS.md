# backend — 工作规则

继承 [根 AGENTS.md](../AGENTS.md)，先读本目录 [ARCHITECTURE.md](ARCHITECTURE.md)。2026-09-09 已获 P05 账号/项目实施及专属空库建表授权，Windows 账号/会话/项目与真实 PostgreSQL 定向验收已通过；P06 具体审阅后的继续指令已核实，专属测试库任务表及本机 SQLite 已建并通过 Windows 定向验收（见 ../docs/testing/p06-jobs-report.md）；其他数据库及再后续 schema 变更仍须按根规则授权，范围见 P05 计划；不扩大为全面业务迁移或部署授权。

- 负责：账号、HTTP、资源、配额、任务、服务端存储。
- 允许依赖：contracts、phonetic_core、数据库/存储接口和服务端第三方依赖。
- 禁止：Qt、浏览器 DOM、本机任意路径、业务算法复制、跨用户全局配置。
- 每次开始定位总计划任务 ID、相关文件、来源和验收项；已有可用代码优先迁移并保留证据。
- API 进程不做长计算；worker 认领必须有租约和 fencing token；空间先原子预留再写；下载与取消同样鉴权。私有数据不由静态 Web 服务器直接暴露。
- 预定验证：python -m pytest backend/tests -q；python -m pytest tests/security tests/contracts -q；WSL/PostgreSQL 集成测试。这些命令需要相应计划中的脚手架先实现，当前不声称可运行或通过。
- 改动边界/算法/平台承诺前同步 ADR 和任务；不通过放宽检查来获得“完成”。
