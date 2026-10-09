# 接口、任务与存储 — 工作规则

继承[根规则](../AGENTS.md)，结构见[本目录架构](ARCHITECTURE.md)。这里只补充本目录约束。

- API 只做身份、协议与编排，长计算交 worker。认领使用租约和 fencing token；空间先原子预留，下载与取消同样鉴权。
- 私有数据不由静态服务器直接暴露，不接受任意本机路径，不使用跨用户全局设置；调用公共 core，不复制声学算法。
- 定向运行 backend/tests、tests/security 与 tests/contracts；存储变更使用独立测试库并遵守迁移授权。
