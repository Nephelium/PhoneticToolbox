# 账号、HTTP、资源、配额、任务、服务端存储 · 组件架构

D0.3 / 2026-09-09 / 实现提案。上级约束见 [总架构](../ARCHITECTURE.md)。

## 职责与依赖
账号、HTTP、资源、配额、任务、服务端存储。允许依赖：contracts、phonetic_core、数据库/存储接口和服务端第三方依赖。禁止依赖或行为：Qt、浏览器 DOM、本机任意路径、业务算法复制、跨用户全局配置。

## 目标代码位置
src/ptb_api/{main,auth,projects,assets,jobs,quota,storage}.py；src/ptb_worker/{main,claims,leases,executor,cleanup}.py；migrations/（实际迁移前审阅）；tests/

这些是待实现的文件/目录，不是本轮已完成的业务代码。必须按主计划逐任务建立，不能创建空实现让导入测试假通过。

## 输入、输出和边界
输入来自版本化 contracts 或本组件明确定义的配置；输出为同一协议可理解的结果、错误、状态和可归属的资源。跨边界错误需含机器可读 code 与用户可理解 message，不暴露完整服务器路径或个人语料。

API 进程不做长计算；worker 认领必须有租约和 fencing token；空间先原子预留再写；下载与取消同样鉴权。私有数据不由静态 Web 服务器直接暴露。

## 实施与验证
python -m pytest backend/tests -q；python -m pytest tests/security tests/contracts -q；WSL/PostgreSQL 集成测试。

测试状态与实现状态分别记录。涉及科研参数时附源版本、参数和数值差异；涉及界面时附浅深色/空态/错误态；涉及文件写入时覆盖失败、取消、额度和清理。组件未通过自身验收不得只靠上层 UI 掩盖问题。
