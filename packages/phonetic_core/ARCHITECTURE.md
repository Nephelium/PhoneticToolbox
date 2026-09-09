# 纯科学计算、模型与可移植研究业务 · 组件架构

D0.3 / 2026-09-09 / 实现提案。上级约束见 [总架构](../../ARCHITECTURE.md)。

## 职责与依赖
纯科学计算、模型与可移植研究业务。允许依赖：Python 标准库、锁定的 NumPy/SciPy/Parselmouth 等科学依赖；定义清晰的 F0/native ports。禁止依赖或行为：Qt、FastAPI、HTTP、数据库、登录、进程全局用户设置、硬编码开发机路径。

## 目标代码位置
src/phonetic_core/{algorithms,models,services,ports}/；pyproject.toml；tests/

这些是待实现的文件/目录，不是本轮已完成的业务代码。必须按主计划逐任务建立，不能创建空实现让导入测试假通过。

## 输入、输出和边界
输入来自版本化 contracts 或本组件明确定义的配置；输出为同一协议可理解的结果、错误、状态和可归属的资源。跨边界错误需含机器可读 code 与用户可理解 message，不暴露完整服务器路径或个人语料。

源数组不原地覆盖；保留原算法默认值、边界与 NaN。迁移第一步做 import/I/O 解耦，不顺便改公式。tdklatt 现有音频设备导入必须移到外层再进入纯核心。移植和参考代码保留 source_id。

## 实施与验证
python -m pytest packages/phonetic_core/tests -q；python -m pytest tests/parity -q；安装 wheel 到干净环境再测试。

测试状态与实现状态分别记录。涉及科研参数时附源版本、参数和数值差异；涉及界面时附浅深色/空态/错误态；涉及文件写入时覆盖失败、取消、额度和清理。组件未通过自身验收不得只靠上层 UI 掩盖问题。
