# packages/phonetic_core — 工作规则

继承 [根 AGENTS.md](../../AGENTS.md)，先读本目录 [ARCHITECTURE.md](ARCHITECTURE.md)。当前仅规划阶段；本文件不构成开始业务编码、安装依赖或部署的授权。

- 负责：纯科学计算、模型与可移植研究业务。
- 允许依赖：Python 标准库、锁定的 NumPy/SciPy/Parselmouth 等科学依赖；定义清晰的 F0/native ports。
- 禁止：Qt、FastAPI、HTTP、数据库、登录、进程全局用户设置、硬编码开发机路径。
- 每次开始定位总计划任务 ID、相关文件、来源和验收项；已有可用代码优先迁移并保留证据。
- 源数组不原地覆盖；保留原算法默认值、边界与 NaN。迁移第一步做 import/I/O 解耦，不顺便改公式。tdklatt 现有音频设备导入必须移到外层再进入纯核心。移植和参考代码保留 source_id。
- 预定验证：python -m pytest packages/phonetic_core/tests -q；python -m pytest tests/parity -q；安装 wheel 到干净环境再测试。这些命令需要相应计划中的脚手架先实现，当前不声称可运行或通过。
- 改动边界/算法/平台承诺前同步 ADR 和任务；不通过放宽检查来获得“完成”。
