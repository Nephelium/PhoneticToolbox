# 测试架构

共用边界见[总架构](../ARCHITECTURE.md)，修改约束见[本目录规则](AGENTS.md)。本页说明结构，不记录逐次验收。

| 位置 | 职责 |
| --- | --- |
| `fixtures/、parity/` | 公开最小基准与独立数值对照。 |
| `contracts/、security/、architecture/` | 契约、账号/路径隔离和依赖方向。 |
| `e2e/、support/` | 实际页面流程及专用宿主/数据工具。 |
| `performance/、staging/` | 性能与隔离环境检查。 |

模块单元测试同时分布在 backend/tests、desktop/tests、frontend/tests 及核心包 tests。测试工具只拥有独立数据和进程，模拟证据不能替代实际原生行为。策略与命令统一见[验证策略](../docs/testing/verification-plan.md)，报告留必要事实，可重建的大产物用后清理。
