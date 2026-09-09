# 跨端数据协议与兼容策略 · 组件架构

D0.3 / 2026-09-09 / 实现提案。上级约束见 [总架构](../ARCHITECTURE.md)。

## 职责与依赖
跨端数据协议与兼容策略。允许依赖：声明式 schema、数据字典、生成脚本及协议测试。禁止依赖或行为：GUI、计算实现、数据库连接与独立手改生成的客户端。

## 目标代码位置
openapi.json；schemas/{audio,track,job,asset,quota,capability}.json；parameter-catalog.json；generated/；versions.md

上述为完整目标结构。P02 已实现的最小包、入口与生成契约以 P02 验收报告为准，其余业务目录仍待对应任务实施；不以空实现冒充业务通过。

## 输入、输出和边界
输入来自版本化 contracts 或本组件明确定义的配置；输出为同一协议可理解的结果、错误、状态和可归属的资源。跨边界错误需含机器可读 code 与用户可理解 message，不暴露完整服务器路径或个人语料。

OpenAPI 由后端模型生成、契约目录保存审阅快照，前端类型只从固定快照生成。JSON Schema 为独立跨进程数据结构；若与 OpenAPI 相同模型，单一生成源避免双份手写。

## 实施与验证
协议 schema 校验；生成后工作区无漂移；前后端契约测试；单位/NaN/时间轴样例往返测试。

测试状态与实现状态分别记录。涉及科研参数时附源版本、参数和数值差异；涉及界面时附浅深色/空态/错误态；涉及文件写入时覆盖失败、取消、额度和清理。组件未通过自身验收不得只靠上层 UI 掩盖问题。

## HTTP 接口草案（v1）
| 接口 | 契约 |
| --- | --- |
| POST /api/v1/auth/login；POST logout；GET me | 服务器会话；桌面使用独立本地会话，不复用云账号 |
| GET /api/v1/capabilities | 模块、实际后端、设备、版本、限制和不可用原因 |
| POST/GET /api/v1/projects | 用户的临时工作组织；不自动延长文件寿命 |
| POST /api/v1/uploads；PUT chunks；POST finalize | 原子预留、校验分块、限额、最终资源 ID |
| GET/DELETE /api/v1/assets/{id} | owner 检查、过期检查；删除支持集合引用提示 |
| GET /api/v1/assets/{id}/content | 支持受控 Range；不能用静态公开路径绕过鉴权 |
| POST /api/v1/jobs | module_id、输入资源 ID、config、幂等键、输出预算 |
| GET /api/v1/jobs/{id}；GET events | 状态与有序进度事件，支持断线续接 |
| POST /api/v1/jobs/{id}/cancel | 幂等取消；终态重复调用返回已终结 |
| GET /api/v1/jobs/{id}/result | 已完成的产物 manifest；不返回临时未提交文件 |
| GET /api/v1/storage/usage | used/reserved/quota、按类统计、最近到期 |
| POST /api/v1/exports | 将所选自有结果打包；流式或先预留 ZIP 空间 |
| GET /api/v1/references | 当前软件版本实际包含的方法/来源清单 |

会话候选：Secure、HttpOnly、SameSite Cookie；写请求 CSRF 防护；账号枚举与登录尝试限制。禁止把长期凭据放 localStorage。桌面 loopback 会话不能共用公网 Cookie。具体认证库在 P05 选型并锁定。

## 数据字典
AudioAsset：id、owner_id（服务端确定）、sample_rate_hz、channels、channel_roles、sample_count、content_hash、created_at、expires_at。
Track：parameter_key、backend、times_s、values、validity、unit、analysis_config_hash。时间和值等长、严格单调；缺失值为 null。
Selection：start_sample、end_sample、sample_rate_hz；区间 [start,end)，播放/导出共享转换。
Job：id、module_id、input_asset_ids、config_snapshot、status、attempt、lease_generation、progress、created_at、deadline_at、result_asset_ids、source_ids、core_version。
Job.status：queued / running / cancel_requested / succeeded / failed / cancelled / interrupted。终态不回跳；重试创建新 attempt 并保留前次记录。输入过期使用明确错误码，资源状态另行管理。
Asset.state：uploading / available / expired / deleting / delete_failed / deleted。删除失败不可读取、继续计量；重试删除保持幂等。
Quota：quota_bytes=5000000000、used_bytes、reserved_bytes、available_bytes；由服务端事务计算。
Error：code、message、retryable、details（脱敏）；code 示例 quota_exceeded / asset_expired / permission_denied / invalid_audio / backend_unavailable / job_interrupted。

## 版本与兼容
请求/响应固定 api_version；新增可选字段兼容，小版本不得修改单位、参数键或状态语义。破坏性变更升 major 并提供迁移说明。前端不尝试猜测旧服务器字段。
科研输出 manifest 同时记录 app_version/core_version/algorithm/backend/source_ids，不把 API 版本当算法版本。

## P02 实施记录

2026-09-09：已完成 Windows 独立环境、包/入口与契约的定向验收，具体文件、命令和边界见 [P02 报告](../docs/testing/p02-scaffold-report.md)。上文完整功能结构仍按后续任务实施，不代表算法、正式 UI、账号/任务或全平台发行已通过。
