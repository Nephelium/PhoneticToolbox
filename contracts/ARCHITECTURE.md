# 协议架构

职责边界见[总架构](../ARCHITECTURE.md)，修改规则见 [AGENTS.md](AGENTS.md)。

| 位置 | 权威内容 |
| --- | --- |
| openapi.json | 由后端模型生成的 HTTP 接口审阅快照，以实际生成路由为准。 |
| schemas/、recording/ | 跨进程及录音数据结构，字段与边界由源模型生成。 |
| generated/ | 前端类型，不独立手改。 |
| parameter-catalog.json、resource-manifest.json | 参数标识、固定资源身份及生成资源的权威来源声明。 |
| versions.md | 版本及兼容规则。 |

- 音频使用真实采样率、声道角色、每声道帧数和内容摘要；轨迹 times_s 与 values 等长，携带单位、有效性和实际后端。
- Job 固定输入、配置快照、owner、attempt、租约、截止和结果版本。取消与中断有明确状态，终态不回跳；重试保留前次证据。
- Asset 生命周期独立于任务。额度由服务端事务计算，政策版本与旧结果兼容不能靠手改生成 JSON 实现。
- 错误包含机器可读 code、用户可理解 message、retryable 与脱敏 details，不暴露服务器绝对路径或其他人的文件名。
- 请求/响应按版本演进，新增可选字段保持兼容；破坏性单位、参数键或状态变更升版本并提供迁移说明。API 版本与算法/core 版本分开。
- 认证与账号存储见[专门规格](../docs/specs/accounts-storage-jobs.md)。桌面本地能力会话与公网账号隔离，不在 URL 或 localStorage 保存长期凭据。

资源清单中的 resources 保存固定文件 SHA-256，第三方文件关联 source_id，自有文件显式标 origin=project 并说明用途。generated_resources 当前仅接受 manual-reader，由说明书源工程、project.json 和 build-report.json 共同验证生成物。该声明不豁免整目录检查，多余旧文件、摘要变化、源稿漂移、缺失和联接都使检查失败。
