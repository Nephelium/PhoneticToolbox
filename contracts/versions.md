# 数据协议与版本 · P02

应用发行版本的唯一维护入口为 [release/version.json](../release/version.json)，当前 Python 包 3.0.0a1，前端 3.0.0-alpha.1，API 1.0.0。运行时分别使用已安装包元数据和生成的前端版本文件。根目录旧 pyproject.toml 的 2.2.0 是历史迁移来源，不参与新包构建。

## 唯一生成链

1. 手写源：[后端模型](../backend/src/ptb_api/models.py) 及 API 路由。
2. `python scripts/generate_contracts.py` 生成 [OpenAPI](openapi.json) 和 schemas 中的 Audio、Selection、Track、Viewport JSON Schema。
3. `npm --prefix frontend run contracts` 从固定 OpenAPI 快照生成 [TypeScript](generated/api.ts)。生成文件禁止手改；同名模型不另写一份前端 interface。
4. 两条命令分别加 `--check` 或改用 `contracts:check` 检查漂移；版本使用 `python scripts/sync_versions.py --check`。

## 科学语义

- sample_count 是每声道的帧数；采样帧为严格整数，最大 2^53−1，防止 JavaScript JSON 往返丢失整数精度。采样率明确为 Hz。
- Selection 使用原始采样率与半开区间 [start_sample,end_sample)，允许空选区和 EOF 单帧；Viewport 校验不能越界，不能改成播放整个文件。
- times_s 是非负、有限、严格递增的真实秒数组；与 values、validity、reason 等长。合法的 0 是真实数值，缺失不能填 0。
- validity 区分 valid / unvoiced / missing / failed。valid 必须有有限数值且 reason=null；其他状态值必须 null 且说明原因。NaN/Infinity 无法作为合法协议结果。
- unit 与 backend、analysis_config_hash、source_ids 必填。P02 不冻结 80 参数的单位映射或实际算法版本；P03/M01 按原源码和黄金样例核定，不能凭参数名猜单位。
- JSON Schema 与 TypeScript 提供结构约束；等长、单调、帧边界等跨字段约束由 Pydantic 执行，不能宣称浏览器类型检查已验证科研正确性。

当前 HTTP 仅提供 GET health 和 capabilities；Viewport 是跨进程数据结构快照，没有虚构的分析 API。两个运行模式的 OpenAPI 一致。请求型业务模型与账号/任务/额度在对应阶段逐项添加。

## 兼容规则

新增可选字段可以兼容扩展；单位、键名或状态语义改变必须升级 API major。应用版本与算法版本分开，科学依赖迁移要有 P03 证据。生成快照、包锁和来源登记与变更同批提交。
