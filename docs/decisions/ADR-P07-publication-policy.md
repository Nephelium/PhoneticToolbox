# ADR-053 补充：按任务快照确定发布期限

2026-09-27。状态：按井井已批准的新政策完成代码与合成验证，具体库迁移待授权。关联[ADR-053](ADR.md)和[收口报告](../testing/p07-policy-closeout-report.md)。

公共发布器从 storage_policy 取得额度/期限与分类，不在各模块散落 TTL。operation 仅提供初始分类，M08 的 saved_copy/source_ref/copy_result 表明复制路径时必须继承受管输入截止。科学新结果在最后 fencing 后原子发布，资产版本与 manifest 版本同时成为 2。历史缺字段按 1 回读，不改旧 JSON/资产截止。

M14 当前 preview/export 每次都重新读取输入表并执行音系分析，随后生成 JSON 或三份文档，依照现有新生成规则明确纳入独立三天结果。未来缓存复制功能须单独继承原结果截止，不能借同 operation 续期。纯切分/ZIP/展开沿用输入更早截止。

代价是 publication 接口须接收可信 snapshot，且响应模型新增版本字段要求严格旧客户端同步升级。保留科学值、来源、M08/M14 结果联合类型、本地无 TTL 和现存数据库独立迁移门。具体代码/hash/验证/公共文件释放见前置交接。
