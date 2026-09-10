# M01-F1 切分执行与持久批次准备

状态：verified（仅F1执行准备，Windows）。井井在 E 布局交付后回复“继续”，按既定 M01-F 计划实施。244项定向测试与三组真实合成产物回读见 [F1报告](../testing/m01-execution-preparation-report.md)。实际数据库迁移仍须单独授权，不用已有 003/004 授权代替；M01-F整体仍in_progress，下一项F2持久接入。

## 本轮文件和行为

- `packages/phonetic_core/src/phonetic_core/segmentation.py`：纯切分计划及参数时间表切片；保留 v2 `int(t*fs)`、原声道/样本类型、静音标签跳过。非法区间/缺失层/重名层明确失败。
- `backend/src/ptb_worker/segmentation.py`、`segment_child.py`：所属 Windows Job 内生成并回读 WAV、可选参数 XLSX/SQLite；独立心跳/取消轮询、超时、内存/输入/输出总预算；返回准备好的产物，不冒充已持久发布。
- `backend/src/ptb_worker/batch_policy.py`：顺序列表/状态汇总/取消后未开始项的规则，17 文件及幂等快照测试。
- `backend/migrations/005_acoustic_batches*.sql`、`scripts/m01_database.py`、`docs/testing/m01-migration-review.md`：PG/SQLite 增量建表内容、只读审阅入口、固定测试库/路径/前置版本/旧行保留保护。默认不执行 DDL，不修改现存数据或配置。
- 核心和 backend 定向测试、独立安装 wheel 与合成导出回读；`generate_contracts.py --check`、`check_architecture.py`、`validate_docs.py`、`git diff --check`。

## 参数同步切分的明确语义

M01-MAN01 当前 v2 说明书有承诺但函数仅写 WAV。补齐派生能力时必须绑定同一个原始 WAV 的 hash、采样率与样本数，不能拿别的结果表切片。保留完整结果的原帧采样点，不重新计算/插值/平滑。选择落在实际 WAV 切片半开范围内的参数帧；派生表 `Time_s` 减去实际切片起点 `int(xmin*fs)/fs`，另保留 `Source_Time_s`。首帧不保证为0。元数据保留原始区间、实际起止样本、父结果 hash 及“未重新估计”的标记。

无父结果时只准备 WAV；有父结果但片段内无参数帧时保留 WAV 并明确 `no_frames`，不生成空表或假值。已有 XLSX/SQLite 的任意旧版导入、实际持久父结果选择和 UI 发布仍需 F2/G 联合接入验证，不以本轮内存派生验收销掉完整功能项。

## 新表设计门

每个批次持久记录 owner/project、幂等键、配置 hash、不可变公共配置、操作、取消标记、创建/关闭时间；每个输入占一行，有序序号、输入与关联 hash、子任务引用。子任务仍复用 P06 的状态/租约/fencing 和 P07 的资源账本。表只增加不改旧表/旧行；本轮不放宽 ZIP 的16条目限制。批次最多1000输入，与既有账号任务记录上限一起做原子接纳，不静默截断。实际服务接入时把尚未创建的批次项也计入任务容量，避免被其它提交绕过。

桌面同样需要持久批次结构，但原生目录能力仍限定本窗口；重启后不能复用失效随机文件句柄。后续受控输入快照/重新授权与结果发布必须在 F2 完成，不能把任意本地路径放进公开协议。公开网页任务/真实桌面保存按钮仍保持未启用直到对应生命周期验收通过。
