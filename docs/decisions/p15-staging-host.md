# P15 决策：单机 staging 宿主与分阶段开放

2026-09-27。准备实施已授权；实际部署 proposed。独立记录，待共享 ADR 释放后由统筹归档。

采用同一专用 Unix 账号的 systemd user manager，API、worker、清理器使用同一发行目录/运行时/数据库/私有存储。P11 的临时计算单元由该 manager 创建，所有准入锁位于同一 `/run/user/<uid>/ptb-resource-admission-v1`。不使用不同 UID、容器隔离或新的锁目录来假装共享一槽。

候选方案比较：①复用现有回环预览入口，固定 HTTP/Host 和两任务默认值，不满足目标 HTTPS 配置，排除。②本任务的薄宿主直接组装现有 API/Storage/JobStore，补配置、TLS/Host、生命周期和维护门，采用。③容器/多服务 UID 增加当前锁与 user bus 隔离问题，首轮不采用。

`scripts/p15_staging/host.py` 属部署适配。算法、配额、租约、结果提交和远程路由归现有后端。配置拒绝未知操作/远程打开，初始空白名单；当前脚本只能服务服务器单独运行候选。B 生产桥未交付前，不同时启动新远程调度器和此处旧 claimant。

排空：control=drain 后停止新上传/新任务/计算预览和新 claim，既有任务继续；已经排队的任务保留。取消和读历史可用。maintenance 更严格关闭业务修改，但登录/会话和任务 GET 仍可能写数据库，因此**维护模式不等于数据库只读、停写或迁移隔离**。迁移时还必须停止三个精确服务及全部其他写者。

停止边界：SIGTERM 只通知本服务，worker 复用现有取消/进程清理。跨 systemd unit 的科学子进程是兄弟组，不能声称 KillMode 自动清掉它们。异常后由 P11 原 journal 的精确 unit 恢复流程确认，禁止通配停止所有 ptb 单元。

当前 P15 API 有补充关闭门：M08/M09/M14、ZIP、通用工程任务、通用 retry、远程 API 暂不可用。M01 仅 acoustic_analysis，textgrid_segment 不开放。通用 retry 缺少部署白名单过滤，保持关闭，待 B 的能力感知路径完成后串行接入。该限制必须在验收/能力矩阵中显示，不能称全功能上线。

独立服务器worker在claim前只读检查queued操作，发现不在部署允许表内的旧任务即保守暂停领取，任务保持queued，待人工审阅。它会同时等待后续有效任务，不宣称具备B的能力分流/公平调度。运行中禁混入其他提交者/claimant；完整路由接B后须替换这个部署安全门并复验。

参考：[systemd 执行配置](https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html)、[Nginx 代理](https://nginx.org/en/docs/http/ngx_http_proxy_module.html)、[Nginx TLS](https://nginx.org/en/docs/http/ngx_http_ssl_module.html)。模板的实际指令兼容性仍需目标安装版本运行 `systemd-analyze verify` 与 `nginx -t`，本地 Python 测试不替代它们。
