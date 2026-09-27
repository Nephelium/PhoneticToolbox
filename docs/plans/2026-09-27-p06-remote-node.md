# P06-REMOTE-NODE 实施计划与局部决策

用户已授权客户端开发、专用入口、本机合成验证、个人电脑 WSL 测试。无系统配置/生产部署授权。
只新增 packages/ptb_node、scripts/node 和节点专项文档，保留当前分支的其他未提交工作。

## 决策与范围

1. 标准库 Python 包，Linux 首发。复用 P11 科学运行接口，禁止私建第二套算法或绕过 trusted-worker 门。
2. B wire protocol 尚未冻结，只实现内部 port 与 HTTPS/分块原语，不猜测路径或生成契约。
3. 私有 AF_UNIX 控制套接字与 flock 单实例。无桌面依赖、TCP 入站端口或系统服务。
4. 租约以请求发出时 CLOCK_BOOTTIME 加服务器授予时长计算，扣除停止余量，绑定 attempt/generation。
5. 每个 attempt 独占随机目录，重启清理旧临时文件，不恢复旧 generation。清理失败阻断新任务。
6. 并行数配置当前仅接受 1；CPU/内存配置是待 P11 实际执行前核验的预算，未执行不宣称已生效。

与在此包自建 systemd runner 相比，复用 P11 保持公共资源治理与科学版本一致。
与猜测草案接口相比，封闭 binding 可避免旧结果/身份模型分叉。代价是 B/P11 接入前客户端不能领任务。
共享 ADR/台账由当前负责人整合，本专项记录不修改公共文件。

## 验证

- WSL NInfer 原生 Linux Python 3.11.14，只读盘点，独立目录与运行时。
- unittest：租约/断网/续租延迟/模拟恢复、hash/offset/重试、临时目录/权限/磁盘/清理、单实例与控制。
- 真实 loopback TLS 合成传输，验证固定 origin、重定向拒绝、服务端认证、chunk 重试。
- 无 systemd/cgroup 时保留明确拒绝。真实进程组预算、B 联验、云回退和实验室实机单列待验。

状态：in_progress。完整任务退出依赖 B 冻结协议与 P11 trusted-worker 接入，不能拿组件测试替代。

本轮结果：组件安装、51 项 WSL 合成测试、真实客户端崩溃重启/残留清理、shell 入口与依赖检查通过。
B 候选已补剩余租约字段，接口交接继续记录于 `docs/specs/p06-node-handoff.md`。
具体证据和仍未开放的能力见 `docs/testing/p06-node-report.md`。
