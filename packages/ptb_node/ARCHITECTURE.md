# 外部计算节点架构

本包负责节点配置、资源盘点、单实例控制、租约、安全临时文件与有界传输。它复用现有科学 adapter，独立节点状态机不复制算法。共用边界见[总架构](../../ARCHITECTURE.md)。

## 协议与执行

`protocol.py` 定义内部 Python port，`service.py` 通过显式 binding 接协议，`runtime.py` 控制科学执行准入。服务端 [remote/1](../../docs/specs/remote-protocol-v1.md)已有版本化规格，客户端 binding 和实际科学准入仍需分别完成。

当前 CLI 保留 `protocol_unavailable` / `binding_pending` 路径，trusted-worker 门仍关闭。配置或任务不能指定动态导入的任意模块和命令，未通过资源硬限制与模块证据门时不执行科学任务。

## 状态与验证

凭据及运行目录独立于公开源码。客户端测试 fake 验证配置、状态机、租约和传输边界，不证明服务端 fencing、结果发布事务或科学任务已接通。验收须区分协议客户端、实际服务器和具体科学模块，不能从接口存在推断生产可用。
