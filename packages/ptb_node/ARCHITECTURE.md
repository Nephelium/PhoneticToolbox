# 节点边界

本包负责本机配置、盘点、单实例/控制、租约、安全临时文件和有界传输。
`protocol.py` 是内部 Python port，不是服务端契约。B 冻结版本化契约后添加固定 binding，
不得从配置或任务动态导入模块/命令。当前 CLI 显式处于 protocol_unavailable。
科学入口交给现有 P11 adapter，当前 trusted-worker 门仍关闭。没有复制算法。
测试 fake 仅验证客户端状态机，不能证明服务器 fencing/发布事务。
