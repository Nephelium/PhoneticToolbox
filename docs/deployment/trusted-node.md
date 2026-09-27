# 可信 Linux 节点客户端基础版

2026-09-27，P06-REMOTE-NODE。**当前可安装、启动、控制和盘点，不能领取科学任务。**
B 的 remote/1 仍在冻结，P11 trusted-worker 尚未开放，WSL 缺少资源硬限制。
这是可继续接线的组件版本，完整节点状态为 in_progress。详见[实测报告](../testing/p06-node-report.md)
及[接口交接](../specs/p06-node-handoff.md)。不要将本包接到未授权的真实语料。

## 目录与安装

每台主机使用自己的普通用户项目目录，不使用系统 Python 包目录，不改全局 PATH/CUDA。
实验室之后独立建立目录、虚拟环境、凭据与资源证据，不能复制个人电脑身份。

下列命令中的路径由管理员按实际机器填写。先核对原生 Linux Python 3.11+ 和 venv/setuptools
是否已有。没有时先给出用户目录运行时方案，不 sudo 安装。现有 WSL 的具体环境见报告。

```sh
umask 077
/absolute/approved/python3 -m venv /absolute/node/venv
/absolute/node/venv/bin/python -m pip install --no-index --no-deps --no-build-isolation /absolute/repository/packages/ptb_node
export PTB_NODE_PYTHON=/absolute/node/venv/bin/python
sh /absolute/repository/scripts/node/ptb-node.sh inspect
```

节点控制包无第三方运行依赖。离线源码构建需要已有 setuptools>=68/wheel，缺少则明确失败，
不要去掉校验或自动联网装全局包。科学运行时是另一个需固定版本/文件 hash 的 P11 环境，
控制包安装成功不代表 phonetic_core、BLAS、字体、模型已合格。

将 [配置模板](../../scripts/node/config.example.json)复制到私有目录并填写真实 origin 和绝对路径。
HTTPS origin 只包含 scheme/host/可选端口，无用户信息、路径、query 或 token。节点不绑定机器名/IP。
`state_dir` 及凭据父目录须归当前用户、0700，无符号链接祖先。
`credential_file` 0600，不能放仓库、Windows 挂载盘或共享目录。

## 启动与控制

```sh
sh /absolute/repository/scripts/node/ptb-node.sh --config /absolute/node/config.json check
sh /absolute/repository/scripts/node/ptb-node.sh --config /absolute/node/config.json start
```

`check` 当前退出码 **2**，返回 ready=false 和具体门状态。`start` 为前台进程，
可用于验证控制入口；显示 protocol_unavailable，不会发假领取请求。
另一终端使用同一配置：

```sh
sh /absolute/repository/scripts/node/ptb-node.sh --config /absolute/node/config.json status
sh /absolute/repository/scripts/node/ptb-node.sh --config /absolute/node/config.json pause
sh /absolute/repository/scripts/node/ptb-node.sh --config /absolute/node/config.json resume
sh /absolute/repository/scripts/node/ptb-node.sh --config /absolute/node/config.json abort
sh /absolute/repository/scripts/node/ptb-node.sh --config /absolute/node/config.json stop
```

pause 仅关闭后续接单意愿，abort 才表示终止当前 attempt。当前没有实际科学任务可终止。
Ctrl+C/SIGTERM 等价于 stop。控制只走 0600 的 AF_UNIX socket，核对同 UID，没有入站 TCP 端口。
flock 持有实例锁；退出不删除锁 inode。崩溃后内核释放锁，下次启动在锁内清理自己的旧 socket。
单实例保护限同一 state_dir，管理员须保证同一节点身份只配置一个实例目录。

无需桌面、托盘、开机启动或 systemd 常驻服务。本轮未安装服务。
未来用户需要后台/开机启动时，先审阅具体 unit、工作目录、账号与资源委托，再另行授权。

## 配额、时段和能力

默认一槽、CPUQuota 目标 50%、科学组内存目标 512 MiB、临时目录预算 1 GiB、磁盘额外留 1 GiB。
一槽当前是唯一可接受 parallelism；CPU 配置范围 1–100，其他值拒绝，避免声称支持未验多槽。
CPU 50% 指一个 CPU 核的 50% 时间，不代表整机 CPU 的 50%。预算会由 P11 adapter 在执行前落地，
当前配置值**不是已生效的 cgroup 限制**。实验室仍需结合日常工作实测调整。

work_hours 是本机时区的 `[start_hour,end_hour)`，默认 `[0,24]`，本版单个非跨午夜时段。
时段结束/暂停只阻止新任务，已接任务仍受服务器 deadline/lease 与本地 abort 约束。
GPU 不扫描、不广告、不作为首轮依赖。模型/字体/原生组件由批准模块 receipt 登记。

## 注册、凭据轮换与撤销

1. B 服务器管理员核实节点作用域（owner/project 配对）、操作白名单、runtime hash、内存/槽数。
   当前 B 为服务器内部 register/issue/revoke 方法，尚无已发布注册 HTTP/UI。本包不猜测登记端点。
2. 管理员通过既有安全方式提供短效凭据，不能发到命令行、URL、源码或 Git。
   在私人交互终端可用下列入口安全保存新文件：

   ```sh
   sh /absolute/repository/scripts/node/ptb-node.sh --config /absolute/node/config.json credential
   ```

   密文输入不回显；非交互输入被拒绝，不接 token 参数。保存成功仅说明本地文件已写入，
   **不表示服务器已登记或连接成功**。已有文件拒绝覆盖；轮换时配置一个新的私有文件路径。
3. B 当前短效凭据默认 15 分钟，后续续期须走其批准机制。本包不制造长期身份或自动续期接口。
4. 撤销先在 B 的管理员入口撤销该节点身份，再本地 stop/cleanup。删除本地凭据本身不能撤销服务器身份。
   尚无管理员入口时等待 B 接线，不直接修改 PostgreSQL。
5. 实验室注册新的节点身份，重新生成/发放自己的凭据。不要把个人电脑 token 或私钥复制过去。

## 数据与清理

输入仅使用 B 批准的 asset ID，binding 禁止任意下载 URL、shell 或路径。
每个 attempt 使用 `state_dir/attempts/a-<random>`，本地文件随机命名。
组件按 256 KiB 分块校验长度/hash，输出原样 offset/hash 重试。最终是否发布由 B 原子事务决定。
当前合成测试没有处理任何研究语料。

```sh
sh /absolute/repository/scripts/node/ptb-node.sh --config /absolute/node/config.json cleanup
```

cleanup 先获得实例锁，只清理带本包 owner 标记的 attempt 目录。未知条目、链接目录、权限错误
均停止并报告，不扩大删除范围。目录内符号链接只删除链接，不跟随其指向。客户端运行中 cleanup 拒绝。
正常完成/取消/失败后，确认科学组已清空才删临时文件。失败保持残留并关闭接单，用户排查具体错误。
重启不恢复旧 attempt，先丢弃本节点拥有的残留，之后重新获得新租约，绝不借新 generation 提交旧输出。
实际 P11 孤儿组恢复接口尚待串行接入，在它完成前不存在科学启动能力。

断电时无法保证物理即时删除，永久离线也无法保证三天墙钟内物理清理。
当前无磁盘加密机制，仅文件权限保护，不向对离线副本有严格限制的研究任务承诺安全擦除。
日志/状态不保存语料、完整标签、科学参数或凭据。

## WSL 的最小环境方案（未实施）

现有 NInfer 无 systemctl/systemd-run，PID1 为 WSL init，无普通用户 cgroup 委托。
因此不能通过设线程数或单进程 RLIMIT 声称内存硬限制合格。

后续可选：

- 在用户另行批准维护窗口后，由管理员核实该发行版的 systemd 支持及所需系统组件，
  审阅启用配置和 user manager/cgroup 委托。此过程可能涉及发行版重启和系统包，当前不执行。
- 直接在已具备 systemd 用户管理器和资源委托的实验室 Linux 上新建普通用户项目目录，
  由获授权管理员提供必要委托；客户端本身不 sudo。

之后必须实际测 MemoryMax/MemorySwapMax/CPUQuota/TasksMax、父子合计 OOM、取消/孤儿恢复、
单槽和清理失败锁定，再核对 P11 receipt。仅工具存在/目录可写仍不构成通过。
Windows 补充路径当前未建设，避免在 B/P11 未冻结时同时维护两套节点。
