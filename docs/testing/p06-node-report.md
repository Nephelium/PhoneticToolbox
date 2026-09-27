# P06-REMOTE-NODE 客户端组件实测

2026-09-27。**组件 verified（下述 WSL 合成范围），完整可信计算节点 in_progress。**
可安装/启动的控制客户端与独立租约/传输/attempt 引擎已经实现。CLI 保持协议/P11 双门关闭，
没有实际领取任务，没有执行科学算法，没有真实云端或实验室节点发布结果。

## 文件范围

- `packages/ptb_node/`：标准库包、内部 port、单实例控制、环境盘点、凭据权限、租约、
  有界 HTTPS/分块、独占临时目录、独立 attempt 引擎和专项测试。
- `scripts/node/`：Linux 前台入口与非秘密模板。
- 本报告、[计划](../plans/2026-09-27-p06-remote-node.md)、
  [使用说明](../deployment/trusted-node.md)、[接口交接](../specs/p06-node-handoff.md)。

未修改云端调度、生成契约、公共政策/AppShell、共享 P11 adapter、第三方依赖、全局环境、
WSL 配置或现存服务。保留工作区其他任务的改动。未 push、生产部署、开机启动或系统服务安装。

## 个人电脑 WSL Linux 节点

只读实测 NInfer / Ubuntu 24.04.4 LTS / x86_64 / WSL2 6.6.87.2。
24 个可见 CPU，MemTotal 33,367,920,640 bytes，采样 MemAvailable 32,650,399,744 bytes，
SwapTotal 8,589,934,592 bytes，采样磁盘可用 995,363,074,048 bytes。
这些是环境瞬时盘点，**不是科学进程峰值、服务器或实验室硬件数据**。

PID1 为 `init(NInfer)`，无 systemctl/systemd-run，cgroup v2 控制器存在但普通用户不能写根组。
资源硬限制明确 unavailable，没有关闭能力检查。全局 python3 命令缺失，已有 P11 项目原生
CPython 3.11.14 可用，本轮用它新建 `/home/ninfer/ptb-node-20260927/venv`，只装本包。
新 venv 未安装科学包，盘点版本均为 null，未把已有 P11 运行时自动认证成节点科学环境。
fontconfig 实测仅 DejaVu Sans/Mono/Serif，无中文/IPA 合格证据。GPU 未广告。

## 已执行与证据

WSL 原生解释器执行：

```sh
/home/ninfer/ptb-p11-20260926/venv/bin/python -m venv /home/ninfer/ptb-node-20260927/venv
/home/ninfer/ptb-node-20260927/venv/bin/python -m pip install --no-index --no-deps --no-build-isolation /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/packages/ptb_node
/home/ninfer/ptb-node-20260927/venv/bin/python -B -m unittest discover -s /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/packages/ptb_node/tests -v
/home/ninfer/ptb-node-20260927/venv/bin/python -B -m ptb_node inspect
```

初轮 41 项通过，追加 attempt 引擎后 48 项通过；最终轮与产物 hash 见本报告末尾。
测试是增量替换，不能累计成 89 项。TLS fixture 用已有 OpenSSL 生成临时本地证书，仅信任该测试 CA，
校验 hostname，不关闭验证。只监听测试期 127.0.0.1 随机端口，结束后自行关闭，产品没有入站 TCP。

| 场景 | 证据和边界 |
| --- | --- |
| 单实例、暂停/继续/状态/退出 | 真实独立 CLI 子进程、第二实例拒绝、AF_UNIX 请求与锁释放；无科学进程 |
| 无关进程保护 | 测试自己创建的对照进程在节点退出后仍存活，再由测试独立收尾 |
| 资源硬限制 | 当前环境缺失、P11 门拒绝；**未验进程组硬限制/OOM** |
| 网络失败/重连 | 回调受控异常、健康重连；真实 TLS 上传存块后断连接、重传同 offset 成功 |
| TLS/跨 origin/凭据 | 真实 CA/hostname 验证、未知 CA 拒绝、redirect 拒绝、403 撤销响应、禁止 query token |
| 下载/上传 | 1 MiB 公开合成字节分块传输，长度/hash/offset/响应 cap 和有界重试 |
| 租约失效 | 假时钟精确边界、迟到回复不复活、generation 不变；真实慢 TLS 下载由 watchdog 中断 |
| 休眠恢复 | 注入 boottime 与 monotonic 偏移，未真实休眠个人电脑 |
| 心跳隔离 | 阻塞续租调用不阻挡独立 watchdog，临时错误后恢复，撤销立即停止 |
| 版本/字体不合格 | 内部 runtime port 注入拒绝，输入未下载、计算未开始；未验证真实字体渲染 |
| 磁盘不足/超限 | 注入可用空间及实际 writer 限额，不填满个人电脑硬盘 |
| 清理失败/残留/链接 | 受控 PermissionError、真实私有目录与链接，保留无关目标、阻止下一 attempt |
| attempt 全流程 | 内部假 binding + 固定合成文件拷贝 runner，绝非 phonetic_core/服务器发布证明 |
| complete 回包丢失 | 客户端保持原 attempt/generation 重试；唯一发布仍须 B 数据库联验 |

## 尚未完成

1. B remote/1 刚新增的候选接口尚未冻结，已将剩余租期与模型缺口交接 B。
   B 已在候选源码补入剩余租期、输入 expiry 和响应模型，最终冻结/协议联验仍待。
   C 没有修改其 schema 或自行定义 wire API。健康、poll、注册、撤销、complete 仍未与真实 B 路由接线。
2. P11 trusted-worker 仍关闭。共享 adapter 的预算、abort/recover、流式输出及模块证据接口已列入交接。
   公共负责人已确认既有固定入口/stop/清理证据/on_chunk 可作为基础，但所需节点接口尚未接通，
   没有因本次交接扩大原 P11-PERF 范围。
3. 个人电脑 Windows 原生节点：未实施/未测试。
4. 云服务器回退：本轮未验证。不能把 P11 历史服务器资源测试当作节点故障接管证据。
5. 实验室 Linux 实机：待验。发行版/架构、CPU/RAM/GPU、磁盘、权限、出网和实际模块运行时未知。
6. B/C/E 的节点执行 → 故障 → 云接管 → 恢复 → 后续回节点、旧结果拒收、单一发布联合门待验。
   当前生产客户端不 poll，自然不会抢走服务器正在执行的任务，但这不等于调度联合验收通过。

断电时物理删除限制已写入使用说明。凭据保存命令只做本地 provisioning，不冒充注册或撤销。

## 最终轮

- 安装最终 wheel 后 **51 tests / OK，1.447 s**，无 skipped。零 skipped 仅表示组件用例均执行，
  不代表把缺失的 cgroup/科学验收包含进来。
- `scripts/node/verify_node.py --output .../smoke.json`：真实 CLI 暂停/恢复、第二实例拒绝、
  精确 kill 自己的客户端 → 重启 → 过期合成残留清理、正常退出全部通过。
  `check` 退出 2、ready=false 为预期的硬门拒绝。没有终止整个 WSL/网络或无关服务。
- `scripts/node/ptb-node.sh inspect` 已由安装环境运行；`pip check` 无破损依赖。
- 最终 wheel：`output/validation/p06-node-20260927/ptb_node-0.1.0-py3-none-any.whl`，
  SHA256 `00ff13fd94be3c4606d8904ef2cb9ae1673fd0a3ba609bd454e531fb68245105`。
- 原始测试/盘点/冒烟/构建输出集中在 `output/validation/p06-node-20260927/`，
  WSL 原始副本在 `/home/ninfer/ptb-node-20260927/`。
- 此 venv 的基础解释器仍是既有 `/home/ninfer/ptb-p11-20260926/python/bin/python3.11`，
  没有复制成独立便携 Python。实验室应按手册使用自己的解释器重新创建 venv，不能搬此目录冒充可移植发行包。
- 只读盘点既有 P11 venv 的结果单独保存 `existing-runtime-inventory.json`。
  该环境未修改，也未作为本包科学能力广告。
