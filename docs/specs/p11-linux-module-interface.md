# Linux 任务执行接口与 M08 接入示例

2026-09-26。P11-LINUX 后端接口，用户已授权实现。未变更 API 请求/响应模型、数据库 schema 或生成契约。全局 ADR、source-registry 和台账由统筹汇总。

## 固定入口

`ptb_worker.acoustic_executor.collect_scientific(entry, request, scratch, limits, stop, on_started=None, evidence=None) -> bytes`

- `entry` 是宿主源码 allowlist 的标识，当前生产业务入口为 `lpc`、`egg`、`acoustic`、`segment`。`request` 是 `ManagedScratch.create()` 返回的受预留保护文件，客户端不可提供解释器、模块或路径。
- Windows 沿用 `collect_pipe`、Job Object、EGG/LPC 既有兼容环境。Linux 调用 `native.linux_runtime.run()`，由临时 user systemd unit 先安装内存/CPU/进程数限制，再启动 `linux_bootstrap.py`。
- Linux 每组上限 1 GiB，更低的调用预算保留；CPUQuota=100%、无 swap、TasksMax=64。`stop` 在父进程中检查取消/租约失效。`on_started` 收到实际科学 MainPID。`evidence` 接收 unit、资源峰值、实际限制和 `cleaned`，失败时仍可审计。
- Linux bootstrap 使用 `-I -B`，只加入宿主 runtime profile 的显式包路径。普通打印转入 stderr，长度前缀 bundle 经 stdout 内核管道返回。子进程不能绕过父级配额直接写最终结果。
- 父级保留输入 hash、bundle 大小/名称/内容 hash 校验、身份/租约/fencing、预留、逐块写入及最后原子可见性门。取消、崩溃、超时及写入失败不发布部分结果。

`fonts` 与 `spectrogram` 是固定的内存请求入口，共用 runtime 和受限组。它们不是任务操作，也不独立开放算法 capability。

## Runtime profile

管理员进程设置 `PTB_LINUX_RUNTIME_PROFILE` 为绝对 JSON 路径。当前 schema 为 `p11-runtime/1`：

```json
{
  "schema": "p11-runtime/1",
  "python": "/task/venv/bin/python",
  "cache": "/task/matplotlib-cache",
  "sys_paths": ["/task/backend-wheel-target", "/task/render-overlay", "/task/pandas-overlay"],
  "versions": {"phonetic-core": "3.0.0a1", "ptb-api": "3.0.0a1", "numpy": "2.2.6", "scipy": "1.16.3", "praat-parselmouth": "0.4.7", "matplotlib": "3.10.8", "pandas": "2.3.3"},
  "fonts": ["/task/fonts/NotoSansSC-VF.ttf"],
  "reaper_binary": "/task/reaper",
  "hashes": {"/task/venv/bin/python": "<sha256>", "/task/backend-wheel-target/ptb_worker/linux_bootstrap.py": "<sha256>"}
}
```

这是结构示例，不能直接作为已验证配置。实际 profile 的 `hashes` 包含解释器、当前 backend/core 源码、NumPy/SciPy 原生库、字体和 REAPER。构建/升级后重新生成 profile 并验证，不能沿用旧报告。hash 校验不等同系统沙箱：这是可信宿主选择的固定程序，不接收第三方任务脚本。

Linux REAPER 来自 google/REAPER 固定提交 `1d6e9b95e6b08b500fccbc9a043989dbda747276`，保留 Apache-2.0 及源码包哈希。既有 Windows EXE 来源提交未知，不能认为两个二进制同源构建完全相同。Linux F0 输出通过继承的匿名管道文件描述符返回，REAPER 留在科学子进程的同一 cgroup。Linux 原生执行失败后拒绝发布缺失 rF0 的默认结果，无声段的正常 native unvoiced 结果允许保留 NaN。

## Capability 凭据

`PTB_LINUX_VALIDATION_RECEIPT` 指向宿主生成的只读配置，包含当前 `profile_sha256` 和各操作的实际 `report` 绝对路径、报告 `sha256`。报告必须成功、phase 对应、profile 一致；解释器/程序/资源 hash 改变即关闭。还需可用的 user systemd 和当前 job store `max_running=1`。M01 的文件适配器必须登记 profile 中的同一 REAPER。

只开放各自验收通过的 `lpc_analysis`/M04、`egg_analysis`/M03、`acoustic_analysis`/M01。当前没有开放 Linux M09、远程节点或未验证分段 capability。该检查只限制当前 store；跨服务实例的全机单槽、资源总和和并发准入仍由 P11-PERF 负责。

## M08 等模块的接入步骤

1. 模块 agent 提供固定 child 模块与版本化配置、输入 hash、输出清单和错误码，不修改共享执行器。函数应直接调用纯 core，返回完整、受大小限制的文件 bundle，不持有数据库或服务器路径。
2. 提交给平台负责人登记唯一 entry、固定 child 和宿主任务 operation 分支。现有 bundle parser 的 kind allowlist 也须按模块契约明确扩展，不能复用错误的 `prepared_analysis` kind 冒充 M01。
3. 宿主调用形态：

```python
request = scratch.create(encoded_header + b'\n' + input_bytes, '.json')
raw = collect_scientific(
    'm08', request, scratch, module_limits,
    stop=lambda: abort.is_set() or stop_event.is_set(),
    evidence=process_evidence,
)
# 登记前 'm08' 明确拒绝。登记后校验模块自己的 bundle、输入 hash、
# 必需输出名，再进入既有 files.output/write/seal/complete 流程。
```

4. Linux runtime profile 补入所需固定源码和资源 hash，定向验证实际输入→持久任务→子进程→结果→下载、取消/崩溃/超时/部分写入恢复及无残留。算法等价与资源测量分别记录。
5. 验证完成后才加入该模块的 capability 凭据和允许表。M08 本轮没有被登记或开放，其模块实现、UI、生成契约仍归原负责人。

## 本轮明确未涵盖

没有生产服务、HTTPS、公网端口、真实 PostgreSQL 网页账号验收或配额/保留政策迁移。Linux HTTP 验证使用现有合成 SQLite 库的隔离副本及本地 token。字体使用显式 Noto Sans SC/DejaVu Sans/Doulos SIL 快照，尚不代表公共 UI 已提供 Linux 字体自动选择或浏览器端完整交互验收。
