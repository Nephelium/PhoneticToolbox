# M15 · 感知实验纯客户端迁移计划

2026-09-27。状态：Windows 开发功能及 A/C 定向验证已完成，完整跨平台条目保持 in_progress。软件验收与未测平台分别见 [报告](../../testing/m15-report.md) 和 [统筹摘要](../../testing/m15-coordination-summary.md)。本轮明确授权已覆盖纯客户端实现，并追加确认文本/图片两帧门与 AppShell/Qt 最小接线。旧 Python core/API、服务器 manifest、P06/P07 存储步骤为通用模板，现已 Superseded，不再作为 M15 依赖。

## 依据与退出门

[源功能映射](../../modules/evidence/M15-source-map.md)覆盖说明书 6.1–6.2 与原 HTML。Python 原服务只打开 HTML，不迁入后端。
[独立架构决定](../../decisions/ADR-M15-client.md)冻结时钟、资源预算、恢复、共享接线与离线 A/B/C；[操作说明](../../manual/perception.md)解释实际使用。

- F01 四范式与全部刺激角色、顺序和作答规则。
- F02 音频/图片/TXT、本地导入/目录拖入、分组、预览、hash 关联与缺失检查。
- F03 序列生成/编辑/排序/拖动/复制/移除、全局/闭区间洗牌；种子增强另存算法与最终顺序。
- F04 默认值保持、0 修复、全局/分段按键、首匹配重叠、阶段提示、手动/自动推进与播放。
- F05 文本/单选/多选问卷、必填、配置/答案会话快照。
- F06 XLSX 模板/导入/导出、JSON 配置、原文件重新关联与本机恢复。
- F07 运行/中断/安全边界/异常 attempt、部分及最终 CSV/XLSX/JSON、关闭保护。

## 阶段、文件归属与验收

| 阶段 | 文件 | 行为与验收 |
| --- | --- | --- |
| M15-A | tests/e2e/m15-baseline.cjs、源映射 | 独立执行 V2 函数体，四顺序、RT 起点、0/缺失/错误基准；不得用 V3 生成 expected |
| M15-B | frontend/src/modules/perception/model.ts、media.ts、runner.ts、storage.ts | 纯客户端模型、单调时钟/音频调度、有界解码、IDB 事务与 CAS/Web Locks；纯逻辑及真实浏览器故障测试 |
| M15-C | PerceptionPage.vue、formats.ts、drop.ts、vendor/* | 公共 ModuleFrame/Toolbar/Section/Status；五页签与正式专注视图、配置/资源/问卷与三格式结果；实际文件回读 |
| M15-D | AppShell.vue 局部、desktop/host.py 下载白名单局部；tests/e2e/m15*.cjs、scripts/verify_m15_qt.py | 重读共享 diff 后串行按需注册、异步保存/关闭保护、主题与按键隔离；Chrome 正式入口 A、实际 Qt C |
| 文档 | 本计划、ADR、source-map、manual、report、coordination-summary | 记录证据及边界，更新本模块状态，不扩大其他模块 verified |

不创建 API/数据库/云端 worker。不依赖 P07 政策迁移。无用户文件上传、全局依赖/系统改变、V2 修改、push、公开部署或 EXE 打包。

## 确切命令

从仓库根目录 Windows PowerShell 执行：

```powershell
node tests/e2e/m15-baseline.cjs
node --test frontend/tests/m15*.test.ts
npm --prefix frontend run typecheck
npm --prefix frontend run build
node tests/e2e/m15.cjs
node tests/e2e/m15-recovery.cjs
node tests/e2e/m15-runtime.cjs
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/verify_m15_qt.py
wsl -d NInfer --exec /home/ninfer/ptb-p11-20260926/venv/bin/python /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/scripts/verify_m15_linux_static.py
```

浏览器端到端使用独立固定生产构建，避免共享目录并行改动触发 HMR 刷新正式试次。runtime 是只用于注入故障的测试页，正式入口仍为 AppShell。运行时测试会发出短合成声音，已事先向用户说明，不调系统音量/默认设备。

## 科研与未测边界

音频仍从作答窗口开放起算；文本/图片经用户确认采用预载入/解码/两次 rAF 后开放。所有时间字段见报告。getOutputTimestamp 和 ended 只记录浏览器可观测估计。物理端到端时延需要回环/外部设备，本轮未测。

A：准备完成后断网运行/导出，必达。C：Qt 内置静态资源定向验收。B：用户明确同意先交付 A/C 和精确缓存方案，本轮不引入 Service Worker；方案在 ADR，后续单独串行整合。浏览器关闭后的断网冷启动不标 verified。

Linux 只分发静态资源，没有 M15 服务器计算进程、账号存储或远程节点。WSL 原生解释器已核验入口与 M15 静态资源，部署/其他浏览器仍未验。不能将 Windows Chrome/Qt 或 WSL 静态检查扩大为 Linux 浏览器验证。
