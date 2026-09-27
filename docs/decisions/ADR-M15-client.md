# ADR-M15-001 纯客户端感知实验

2026-09-27，用户明确授权。状态：accepted；实现与验证状态另见 M15 报告。

## 架构

M15 使用现有 Vue 3.5.42 工作台与按需加载的本地模块，Web Audio、File/Blob、IndexedDB 负责呈现与数据。没有 Python core、服务端 API、任务 manifest、账号/数据库/远程 worker。用户选择的文件不上传，不计入服务器额度/有效期。旧 React、Babel、Tailwind、lucide-react CDN 不进入新版运行资源。SheetJS CE 0.20.3 官方 ESM 原件随模块构建，Apache-2.0 许可保留，无运行时 CDN。

## 方法冻结与明确修正

音频保持 X、A-X、A-B-X、A-X-B，完成整段序列后开放作答，RT 从作答窗口开放起算。performance.now 为主 RT 时钟，event.timeStamp 单列为诊断，Date 仅作日历。AudioContext 秒与 performance 毫秒以成对取样记录映射，不直接相减。getOutputTimestamp 为浏览器估计，不能当耳机测量；ended 回调只记观测，不作声学终点。

用户已在本轮审阅并同意：文本/图片读取解码后呈现，经过两次 requestAnimationFrame 再开放作答。记录呈现回调与开放时间，不声称物理显现精度。X 文本通路恢复，旧上传器排除 TXT 与旧运行器支持文本的冲突单列修复。复杂范式只接受音频，避免凭空定义混合媒体 ISI。

0 ISI/间隔合法，负数/非有限值拒绝；缺失/解码/调度失败停止并保存异常 attempt。重复响应、repeat、组合键、IME、跨阶段同一事件不能作答。分段按键重叠保留旧“列表首个匹配优先”，界面明示；非法/越界范围拒绝，洗牌范围重叠按列表顺序执行并记录最终序列。种子为增强 mulberry32-v1，未洗牌也保存最终顺序。旧 autoPlay/allowReplay 没有运行分支，其中手动播放按说明书补齐，重新呈现保留独立 attempt。

## 持久化与中断

项目/刺激 Blob/会话分对象存储；独立项目、被试、会话 ID，配置与刺激哈希快照。会话更新使用 IndexedDB revision CAS，浏览器支持时加 Web Locks 防多标签并发。每次 trial 的 running 边界先落盘，完成/异常结果落盘成功后推进；失败停在 saving-error，可导出内存记录并重试。恢复将 running 标为 interrupted，不续旧 timeOrigin RT；重新呈现须显式确认。浏览器不能检测的硬件故障不伪造事件。

## 资源预算

最多 2000 素材、10000 试次、单文件 64 MiB、总原始文件 512 MiB、解码缓存 128 MiB。全刺激预检顺序解码，累积超预算切换按试次分块，试次间预备完成后才启动。单试次超过缓存预算明确拒绝。无裁剪/归一化/声道修改。WAV 原采样率从头读取，其他容器无法可靠读到的原率记 null，并告知不可追溯项；播放采样率、声道数/时长来自实际解码。

## 已批准的共享接线与离线范围

本轮第二次明确确认：AppShell 增量按需注册、异步保存/未导出保护与专注按键/主题隔离；冲突任务仅提示。Qt 当前页面 blob 下载白名单增加 json/csv/xlsx，继续原生路径选择。修改共享文件前重读 diff。

A：静态脚本/字体与刺激准备完成后断网完成与导出为必达。C：实际 Qt 内置 ptbapp 资源另验。B：用户同意本轮交付精确方案，暂不实施 Service Worker。方案为 `/m15-offline/` 的同一 AppShell 静态入口、仅该 scope 的版本化缓存、构建清单中的公开 JS/CSS/字体，拒绝 API/非 GET/跨源与私有响应；不 skipWaiting/clients.claim，不在实验中激活，首页显式准备/更新。需要宿主静态路由和构建输出串行整合后另验离线冷启动。当前不保证浏览器关闭后断网重开或 file://。

来源：[Web Audio](https://www.w3.org/TR/webaudio/)、[SheetJS 分发](https://docs.sheetjs.com/docs/getting-started/installation/standalone/)、[SheetJS 许可](https://docs.sheetjs.com/docs/miscellany/license/)。物理声音/键盘端到端延迟留硬件测量门，不阻挡软件交付。
