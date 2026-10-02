# M16 首版开发态验收

2026-10-02。状态 `in_progress`，已交付可运行本地开发态，完整硬件/平台验收尚未完成。井井最新指定默认双声道音频，EGG 手工可选，已落实。

## 实现范围

自由录音、可选任务清单、CSV/TSV/XLSX 预览映射、无损 JSON 模板、任务编辑/复制/顺序/跳过/停用/移出、多个 take 和独立选用版本。采集设备与输出设备分别选择，同声卡 1–8 通道显式角色；默认双音频，单输入设备明确降为单音频。

原始 float32 分块、有界队列、5 秒显示缓存、满幅/数字超幅/低电平/DC/相关性提示、可关实时 STFT。录后整数帧选区、剪切/复制/粘贴/删除/保留/撤销/重做/恢复原始，原始/当前/差分预听。降噪和增益在本机派生处理，用户勾选音频通道，EGG 排除并逐样本保护。导出支持 FLOAT/PCM24/PCM16、选区、全部 take/历史、多段 WAV 和逐项 JSON/CSV，取消/失败保留已有成功项。

工程独立目录、操作系统单写者锁、原子修订入口和损坏尾块恢复，不进入 P06/P07、现存数据库或服务器。普通浏览器和未准入平台无隐式上传回退。工作台公共接线由 root 串行维护。

## 已执行验证

以下均使用既有 `.venv/m09-ui`，没有安装依赖或覆盖环境。`PYTHONPATH` 指向本仓库 `packages/phonetic_core/src;desktop/src`；Qt 额外包含 `backend/src`。

| 命令 / 证据 | 结果与范围 |
| --- | --- |
| `python -m pytest -o addopts='' desktop/tests/test_m16_recording.py packages/phonetic_core/tests/test_recording_core.py -q` | 最终 **34 项通过**（含双音频默认、EGG手选、输出差分、采集及播放构造/开始/中止/关闭失败与睡眠中断）。覆盖原始/数字增益分离、分段、溢出、非有限、磁盘写失败、单写者、原子替换失败、损坏尾块、EDL/历史/raw hash、任务快照、三种量化与跨段导出、降噪/取消/EGG/分块一致性。根 legacy pytest 配置依赖未安装的 pytest-cov，覆盖率参数不适用于此独立测试，故显式清空 addopts，测试断言未放宽。 |
| `node --test tests/m16-recording.test.ts`（frontend） | 最终 **11 项通过**，含双音频默认/单输入回退、键盘上下文、无损JSON模板和XLSX前导零/公式拒绝。导入规则、展开即限制 10000、选区、模板公式防护、自由任务/跳过。 |
| `npm run typecheck`（frontend） | 已通过；根统一执行最终构建。 |
| `node tests/e2e/m16-recording.cjs` | 真实 Chrome + 生产 adapter/service，QWebChannel 传输替身、明确合成 PortAudio；**14 组通过**，无 pageerror / 公网请求。证据 `output/validation/m16/chrome-6048ffe2c48e4caeb78f43669b6fde9c/browser-report.json`。1920/1440/900/2560 宽度、浅/深主题，导入、任务ID编辑与复制、空格边界、采集/编辑/降噪/导出、跨模块继续录制。 |
| `python scripts/verify_m16_qt.py` | 实际 Qt/QWebChannel + 生产 bundle，明确合成输入/输出，不打开声卡；**8 组通过**。最终证据 `output/validation/m16/qt-4c74395fa9fb4c759bd4bc2dc5376508/report.json`。早期offscreen过渡旧帧截图保留为失败证据，修复抓图同步后已亲看浅/深稳定截图，不通过改业务CSS掩盖。 |
| `python scripts/verify_m16_long.py` | 受控加速 60 分钟、48kHz×2、172800000 帧、1382400000 raw 字节、1319 段，哈希一致，50.08 秒写入/回读，抽样工作集峰值104632320 B（99.8 MiB）、固定64块队列、显示235520帧。证据 `output/validation/m16/long-2d09a9a45246454cae8fae4f8bdeb4ec/report.json`。不是60分钟物理流或墙钟测试。首个探针working_set=0无效，旧证据保留、不引用其内存数据。 |
| WSL NInfer 既有 `/home/ninfer/ptb-m06-20260927/bin/python` | **11 项纯核心通过**，最终 `-p no:cacheprovider` 复跑无pytest cache警告。Windows外正式采集能力关闭，WSL启动器的localhost代理提示保留，不影响Linux核心测试。 |

WSL 可复现命令：

```powershell
wsl -d NInfer -- bash -lc "cd /mnt/d/PhoneticToolbox/PhoneticToolbox_v3 && PYTHONPATH=packages/phonetic_core/src:desktop/src /home/ninfer/ptb-m06-20260927/bin/python -m pytest -c /dev/null -p no:cacheprovider packages/phonetic_core/tests/test_recording_core.py -q"
```

root 联合 Qt 另有 **4 组通过**：`output/validation/m16-m17-integration/qt-545d1d1512444f359c87567daacf38c6/report.json`，覆盖默认双音频自由录制、切 M17 输入空格时继续采集、隐藏 M16 标签取消关闭/停止保存关闭、重新打开同一 take。root 宿主额外 5 项单测通过，其全量前端/构建结果由根收口报告记录。

## 修复的真实失败

保存任务清单曾在还没有 take 时清空当前 task ID，Qt 实际录制暴露任务快照变成自由录音。已修为保留有效的选中任务，Qt 快照 `SYN-001` 验证通过。Vue Proxy 的 structuredClone 改用 toRaw。打开新工程先保存旧任务草稿并重置选中归属。关闭时拒绝尚在进行的开始/停止/处理请求，防止迟到开始产生无人拥有的流。

Canvas ResizeObserver 调整为下一帧绘制且尺寸变化才重设 backing store，Chrome 无循环告警。实时谱从覆盖最后2秒改为完整5秒输入，按实际 STFT 中心时间对齐，不拉伸到错误时间范围。队列满、写入失败、流停止/关闭失败均保留错误与所有权。线程型导出超时只请求取消，真实退出前不会释放工程。

## 未验证与当前限制

- 没有实体声卡、麦克风或 EGG 接线验收，没有模拟削波/驱动AGC标定、独立时钟/热插拔、60分钟墙钟录制、声卡循环延迟或人耳比较。合成测试不扩大为设备 verified。
- 自然气声/擦音处理及科研参数变化需要人工验证，默认不自动降噪。噪声样本只做统计提示，不宣称自动识别语音。
- WSL只纯核心；Linux原生/macOS/普通浏览器采集、RF64、真实实体多屏/DPI未准入。
- 历史波形是流式保峰显示，语谱显示限于当前预览窗口前2秒并明确标注，可放大选区查看；实时谱覆盖整个有界采集窗。高级STFT参数当前固定在方法版本并明示，没有等价Praat分析承诺。
- 原始 `.f32` 块须通过工程/导出读取，float32只表示应用收到的PCM；磁盘断电/fsync硬件持久性和存储损坏仍受系统影响。
- 导出中未完成文件保留 `.wav.partial`，只有格式/帧数/哈希校验通过后才发布 `.wav`，manifest列出partial_files。
- 没有 EXE 打包、公开发行、push、DDL、全局依赖、旧项目修改或真实博士论文语料访问。所有合成输入与输出均放隔离 `output/validation/m16/`。

开发入口：`scripts/Start-M16-M17-Workbench.ps1`，使用已存在的独立环境和根统一构建，不自动安装依赖或创建业务数据库。

来源和算法参数：[ADR](../decisions/ADR-M16-local-recording.md)、[来源增量](../references/m16-source-additions.json)、[使用手册](../manual/recording.md)、[本地契约](../../contracts/recording/README.md)。


追加交付：井井后续授权打包，最终R2实际EXE检查通过，范围与修复见[成品报告](2026-10-02-m16-m17-exe-report.md)。本文件之前的无EXE说明为开发态阶段记录。
