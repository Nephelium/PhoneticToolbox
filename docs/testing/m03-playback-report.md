# M03-E3 连续试听与 IF 角色收口

2026-09-13，verified，限定 Windows 开发态公共试听状态、EGG IF 播放角色和本轮连续操作。完整 M03 仍 in_progress，A20 的实际声卡/听觉、多屏DPI、来源与冻结发行范围未完成。见[实施计划](../plans/2026-09-13-m03-playback-closeout.md)。

## 修复与实际证据

1. 原 IF 结果按清单位置标注音频角色，而实际清单按文件名排序，`egg_IF.wav` 位于 `egg_ORIG.wav` 前。原文件内容与导出名称正确，页面试听角色对调。现按文件名识别并排列原音频/IF，标签绑定角色，不能以后台列表顺序推断。首次 Chrome 节点样本断言失败，读取该次任务清单确认根因；修复后两个节点分别与各自 WAV 解码的 Float32 样本及采样率精确一致。
2. 两个播放器此前共同显示暂停和同一进度，点击第二个会先暂停第一个。公共状态现记录资产身份及声道，各播放器只显示自己的状态与进度，另一条可直接开始；保留全工作台单路播放、同一条暂停续播。等待设备启动期间立即显示新选区起点。
3. 旧 `AudioContext.resume` 延迟失败会调用全局 stop，打断已成功的新播放。回归用例先复现失败，现过期请求在成功和失败路径都不改变新播放。当前请求的设备错误仍明确显示。

未修改科学核心、样本、选区换算、WAV/CSV/PNG导出或来源资源。历史双WAV文件回读只能证明文件正确，本轮补足按钮实际样本对应。

## 验证命令与产物

| 命令 | 实际结果 |
| --- | --- |
| `npm --prefix frontend run test` | 62 passed，含7项公共音频异步/归属/进度用例和IF文件排序用例。音频设备竞争采用确定性模拟，不冒充声卡故障实测。 |
| `npm --prefix frontend run typecheck` / `npm --prefix frontend run build` | 通过。 |
| `python scripts/validate_docs.py` / `python scripts/check_architecture.py` / `git diff --check` | 通过，566文件、330来源、32任务，无新增错误；历史快照缺失链接继续单列。 |
| `node tests/e2e/m03-playback.cjs` | 4组独立Chrome操作通过，真实本机服务与MKL核心，原生Web Audio节点未替换。仪器化仅记录传给节点的样本和开始/停止参数。归一化音频、交换后的音频、原始分析片段与IF各自精确对应；暂停续播、直接切换、换文件/标签停止、取消关闭保留及焦点返回、保存关闭均通过。`output/validation/m03-ui/chrome-cb76a56d030f4f11847af57f00c9ea10/report.json`，0页面错误。 |
| `python -X utf8 scripts/verify_m03_qt.py` | 9组实际Qt回归通过，包含四图、选区失效、单文件/IF/批次真实保存、窄窗和模块重开。`output/validation/m03-ui/qt-b85babf137b946ee9d210d8980d3f5f8/report.json`。Qt未做实体声卡回录。 |

Qt使用 `.venv/m09-ui/Scripts/python.exe`，`PYTHONPATH`为本项目backend/src与desktop/src，计算使用既有 `.venv/m03-compatible/python.exe`。测试数据库由已有合成测试库只读复制，不执行DDL。原有Qt PNG元数据警告保留，未屏蔽。最终截图已查看；播放器动作及角色依据实时状态与样本断言，截图不替代音频验证。

失败记录：`chrome-8af80e0853b14e66b2506d95766ba1be` 为真实IF角色错配；后两轮 `chrome-a42631ff4a7e4cd3ac762a69cf1080d2`、`chrome-3d6e8af2ddff48dab25ab5079b5f5206` 为测试定位器遗漏未保存标记、误写按钮名，按现有DOM修正。未通过降低样本断言或改科学结果获取通过。公共异步用例先失败后通过。

## 来源和下一阶段

仅复核当前本地方法审阅及来源记录，无新增论文或授权证据。`PENDING-EGG`许可、SQ定义引用链和简化CP实现来源继续未闭合，不宣称本轮解决。未下载字体/论文、升级环境或联系作者。

[M03-F候选范围审阅](../plans/2026-09-13-m03-f-candidate-review.md)已形成 planned 方案：旧 Research-Fix1 脚本排除Matplotlib，当前EGG依赖外部固定Conda Python与构建记录，后台字体资源也需独立核对。不能直接复用旧打包脚本宣布EGG独立EXE可用。下一项按候选范围审阅推进隔离运行时探针和剩余适用验收，实际打包/发行状态单列。未改v2、旧EXE、数据库schema、全局依赖或CI/CD，未push、发布或部署。
