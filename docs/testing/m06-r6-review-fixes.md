# M06 六项审查问题修复与定向验证

2026-10-21。状态：**verified，限当前共用源码、Windows Qt/Chrome、WSL 纯核心和浏览器 Pyodide**。macOS 实机与最终发行物另验。

## 修复范围

- 网页设置通过独立偏好对象恢复，再事务性缩放时长、重建 F0 平移并统一校验。完整草稿曲线保留；失败保留原配置。
- 元音生成完整保留 AV 点与标量覆盖，静音只登记独立区间，重复 `a a → a` 不遗留旧静音。旧文件中显式 AV 零值保留，不猜测其来源。
- 静音登记先于 10 ms 曲线网格检查，按绝对整数半开采样边界放置，避免多段分别取整后整段拉伸造成边界漂移。诊断记录 `m06-silence/2` 与 `silence_sample_intervals`，串联声源和 HNR 公式仍为 `klatt/2`。
- A1–A5 当前串联路由未使用，三端共同禁用新编辑，旧文件/导出字段保留。HNR 新覆盖与手绘为 0–60 dB，旧文件 60–100 dB 可读取、切换参数、保存及导出，明确其等效上限。
- 网页本机失败/取消发布对应终态；已知本机任务不发往服务器取消。存储失败保留最近一项已计算数据和分文件保存进度，重试只补存，不重算，修改配置后仍拒绝迟到结果。
- 浏览器运行时原先的打包源码不会自动跟随 Python 编辑。共用打包脚本增加 sources-only 模式，对既有本机二进制逐项核对锁定 SHA，再生成当前源包和锁定清单，无网络下载或全局安装。

实现以[本轮计划](../plans/2026-10-21-m06-review-fixes.md)为准。文件格式仍为 m06/2，无用户文件迁移或历史 WAV 重算。Windows/macOS 使用同一 Vue 页面、平台接口和 phonetic_core，网页的 Pyodide 包由相同源码重新生成。

## 已执行的检查

| 范围 | 命令与结果 |
| --- | --- |
| Windows 科学/兼容回归 | `.venv/m14/Scripts/python.exe -B -X utf8 -m pytest -c tests/pytest.ini packages/phonetic_core/tests/test_m06_r6.py packages/phonetic_core/tests/test_m06_revised_sources.py packages/phonetic_core/tests/test_m06_r3.py backend/tests/test_m06_r5.py -q -p no:cacheprovider`，37 通过，保留 5 条已有 HNR 空切片警告 |
| WSL NInfer 纯核心 | 既有 `/home/ninfer/ptb-m06-20260927/bin/python`，PYTHONPATH 指向当前 v3 源码，同三组核心测试 33 通过、同 5 条警告；无 Linux 桌面或服务器部署结论 |
| 共用前端 | `node --test tests/m06-r6.test.ts tests/m06.test.ts tests/m06-r2.test.ts tests/m06-r3.test.ts tests/m06-r5.test.ts tests/browser-portable-modules.test.ts tests/web-settings.test.ts`，42 通过 |
| 静态与构建 | typecheck、contracts:check、ui-data:check、build:workbench 通过；保留既有大分块提示，未清空正在运行窗口的旧资源 |
| 说明书 | strict 校验 21 章、1113 素材无错误/警告，software 阅读资源已生成；本章更新正文，无新录音/图片素材 |
| Windows 真实任务 | `backend/tests/test_m06.py test_m06_r3.py test_m06_r5.py`，17 通过，包含真实 CC/AC、原生 REAPER、三类任务、不可变快照、取消、导出和清理。与前述测试去重后 Windows Python 共 50 项 |
| 实际 Qt | `scripts/verify_m06_r6.py --qt-only ...` 四组通过：真实桥接/worker 的 AV 与重新生成、35 采样点静音及导出、A1–A5/HNR/旧文件兼容、浅深截图。Qt 为本机隔离 offscreen，日志保留 GPU 软件回退，不作为实体声卡或 DWM 验收 |
| 实际 Chrome/Pyodide | 同入口 `--browser-only ...` 五组通过：实际页面设置恢复与生成、AV、参数兼容、受控存储失败后真实 IndexedDB 补存且不重算、原生与浏览器比较。最终 pageerror 和 consoleWarnings 均为空，浅深截图已检查 |
| 数值比较 | 同一 seed=23 的两份 WAV（移除旧静音、约 2.216 ms 新静音）与 Windows 原生输出逐样本一致，最大 PCM 差 0；另外三组生成配置的 AV 完全一致、时间边界在 1e-15 秒内一致 |
| 目录/文档/资源 | M06 catalog 24 参数一致；架构无错误；文档检查 2407 文件、448 来源、175 任务无当前错误，旧历史快照缺链单列保留。M06 源工程/public/dist 正文一致，浏览器 ZIP 内 engine/api 与当前源文件及锁定 SHA 一致 |
| 测试产物 | 10 份 scratch 回执均为 deleted，测试 WAV、数据库、Chrome 配置和临时参数文件均已清理；保留脚本、日志、截图、对照摘要与作为后续构建输入的浏览器源码包 |

## 证据与复核

- Qt：`output/validation/m06-r6/qt-1d2de7a2ccf441ec879e9969460aabe0/qt-report.json`。
- 最终 Chrome：`output/validation/m06-r6/browser-a4c77c75c7f9407abf5f0adef9f12db7/browser-report.json`，同目录 `browser-comparison.json` 与 `stdlib-readback.json`。
- 真实任务：`output/validation/m06-r6/worker-tests-570d28a220b44b5fbf4297b8e60e1a35/pytest.log`。
- 汇总与逐源文件摘要：`output/validation/m06-r6/completion.json`。
- 浏览器源码包位于 `output/validation/m06-r6/browser-runtime/`，SHA-256 为 `ba322efbbd9e0b58d944222e88b3791bbbcccd01eb4d0524ac373e83a034a689`。既有 Pyodide/原生/字体文件逐项验证后只读复用，不进入桌面 public 树。

```powershell
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/prepare_browser_science.py --sources-only --runtime-source output/validation/browser-compute-20261017/browser-science --output output/validation/m06-r6/browser-runtime --update-lock
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/verify_m06_r6.py --browser-only --science output/validation/browser-compute-20261017/browser-science --overlay output/validation/m06-r6/browser-runtime
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/verify_m06_r6.py --qt-only --science output/validation/browser-compute-20261017/browser-science --overlay output/validation/m06-r6/browser-runtime
```

重建源码包会产生新冻结摘要，须同步锁定清单并先执行 `npm --prefix frontend run build:workbench`，再验证其构建 Worker。

## 诊断与修正

首轮测试中的 Vite 开发态经典 Worker 含模块 import，改为实际构建 Worker；Qt 原生 dialog 没有显式 role 属性，检查改为 `dialog[open]`，未改产品来适配测试。网页独立容器补齐固定高度后消除 ResizeObserver 循环，最终控制台为空。

本机回环 HTTP 下，服务端发送 2423989 字节的 `python_stdlib.zip`，浏览器却读到零字节，读回 SHA 为标准空串摘要；因此 Python 启动报 encodings 缺失。隔离测试将 ZIP 请求路径映射为 `.ptbdata` 别名，读回恢复为 2423989 字节，SHA 与锁定文件完全一致。别名只改变测试资源传输，科学源码、原二进制和摘要检查保持，未改变用户浏览器、下载器或系统设置，亦未将其描述为生产环境修复。Qt 首轮收尾有子进程暂未退出而保留 scratch，拥有进程退出后由同一清理入口复核并删除，最终十份回执均已闭合。

## 平台与交付边界

不重建发行包，不 push、不公开部署。macOS Qt/Safari、实体音频设备、自然度听辨及最终 EXE/Mac 应用本轮未验。当前 Python 包的已有空切片警告不等同于本轮六项修复失败，仍保留诊断。
