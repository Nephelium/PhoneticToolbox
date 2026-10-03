# 2026-10-03 桌面 R2：原生复制、窗口调查与 M16 完成状态

状态：原生复制已完成限定 Windows 开发态验证，最终 R2 静态构建的实际 Qt 按钮、原生系统剪贴板回读与恢复全部通过。最大化问题已完成限定隐藏窗口调查，未复现用户描述的整页黑屏数秒，不标为已修复。冻结 EXE 验证由本轮成品报告另记。

## 范围和实现

- `desktop/src/ptb_desktop/host.py`：增加 `Bridge.writeClipboard(text)`，QWebChannel 在 Qt GUI 线程调用，直接写入 `QApplication.clipboard()` 并核对原文。只返回成功或错误，不向页面提供剪贴板读取。超过 2,000,000 个 Python 字符明确拒绝，原剪贴板保留。
- `frontend/src/platform/clipboard.ts` 与 `desktop.ts`：桌面握手后安装原生写入能力，`copyText(text)` 优先使用该能力，浏览器保留 Clipboard API。桌面错误与 5 秒无回应明确向页面传播。M17 的按钮接线由本轮 UI 修改完成，失败时保留全选兜底。
- 保留既有媒体权限门禁、WebGL、音视频与 GPU 配置。未变更科研计算、数据库、环境、系统设置或 CI。

## 复制失败的直接证据

旧构建在真实 Windows QPA 后端、隐藏的自有 Qt 窗口中，使用 `QTest.mouseClick` 向复制全部按钮发送 Qt 输入事件。`navigator.clipboard.writeText` 存在，但实际 `ClipboardReadWrite` 权限请求被当前只允许媒体类型的 `MediaPermission.requested` 以 `denied_gate` 拒绝，剪贴板逐字比较失败。

证据：`output/validation/desktop-r2/qt-20261003-155500-f2b98e/report.json`。

Qt 的 Clipboard API 权限与设置见 [QWebEngineSettings 官方说明](https://doc.qt.io/qt-6/qwebenginesettings.html)。本次以独立纯文本写入能力解决，不打开页面读取剪贴板的能力，也不修改录音/摄像头权限处理。

最终 R2 构建使用相同 `QTest.mouseClick` 输入路径通过：中文、清化圈、NFD `e+U+0301` 与 NFC `é`、非 BMP 音标/修饰字母、连音线、换行和制表符逐字一致，页面显示已复制全部文字。空串和 23,000 字符长文本写入/回读通过，2,000,001 字符请求明确拒绝且保留前一剪贴板。页面仍保持 `JavascriptCanAccessClipboard=false` 和 `JavascriptCanPaste=false`，本次复制未触发浏览器权限事件。测试结束已逐 MIME 格式恢复原剪贴板，报告 `clipboard_restored=true`。

最终证据：`output/validation/desktop-r2/qt-20261003-160238-734463/report.json`，同目录末轮截图可见已复制提示。

## 最大化调查

读取宿主代码确认：没有自定义 `resizeEvent` 或窗口状态回调反复调整尺寸，没有生产用 `--disable-gpu`；页面背景实测是 alpha 255 的 `#ffffff`。`fit_screen` 只处理启动和屏幕变更，最大化时不调用其中的普通窗口 resize。背景色匹配主要影响页面加载底色，不能据此证明能解决 GPU/桌面合成停顿，见 [QWebEnginePage 背景与渲染进程信号](https://doc.qt.io/qt-6/qwebenginepage.html)。

Windows Qt 6.11.2 原生 QPA，`WA_DontShowOnScreen` 隐藏，Chromium 仅 `--mute-audio`，WebGL 开启。执行 20 轮最大化/还原、40 次状态及客户区尺寸变化，200 次目标时点 0/16/50/150/500 ms 的图像与 JS 采样。隐藏窗口不由窗口管理器自动设置最大化客户区，因此测试显式调整为屏幕可用尺寸。实际 DOM 最大尺寸 1707×1019，普通尺寸覆盖 1366–1566×768–848。

- 未出现整页全黑、renderer 退出/PID 变化或 WebGL context lost，DOM 和 rAF 持续响应。JS 回应平均约 2.04 ms，最大 16 ms。
- 最早采样确实出现旧尺寸画面外侧的新增区域暂时为黑色，最高约 37.7%。这与整页黑屏数秒不等价，不能报告为没有任何黑色过渡帧。
- 150 ms 目标采样与 500 ms 目标采样均无黑色像素。截图本身有开销，实际采样时间完整保存；150 ms 组最晚为 328 ms，500 ms 组最晚为 516 ms，不能将目标时点当作精确呈现时延。
- 已人工查看首轮 0 ms 与 500 ms PNG，分别确认右/下黑边和完整重绘。默认深色页面本身的蓝灰背景不被当作纯黑。

证据：`output/validation/desktop-r2/qt-20261003-155105-75ec17/report.json` 及同目录 PNG。

最终 R2 静态构建再次完成同样 20 轮/200 次采样，结论一致。改用高精度 `perf_counter` 计时，JS 响应平均约 3.28 ms、最大 29.60 ms；早期新增黑边最高仍为 37.7%，150/500 ms 目标组均无黑色像素，实际最晚采样分别约 374/519 ms。renderer PID 不变，无终止事件，WebGL context 保持有效。已人工查看最终末轮完整重绘图，未将隐藏窗口结果扩大为实体呈现验证。

另一个保留 GPU 的 offscreen QPA 试验触发 D3D shared-image/context-lost 错误，截图仅有单色，记录在 `output/validation/desktop-r2/qt-20261003-154820-d97ac0`。同机原生 Windows 隐藏 QPA未出现该错误，故将其列为测试后端差异，不以此更改正式程序 GPU 配置。

仍待：实体标题栏双击、窗口曝光/DWM 合成、多屏/不同 DPI、用户实际显卡负载下的偶发数秒黑屏。这些检查未用隐藏测试替代。本次没有背景遮罩、强制刷新页面、自动重启 renderer 或全局禁用 GPU。

## 已执行验证

```powershell
node --test frontend/tests/clipboard.test.ts
$env:PYTHONPATH='D:\PhoneticToolbox\PhoneticToolbox_v3\desktop\src;D:\PhoneticToolbox\PhoneticToolbox_v3\packages\phonetic_core\src;D:\PhoneticToolbox\PhoneticToolbox_v3\backend\src'
& '.venv\m09-ui\Scripts\python.exe' -m pytest desktop/tests -q -o addopts=''
& '.venv\m09-ui\Scripts\python.exe' -m pytest desktop/tests/test_clipboard.py -q -o addopts=''
& '.venv\m09-ui\Scripts\python.exe' -X utf8 scripts/verify_desktop_r2_qt.py --native-hidden --baseline-copy --cycles 20
& '.venv\m09-ui\Scripts\python.exe' -X utf8 scripts/verify_desktop_r2_qt.py --native-hidden --baseline-copy --cycles 0
& '.venv\m09-ui\Scripts\python.exe' -X utf8 scripts/verify_desktop_r2_qt.py --native-hidden --cycles 20
```

前端 helper 两项通过，覆盖 Unicode 原文、浏览器路径、原生错误传播及不支持剪贴板的浏览器。原桌面 95 项通过，2 项 POSIX 专属测试在 Windows 跳过；随后新增的剪贴板边界 4 项单独通过，覆盖 Unicode、超长拒绝、系统未接收写入和剪贴板不可用，检查错误返回不包含旧剪贴板内容。首次未覆盖 `addopts` 的 pytest 命令因当前环境未安装 pytest-cov、旧顶层配置含 `--cov=phonetic_toolbox` 而在收集前退出；按本项目独立环境测试方式清空这一旧覆盖率参数后执行全部桌面用例，没有安装插件或跳过失败测试。

所有探针只创建自己的服务、窗口、配置和输出目录。剪贴板探针不记录原始内容，测试后在剪贴板仍为本次写入内容时恢复原 MIME 格式；若用户期间复制了新内容，不覆盖该内容。未运行实体设备。

## P19 候选包发现的 M16 提前完成竞态

成品检查在 `output/validation/v3-preview-exe-d78e2e63871d49b6a83fa17065719189/results/m16-m17` 发现：降噪子进程已经写出 `process.json: complete`，但进程仍在退出收尾。旧 `poll_job()` 直接返回这个终态，页面显示后台处理已结束，此时宿主尚未执行 `append_version`，当前版本仍为 `restored_raw`。冻结验证的 `head.kind == denoised` 断言正确发现问题，该断言完整保留，未加等待绕过。

本轮仅调整 `desktop/src/ptb_desktop/recording/service.py:poll_job`：

1. 先观察拥有的进程/线程是否结束，再读取结果文件，避免先读到旧 running、随后观察到退出而误报失败。
2. owner 仍运行时，结果文件中的 complete/failed/cancelled 不作为对外终态。返回 running 或 cancelling，保留 `self.job` 和原生 busy 限制。
3. owner 结束后仍走既有校验与项目提交。提交成功才能返回 complete 并释放任务；提交失败抛出错误，任务引用和原版本保留，可重试提交。

音频处理算法、输出数据、UI 与冻结断言未改变。导出线程和失败/取消状态也受同一生命周期限制。

`desktop/tests/test_m16_job_publication.py` 新增 9 个确定性用例，不用 sleep 或真实设备：增益/降噪完成两阶段、导出完成/失败、处理失败/取消、结果写完后取消、进程退出瞬间发布结果，以及项目提交失败后的重试。

旧方法红测试在独立 pytest 进程中从 `git show HEAD:desktop/src/ptb_desktop/recording/service.py` 提取旧 `poll_job` 并临时替换内存方法，未改工作区源码，结果 **8 failed、1 passed**；失败全部对应上述提前终态或陈旧结果读取。证据 `output/validation/m16-publication-r2/old-method-red.log` 与同目录插件。

新实现定向验证：

```powershell
& '.venv\m09-ui\Scripts\python.exe' -m pytest desktop/tests/test_m16_job_publication.py desktop/tests/test_m16_recording.py desktop/tests/test_m16_host_integration.py -q -o addopts=''
```

结果 **37 passed in 3.30s**。首次 36 项通过后仅新增了第 9 个提交失败用例，再执行三组得到 37 项通过。实际完整源码集成及最终重打包的冻结验证由统筹继续执行，候选包不能据此充当最终已验收包。
