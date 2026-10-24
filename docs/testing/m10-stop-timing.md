# 生理参数合成：爆破时序与正负 VOT 验收

2026-10-20追加：本页共享实现已同步公开网页，相关操作在实际HTTPS Chrome155重新验证，Linux声道回退为R21。当前网页部署以[同步验收](2026-10-20-web-module-sync.md)为准。以下保留原本机/各平台验收范围，最终Windows/macOS成品与实体设备边界不扩大。

2026-10-19。范围：三端共用源码，Windows 原生/Qt与本机独立 Chrome 真实 Wasm。决策见 [ADR](../decisions/ADR-M10-stop-timing.md)。

## 修复与审阅结论

- 负 VOT 之前使用开放、不振动的声门，闭塞腔提前充压，随后开启振动时压差不足，原生输出的前浊极弱。现在先内收声门，再于指定时刻用 2 ms 过渡恢复目标声源。闭塞构型保持，前浊来自原生求解及皮肤辐射。
- 浊音起始恰在帧末时，旧采样直接取下一帧的全振动声源，原生块插值可提前起振。现在将 2 ms 过渡连续延至普通后续帧，并在时序帧起止切块、取左极限终点、零长度更新边界声源。
- 新编辑共用一套时序：送气=max(0,VOT−释放)，总长=闭塞+释放+送气。改送气同步 VOT，改释放保留 VOT；负 VOT 的前浊计入闭塞。非法提前量报错并保留已保存值。最短时序帧为 11 ms，普通帧仍为 50 ms。
- 删除添加送气帧入口，旧送气帧仍可读。启用/关闭时序保留原声源。旧时序文件保留四项数值及总时长，提示按新方法重算；新文件标记 stop-timing/2，未知方法拒绝。
- 修复编辑一次后时序面板意外收起：保留展开状态，忽略已脱离页面的 toggle 事件。时序说明和编辑说明默认折叠，采用浊音起始术语；嵌套说明的加减号按自身状态显示。

VOT 是输入的控制时间差，声学 VOT 应由实际输出波形测定。这里的检查覆盖控制时刻、原生 PCM 周期性与总采样数，不将指令时刻等同于测得的声学起点。

## 实际验证

- Windows 定向测试：工作台源码绑定后运行 parity/backend/desktop 的 test_vocal_*、test_m10_*，169 项通过，110.49 秒，保留两条既有 Starlette/AnyIO 弃用提示。新增双模型 10 项覆盖负 VOT 原生前浊、帧末正 VOT、11 ms 边界、帧开始切块、旧文件与构型包兼容。
- 前端：`node --test tests/vocal-stop-timing.test.ts tests/vocal-profile.test.ts tests/vocal-web.test.ts tests/vocal-wasm.test.ts tests/m10-model-pinch.test.ts`，22 项通过。类型、契约、UI 数据及 `npm run build:workbench` 通过，既有 Vite 大包提示保留。
- Windows Qt：`scripts/verify_m10_shared_qt.py`，5 组通过。实际 IPC/原生 PCM、文件读写、模型与套装恢复、正负 VOT 和折叠状态检查。最终报告：`output/validation/m10-shared-20261018/qt-1e3d56bfaa53421f9d86b05987f4b8de/report.json`。测试模式未打开实体音频设备。
- Chrome：`scripts/verify_m10_wasm.py --stop-timing`，5 组通过。断言执行端为 browser-wasm，检查连续参数编辑、前浊 PCM、导出/旧文件验证、真实浏览器播放完成、折叠与展开以及加号状态。最终报告：`output/validation/m10-wasm-20261018/browser-3bea37c486cf41fcaf6a1f5c9b08be39/stop-timing.json`。浅深主题与 Qt 最终截图已复核。

实际 Chrome JD2，闭塞 150 ms、释放 8 ms、VOT −100 ms、后接 200 ms 元音：总长 358 ms / 17184 样本。75–140 ms 的前浊相关系数 0.9997801，RMS 0.0108572，正 VOT 对照为 0.00024768。VOT 改 +50 ms 后送气 42 ms，总长 400 ms / 19200 样本。再改送气 70 ms、释放 20 ms，VOT 保持 78 ms，总长 428 ms / 20544 样本。VOT 改 0 或 −40 ms，总长均 370 ms / 17760 样本。所有长度按 48 kHz 检验。

复现命令（PowerShell，工程根目录）：

```powershell
.venv/m14/Scripts/python.exe -E -s -B -X utf8 scripts/verify_m10_shared_qt.py
.venv/m14/Scripts/python.exe -E -s -B -X utf8 scripts/verify_m10_wasm.py --work output/validation/m10-smoothing-20261018 --science output/validation/browser-compute-20261017/browser-science --frontend frontend/dist --vocal-runtime output/validation/m10-smoothing-20261018/wasm/vocal-wasm --stop-timing --port 18786
```

资源登记 72 项幂等，架构检查零错误，文档检查 2338 文件 / 447 来源 / 168 任务零错误，未修改历史快照的旧断链单列。定向 diff 检查通过。

## 失败修复与范围

首轮完整测试发现最短帧错误文案回归，恢复按实际最小值显示后全套通过。新前端测试最初直接比较二进制浮点时长，改为核对整数微秒。首次实际浏览器连续编辑暴露面板意外折叠，修复事件与重绘状态后复验通过。最终图像检查发现内层说明继承父层减号，已按自身展开状态修正并增加真实页面断言。没有删断言、放宽科研容差或增加跳过标记。

最终 Qt/Chrome 的自有 scratch 清理回执为 deleted，保留日志、报告与截图。未触碰用户已有构型和当前工作台进程。Mac 本轮仅同步共用源码，R21 原生构建待完成。本轮未重打最终 EXE/Mac 应用，未公开部署，未新增 Safari 或实体音频设备验证。
