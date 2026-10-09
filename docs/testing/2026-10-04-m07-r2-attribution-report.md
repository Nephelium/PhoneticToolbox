# M07-R2 三栏底部来源与致谢

2026-10-04，状态：**verified，限定 Windows 源码工作台**。井井先要求规划，随后确认三栏最下方的位置，并明确授权修改界面。本轮完成主页底栏及对应来源登记。入口：[Start-M07-Workbench.ps1](../../scripts/Start-M07-Workbench.ps1)，旧 EXE 未更新。

## 实现

三栏工作区下方增加横跨总宽度的常驻来源与致谢栏，显示陆尧、梁昌维、孔江平及北京大学语言学实验室的署名致谢、2026-09-10 作者邮件许可、完整 Lu et al. (2025) 引用。论文、原始仓库、复制引用、改写说明入口直接可见。

底栏在三栏内部滚动区之外预留实际高度。桌面工作区使用剩余空间，窄窗三栏重排在独立滚动容器内，底栏保持可见。1920×1080、1440×900 普通字号下四图未被底栏遮挡。较窄布局和 150% Qt 缩放需要滚动查看工作区，不宣称四图在全部尺寸单屏可见。

底栏引用、链接、致谢与改写文案来自公共 `SRC-ZAIWA` / `REF-ZAIWA` 登记。复制引用复用既有剪贴板适配，外链复用 Qt 用户点击适配。改写说明窗口介绍原 MATLAB 实现、经许可 Python 改写、F0 后端及 v3 交互适配，并可进入完整方法与来源。

`SRC-ZAIWA` 状态由历史 review-required 更新为 author-permission-granted，依据井井提供的 2026-09-10 作者邮件回复。许可范围与约定致谢记录在[摘要](../../third_party/evidence/SRC-ZAIWA/permission-summary.md)，未存完整邮件、联系方式或签名图片。仓库未发现标准 LICENSE 的 2026-09-09 观测保留；论文、原录音、统计数据和第三方组件许可分别记录。其他模块 UI 未增加底栏。

计划：[主页方案](../plans/2026-10-04-m07-home-attribution.md)。来源审计：[M07 映射](../modules/evidence/M07-source-map.md)。说明书：[M07 手册](../manual/phonation-synthesis.md)。

## 实际验证

| 检查 | 结果与证据 |
| --- | --- |
| 类型、单元、构建 | `typecheck` / `build` 通过，**285 前端测试通过**；[类型](../../output/validation/m07-r2/typecheck.log)、[单元](../../output/validation/m07-r2/frontend-tests.log)、[构建](../../output/validation/m07-r2/build.log)。既有大 chunk 提示保留 |
| 来源单一登记 | `ui-data:check` 通过，358 个非 retired UI 条目。与本轮开始的两个快照比较，源登记和生成 UI 数据均**仅改变 SRC-ZAIWA / REF-ZAIWA 两条**，其他记录相同；[范围对照](../../output/validation/m07-r2/source-scope.json) |
| Chrome | **4 组通过，pageerror=0**：真实宿主一组三步结果、八种浅深尺寸 1920/1440/1280/960、21px 字号、底栏总宽度 / 可见性 / 滚动不动、完整引用复制与改写说明、其他模块无此底栏；[报告](../../output/validation/m07/host/1f0523689d784895b2394f2b1bea382a/r2-report.json) |
| 实际 Qt | **3 组 / 八种布局及 150% Qt 缩放通过**。QTest 原生复制、论文指针点击、仓库 Enter，QWebChannel 与系统外链适配每次只到达一个记录终点；引用原文与成功反馈核验。真实分析及三步生成、底栏滚动与四图几何通过；[报告](../../output/validation/m07/host/6f9de35da99e4f959bdebad4ec4dc56c/r2-qt-report.json) |
| M07 原交互回归 | **7 组通过**：六组默认整组、单步 / 整组 WebAudio 帧数、任务历史、刷新 / 重开、保存、九步及迟到 F0 归属；[报告](../../output/validation/m07/host/9f44bba3476d4fc8b7da678c8b9f3fba/r1-report.json) |
| 文档全库 | 1451 文件、362 来源、41 任务。**退出码 1，15 条缺链均为既有 EXE 成品路径**，本轮 M07 / 来源文档未报缺链；[日志](../../output/validation/m07-r2/docs.log)。新增报告的本地链接另做定向核验 |

截图：[Chrome 完成态](../../output/validation/m07/host/1f0523689d784895b2394f2b1bea382a/r2-1920-light.png)、[Qt 完成态](../../output/validation/m07/host/6f9de35da99e4f959bdebad4ec4dc56c/r2-qt-1920-light.png)、[Qt 150%](../../output/validation/m07/host/6f9de35da99e4f959bdebad4ec4dc56c/r2-qt-150percent.png)、[窄窗深色](../../output/validation/m07/host/1f0523689d784895b2394f2b1bea382a/r2-960-dark.png)。已目视核实底栏、引用、图窗与正常按钮渲染。

## 命令及边界

```powershell
npm --prefix frontend run ui-data
npm --prefix frontend run ui-data:check
npm --prefix frontend run typecheck
npm --prefix frontend run test
npm --prefix frontend run build
node tests/e2e/m07-attribution.cjs
node tests/e2e/m07-r1.cjs
$env:PYTHONPATH='D:/PhoneticToolbox/PhoneticToolbox_v3/backend/src;D:/PhoneticToolbox/PhoneticToolbox_v3/desktop/src;D:/PhoneticToolbox/PhoneticToolbox_v3/packages/phonetic_core/src;D:/PhoneticToolbox/PhoneticToolbox_v3/scripts'
.venv/m09-ui/Scripts/python.exe -B -X utf8 scripts/verify_m07_attribution_qt.py
.venv/m09-ui/Scripts/python.exe -B -X utf8 scripts/validate_docs.py
```

- 首次 Chrome 定位把来源 tab 当作 button，Qt 首次使用不存在的弹窗 class，按真实角色 / 元素修正测试后重跑。首个 Qt 记录剪贴板未实现宿主要求的 text 回读，因此虽记录写入，UI 仍显示失败；补全记录终点并同时核验成功反馈后通过，未修改产品复制逻辑。
- 曾尝试 CSS body zoom 模拟缩放，100vh 被整体放大，产生超出视口与 ResizeObserver 通知错误，此模型不能作为浏览器缩放验收。Chrome 改用公共字号变量 21px，Qt 独立使用真实 `setZoomFactor(1.5)`，最终均通过。未减少底栏可见性和不遮挡几何要求。
- 原生测试使用隐藏 windows 平台窗口及测试进程软件渲染标志，日志中的 libpng / GLES 回退诊断保留。产品渲染设置未改，物理 DPI / GPU / DWM、实体音频和 Linux GUI 未验。
- 剪贴板与系统浏览器末端换为记录适配器，既有 QWebChannel / 用户点击通道实际运行；未改用户剪贴板，未弹系统浏览器。测试音频静音，未做实体声卡听辨。
- 原科学核心、worker、接口契约、数据库 schema 无本轮修改。隔离任务库复制既有 schema，`schema_applied=[]`。无本轮 EXE、push、公开发布、依赖安装或用户文件清理，保留同期差异。
- 页面本轮对照为 [page-diff.patch](../../output/validation/m07-r2/page-diff.patch)，修改前页面及两份登记快照保留在验证目录。
