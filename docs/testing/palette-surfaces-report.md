# 浅色配色与显示模式预览

任务 `P04-LIGHT-PALETTE` 在 Windows 当前源码、独立 Chrome 和实际 Qt 的以下范围为 `verified`，本轮未构建 EXE。

## 问题与修复

公共浅色面板原来混入55%白色，使非白底配色的主区、侧区与嵌套卡片明显分裂。现仅轻微提亮底色，各29套浅色方案的面板与页面RGB每通道差≤3。导航与标签顶栏共用略灰的sidebar底色，模块贯穿侧栏共用微深的rail底色，内部卡片和中央区域保留轻亮panel。中性白底的rail沿用app，不强加层次。暗色新增rail沿用既有sidebar，不重新调整科学图像；文字/强调/状态色在包括rail的全部底色上检查对比度。默认 Everforest，已保存的其他配色保留。

审阅全部模块后，录音语谱周围、语谱频率轴标签、图片视口周围及比较图片留白改为公共主题令牌。声道旧页面的studio/analysis/charts/dialog、工具栏与控件同步宿主配色，修复仅深色收到完整覆盖的问题。科学语谱像素、图片重建白黑dB/笔迹、PDF纸面及器官颜色保留原意义，不重赋曲线颜色或改变算法。

设置外观增加跟随系统示意卡，以分色显示器表现自动深浅模式；三卡与三按钮使用同一列宽及间距，宽窄窗口上下对齐。系统模式仍根据系统外观响应变化。

## 实际检查

- `node --test frontend/tests/appearance.test.ts`：所有29套配色浅深、文字/状态/按钮对比度≥4.5、保存选择及轻微面板差异通过；纳入最终25项相关前端检查。
- `node tests/e2e/palette-surfaces.cjs`：18模块×Everforest/Solarized/Rose Pine/Catppuccin/Tokyo Night/Codex，共108实际入口页面，通过可见大块界面背景检查。科学图像与PDF纸面明确排除，未以这些入口空态证明结果/算法正确。
- 同一独立Chrome检查1440/800宽度下三预览/三按钮位置和宽度，以及真实媒体模式切换后的跟随系统深浅响应。
- 声道网页使用实际父页面主题消息，在Everforest/Rose Pine/Codex各浅深模式下检查面板、分析、图窗与弹窗背景。浏览器工作台的桌面能力提示与独立声道页面验收分开，未运行引擎或器官/动作计算。
- 当前实际Qt LPC、参数显示空态与四设置页捕获通过，LPC页面无大块写死白色背景。参数显示实测导航/顶栏rgb(244,238,224)、模块侧栏rgb(249,243,225)、内部卡片/中央rgb(253,246,228)，左右长栏与卡片有柔和层次。科研曲线、完整非空结果、外部PDF及设备模式不扩大本轮结论。
- typecheck、frontend build、资源登记及check_architecture通过。说明书外观说明和正式Qt截图已同步，严格校验无错误/警告。

证据 `output/validation/palette-surfaces/chrome-1791695539740/` 包含108入口、三预览/系统响应、声道主题桥报告及必要截图。Qt证据见 `output/validation/settings-navigation/manual-qt-92ce7f8df99c4fb099e2ced5e8a8b596/`，相关设置与默认图标见[设置报告](settings-navigation-report.md)。两类scratch最终均deleted。

分层检查曾把M01内部文件卡片误归长栏，改为按容器身份核对；中性Codex的rail与panel允许一致，改用实际共享rail令牌核对声道侧栏，完整108入口和声道浅深检查通过。失败报告保留。

EXE、其他平台、物理系统DPI/多屏和全部非空科研任务未重验。
