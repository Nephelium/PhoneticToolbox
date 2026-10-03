# P19 外观设置、配色与任务栏图标

状态：verified，限定 Windows 开发态，见[验证报告](../testing/2026-10-03-p19-appearance-report.md)。井井在本轮明确要求配色选择与深浅联动、紧凑左右两栏设置、内置代码字体，以及放大任务栏图标。

## 范围与设计决定

后续外观 R1：井井要求修复内置字体无法选择、移除 PhoneticToolbox 配色、默认 Everforest 并推荐尝试更多配色。限定改动 FontSettings、AppearanceSettings、themes 与 AppShell 默认值，保留已有用户字体和其他配色。按用户截图验证 Georgia → JetBrains Mono、自定义与取消、刷新持久化，旧 `ptb` → Everforest 和其他主题保留。初轮历史范围如下，R1 结果见[续修报告](../testing/2026-10-03-p19-appearance-r1-report.md)。

- 保持 U2 工作台结构。设置内容限宽，左栏外观、模式、缩放，右栏字体与紧凑预览；窄窗自动单栏。保留字体草稿、应用与关闭保护。
- 配色方案与 light/dark/system 分开保存。保留原 PhoneticToolbox 配色及现有模式偏好，增加 Codex 当前 29 主题的工作台适配版。所有方案提供两种模式，原主题缺失的模式由本项目补齐，不声称逐像素复制。
- 配色仅影响界面语义令牌。科学轨道蓝/橙等含义、图像数据、实验刺激颜色与计算保持原行为。M10 同源页面转发同一界面令牌。
- 内置 JetBrains Mono 2.304 Regular 与 OFL，不安装系统字体。旧显式字体选择保留，空代码字体使用内置字体。IPA 固定 Doulos SIL。
- 桌面图标由原 K2 资源按可见边界归一化占用，保持比例与防裁切边距，支持 16/24/32/48/64/128/256。原设计图保留。源码窗口和后续打包入口使用同一生成逻辑。本轮未获 EXE 打包请求。

## 文件与验收

AppShell.vue、FontSettings.vue、AppearanceSettings.vue、design/themes.ts、state/appearance.ts、字体资源与加载、M10 主题桥、desktop/app_icon.py、后续打包的图标参数，来源登记和手册。

运行 `npm --prefix frontend run typecheck`、`npm --prefix frontend test`、`npm --prefix frontend run build`，新增主题对比度/非法偏好回退测试、独立 Chrome 主题/布局/持久化/字体/草稿回归及 Qt 离屏离线验收。检查运行时图标可见边界与 16–256 各尺寸。Windows verified 与 WSL 静态检查、实体任务栏/DPI 未测分开记录。

不修改数据库、科学算法、系统环境、CI/CD，不 push、不生成或覆盖旧 EXE。保留本轮开始时已有的 M05/P18/M17 等差异。
