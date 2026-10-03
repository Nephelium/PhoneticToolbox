# P19 外观来源与适配边界

查验日期：2026-10-03。

## 配色

参考井井提供的 Codex 外观截图，以及本机安装包版本 26.930.3930.0 中主题名称与模式元数据，只读核对 29 个名称。未把 Codex 的 JavaScript、CSS、图标或字体复制进本项目。原应用实现的许可不在此声称为开源。

本项目 `frontend/src/design/themes.ts` 是独立编写的工作台配色：包含 PhoneticToolbox 原主题以及 29 个同名适配方案，每个提供 light/dark。基本色参考相应主题的视觉习惯，边框、弱文字、按钮、状态颜色按工作台语义生成并校验对比度。Everforest 深色背景 #2D353B、前景 #D3C6AA、强调 #A7C080 与截图及安装包内对应主题基础色核对。不能声称其他每个颜色都与 Codex 逐项相同。

上游名称/模式核对清单：Absolutely、Ayu、Catppuccin、Codex、Dracula、Everforest、GitHub、Gruvbox、Linear、Lobster、Material、Matrix、Monokai、Night Owl、Nord、Notion、OG、Oscurange、One、Proof、Raycast、Rose Pine、Sentry、Solarized、Temple、Tokyo Night、Vercel、VS Code Plus、Xcode。原元数据中 Ayu/Dracula/Lobster/Material/Matrix/Monokai/Night Owl/Nord/OG/Oscurange/Sentry/Temple/Tokyo Night 仅深色、Proof 仅浅色；本项目为这些补齐另一模式。

关系：视觉参考与本项目独立适配。公共参考入口：[Codex](https://openai.com/codex/)。这些名称用于说明配色参考来源，不表示 OpenAI 或主题作者背书。科学轨道配色不随主题重新赋予意义。截图的相对路径仅用于本轮读取，不写入产品运行路径。

## JetBrains Mono

- 作者：The JetBrains Mono Project Authors。
- 实际资源：2.304 Regular，原文件未修改，只嵌入 Web 前端，无系统安装。
- [固定版本字体](https://github.com/JetBrains/JetBrainsMono/blob/v2.304/fonts/webfonts/JetBrainsMono-Regular.woff2)
- [固定版本 OFL 1.1](https://github.com/JetBrains/JetBrainsMono/blob/v2.304/OFL.txt)
- 文件 `frontend/src/assets/JetBrainsMono-Regular.woff2`，92,164 字节，SHA-256 `a9cb1cd82332b23a47e3a1239d25d13c86d16c4220695e34b243effa999f45f2`。
- 许可文件 `third_party/licenses/JetBrainsMono-OFL.txt` 与前端显示副本，SHA-256 `30f0c136e3c88e422d0791acd97238870f9054a9729bc34cf2ff0d4ed8cac4ad`。
- 关于页同时显示 Doulos SIL 与 JetBrains Mono 许可。
- 宋体、Times New Roman 使用设备已安装字体，不随应用分发商业字体文件。

## K2 图标

复用既有 `frontend/src/assets/k2.png`。显示时按 alpha >=32 的可见边界裁定占用，保留比例和边距，生成 16/24/32/48/64/128/256 尺寸图标。没有改画 logo 或覆盖原图片；Windows 窗口 QIcon 与后续打包 ICO 共用逻辑。已有冻结 EXE 保持原资源，不代表其图标自动更新。
