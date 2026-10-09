# P19-R11：29 套配色的上游与许可核查

查验日期：2026-10-05。当前名称、色值和偏好保持现状。

本次核查对应当前源码中的 29 个 palette ID。11 套取得公开上游的完整 MIT 许可原文；Gruvbox 取得 MIT/X11 声明，Monokai 取得 Microsoft 公开实现的 MIT，但两者各保留一项许可缺口；其余 16 套没有取得可可靠对应的主题再分发许可。

公开上游的本次查验 commit 与 v3 最初引入版本分列。历史引入 commit 全部 unknown。本次核对支持配色家族及适配署名，不能证明 Codex 本身采用哪一上游版本，也不授予品牌使用权或把未决许可改成已授权。

机器可读原始证据见[逐项审计](../../third_party/evidence/p19-palettes/audit.json)，应用致谢从[统一登记](../../third_party/source-registry.json)生成。

## 完整清单

| 配色 | 公开上游或参考入口 | 查验结果 | PTB 适配与未决项 |
| --- | --- | --- | --- |
| Absolutely | [Codex 历史视觉参考](https://openai.com/codex/) | 来源或许可待确认 | 仅能确认 Codex 同名内置方案；未取得可对应的原作者声明或主题许可。社区移植的许可证不能替代原方案授权。 |
| Ayu | [ayu-theme/ayu-colors](https://raw.githubusercontent.com/ayu-theme/ayu-colors/b0fd979a1ddf050101b43311fa598a1a9c5f1bbc/themes/dark.yaml) | 公开上游 MIT 原文已核对 | 最初引入版本未记录；当前公开上游查验不能倒推历史复制版本，也不证明与 Codex 全部颜色一致。 |
| Catppuccin | [catppuccin/palette](https://raw.githubusercontent.com/catppuccin/palette/07d02aa110ef9eb7e7427afca5c73ba9cf7f8ebd/palette.json) | 公开上游 MIT 原文已核对 | 最初引入版本未记录；当前公开上游查验不能倒推历史复制版本，也不证明与 Codex 全部颜色一致。 |
| Codex | [品牌视觉参考](https://openai.com/codex/) | 来源或许可待确认 | OpenAI 官方应用及品牌指南可查；未找到桌面 Codex 此配色的独立开源许可。Codex CLI 的开源许可不外推到桌面应用主题。 |
| Dracula | [dracula/visual-studio-code](https://raw.githubusercontent.com/dracula/visual-studio-code/a08a206f2c8420ba3c05f0e8d01d43b2f933fdf8/src/dracula.yml) | 公开上游 MIT 原文已核对 | 最初引入版本未记录；当前公开上游查验不能倒推历史复制版本，也不证明与 Codex 全部颜色一致。 |
| Everforest | [sainnhe/everforest](https://raw.githubusercontent.com/sainnhe/everforest/85a86eb62409e3ec88713bff3d1b9d7374e112e4/palette.md) | 公开上游 MIT 原文已核对 | 最初引入版本未记录；当前公开上游查验不能倒推历史复制版本，也不证明与 Codex 全部颜色一致。 |
| GitHub | [primer/github-vscode-theme](https://raw.githubusercontent.com/primer/github-vscode-theme/cd78e5e4e7bcf132a6f428ae0f32264bb1b729cf/src/classic/colors.json) | 公开上游 MIT 原文已核对 | 最初引入版本未记录；当前公开上游查验不能倒推历史复制版本，也不证明与 Codex 全部颜色一致。 |
| Gruvbox | [morhetz/gruvbox](https://raw.githubusercontent.com/morhetz/gruvbox/ef8864bb42bf244f0295d1c5a403b27e3d139695/colors/gruvbox.vim) | MIT/X11 声明已核对，完整通知待补 | 上游 README 声明 MIT/X11，已保存声明原文；未取得独立完整版权及许可通知，未伪造作者版权年份或完整许可文件。历史引入版本未记录。 |
| Linear | [品牌视觉参考](https://linear.app/brand) | 来源或许可待确认 | 仅以 Linear 官方品牌页记录视觉参考；没有证据把 Codex 的同名适配绑定到 Linear 或某个社区主题仓库的许可。 |
| Lobster | [Codex 历史视觉参考](https://openai.com/codex/) | 来源或许可待确认 | 仅能确认 Codex 同名内置方案；未取得相应主题作者及许可。与宠物或其他同名项目无已证实来源关系。 |
| Material | [历史项目入口](https://github.com/material-theme/vsc-material-theme) / [当前后继项目](https://github.com/vira-soft/vira-assets) | 来源或许可待确认 | Mattia Astorino / Equinusocio 的旧 Material Theme 仓库现重定向至 vira-soft/vira-assets，当前仅有 live 分支、无 tags，未取得许可证。第三方保存的历史 Apache-2.0 标签未作为本项授权证据。Google Material Design 同名资源的许可也不能套用。 |
| Matrix | [Codex 历史视觉参考](https://openai.com/codex/) | 来源或许可待确认 | 仅能确认 Codex 同名内置方案；绿色终端视觉和同名社区主题不能证明本项的具体来源或许可。 |
| Monokai | [microsoft/vscode](https://raw.githubusercontent.com/microsoft/vscode/40bb6fae207507042de8b6d7b74e9aa374a212a4/extensions/theme-monokai/themes/monokai-color-theme.json) | 公开实现 MIT 已核对，原版许可链待确认 | 经典 Monokai 的三项深色 seed 与 VS Code 公开实现一致，Microsoft 仓库 MIT 原文已保存；原设计及 Codex 此方案的具体许可链未闭合。Monokai Pro 有另外的社区移植条件，未套用其许可或将本项称为 Pro。 |
| Night Owl | [sdras/night-owl-vscode-theme](https://raw.githubusercontent.com/sdras/night-owl-vscode-theme/cc291eba7976b20d7c66bde6883c27b902196b07/themes/Night%20Owl-color-theme.json) | 公开上游 MIT 原文已核对 | 最初引入版本未记录；当前公开上游查验不能倒推历史复制版本，也不证明与 Codex 全部颜色一致。 |
| Nord | [nordtheme/nord](https://raw.githubusercontent.com/nordtheme/nord/1cef71605416a222e57225b544540ce0fcec18d4/src/nord.css) | 公开上游 MIT 原文已核对 | 最初引入版本未记录；当前公开上游查验不能倒推历史复制版本，也不证明与 Codex 全部颜色一致。 |
| Notion | [品牌视觉参考](https://www.notion.com/) | 来源或许可待确认 | 仅记录 Notion 官方产品视觉参考；公开产品页面没有提供可对应此配色的再分发许可。 |
| OG | [Codex 历史视觉参考](https://openai.com/codex/) | 来源或许可待确认 | 仅能确认 Codex 内置 OG 名称；未取得可对应的独立上游及主题许可。 |
| Oscurange | [Codex 历史视觉参考](https://openai.com/codex/) | 来源或许可待确认 | 仅能确认 Codex 同名内置方案；公开检索未取得可对应的作者仓库及主题许可。 |
| One | [atom/one-dark-syntax](https://raw.githubusercontent.com/atom/one-dark-syntax/9c96f4454362267ac45322063e193ccf9d2debb1/styles/colors.less) / [atom/one-light-syntax](https://raw.githubusercontent.com/atom/one-light-syntax/d84579027410c576086dfca14d934c4bd74b0438/styles/colors.less) | 公开上游 MIT 原文已核对 | 最初引入版本未记录；当前公开上游查验不能倒推历史复制版本，也不证明与 Codex 全部颜色一致。 |
| Proof | [Codex 历史视觉参考](https://openai.com/codex/) | 来源或许可待确认 | 仅能确认 Codex 的浅色 Proof 方案；未取得可对应的原作者及许可。PTB 深色是自行补齐。 |
| Raycast | [品牌视觉参考](https://www.raycast.com/pro) | 来源或许可待确认 | 仅记录 Raycast 官方外观/产品参考；未找到可对应本项的官方主题开源仓库。Raycast 扩展仓库的 MIT 不外推到应用界面配色。 |
| Rose Pine | [rose-pine/rose-pine-palette](https://raw.githubusercontent.com/rose-pine/rose-pine-palette/92af52b465ab6e47437aca223c9b8d3009a2023b/palette.json) | 公开上游 MIT 原文已核对 | 最初引入版本未记录；当前公开上游查验不能倒推历史复制版本，也不证明与 Codex 全部颜色一致。 |
| Sentry | [品牌视觉参考](https://sentry.io/) | 来源或许可待确认 | 仅记录 Sentry 官方产品视觉参考；Sentry 代码库或 SDK 的许可未被用作 Codex 同名主题的授权。 |
| Solarized | [altercation/solarized](https://raw.githubusercontent.com/altercation/solarized/62f656a02f93c5190a8753159e34b385588d5ff3/README.md) | 公开上游 MIT 原文已核对 | 最初引入版本未记录；当前公开上游查验不能倒推历史复制版本，也不证明与 Codex 全部颜色一致。 |
| Temple | [Codex 历史视觉参考](https://openai.com/codex/) | 来源或许可待确认 | 仅能确认 Codex 同名内置方案；未取得可对应的独立作者仓库及主题许可。 |
| Tokyo Night | [tokyo-night/tokyo-night-vscode-theme](https://raw.githubusercontent.com/tokyo-night/tokyo-night-vscode-theme/7c0f11eaef322f293621ca7befe462214b7ea468/themes/tokyo-night-color-theme.json) | 公开上游 MIT 原文已核对 | 最初引入版本未记录；当前公开上游查验不能倒推历史复制版本，也不证明与 Codex 全部颜色一致。 |
| Vercel | [品牌视觉参考](https://vercel.com/geist/introduction) | 来源或许可待确认 | 仅记录 Vercel Geist 官方设计参考；Geist 字体等资源的开源许可不外推到 Codex 同名配色。 |
| VS Code Plus | [主题名称记录](https://github.com/openai/codex/issues/15987) | 来源或许可待确认 | 可在 openai/codex 问题记录中确认 VS Code Plus 桌面主题名称。尚无证据证明它等同于 Microsoft 的 Dark+ / Light+，因此未套用 VS Code 仓库 MIT。 |
| Xcode | [品牌视觉参考](https://developer.apple.com/xcode/) | 来源或许可待确认 | 仅记录 Apple Xcode 官方产品视觉参考；未取得可对应此配色的独立再分发许可，未引入 Apple 图标或界面资源。 |

## 作者、固定版本及许可原文

下面的原文按本次公开查验版本保存。完整 MIT 原文用于保留版权及许可通知，不据此把整套软件改标为 MIT。

### Ayu

作者/版权主体：Konstantin Pschera / ayu-theme contributors。

- 本次查验 commit：`b0fd979a1ddf050101b43311fa598a1a9c5f1bbc`；[固定上游文件](https://raw.githubusercontent.com/ayu-theme/ayu-colors/b0fd979a1ddf050101b43311fa598a1a9c5f1bbc/themes/dark.yaml)。
- [上游许可原文](https://raw.githubusercontent.com/ayu-theme/ayu-colors/b0fd979a1ddf050101b43311fa598a1a9c5f1bbc/license) · [本地原文](../../third_party/licenses/p19-palettes/ayu-MIT.txt)；SHA-256：`0e516038e9348d278dd886a2f07826b78e851f6e5f002c62096529d508c55be1`。

### Catppuccin

作者/版权主体：Catppuccin contributors。

- 本次查验 commit：`07d02aa110ef9eb7e7427afca5c73ba9cf7f8ebd`；[固定上游文件](https://raw.githubusercontent.com/catppuccin/palette/07d02aa110ef9eb7e7427afca5c73ba9cf7f8ebd/palette.json)。
- [上游许可原文](https://raw.githubusercontent.com/catppuccin/palette/07d02aa110ef9eb7e7427afca5c73ba9cf7f8ebd/LICENSE) · [本地原文](../../third_party/licenses/p19-palettes/catppuccin-MIT.txt)；SHA-256：`814096d2c34cc216c624738a49356f32b7237733b4f7edb0685f4e50ef5074ba`。

### Dracula

作者/版权主体：Zeno Rocha / Dracula Theme contributors。

- 本次查验 commit：`a08a206f2c8420ba3c05f0e8d01d43b2f933fdf8`；[固定上游文件](https://raw.githubusercontent.com/dracula/visual-studio-code/a08a206f2c8420ba3c05f0e8d01d43b2f933fdf8/src/dracula.yml)。
- [上游许可原文](https://raw.githubusercontent.com/dracula/visual-studio-code/a08a206f2c8420ba3c05f0e8d01d43b2f933fdf8/LICENSE) · [本地原文](../../third_party/licenses/p19-palettes/dracula-MIT.txt)；SHA-256：`f08d0bba7f8906ed4e7a76d7e555a883968d83eb550ae298ce893a65aee15dff`。

### Everforest

作者/版权主体：sainnhe。

- 本次查验 commit：`85a86eb62409e3ec88713bff3d1b9d7374e112e4`；[固定上游文件](https://raw.githubusercontent.com/sainnhe/everforest/85a86eb62409e3ec88713bff3d1b9d7374e112e4/palette.md)。
- [上游许可原文](https://raw.githubusercontent.com/sainnhe/everforest/85a86eb62409e3ec88713bff3d1b9d7374e112e4/LICENSE) · [本地原文](../../third_party/licenses/p19-palettes/everforest-MIT.txt)；SHA-256：`6504b9e62794cb58b61c90ea5fe598274086119f2126b8d8245708b2bfcc0e36`。

### GitHub

作者/版权主体：Primer / GitHub。

- 本次查验 commit：`cd78e5e4e7bcf132a6f428ae0f32264bb1b729cf`；[固定上游文件](https://raw.githubusercontent.com/primer/github-vscode-theme/cd78e5e4e7bcf132a6f428ae0f32264bb1b729cf/src/classic/colors.json)。
- [上游许可原文](https://raw.githubusercontent.com/primer/github-vscode-theme/cd78e5e4e7bcf132a6f428ae0f32264bb1b729cf/LICENSE) · [本地原文](../../third_party/licenses/p19-palettes/github-MIT.txt)；SHA-256：`e2c871bb6effe9f6ec8d94d6596db33c4bfe4dd94969a6a9a2fb0473ebe6442e`。

### Gruvbox

作者/版权主体：morhetz。

- 本次查验 commit：`ef8864bb42bf244f0295d1c5a403b27e3d139695`；[固定上游文件](https://raw.githubusercontent.com/morhetz/gruvbox/ef8864bb42bf244f0295d1c5a403b27e3d139695/colors/gruvbox.vim)。
- [上游 MIT/X11 声明](https://raw.githubusercontent.com/morhetz/gruvbox/ef8864bb42bf244f0295d1c5a403b27e3d139695/README.md) · [已保存声明](../../third_party/evidence/p19-palettes/gruvbox-license-declaration.txt)。此文件为 README 声明片段，不冒称独立完整许可证。

### Monokai

作者/版权主体：Wimer Hazenberg（原设计）；Microsoft Corporation（公开实现）。

- 本次查验 commit：`40bb6fae207507042de8b6d7b74e9aa374a212a4`；[固定上游文件](https://raw.githubusercontent.com/microsoft/vscode/40bb6fae207507042de8b6d7b74e9aa374a212a4/extensions/theme-monokai/themes/monokai-color-theme.json)。
- [上游许可原文](https://raw.githubusercontent.com/microsoft/vscode/40bb6fae207507042de8b6d7b74e9aa374a212a4/LICENSE.txt) · [本地原文](../../third_party/licenses/p19-palettes/monokai-vscode-port-MIT.txt)；SHA-256：`9480271317925265e806a9a196aaa33410a962fa9d4d1e248a4a5187bc8c9df9`。

### Night Owl

作者/版权主体：Sarah Drasner。

- 本次查验 commit：`cc291eba7976b20d7c66bde6883c27b902196b07`；[固定上游文件](https://raw.githubusercontent.com/sdras/night-owl-vscode-theme/cc291eba7976b20d7c66bde6883c27b902196b07/themes/Night%20Owl-color-theme.json)。
- [上游许可原文](https://raw.githubusercontent.com/sdras/night-owl-vscode-theme/cc291eba7976b20d7c66bde6883c27b902196b07/LICENSE.md) · [本地原文](../../third_party/licenses/p19-palettes/night-owl-MIT.txt)；SHA-256：`7167d2a386e197ad4bf85f9bc60653ddce195bfc1243733dc92975e77dc94968`。

### Nord

作者/版权主体：Sven Greb / Nord contributors。

- 本次查验 commit：`1cef71605416a222e57225b544540ce0fcec18d4`；[固定上游文件](https://raw.githubusercontent.com/nordtheme/nord/1cef71605416a222e57225b544540ce0fcec18d4/src/nord.css)。
- [上游许可原文](https://raw.githubusercontent.com/nordtheme/nord/1cef71605416a222e57225b544540ce0fcec18d4/license) · [本地原文](../../third_party/licenses/p19-palettes/nord-MIT.txt)；SHA-256：`25ac8188d670bd2ad2ce2f4f55ab88573010ee9f7a4502543cb1eea1e2274f8a`。

### One

作者/版权主体：GitHub Inc. / Atom contributors。

- 本次查验 commit：`9c96f4454362267ac45322063e193ccf9d2debb1`；[固定上游文件](https://raw.githubusercontent.com/atom/one-dark-syntax/9c96f4454362267ac45322063e193ccf9d2debb1/styles/colors.less)。
- [上游许可原文](https://raw.githubusercontent.com/atom/one-dark-syntax/9c96f4454362267ac45322063e193ccf9d2debb1/LICENSE.md) · [本地原文](../../third_party/licenses/p19-palettes/one-1-MIT.txt)；SHA-256：`e1bd6bab503e4d7990504df1e646f6e7465a9096648304335e4c0b55c88f1f54`。
- 本次查验 commit：`d84579027410c576086dfca14d934c4bd74b0438`；[固定上游文件](https://raw.githubusercontent.com/atom/one-light-syntax/d84579027410c576086dfca14d934c4bd74b0438/styles/colors.less)。
- [上游许可原文](https://raw.githubusercontent.com/atom/one-light-syntax/d84579027410c576086dfca14d934c4bd74b0438/LICENSE.md) · [本地原文](../../third_party/licenses/p19-palettes/one-2-MIT.txt)；SHA-256：`e1bd6bab503e4d7990504df1e646f6e7465a9096648304335e4c0b55c88f1f54`。

### Rose Pine

作者/版权主体：mvllow / Rosé Pine contributors。

- 本次查验 commit：`92af52b465ab6e47437aca223c9b8d3009a2023b`；[固定上游文件](https://raw.githubusercontent.com/rose-pine/rose-pine-palette/92af52b465ab6e47437aca223c9b8d3009a2023b/palette.json)。
- [上游许可原文](https://raw.githubusercontent.com/rose-pine/rose-pine-palette/92af52b465ab6e47437aca223c9b8d3009a2023b/LICENSE) · [本地原文](../../third_party/licenses/p19-palettes/rose-pine-MIT.txt)；SHA-256：`0e691132d5d92a848a1688b90fd4c662f62f78f3bd550ba46826ec31ec98fc70`。

### Solarized

作者/版权主体：Ethan Schoonover。

- 本次查验 commit：`62f656a02f93c5190a8753159e34b385588d5ff3`；[固定上游文件](https://raw.githubusercontent.com/altercation/solarized/62f656a02f93c5190a8753159e34b385588d5ff3/README.md)。
- [上游许可原文](https://raw.githubusercontent.com/altercation/solarized/62f656a02f93c5190a8753159e34b385588d5ff3/LICENSE) · [本地原文](../../third_party/licenses/p19-palettes/solarized-MIT.txt)；SHA-256：`494aefdabf86acce06bd63001ad8aedad4ee38da23509d3f917d95aa3368b9a6`。

### Tokyo Night

作者/版权主体：Enkia / Tokyo Night contributors。

- 本次查验 commit：`7c0f11eaef322f293621ca7befe462214b7ea468`；[固定上游文件](https://raw.githubusercontent.com/tokyo-night/tokyo-night-vscode-theme/7c0f11eaef322f293621ca7befe462214b7ea468/themes/tokyo-night-color-theme.json)。
- [上游许可原文](https://raw.githubusercontent.com/tokyo-night/tokyo-night-vscode-theme/7c0f11eaef322f293621ca7befe462214b7ea468/LICENSE.txt) · [本地原文](../../third_party/licenses/p19-palettes/tokyo-night-MIT.txt)；SHA-256：`e3f5d0d772cda0f4f67405696f15a0f335460a2b3944eff40f00675aadacb31b`。

## 解释边界与发行前未决项

- 软件代码许可、主题设计许可、名称/商标使用条件分别判断。品牌页、同名主题、社区移植或字体许可均不能自动覆盖当前方案。
- 11 套 MIT 公开上游的许可文本与作者已保存。Gruvbox 需取得完整版权/许可通知；Monokai 需闭合经典原设计与所选公开实现的对应许可链。
- Material 历史上游重定向到 Vira 的当前仓库。未取得官方可核对的历史许可原文及对应版本，因此没有沿用第三方 fork 的 Apache-2.0 标记。
- Monokai 原设计与 Monokai Pro 分列。官方 [Pro 社区移植条件](https://monokai.pro/contribute)带有特定要求，不能用其名称给本项补授权。
- 16 个待确认项只登记视觉参考和已知缺口。声明独立适配、保留署名、修改名称或调整少量色值，都不能单独替代必要的许可。对应公开发行的未决项仍保留。
- 历史记录曾读取 Codex 安装包主题元数据。本轮只访问公开网页/仓库，不进一步读取安装包；是否符合当时适用的[OpenAI 使用条款](https://openai.com/policies/terms-of-use/)仍属单独问题。名称/品牌参考另见[OpenAI 品牌指南](https://openai.com/brand/)。

## 应用与随包内容

关于页的软件与代码来源中新增 29 条配色记录，可按名称搜索，并查看作者、许可状态、固定上游与许可链接。记录保持未决标记；上游许可原文存于 `third_party/licenses/p19-palettes/`，沿用现有打包脚本收集许可目录。此次未重新打包 EXE，也未验收冻结成品中的展示。

[实施与验证报告](../testing/2026-10-05-p19-r11-theme-sources-report.md)。
