# GitHub 图文 README 检查

任务：`DOCS-GITHUB-HOMEPAGE`。日期：2026-10-11。状态：`verified`，限本机文档、真实界面配图与本地渲染。

## 交付内容

- 根 README 改为访客入口，包含工作台概览、真实录音分析、EGG、TextGrid、LPC、音标表、18 个模块链接、安装架构、MFA 独立组件与开发导航。
- `docs/assets/readme/` 保存六张原始 Qt 截图、来源说明和 SHA-256 清单，共约 2.97 MB PNG。
- 输入由用户在当前任务中明确指定。语句示例含同名原始 TextGrid，EGG 输入为真实双声道记录；WAV / TextGrid 仅复制并使用通用名称，字节保持一致。公开文档不含原始语料、参与者身份或私人绝对路径。
- 来源登记 `DOC-README-NATURAL-AUDIO` 仅服务文档，不进入软件致谢或随包许可生成项。原始语料没有据此获得开放再分发许可。

## 已执行的检查

| 项目 | 实际结果 |
| --- | --- |
| 模块入口 | 18 个注册模块均有独立 README，根 README 包含全部 18 个唯一链接 |
| 真实 Qt 配图 | Windows Qt 原生工作台实际加载真人语句与 TextGrid，完成 pF0 / pF1 / pF2、LPC 与 EGG 交互分析；共六张最终 PNG，2561 × 1601，DPR 1.5、页面 100%；原始截图未裁剪或缩放 |
| 来源一致性 | 三份输入的临时副本与原件 SHA-256 一致，结束后再次检查原件未变；最终输入和图片摘要见 provenance.json |
| 本地 README 渲染 | 既有 marked + 独立无头 Chrome / Playwright 渲染 GFM，1200 px 与 390 px 两种宽度；6 张图均完整解码，无页面横向溢出，页内导航有效 |
| 文件引用 | 根 README 的 44 个本地文件引用有效；含导航、来源和检查报告共核对 75 个本地文件引用，全部有效 |
| 生成数据 | `node frontend/scripts/generate-ui-data.mjs --check` 通过，80 参数、391 个界面来源；隐藏的文档来源没有改变应用生成数据 |
| 文本与仓库边界 | 新文档 UTF-8 无 BOM；配图目录仅含 PNG、Markdown、JSON；临时脚本和私有来源路径位于 Git 忽略的 output 中；无 `.github/README.md` 覆盖根 README |
| 临时材料清理 | 截图进程退出后执行 `scripts/cleanup_test_scratch.ps1`，复核归属、联接与进程，三轮完成截图的 scratch 回执均为 deleted；不涉及原始资料 |

## 本机证据与修正

证据根目录为忽略的 `output/validation/readme-homepage/`，脚本、截图、输入摘要、渲染报告和清理回执保留在本机。

- `local-audio-47cf3f673eb9447db52e4b518132dbb2/report.json`：六个真实界面状态及输入来源摘要，原件不变。
- `m01-spectrogram-b6282c98a6854a0eb104ce0fbc348829/report.json`：M01 补开实际 Praat 语谱图，替换最终主页 M01 配图。
- `render-report.json`：两种宽度、六张图片、模块入口和页内锚点结果；`preview.html` 与 `readme-1200.png` / `readme-390.png` 供本机预览。
- 首次探索公开语音时，Python wave 不接受输入的浮点 WAV，随后改用 soundfile 读取元数据。用户指定本机数据后，最终图片全部改用指定本机录音，公开朗读图未进入交付目录。
- 本地渲染脚本初次存在一个正则表达式语法错误，改为明确的路径前缀判断后执行通过。截图程序退出上下文时 Qt 子进程尚未完全释放，自动清理先保留 scratch；主进程结束后使用既有安全清理入口成功清理。

## v2 / v3 远端只读核对

执行 `git ls-remote --symref origin HEAD refs/heads/main refs/heads/codex/v3-rebuild refs/tags/v2.1.4`：

| 对象 | 观察到的指向 |
| --- | --- |
| 默认 HEAD | `refs/heads/main` |
| `main` | `4eeff893eb6cdcddb40456072571dd5d30c51a76`，v2 |
| `codex/v3-rebuild` | `ef764e2d4e32ba442b02350e1168d4e2b538e87d`，v3 |
| `v2.1.4` 标签 | 与上述 `main` 指向相同提交 |

改变默认分支只影响默认展示与相关工作流，不自动删除其他分支，也不自动建立名为 v2 的分支。[GitHub 默认分支说明](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/changing-the-default-branch)与[根 README 展示规则](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)已核对。未来若要求 v3 使用 main 名称，应先明确保留 v2 分支及旧版本标签，再执行指定的切换步骤。

## 未包括的范围

没有提交、push、上传图文、重命名分支、修改默认分支或重新构建应用。渲染采用本地 GFM 与近似 GitHub 样式，尚未在 GitHub 页面实际发布验收。界面截图证明示例路径完成，不外推科学准确率、物理设备、安装包或跨平台验证。
