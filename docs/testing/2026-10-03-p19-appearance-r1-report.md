# P19 外观 R1：字体选择与默认配色

状态：**verified，限定 Windows 开发态、独立 Chrome 和实际 Qt 离屏**。源码与前端构建已更新，本轮未重打冻结 EXE。WSL 仅静态资源读取，Linux GUI 与实体 DPI 未验。

## 修复

- 原代码字体框使用 datalist，当前保存 Georgia 时浏览器会按该文字筛选候选。字体资源已经内置，但这一入口无法直观展示其他选项。改为完整原生下拉列表，首项为 JetBrains Mono（内置），保留原保存值、已读取的本机字体和自定义名称输入。
- 成功应用字体后清除临时预览覆盖，避免先预览 Georgia 再应用 Mono 时仍看到旧预览。中文宋体、英文 Times New Roman 的默认值及已有用户选择保持。
- 移除 PhoneticToolbox 配色选项，剩余 29 套、58 个深浅组合。首次使用、旧 `ptb` 与无效偏好采用 Everforest，保留用户已经选择的 Matrix 等其他有效方案及显示模式。
- 设置增加提示：默认使用 Everforest。也欢迎试试更多配色，选一款自己喜欢的，让工作台更合心意。

## 验证

| 检查 | 结果 |
| --- | --- |
| `npm --prefix frontend run typecheck`、`npm --prefix frontend run build` | 通过，原大 chunk 提示保留 |
| `npm --prefix frontend test` | 242 项通过，含旧默认回退、其他方案保留及全部配色语义文字对比度 |
| `node tests/e2e/p19-appearance.cjs` | 11 组通过，58 主题/模式及 15 窗口/缩放组合；默认与旧偏好迁移、草稿、系统模式、字体资源实载/等宽测量；实际自定义保存 Georgia → 预览 Georgia → 下拉选择 Mono → 应用后预览与重启保持；缺字体拒绝及取消 |
| `node tests/e2e/fonts.cjs` | 12 组通过，含真实 M02 PNG/SVG、IPA 像素参考、字体偏好与账号切换 |
| 既有 m09-ui 环境、源码 PYTHONPATH 下 `scripts/verify_p19_qt.py` | 4 类通过，58 主题/模式、6 布局，离线字体、实际 Georgia → Mono 应用及 M10 已开 iframe 配色同步 |
| WSL NInfer 既有 Python | 4 项静态回读通过：字体哈希、许可一致、构建入口与源码/构建 UTF-8；不作为 Linux 交互验证 |
| 定向 `git diff --check` | 通过 |
| `scripts/validate_docs.py` | 检查 1284 文件，仍有 8 条既有旧 EXE 缺失链接，全库文档检查未全绿；无本轮新增错误 |

最终证据位于忽略目录：

- `output/validation/p19/chrome-1791038868575/report.json`，浅深/窄窗截图。
- `output/validation/p19/qt-49f0f6a796b94781a31ebb977f9132d6/report.json`，实际 Qt 浅深截图。
- `output/validation/fonts/chrome-1791038868574/report.json`，字体导出回归。
- `output/validation/p19/wsl-r1.json`。

浅深截图已目视检查，完整下拉框及双栏内容无裁切。Qt 离屏存在原 GLES 上下文告警，实际 DOM、字体与截图检查通过，未修改系统或产品渲染配置。未改科学算法、数据库、用户数据、系统字体或其他任务改动，无 push 或公开发布。启动源码工作台后生效，现有 EXE 内容不变。
