# M13-R1 / P04-ICON 验收记录

2026-09-29。状态：verified，限定以下实际检查范围。旧 EXE 未重打，Linux 原生桌面、macOS 与固定任务栏快捷方式未测。

## 修复

- 左右/上下排布均保留最右侧操作按钮、标准、样式和四个滑块。上下排布仅改变输入与结果位置，保留并行公共栏宽拖动接线。
- 多音字浮窗为 240×240 逻辑像素，锚定当前字，视口边缘避让，滚动/页面缩放重新定位，长列表内部滚动。选择、Escape、× 和外部点击关闭；选择/Escape 后焦点归还字按钮。
- 公共侧栏的文字隐藏规则排除 `.ipa-icon`，主动折叠及窄窗均保持可见。
- Qt 应用和 Workbench 使用前端已有 K2 图像；Windows 进程使用 `PhoneticToolbox.Desktop.3` 应用身份。没有新图像来源或科学数据修改。
- 公共 WorkspaceView 正在进行的模块壳替换留下 `</section>`，本轮只将闭合标签修正为匹配的 `</ModuleFrame>`，保留其余改动。

## 实际命令与证据

| 命令 | 结果 |
| --- | --- |
| `npm --prefix frontend run typecheck` | 通过 |
| `npm --prefix frontend test` | 178 项通过，0 失败 |
| `npm --prefix frontend run build` | 通过，保留大分包警告，未提高阈值 |
| `node tests/e2e/m13-layout.cjs` | 两主题 × 三缩放（70/100/150%）× 两排布共 12 组合通过，另验浮窗选音/关闭/焦点/实时缩放/下边缘/滚动、折叠图标和 800×700 窄窗 |
| `node tests/e2e/m13.cjs` | 原有 11 标准、逐位置选音、长文本、断网 PNG、字体/编码失败、空输入及 AppShell 草稿/关闭/恢复全部通过 |
| `.venv/m09-ui/Scripts/python.exe -m pytest desktop/tests -q -o addopts=` | 31 项通过 |
| `.venv/m09-ui/Scripts/python.exe scripts/verify_m13_qt.py` | 实际 Qt/ptbapp 10 阶段通过；应用及窗口 QIcon 非空，Windows HWND 的 WM_GETICON 大/小图标句柄均存在 |
| `wsl.exe -d NInfer --exec perl /mnt/d/PhoneticToolbox/PhoneticToolbox_v3/scripts/verify_m13_linux_static.pl` | Linux 静态构建 5 项 HTTP 回读/哈希一致 |
| `git diff --check` | 通过 |

Python 验证使用 `PYTHONPATH=desktop/src;backend/src;packages/phonetic_core/src;scripts`。原项目 pytest 默认继承 V2 的 `--cov=phonetic_toolbox`，现有 v3 环境未安装 pytest-cov，首次命令因此未启动测试；显式清空无关覆盖参数后执行全部 desktop tests，没有跳过测试。初轮完整构建/浏览器被上述 WorkspaceView 闭合标签阻断，修复后重跑通过。

证据目录：

- `output/validation/m13-layout/1790690259143`：12 组合报告、上下深色和左右浅色/收起侧栏真实截图，已查看。
- `output/validation/m13-browser/1790690259126`：原 M13 回归与真实离线 PNG。
- `output/validation/m13-qt/912119f0171444f8a57cb1e56c345b2b`：Qt 报告、浅深截图、实际窗口图标回读，已查看。保留既有 libpng 元数据警告。
- `output/validation/m13-linux-static/20260929-135815`：WSL 静态回读报告。该证据不代表 Linux GUI 验证。

另在独立隐藏 Qt 窗口中用 Windows API 实查应用身份为 `PhoneticToolbox.Desktop.3`，大/小图标句柄均有效。没有打开可见测试窗口、修改系统配置、现存数据库 DDL、V2、公开发布或 push。

本轮结果为源码及前端生产构建。已有冻结 EXE 不会自动获得修复。并行公共布局与其他模块差异不归入本报告的验证结论。
