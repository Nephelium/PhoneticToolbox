# P19-R2 导航折叠按钮与任务栏图标

状态：verified，限定 Windows 开发态、独立 Chrome 和实际 Qt 离屏宿主。井井于 2026-10-04 要求导航三条线采用任务与记录区域的折叠按钮样式，并略微放大任务栏图标。

## 修改

- `frontend/src/app/AppShell.vue`：导航展开时显示 ‹，收起时显示 ›；采用有边框按钮，悬停提示收起侧栏／展开侧栏，补充 aria-expanded 与 aria-controls。原本的折叠状态记忆和窄窗自动折叠保持。
- `frontend/src/design/tokens.css`：按钮 28×30px，背景和边框使用现有主题令牌。
- `desktop/src/ptb_desktop/app_icon.py`：32px 及更大的图标使用完整输出画布，保留原裁剪安全边距、比例、圆角和透明背景；16／24px 保留一像素边距。运行时 QIcon 与后续打包 ICO 使用同一处理。
- 调整既有 `desktop/tests/test_p19_icon.py` 的可见占用验收，保留比例、ICO 解码和源图哈希检查，增加圆角透明性检查。无第三方素材或依赖变化。

可见主体宽度采用 alpha ≥32 实测：

| 输出尺寸 | 原可见宽 | 新可见宽 | 宽度增加 |
| --- | --- | --- | --- |
| 16 | 14 | 14 | 0% |
| 24 | 22 | 22 | 0% |
| 32 | 30 | 32 | 6.67% |
| 48 | 46 | 48 | 4.35% |
| 64 | 60 | 64 | 6.67% |
| 128 | 122 | 128 | 4.92% |
| 256 | 242 | 254 | 4.96% |

## 实际验证

| 命令 | 结果 |
| --- | --- |
| `npm --prefix frontend run typecheck` | 通过 |
| `npm --prefix frontend test` | 250 项通过 |
| `npm --prefix frontend run build` | 通过，保留既有大 chunk 告警 |
| 源码 PYTHONPATH 下 `.venv/m09-ui/Scripts/python.exe -m pytest -c tests/pytest.ini desktop/tests/test_p19_icon.py -q` | 2 项通过，七尺寸、比例、圆角、ICO 回读、源图哈希保持 |
| `node output/validation/p19-r2-20261004/sidebar.cjs` | 浅深两组状态／箭头／提示／刷新记忆／Enter／空格通过；六组宽度及缩放几何通过；窄窗自动折叠通过，页面错误为空 |
| 源码 PYTHONPATH 下 `.venv/m09-ui/Scripts/python.exe output/validation/p19-r2-20261004/icon_qt.py` | 实际 Qt 构建页折叠与展开通过，运行时 QIcon 七尺寸及32px可见主体通过，生成七份PNG和ICO并记录新旧宽度 |
| `git diff --check` | 通过 |

证据位于忽略目录 `output/validation/p19-r2-20261004/`，含两个验证脚本、两个 JSON 报告、浅深展开／收起局部截图和实际 Qt 截图。截图已人工回读。

Vite 开发预扫描对既有声道静态资源的 `three` 裸导入发出解析告警，导航验证正常完成；本轮未加载声道模块，不能据此判断声道功能。Qt 离屏运行保留 GLES 上下文告警，页面、截图和图标读取实际通过。测试进程的离屏参数没有改动产品或系统配置。

## 边界与入口

使用[源码工作台入口](../../scripts/Start-M16-M17-Workbench.ps1)。本轮未重新生成 EXE，既有 EXE 不含本轮修改。实体 Windows 任务栏、系统 DPI 和其他平台 GUI 未实测，所列百分比是图标输出像素测量，实际显示由 Windows 缩放决定。保留原 K2 图片、旧包、V2、科研数据与同期其他改动。无删除、数据库迁移、环境安装、push 或公开发布。
