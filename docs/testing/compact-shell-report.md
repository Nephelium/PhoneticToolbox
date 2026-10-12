# 顶边标签栏与阅读侧栏

`P04-COMPACT-SHELL` 的当前 Windows 源码交互已验证，最终 EXE 未重建。Windows 将单一品牌、标签与窗口按钮合并到顶边，激活标签整体采用面板背景，文字和关闭按钮同色。首页试用版行移除，致谢入口在首页内容区右上角。说明书定位标题不再绘制框线，当前目录项通过背景和文字颜色提示。说明书目录和论文库复用公共分栏，分别记忆宽度。

## 验证范围

- `.venv/m14/Scripts/python.exe -X utf8 scripts/verify_compact_shell_qt.py`：实际 Windows 11 Qt WebEngine、独立 profile、既有 schema 数据库副本。最终交互证据为 `output/validation/compact-shell/qt-99a0fc40c3ad436e809eed92d56efe20/report.json`，`success=true, closed=true`。
- 顶行与品牌从客户端顶部开始，单一品牌，首页内容区右上致谢对话框可打开，标签栏没有该入口。实际最小化/最大化/还原、最大化屏幕工作区与全屏范围通过；其中旧最小化检查只将普通窗口用 showNormal 恢复，未覆盖最大化窗口的 Windows 原生恢复像素，该缺口由 [窗口恢复验收](window-restore-report.md)补充。本窗口左右边缘及角落的原生命中测试通过。
- 浅深主题的完整标签底色与文字/关闭按钮同色通过。真实 Qt 指针拖动说明书目录和论文库，两者分别保存，切换标签及临时窄窗口后恢复。目录当前项没有框线或阴影，正文定位标题无 outline。
- 用真实 Qt 点击关闭按钮，未应用字体触发既有原生未保存确认；选择否保留窗口和草稿。仅取消隔离测试实例的编辑后退出。
- `node tests/e2e/manual-outline.cjs`：独立 Chrome profile，最终 `output/validation/manual-outline/navigation-1791701609545/report.json success=true`。18模块目录图标、5组折叠/定位/滚动/搜索行为、六组浅深色与14/18/24 px字号及窄窗口通过。覆盖悬停时的通用按钮阴影覆盖问题。
- `node --test frontend/tests/manual-reader*.test.ts frontend/tests/panel-widths.test.ts frontend/tests/papers.test.ts`：23项通过，appearance另5项通过。前端类型、contracts/ui-data与生产构建通过。

## 说明书与边界

正式说明书同步入门、设置与论文精读文字，以及首页、目录与四设置页配图。最终配图证据为 output/validation/settings-navigation/manual-qt-35e0d0489e2e4b1c9b1d219eaf294fc3/report.json，包含最大化启动时还原按钮初始状态检查。说明书生成20章1042素材、warnings=[]；文档/架构errors=[]。截图来自当前源码 Qt、Everforest 浅色、原生 DPR1.5、页面100%、2560×1600的独立实例，登记为 software-only。图片与正式工程不因未引用而删除。

测试报告和截图保留；11个自有Qt scratch 在进程退出后已通过项目清理入口清理，cleanup.json均为deleted。Chrome测试由其脚本清理。

本次保留 Windows 缩放与贴边样式，完整贴边交互、多屏/其他物理 DPI、Windows 10与其他平台尚未实测。窗口移动接入 Qt startSystemMove，未以真实鼠标跨屏拖动扩大验收。科研算法、数据格式与论文内容未改动。
