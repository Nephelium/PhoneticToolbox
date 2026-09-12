# P04 深色主题下拉弹层修复

2026-09-11，verified，限定公共前端及Windows Qt开发运行时。用户截图显示深色主题下拉框白底浅字。本轮仅修公共控件颜色，不修改分析、任务、数据库或EXE。

## 实现

`frontend/src/design/tokens.css` 中将主题select透明背景改为公共panel背景，option/optgroup显式指定背景与文字，保留原生选择和键盘行为。沿用既有U2浅深色令牌，其他公共下拉框也获得相同修正。没有新增依赖或外部素材。

## 验证

- `npm --prefix frontend run typecheck`：通过。
- `npm --prefix frontend test`：33 passed，新增弹层显式背景回归，并加入text/selected对比度检查。
- `npm --prefix frontend run build`：通过，开发前端产物已更新。
- `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_theme_popup.py`：通过。使用独立、无持久用户配置的Qt页面加载真实前端产物，无后台服务。展开实际原生弹层，截图核对深色、浅色与跟随当前系统（dark），Escape关闭保留选择。
- 证据：`output/validation/theme-popup/7f801b337c72468e88b7e22cdd206cef/report.json`及三份popup截图。深色背景rgb(24,36,50)、文字rgb(233,240,249)，选中行由Qt原生高亮绘制，实图可读。

测试开发中首轮错误地把选中项背景与普通项背景比较，已改为检查未选中项。另一轮发现QStyleHints应用级配色覆盖未同步Chromium媒体查询，不能借此冒充系统切换测试；最终跟随系统核对真实matchMedia结果。未修改Windows配色，未验证真实系统设置动态切换。Qt打印既有资源PNG配置警告，不影响本次断言与截图。

当前录制专用M10-R5 EXE没有重新打包，此报告不声称旧EXE已获得修复。后续实际发行入口修复时需纳入新前端并复验。
