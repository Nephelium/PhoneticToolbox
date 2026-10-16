# 语谱图转音频小数输入验收

2026-10-13。状态为 `verified`，范围为 Windows 当前源码与独立 Chrome。未打包、发布或推送。

井井后续要求检查全部同类框，公共播放条等延迟提交字段及 F0 数值表已继续修复。扩展范围与当前证据见[全项目数值输入检查](numeric-input-report.md)，本页保留语谱图原问题的复现与修复记录。

## 原因与改动

公共 `vTimePrecision` 指令在 input、updated 与提交事件中无条件赋值 `input.value`。Chromium 数字框输入 `0.` 时，公开 value 暂为 `0`，其内部仍保留小数点；把相同的 `0` 写回会清除中间输入并改变光标。修改前实际语谱图页面逐键输入 `0.1` 得到 `10`，与输入法切换无关。文字框也会因相同内容回写而把光标移到末尾。

`frontend/src/design/time-precision.ts` 增加条件回写，只在格式化前后文本不同的时候赋值。原秒五位、毫秒两位精度规则、显示与科研状态分离规则保持。没有修改算法、默认值、输入单位或后端请求契约。

## 检查与结果

- `node tests/e2e/m09-decimal-input.cjs`：真实 M09 Vue 页面、8×8 内存 PNG、独立 Chrome 用户目录。浅／深主题共10组通过，覆盖时间起点 `0.125`、终点 `0.5102`、窗长 `10.25` 的逐键输入，未完成小数时其他参数触发页面更新、删除后重输、`.5`、中间修改、超限精度、负 dB 小数、数值提交快照，以及公共文字时间框中间插入和末尾零。页面异常0。
- `node tests/e2e/time-precision.cjs`：已有公共播放条、EGG、文字与参数设置、聚焦时外部数值更新、单位切换、M10 原生关键帧等4组回归通过，源状态保留原精度。
- `node --test tests/time-precision.test.ts tests/m09*.test.ts`（frontend）：4项通过。
- frontend 的 `npm run typecheck`、`npm run contracts:check`、`npm run ui-data:check`、`npm run build` 通过。Vite 保留现有大块体积提示，不影响构建退出成功。
- 浅深主题参数栏截图已目视检查，数值与标签可见，布局保持。
- `python -B -X utf8 scripts/validate_docs.py`：2022文件／427来源／107任务，errors为空。既有历史快照链接单列，未修改；定向 `git diff --check` 通过。

本机证据为 `output/validation/m09-decimal-input/chrome-1791898695928/` 的报告、浅深参数栏截图与清理回执。修改前回归失败保留在 `chrome-1791898671734/`；更早的 `chrome-1791898544179/` 为夹具把图片下拉误作按钮造成的定位失败，已按真实页面改成选择图片，未改变输入断言。三个新增脚本运行目录的 scratch 均为 `deleted`，只保留报告、截图与输入配方。公共控件回归证据为 `output/validation/time-precision/chrome-1791898724257/`。

## 范围

重建提交由内存测试 port 收集参数，用于确认输入结果以数值进入快照，没有执行科学重建或保存文件。当前浏览器输入验收不代替实际 Qt、最终 EXE、IME 组合输入、实体键盘／DPI、跨平台或听辨验收。图片反相保留原任务未做专项行为验收的范围，本轮前端构建成功不能扩大该结论。
