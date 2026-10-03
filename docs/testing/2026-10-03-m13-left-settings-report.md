# P19 普通话转IPA设置栏与音标导航图标

2026-10-03，限定Windows独立Chrome开发态。

M13左右排布原先明确采用 `input result settings`，拖动指令也将设置栏定义为right。现改为 `settings input result`，上下排布继续 `settings input / settings result`，DOM也将设置区放在输入区之前。两模式设置栏均在左，其右边界向右拖动即增加宽度。沿用原 `--panel-right` 存储变量读取已保存的设置栏宽度，未清除草稿或偏好。该变量仅是兼容旧存储的键名，视觉方向固定left。

M13保留æ图标，M17改为符号键盘SVG图标，沿用公共24×24画布与描边样式。变更应用于侧栏、主页及使用同一登记表的入口。

## 验证

```powershell
& 'C:/Users/13680/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' tests/e2e/m13-layout.cjs
```

最终证据：`output/validation/m13-layout/1791013657017/report.json`。

- 浅/深色 × 70%/100%/150%缩放 × 左右/上下，共12组设置栏左侧与输入结果顺序检查。
- 多音字弹窗相邻定位、视口内显示、选择、外部点击关闭、Escape关闭与焦点返回。
- 800px窗口及侧栏折叠下的菜单可达性和图标显示。
- 两种排布中分别进行方向键及真实指针拖动，向右增长10px和20px，正文仍保留。
- 保存后重载，布局、文本、栏宽保持，设置折叠/恢复后宽度不变。
- M13和M17分别显示æ和SVG符号键盘，浏览器pageerror为0。

前端最终类型/全套测试、生产构建、实际Qt/EXE由P19联合报告追加，不能将本页开发态结果扩大为全部成品验证。
