# M12-R6 图窗优先与完整标注剪贴

2026-09-19，Windows 开发前端、独立 Chrome 与本地文件桥接的本轮范围 verified。用户明确授权的 R6 临时单文件 EXE 已构建，冻结宿主 11 步加载、编辑、保存和下载验证通过。

## 行为

- 中央第一张卡片为波形、语谱和标注层，文件列表与保存 TextGrid 放在卡片顶部。强度贴合、词典词表、搜索替换依次放到图窗下方。
- 移除音素自动填充、撤销、复制词、连续粘贴四个按钮。双击音素编辑和 Ctrl+Z 保留。
- 用户已确认整段标注语义。Backspace 删除选中标注并在原位置留空，词层对应音素一起处理；选中边界时仍合并边界。Ctrl+X/C 捕获真实时长、原文字、内部音素边界与多选间距，Ctrl+V 从点击的空白时间粘贴，不递增声调。
- 剪切/粘贴分别一步撤销。跨越所选外边界的非空音素、目标重叠、越界、层角色不匹配均在修改前拒绝。音素层独立操作不修改词层。输入框继续使用原生文字快捷键。
- R5 的原始 TextGrid 优先、三图编辑、选区同步、毫秒窗长和 Ctrl 拉开边界保持。

## 深色截图调查

按钮沿用全局 tokens.css。旧测试在暂停的浏览器时钟下修改主题，只推进 40 ms，截到了 120 ms 背景过渡中间态。恢复计时并等待实际 computed background 后为 rgb(24,36,50)，即全局 #182432。本轮没有修改产品主题配色。对照证据 `output/validation/m12-ui/1c6c063e634d4e6a866c4855f0bc7942/theme-colors.json`，R6 截图等待最终颜色。

## 验证

- `node --test frontend/tests/annotation-r6.test.ts`：6 项通过。包含剪贴后序列化回读、无关点层保留、单步撤销、拒绝操作不改变文档、组选择与跨界音素。
- `npm --prefix frontend test`：119 项通过，0 失败/跳过。
- `npm --prefix frontend run typecheck` 与 `npm --prefix frontend run build`：通过。
- `node tests/e2e/m12-r6.cjs`：6 组 Chrome 检查通过，真实按键及本地保存回读、布局、原生文字退格、浅深/窄窗。证据 `output/validation/m12-ui/a74d447afff14c56b3069eb767d7ef82`。
- `node tests/e2e/m12-r5.cjs`：8 组回归通过。证据 `output/validation/m12-ui/055c6b0b316943ada3b53668e6ef5792`。旧 R5 截图仍是冻结时钟流程，颜色结论采用 R6 的稳定截图。

## 范围

仅修改 M12 编辑交互、布局、模块内剪贴和相应测试说明，无新增依赖。没有修改 V2、用户原语料、现存数据库或其他模块业务。保留工作区原有改动，不将其他任务的全局侧栏改动归为本轮成果。临时包从当前工作区构建，保留当前公共外壳；没有重新验收所有模块。跨平台、生产服务器、声卡、长录音及新快捷键的真实 Qt 键盘事件尚未作为本轮验收范围。

新临时包仍不包含 M03/M04 独立兼容运行环境，供本机 M12 试用，不代表完整正式发行。旧 R4 包保留。

## 临时 EXE 与冻结验证

产物 `dist/m12-preview-r6/PhoneticToolbox-v3-M12-R6.exe`，326685982 字节，SHA256 `f5aaf8a18cf9e9432b89ed39245efe367e489fc93f6efb44ec547f99432f7c4c`。同目录附使用说明。使用项目 `.venv/m09-ui` 和既有 PyInstaller 单文件配方，未升级第三方包。构建约 196.5 秒，日志 `output/validation/m12-r6-build.log`。

命令：

```powershell
.venv/m09-ui/Scripts/python.exe -X utf8 scripts/build_m12_preview.py
.venv/m09-ui/Scripts/python.exe -X utf8 scripts/run_research_repair_check.py --exe dist/m12-preview-r6/PhoneticToolbox-v3-M12-R6.exe --verification m12
```

首次在工具沙箱内启动，页面未返回响应，240 秒超时，检查器终止自身持有的测试进程。保留证据 `output/validation/m12-exe/frozen-b268532a6c0b4fcca16fbaaf74d7e568`，不计通过。获得明确沙箱外执行授权后，相同 EXE 复跑通过，证据 `output/validation/m12-exe/frozen-6db9bc4ed1534f09994ca676d13275a4`。

实际冻结 EXE 从系统临时目录启动，清除开发 Python/Qt 环境并仅保留系统 PATH，使用新建独立测试状态及合成文件。11 步通过，中文/IPA 标注保存与 TextGrid 下载逐字节回读、唇偏 -21 ms 保存及安全 JSON 下载回读通过，波形/语谱高度均 175 px。宿主退出码 0，没有可见测试工作台窗口、没有残留自有子进程，3 种非法子进程分派参数均拒绝。日志保留素材 PNG 配置警告，不影响上述检查。

这项冻结检查覆盖桌面集成链路，R6 新键盘剪贴的端到端证据来自 Chrome。没有将 11 步冻结检查扩大为所有新交互/全部模块验收。

旧 R4 包构建前后 SHA256 均为 `769398a6de755817654fc7c7166b0bde66b26652fdc0d98061699c9478e85964`，未覆盖。contracts:check、ui-data:check 与 git diff --check 同样通过（仅有既有换行提示）。
