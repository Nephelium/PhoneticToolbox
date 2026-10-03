# P19-R4 波形颜色验收

2026-10-04，Windows 开发态本轮范围 verified。入口沿用现有开发工作台。EXE 未重新打包。

## 实施

设置 → 外观 → 波形线颜色提供默认蓝色、跟随主题色、自定义颜色。默认保留现有浅/深蓝色。主题色随配色与显示模式变化，自定义颜色保持固定值。色盘与 HEX 输入同步，支持 `#RGB` 和 `#RRGGBB`，无效输入提示并保留已应用颜色。设置立即作用于已打开的波形并在本机记忆，保存失败显示提示。

波形独立使用 `--waveform-color`，没有替换通用科研曲线令牌 `--wave`。公共波形及总览、M16 录音波形、M03 音频微观和逆滤波音频轨道、M05 偏移量音频图、M10 监视画布接入。频谱、F0、CQ、EGG 和 IF 曲线保留原有配色。M10 通过已有主题桥传递已解析的 HEX 值，并在主题事件重绘现有画布。

M02 整幅图和 M03 逆滤波前端导图使用显式选择的主题或自定义色。默认蓝色保持既有印刷蓝色，M02 为 `#245ab5`、M03 音频为 `#174b82`。历史 PNG、后端正式任务结果和原始样本未修改。

## 验证

- `npm --prefix frontend run typecheck`、`test`、`build` 通过。前端测试 266 项，无失败，两项新增测试覆盖 HEX 校验和独立偏好恢复。
- `node tests/e2e/p19-waveform-color.cjs` 通过 5 个功能组：29 配色 × 2 显示模式、色盘/简写 HEX/非法输入/主题切换/偏好重启、390 宽与 150% 页面缩放、实际公共/M16/M03 组件的三模式 × 两主题及 M10 合成画布重绘、M02/M03 导图。改变颜色前后的波形路径和坐标逐值相同，其他科研轨道的导出色保持原值。
- Chrome 证据：`output/validation/p19-r4/chrome-1791054603721/report.json`，设置截图 `settings-custom-dark.png`。
- 两份 PNG 回读：`whole-custom.png` 3750 × 1297、`scientific-custom.png` 7550 × 616；均为约 299.9994 dpi。精确自定义色 `#b52f91` 像素分别 196497、15926。记录为同目录 `png-readback.json`。
- 实际构建 Qt：`scripts/verify_p19_waveform_color_qt.py` 通过 9 条记录，含设置三模式 × 两主题、自定义与主题 iframe 桥接、非法输入保持前色、切回自定义保留原色及 1024 宽布局。证据：`output/validation/p19-r4/qt-7bc71e7632c94bfb8545e3c7cb2d43dd/report.json`。只读设置和主题桥，无设备采集。
- WSL NInfer 使用既有 Python 读取 4 份资源，SHA-256 与 Windows 一致，记录为 `output/validation/p19-r4/wsl-static-hashes.json`。静态一致性不代表 Linux GUI 验证。
- `scripts/validate_docs.py` 检查 1353 份文件，当前仍有 8 条既有旧 EXE 缺失链接，本轮新增计划、报告、设置说明和 UI 规范链接无新增错误。全库文档检查未整体通过，未绕过历史错误。

## 验证过程与限制

早期 Chrome 检查有测试侧颜色序列化断言差异，及 Vite 对 public 脚本的测试加载路径限制。修正测试以 RGB 比较并按生产页面直接加载 public 模块。首轮 Qt HEX 测试 helper 的选择器转义错误已修正。失败证据保留，最终通过记录如上，未降低产品断言。

构建保留既有大分块提示，Qt 离屏保留既有 GLES 日志。本轮没有全平台、实体 DPI 或硬件验证。Chrome 使用合成显示输入，不能作为科研算法等价证明。M10 实际 Qt 验证颜色桥，画布重绘由 Chrome 的受控合成监视输入验证。未改数据库、V2、原资料、环境或 CI，未 push、发布、删除或新打 EXE，保留同期其他任务改动。
