# P08 成果收尾与 M02 整幅 PNG 验收

2026-09-12。**M02-F05 整幅 PNG verified，限定Windows本机开发态Qt和独立Chrome组件验证。** M03只完成规划，仍planned。Research-Fix1及M10旧EXE未重新打包，不能认为历史EXE已包含本轮前端修复。

## 成果整理

- 本轮开始分支 `codex/v3-rebuild`，最后提交为 `6ea58f2`。原工作区47个已跟踪文件修改、73个未跟踪条目，展开后共178个源码/文档/资源文件。
- 已审阅归属和目录边界，形成本地检查点 `c800ce8`，覆盖既有交织的M01收口、M02/M09、M10/R5及Research-Fix1。未包含环境、数据库、语料或临时输出，未push。忽略目录中的 `output/validation/20260912-closeout/existing-files.json` 保留提交前逐文件哈希。
- 将当前不存在的R4下载链接改为历史产物记录，保留原大小/哈希与验收。此次未删除或重建R4，不能据此判断其此前何时消失。
- 初次完整暂存检查发现此前未跟踪的M10文件、精确上游three.core.js及几何补丁含空白格式问题。为保留冻结来源和补丁上下文，原样进入历史检查点，不宣称该检查点的全量空白检查通过。本轮新修复差异单独检查。
- 文档、架构、契约、来源生成校验通过。历史归档README等原已登记的失效链接保留，未将它们伪装成当前可用入口。

## 产品行为

每个M02图窗新增“保存整幅 PNG”，输出当前显示声道的波形、文字标注、可选已完成的Praat语谱图，以及该图窗参数/左右轴/图例。各轨按同一时窗和同一绘图区宽度排列。白底深字，保留选区、曲线原值和时间。快照时先冻结图形，生成期间防重复点击。

保留原“保存当前图”的参数SVG。PNG按300/96倍尺寸栅格化，pHYs为11811像素/米，解码读取299.9994dpi（整数像素/米的正常舍入）。最大3200万像素和单边16384，超过时明确提示缩小图窗，不静默降分辨率。语谱图未就绪或失败时拒绝不完整导出，允许关闭语谱图后重新保存。

图像不包含工具按钮、参数侧栏或其他图窗。导出IPA采用独立字形尺寸，避免随波形viewBox非等比拉伸；不改变页面计算或原始文件。

## 实际验证

| 命令/检查 | 结果 |
| --- | --- |
| `node --test frontend/tests/m02-export.test.ts` | 先记录缺实现失败，补实现后2项通过。独立Node zlib CRC核对pHYs，IDAT保持不变，已有pHYs替换幂等，截断与尺寸预算拒绝 |
| `npm --prefix frontend test` | 37项通过，包含M02和公共回归；不能解释为M10/R5完整验收 |
| `npm --prefix frontend run typecheck` / `run build` | 通过 |
| `.venv/m09-ui/Scripts/python.exe -X utf8 scripts/verify_m02_png_qt.py`，`PYTHONPATH=backend/src;desktop/src;packages/phonetic_core/src` | 实际工作台、合成双声道WAV、旧XLSX受限读取、真实Praat及原生文件保存对话框，12步通过；正常关闭，输入哈希不变，无DDL |
| `node tests/e2e/m02-png.cjs` | 独立Vite/Chrome，合成组件输入下实际下载通过。浅/深色、双声道、自动双轴、多图窗/空窗禁用、0.2–0.7秒缩放、390px窄窗、语谱图未就绪错误均覆盖；自有服务/浏览器正常关闭 |
| PNG独立回读 | Qt4份、Chrome5份均解码成功，全部块CRC经Python zlib核对，300dpi、白底、真实波形/参数像素。未把下载按钮存在当成保存完成 |

最终Qt证据：`output/validation/m02-png/qt-f328f0c1d16d47b38453282fa80a3682/report.json`，图片尺寸依次1969×2197、1969×2854、1969×3729、1969×3729。已人工查看含双声道/Praat与缩放的PNG，检查中文、IPA、缺失曲线断线和时间轴对齐。

Chrome证据：`output/validation/m02-png/chrome-1789192914671/report.json`、`snapshot.svg`、`image-readback.json`。五份PNG从1063×2782至2669×2854。组件夹具明确与真实文件/后端验收区分，真实旧表和Praat路径由Qt覆盖。

开发中的失败保留：首轮Qt测试导入了本环境没有的Pillow，改用已有OpenCV和标准库解码/CRC，不安装依赖；首轮Chrome夹具错误返回ArrayBuffer，未满足read接口对象结构，修正夹具后通过。首次图片审阅发现波形IPA字形随viewBox拉伸，已修复产品导出并复跑Qt。对应失败文件保留在输出目录，不作为通过证据。

## 来源与限制

新登记 `REF-PNG`：[W3C PNG Third Edition, 24 June 2025](https://www.w3.org/TR/2025/REC-png-3-20250624/)，关系为规范参考，使用5.5 CRC与11.3.4.3 pHYs。未复制规范正文、示例实现或增加编码器依赖，PNG压缩沿用浏览器Canvas编码器。来源已生成到共同致谢。

未验证：本轮修改后的真实冻结EXE、macOS/Linux、所有字体/打印软件、超长标签全部排版组合和3200万像素附近持续压力。Chrome为合成组件夹具，未重跑网页账号/PG集成，已有M02报告边界保留。旧Research-Fix1和M10-R5文件均未改变。

开发入口仍为 `scripts/Start-Research-Workbench.ps1`，读取本轮构建的 `frontend/dist`。下一步为审阅 [M03模块计划](../plans/modules/M03-egg-analysis.md)后进入M03-A，不能把规划完成标为算法迁移完成。

最终静态检查：validate_docs.py检查461个文件、327条来源、32项任务，errors=[]；check_architecture.py errors=[]；contracts:check、ui-data:check、typecheck和最终build通过。本轮git diff --check通过（历史c800ce8的空白记录仍保留）。最终Chrome脚本已内置五份PNG独立解码与全部块CRC回读，退出码0。Research-Fix1 SHA256仍为da3a5fe82840e6cb4705e1517cf6a5b9919a61da866cbe78695ca3724122341e。
