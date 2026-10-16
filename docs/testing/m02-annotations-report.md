# 参数显示连续标注重复修复

2026-10-13，状态 verified，范围为 Windows 当前前端源码与独立 Chrome。本轮未重建 EXE。

## 根因与修改

SQLite 多表窗口查询返回每个文字列的时间轨迹 `tracks`。`annotationRuns` 的轨迹分支将每一帧直接作为一段标注，未执行旧行表分支已有的连续同名合并。放大后同一标签反复出现，全长时密集的边界虚线形成横条。用户截图中的波形与参数图都调用这个函数。

`frontend/src/modules/parameter-display/state.ts` 将轨迹与旧行表统一为相同的区间生成逻辑。仅合并时间连续且文字相同的帧，保留文字变化、空白间隔及间隔后的同名文字。裁剪当前时间窗，并排除边缘零长度段。数值曲线、源文件、查询协议和帧时间不变。原始 TextGrid 精确边界仍受既有帧采样与查询抽样限制。

## 验证

| 命令 | 结果 |
| --- | --- |
| `node --test frontend/tests/m02.test.ts frontend/tests/m02-export.test.ts` | 12 项通过；新增用例在修复前失败，确认逐帧重复，修复后通过 |
| `npm --prefix frontend run typecheck` | 通过 |
| `npm --prefix frontend run build` | 通过，保留既有大 chunk 提示 |
| `node tests/e2e/m02-annotations.cjs` | 独立 Chrome 12 组通过，页面错误 0；流式及旧表各覆盖浅深主题、全长、0.64947–0.70012 秒缩放窗及 0.7 秒标签切换边缘；波形和参数图的文字与边界数量符合独立预期，数值曲线仍有路径及有效原值轴范围 |
| 图片导出 | 两类表分别导出当前 SVG 与整幅 PNG；SVG 中 shang4、yi1 各一次，PNG 签名与非空字节通过；全长及缩放截图已人工查看 |
| `python -B -X utf8 scripts/validate_docs.py`、定向 `git diff --check` | 2020 文件／427 来源／107 任务，errors=[]；定向差异检查通过，保留既有历史快照链接提示 |

浏览器夹具只使用内存生成的 8 kHz PCM、4,393 帧及 shang4 / yi1 / 空白 / shang4 标签序列。流式 port 模拟窗口读取与统计数据，不代表真实 SQLite 原生读取。没有读取用户录音或数据库，没有新增依赖与外部来源。

最终证据在 `output/validation/m02-annotations/chrome-1791897901342/`，`report.json` 为 success，12 组记录与截图保留。初轮 `chrome-1791897843380/` 标签检查也通过，但查看截图发现合成流式夹具漏传数值统计，导致曲线超出默认轴范围；补齐夹具统计并增加数值曲线路径与轴范围断言后复验。两轮浏览器独立 profile 与测试导出 scratch 均已关闭后清理，`cleanup.json` 为 deleted。

未验范围：实际 Qt、原生 SQLite 输入、用户截图所用文件、最终 EXE、实体 DPI 与其他平台。未提交、推送或公开发布。
