# M14 工作区、Word 预览与二维表

2026-10-11，`M14-R2` 在当前 Windows 源码、独立 Chrome、真实 Qt 和重建的说明书阅读资源范围内为 `verified`。本次未重新打包 EXE，未执行现存库迁移或外部发布。

## 改动与验收对象

- 四步中央区使用剩余高度。样例表、调类表、声韵列表内部滚动，底部操作留在当前工作区。文本编码和分隔符移至左侧导入设置，避免矮窗口中主表单挤出确认按钮。
- 两份 DOCX 预览沿用导出文档的居中标题、按列排列的双栏统计表、前五个不同字头／备注组合、连续分类正文和备注下标。同一外层分类不再逐组合重复标题。生成后保存同次导出字体快照。屏幕每批 100 条，实际 Word 纸张分页仍以文件为准。
- 二维表按可见窗口预计算单元格切片，取消逐条记录重复截取列表；缓存未变格，分帧呈现行。固定行列标题，长格内部滚动。维持 12 行×10 列窗口与每格每批 60 条，全表搜索可定位末尾格和格内末页。
- 说明书本章同步操作正文与五张真实 Qt 图。五图均为原生 DPR 1.5、浅色、页面 100%、最大化、2560×1600；原图 2561×1601 时仅裁右下各一像素。正式旧素材保留，新图仍为 software-only，不进入公开 Git。

分类、调类顺序、归并、重复记录、原始 IPA、备注和三文件导出算法未修改。

## 实际检查

| 检查 | 结果 |
| --- | --- |
| `npm --prefix frontend run typecheck` | 通过 |
| `node --test frontend/tests/m14.test.ts` | 6 项通过，新增统计例项、备注身份、归并与双栏次序检查 |
| `.venv/m14/Scripts/python.exe -B -m pytest -c tests/pytest.ini tests/parity/test_phonology_induction.py backend/tests/test_m14_r1.py -q -p no:cacheprovider --tb=short` | 当前源码 PYTHONPATH 下 51 项通过 |
| `node tests/e2e/m14-r1.cjs` | 原有完整流程回归通过；5 种既有格式、DOCX 表选择、GB18030、列映射、多选排序、归并撤销、三预览、搜索、三文件保存冲突、旧结果快照、草稿恢复、32 组窗口/主题 |
| `node tests/e2e/m14-r2.cjs` | 5452 条合成字表、10000 条单格字表；22 组按钮和中央区几何检查通过，另有 6 组二维表窗口/主题截图；同格连续搜索、末条搜索和 167 页格内分页通过；实际阅读器打开 M14，更新正文及五图解码通过 |
| `.venv/m14/Scripts/python.exe -B -X utf8 scripts/manual/capture_m14_r2.py --source <工程内现有测试字表副本>` | 真实 Qt 保留 5453 条记录、成功生成三文件、两种预览及矩阵搜索通过，源 SHA-256 未变 |
| `scripts/manual/register_m14_r2.py --report <Qt report.json>` | 更新本章五图和五处正文，稳定章节/图 ID 保留 |
| `scripts/manual/validate.py --project manual --strict` | 零错误、零警告 |
| `scripts/manual/build.py --project manual --output frontend/public/manual --distribution software` | 阅读资源生成通过 |
| `scripts/check_architecture.py` | 零错误，源章节、阅读资源、媒体摘要一致 |
| `npm --prefix frontend run build` | 前端构建通过，保留既有大 chunk 提示 |

R2 导入按钮几何检查覆盖 800×500、1366×768、1920×1080、2560×1440，以及 1920×1080 的 150% 页面缩放，各含浅深主题。调类和声韵底部操作检查覆盖后三种窗口及浅深主题。页面缩放使用产品的 zoom 与 `--page-scale` 配对设置。

### 性能比较

同机独立无头 Chrome、相同 5452 条合成字表、相同当前宿主，分别切换八次。基线由 Vite 在内存加载 `HEAD` 中的原 `ResultPreview.vue`，没有覆盖工作树或修改 Git 历史。

从点击二维表到第二个绘制帧的中位耗时，原预览为 **25.10 ms**，本次为 **13.75 ms**。这是首屏响应时间，当前方案继续分帧呈现后续行，不能据此宣称全部单元格完成时间、自然语料准确率或任意电脑性能。后续完整验收再跑一次得到 11.75 ms，不与另一轮基线拼接扩大结论。

## 证据位置

均在本机忽略的 `output/validation/m14/` 下：

- 完整 R2 与阅读器：`r2-55fdca4743334c6b823f3536857127c5/browser-report.json`。
- 配对性能：`r2-0dae8756082b43e0b1b0d0e81435cd7a/browser-report.json`、`r2-53b4551d8e7c48f8a3fa323b28a486fd/baseline-report.json`。
- R1 回归：`r2-8a8238437ffc4fba86cd2978fcf9eaec/r1-browser-report.json`。
- 最终说明书图：`r2-manual-qt-a8fc306440cc42fa842c2e1a26153edd/report.json`，正式媒体登记在 `manual/project.json`。

初次 800×500 检查暴露文本选项占用过多高度，已移入左侧并重新通过。验收脚本的标签匹配、搜索后的矩阵页位置和双调类单格的预期页号也曾导致失败，分别修正定位与测试输入，保留失败报告，没有放宽产品断言。新验证入口将可重建输入、隔离库、缓存和导出放入 scratch，服务退出后清理，截图与报告保留。Python 退出前的清理因子进程生命周期保留，关闭所有本次进程后按既有 PowerShell 清理入口重试，11 个本次隔离 scratch 的最终 `cleanup.json` 均为 deleted，目录已不存在；源字表、正式素材、报告及截图保留。

## 未验证范围

最终 EXE、其他实体电脑、完整物理 DPI/IME 矩阵、Linux GUI、远程联合任务和长期压力未在本次检查。Word/WPS 实体程序没有重新打开，预览按既有 DOCX 导出代码及结构测试对齐，实际纸面分页不作逐像素一致承诺。本次 Qt 使用工程内已有作者字表，调类名称为操作暂拟，未据此验证自然语料的音位分类。
