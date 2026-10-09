# M14 说明书真实字表示例与完整窗口截图

日期：2026-10-06。状态：`verified`，限定 Windows 本机说明书源工程、生成阅读资源及实际 Qt 阅读器。

井井要求音系归纳章节改用测试.xlsx 举例，并把本章所有截图设为最大化全应用窗口，无法最大化时采用 2560×1440。调类允许暂拟，须明确仅作展示。本轮完成 M14 正文、全部 17 张图及阅读资源更新。

## 字表与展示边界

- 原表 Sheet1 共 5,453 行、3 列，无表头、无公式。字头、IPA、备注依次为第 1、2、3 列，从第 1 行开始导入。
- 真实 Qt 操作纳入全部 5,453 条记录，910 种原始 IPA、34 个声母与 44 个韵母。11 条重复保留，未跳过记录、未产生符号警告。该结果只证明本表在当前规则下完成整理流程。
- 原表 SHA-256：`a4696b9399aeb67134111c795d0c8d2230c6f13ce0c8e12b708a9e9dbb6f1501`，执行前后逐字节身份一致。
- 六个编码分别使用 1→阴平（暂拟）、2→阳平（暂拟）、5→阴去（暂拟）、6→阳去（暂拟）、7→阴入（暂拟）、8→阳入（暂拟），对应记录数为 1,933、1,150、676、807、489、398。没有补造编码 3、4 或上声类别。
- 正文开头、调类操作、结果与方法边界明确写出当前例子仅作操作展示，不代表真实调类。尾部数字按源表编码理解，未认定为实测五度值。
- 列映射 CSV 为原表全部记录的 UTF-8 派生副本，另加说明行、表头与编号列；分号 TXT 摘录原表前三条。正文明确区分派生文件与原 XLSX。
- 归并操作用原表第 3,193 行捧 phoŋ1 演示 ph→pʰ，影响 1 条记录。生成前撤销此映射，最终仍保留 34 个声母。该操作没有被解释为已证实的音位归并。

## 全部 17 张截图

实际 Qt 隐藏测试窗口的最大化尺寸无法达到指定目标，因此按用户授权采用完整客户区 2560×1440，DPR=1。每张图均保留全局导航、标签栏、模块页和状态栏，窗口、网页视口及 PNG 原图尺寸全部一致，无局部裁剪。使用浅色主题，字体加载就绪后抓图。

截图涵盖四步总览、原 XLSX 导入、列映射冲突、原编码与暂拟调类、声韵排序、多选、归并确认及撤销前状态、生成结果、两种同音字表、二维声韵表、全表搜索审阅、同名保存拒绝，以及两份派生文本的导入设置。

矩阵和同音字表仍按产品现有分组、分页及内部滚动展示。完整窗口能显示更多内容，不表示单图显示全部 5,453 条记录。说明书同时解释每页 100 条的同音字表、12×10 的矩阵窗口和全表搜索。

保留 31 个标题和 17 个图节点的稳定 ID，替换图节点的素材引用。新图登记为 `software-only`、`git:false`，私有素材路径不进入公共来源记录；旧素材与先前候选均保留。

## 实际验证

| 验证 | 结果 |
| --- | --- |
| 真实 Qt 全表导入、编辑、演示归并与撤销、三文件生成 | 通过 |
| 保存、同名拒绝、另目录重试，三文件 SHA 回读 | 通过 |
| 两份 DOCX 与一份 XLSX 的独立字项 Counter 回读 | 各 5,453 条，逐字及重复频数与源表完全一致 |
| 17 张 PNG 的窗口/视口/原图尺寸及 SHA | 全部 2560×1440，通过 |
| 严格说明书校验 | 19 章、447 素材，0 错误；仅保留 M10 deferred 的两条既有提示 |
| 生成的源工程/public/dist 中 M14 正文与 17 张新图逐文件 SHA | 一致 |
| 前端生产构建 | 通过，保留既有大 chunk 提示 |
| 实际 Qt 阅读器 | 全部 17 张原图加载；浅深色帮助入口与返回通过 |
| 实际 Qt 二维表图片放大/关闭 | 原图 2560×1440，图注正确，通过并回看 |
| 非目标说明书与 M14 产品文件 | 其他 18 章字节不变，23 个 M14 产品源码文件与开始快照一致 |

执行入口：

```powershell
& '.venv/m14/Scripts/python.exe' -B scripts/manual/capture_m14_test_table.py --source (Join-Path $env:USERPROFILE 'Desktop\PhoneticToolbox\测试.xlsx')
& '.venv/m14/Scripts/python.exe' -B scripts/manual/verify_m14_test_exports.py --capture-report 'output/manual-work/m14-test-table/4aa289250c114d5ab6ef7054f05961f0/report.json'
& '.venv/m14/Scripts/python.exe' -B scripts/manual/register_m14_test_table.py --capture-report 'output/manual-work/m14-test-table/4aa289250c114d5ab6ef7054f05961f0/report.json'
& '.venv/m14/Scripts/python.exe' -B scripts/manual/validate.py --project manual --strict
& '.venv/m14/Scripts/python.exe' -B scripts/manual/build.py --project manual --output frontend/public/manual --distribution software
npm --prefix frontend run build
& '.venv/m14/Scripts/python.exe' -B scripts/manual/verify_m14_test_reader.py --output 'output/manual-work/m14-test-table/4aa289250c114d5ab6ef7054f05961f0/reader-review'
```

独立作者审计：`output/manual-work/chapter-audit-m14-test-table-20261006.json`。

最终完整证据目录：`output/manual-work/m14-test-table/4aa289250c114d5ab6ef7054f05961f0/`，含 `report.json`、`record-count-readback.json`、`manual-validation.json`、`preservation-readback.json`、`reader-review/reader-report.json` 与可直接浏览的 `音系归纳说明书预览.html`。HTML 使用同一正文和 17 张完整原图，点击可查看原图，作为应用阅读器之外的补充预览。

初始候选的两张文本导入图因脚本用了不存在的编码选项而显示空下拉。已改用实际 `utf-8-sig` 选项、增加选项存在断言并重拍全套，最终图显示 UTF-8。阅读器初次验收脚本漏算章号 17，已修正标题判断后通过；失败日志保留。

## 保留与未验证范围

本轮修改说明书和截图工作脚本，未改变 M14 解析、科研计算及用户原字表。快照比对发现同期桌面宿主、展示窗口和声道文档三个非 M14 产品文件有其他工作改动，未覆盖或回退，具体文件见 `preservation-readback.json`。

未重新打 EXE，现有交付包尚未包含本轮说明书修改。未验实体 2560×1440 显示器最大化、DPI/DWM、Office 纸面分页、跨设备字体、自然语料分类准确率或暂拟调类真实性。Qt 使用隐藏隔离窗口与测试进程的绘图设置，终端仍有既有 PNG profile/GPU context 提示，验收限定于实际成功的正文与图片呈现。未执行发布、上传、push、现存数据库变更、全局安装或用户文件清理。
