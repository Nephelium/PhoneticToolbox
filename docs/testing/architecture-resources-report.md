# 资源登记与计算依赖检查验收

状态：Windows 当前源码与静态检查 verified。此页原位维护对应验收范围，任务 ID 为 `ARCHITECTURE-RESOURCE-CHECK`。

## 改动与原因

原检查报告有 1,635 条失败，其中 1,634 条是文件未进入统一资源清单，1 条是 M09 整包导入 cv2。说明书资源被逐文件计数，不能据此推断有同等数量的算法错误。

- 固定资源补齐 5 个字体/许可文件及 2 个公共交互文件。第三方条目关联现有来源，自有交互代码显式记录 project 归属和用途，均固定 SHA-256。
- 资源清单以一条 `manual-reader` 声明关联 `manual` 源工程和 `frontend/public/manual`。检查器根据权威源工程重新计算阅读索引、章节和媒体摘要，与实际输出及 build-report 比较，不手工登记每份生成媒体，也不豁免整个目录。
- 已有输出中的缺失、修改、索引/源稿漂移、多余旧文件、符号链接或 Windows 联接均使检查失败。即使一起修改媒体和输出报告也不能绕过源工程比较。干净检出尚未生成整棵阅读目录时允许缺省。
- M09 使用明确的 OpenCV 数组函数导入：LINE_8、circle、line、getPerspectiveTransform、warpPerspective。保留原函数、参数和计算顺序。整包、通配符、动态导入、摄像头/视频、窗口、文件编解码仍被核心层规则拒绝，编解码继续由 worker 负责。
- software 阅读构建成功写入新索引及报告后，只清上一版共同登记、当前不再使用且摘要未变的生成文件。未知文件和被改动的旧文件报错保留，失败不清旧版本，独立写入暂存正常收尾。public 输出仍拒绝混有旧文件的目标，不自动删除。

## 实际验证

| 检查 | 结果 |
| --- | --- |
| 全库 `scripts/check_architecture.py` | 零错误 |
| 文档检查及 Git 差异格式 | 1,801 文件零错误、配置内 diff --check 通过；36 条未改动历史快照缺链另列 |
| 架构边界、生成资源、阅读导出、M09 数值及 v2 parity | 65 项通过，无跳过 |
| 作者工具 `npm --prefix tools/manual-studio test` | 38 项通过 |
| M09 修改前后固定输入比较 | 6 个数组的形状、类型、SHA-256 完全一致，结果元数据一致 |
| 正式说明书完整性 | 工程索引、20 章及 953 个素材共 974 文件 SHA-256 不变 |
| 旧生成媒体清理 | 当前索引/报告/正文不引用的 packed 345 个、png-safe 313 个，合计 158,139,933 字节已删除 |

测试覆盖媒体损坏/缺失、伪造输出摘要、源稿变化、未知文件、旧媒体被修改、生成失败、替换失败、公开版私有媒体拒绝、路径逃逸、真实 Windows 联接及源工程保护。最初未绑定源码的 pytest 收集误入旧环境，已用下面的明确源码路径重跑。链接用例先受 Windows 符号链接权限限制，随后以本任务创建的真实目录联接完成验证并清理，最终无跳过。

```powershell
$env:PYTHONPATH = "$PWD\packages\phonetic_core\src;$PWD\backend\src;$PWD\desktop\src"
.\.venv\m14\Scripts\python.exe -B -X utf8 -m pytest -c tests/pytest.ini tests/architecture/test_boundaries.py tests/architecture/test_generated_resources.py tests/test_manual_reading_export.py packages/phonetic_core/tests/test_m09_editing.py tests/parity/test_spec2wav.py -q
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/check_architecture.py
npm --prefix tools/manual-studio test
```

详细本机摘要、删除前清单、回执和命令输出位于 Git 忽略的 `output/maintenance/architecture/`，当前架构结果在 `output/maintenance/architecture-check.json`。测试输入为临时小文件或内存数组，结束后清理，没有新增长期 WAV 或测试工程。

## 验收边界

静态依赖检查不充当安全沙箱，也不代替来源许可、科研自然语料准确性、实体设备或发布验收。本次未更改科学公式/输出协议、安装依赖、迁移数据库、打包 EXE、公开发布或 push。旧 EXE 保持原有快照，当前 M09 源码仅导入形式改变且数值比较一致；完整 GUI、实体设备、其他平台和长期运行没有重验。
