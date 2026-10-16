# 栏目说明删除与往期找回

日期：2026-10-15。范围：Windows 当前源码与隔离可见 Qt WebEngine 默认 GPU。状态：`verified`，限定于下述源码及 Qt 范围。

## 行为与实现

- 请看这里~ 支持右键或 Shift+F10 打开删除本机副本菜单，并以删除栏目说明窗口确认。取消保留页面及文件。
- 删除复用 `removedRecommendations` 偏好，移除该说明的缓存并隐藏置顶入口，批注按 PDF 摘要保留。刷新目录、重建宿主服务及启动检查均尊重删除状态；服务端同 ID 更新也不会自动恢复。
- 获取往期按原发布日期列出栏目说明，支持年月筛选、勾选及重新获取。只下载单份原始 PDF，预计大小不重复计算译文。已获取条目隐藏且禁选，主目录推荐论文数量继续排除栏目说明。
- 下载完成写入前复核删除版本，阻止正在进行的自动或手动获取恢复刚删除的说明。共用摘要缓存仍为其他未删除条目保留，偏好写入失败恢复文件原位。
- 使用现有平台接口与偏好格式，无数据迁移。README、说明书正文、生成阅读资源和台账同步更新。

## 验证

桌面定向检查使用工程现有 `.venv/m14`，先经 `workbench_source.configure()` 绑定当前源码，再运行：

```powershell
.\.venv\m14\Scripts\python.exe -B -X utf8 -c "import sys; sys.path.insert(0,'scripts'); from workbench_source import configure; configure(); import pytest; raise SystemExit(pytest.main(['-c','tests/pytest.ini','desktop/tests/test_paper_reader_updates.py','desktop/tests/test_papers.py','desktop/tests/test_paper_annotations.py','desktop/tests/test_paper_library.py','-q']))"
```

60 项通过。新增覆盖删除后重启与显式取回、关闭自动获取时说明独立下载、服务端内容修订保持删除、自动及手动下载期间删除、共用摘要双向保护和偏好写入失败回滚，既有论文与个人 PDF 库回归通过。初次直接 pytest 未绑定工程源码导致导入失败，经统一入口绑定后通过。

前端在 `frontend` 目录执行 `node --test tests/papers.test.ts tests/paper-citations.test.ts`，8 项通过；`npm run typecheck`、`npm run contracts:check`、`npm run ui-data:check`、`npm run build` 通过。构建保留既有大分块提示。

说明书执行 `scripts/manual/validate.py --project manual --strict` 与 `scripts/manual/build.py --project manual --output frontend/public/manual --distribution software`，20 章、1053 素材，0 警告/错误；生成资源后再次构建前端通过。

Qt 入口：

```powershell
.\.venv\m14\Scripts\python.exe -E -s -B -X utf8 scripts/verify_m18_notice.py
```

隔离的当前 Workbench 与真实 WebChannel/PaperService 使用本机已有说明 PDF 和获许可论文的测试副本。浅深两主题已通过键盘取消、右键确认删除、其他论文继续阅读、原生服务重建与页面重载、启动仅读目录、年月选取、只下载说明单份文件、笔记保留和已获取禁选。置顶页恢复后仍不增加论文计数。确认及往期截图已人工查看，控件、文案可读。

首轮脚本在页面重载完成前读取旧 DOM，导致获取往期按钮仍处于旧请求禁用状态。脚本已改为等待 `loadFinished` 和实际按钮可用，完整两主题复核通过，产品逻辑未为该问题添加等待或绕过。

## 证据与边界

最终证据：`output/validation/m18/notice-qt-d948c851c0b1417ba350461d5f5a52cd/`，两主题主流程、删空列表后无残留阅读器、只恢复说明且论文计数为零，以及更新后模块帮助正文读取全部通过。首次主流程成功证据为 `output/validation/m18/notice-qt-0003f0a55d3e48daaea789de4dcc36ec/`。脚本诊断证据为 `notice-qt-b4552efc4a2d4fa59c9c817933bfc232` 和 `notice-qt-368cf8208e9b47139febb2697988748e` 两轮，后者为附加空态准备从主线程直接操作 PDF 文档导致文件占用；脚本已改为在序列化阅读线程关闭旧文档，并经真实菜单删除。最终全部检查通过，产品文件占用错误按预期保留原位。各轮 `cleanup.json` 均为 `deleted`，保留报告、截图、输入摘要与配方，测试论文库及临时配置已清理。

`scripts/validate_docs.py` 检查 2043 文件、436 来源、110 任务，0 错误。定向 `git diff --check` 通过。

本次验证使用隔离测试数据，不删除井井现有论文或说明页。未构建 EXE、发布应用或修改服务器目录，其他设备和跨平台未验。
