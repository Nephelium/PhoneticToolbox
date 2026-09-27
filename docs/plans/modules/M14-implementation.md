# M14 实施记录 · 2026-09-27

状态：in_progress。已完成的限定 Windows/核心及未完成的平台接口详见 [验收报告](../../testing/m14-report.md)。用户已授权完整迁移及限定平台验收。仅公开合成输入。根目录历史统筹停止点由本轮 M14 明确授权覆盖。

## 阶段和文件

1. A：V2 第 12.1–12.2 章、parser/service/models/widget 的独立捕获。`tests/support/m14_baseline.py`、`tests/fixtures/m14/`、`docs/modules/evidence/M14-source-map.md`。
2. B：`packages/phonetic_core/src/phonetic_core/transcription/phonology/` 纯解析、归并、排序和内存文档生成。文件解码在 `backend/src/ptb_worker/m14_import.py`。先原样拆分，再独立修复输出一致性。
3. C：`backend/src/ptb_api/m14_models.py`、`backend/src/ptb_worker/m14_{jobs,child,task}.py`、`desktop/src/ptb_desktop/m14_bridge.py`、`frontend/src/modules/phonology-induction/{PhonologyInductionPage.vue,state.ts,port.ts}` 及模块平台适配器。正式宿主接口需待共享文件占用释放后串行接线。
4. D：Windows 正式工作台、现有 WSL 原生 Python、结构回读与视觉检查。`tests/parity/test_phonology_induction.py`、`frontend/tests/m14.test.ts`、`tests/e2e/m14.cjs`、`scripts/verify_m14_*.py`、`docs/manual/phonology-induction.md`、`docs/testing/m14-report.md`。

## 接线需求（Windows 已串行完成，Linux 仍依 P11）

- AppShell：M14 async import、真实 pane、dirty 与 save 方法；不显示音频播放器。
- ResearchFiles：可选 m14 port，授权文件 ID／字节传递，前端不接触路径。
- desktop task bridge／API：模块请求分派，持久输入和三结果复用既有文件存储、owner/project、配额和到期。
- P11：固定 `m14` child entry 与 `phonology_induction` operation；M14 不修改公共资源/进程代码，交付请求/结果协议给负责人。
- 若共享执行器尚未交付接口，完成核心、页面和模块适配并明确正式链路未验证，不以测试 adapter 代替。

## 基准与修正边界

- V2 parser 的元音集合包含 Klatt 表中的 ɹ/ɻ，保留其单辅音优先和扫描顺序，不重新解释。
- GUI 默认跳首行（默认按钮为“是”），底层 load_rows 默认 False。UI 保留 True，服务 API 显式传值。
- 常见表头始终过滤、缺失行跳过、重复行保留。新增可审阅跳过统计，不悄悄清洗。
- 调值拖动是交换两行，声韵拖动是移动选择块。调类为空回退原调值，同名调类归组。
- FIX01：XLSX 富文本分支忽略调值排序，独立修正为与 DOCX 相同顺序。
- FIX02：V2 空韵无法作为归并源/目标，明确修正空字符串判定，使空韵也可选择，仍禁止自归并及循环。
- FIX03：保存不得无提示覆盖同名结果，三个输出完整提交后才显示成功。
- DOCX/XLSX 字体接提交时公共快照，IPA 固定 Doulos SIL；与 V2 Times New Roman 单列呈现差异。

## 依赖与规模

现有工作台 pandas 2.3.3、openpyxl 3.1.5；V2 python-docx 1.2.0，V3 缺失；两者均缺 xlrd。提议项目专用环境 python-docx 1.2.0、xlrd 2.0.2，测试 xlwt 1.3.0。井井已批准项目内隔离安装，Windows/WSL/授权服务器均落实；lxml 6.1.3、typing-extensions 4.16.0 锁定于 additions.lock。禁止全局安装和改变 V2。

规模门由公开合成输入实测后冻结，读入和导出均不在 API 事件循环内执行。导出复杂度同时受记录数和声韵笛卡尔积约束。

## 原定验收命令（实际执行和完整路径见验收报告）

```powershell
& .venv/m14/Scripts/python.exe -m pytest -c tests/pytest.ini tests/parity/test_phonology_induction.py backend/tests/test_m14.py -q
node --test frontend/tests/m14.test.ts
npm --prefix frontend run typecheck
npm --prefix frontend run test
npm --prefix frontend run build
node tests/e2e/m14-host.cjs
```

另记录实际 Qt 宿主、WSL 核心/导出、Windows 浏览器访问 Linux 服务与 Linux 原生浏览器的不同证据；DOCX/XLSX 回读不替代视觉检查。全局台账由统筹汇总本模块状态摘要。
