# M01-D 契约与科学结果语义验收

2026-09-09至10，Windows，状态 **verified（限定共享协议、输入策略与真实结果往返）**。井井在M01-C后回复“好，请继续”；本轮在`codex/v3-rebuild`的`b03985c`之后实施。完整M01/P08仍in_progress，下一项M01-E。具体方案见[契约计划](../plans/2026-09-09-m01-contract.md)。

## 完成范围

- 新增`acoustic_models.py`：14设置、80目录键与显示名、显式legacy_service模式、实际后端、解码元数据、配置hash、逐列单位和nonfinite mask、完整单文件双产物及全列表批次汇总。`acoustic_boundary.py`校验可信资源owner/project/ready/hash/类型/期限，拒绝同名关联歧义，定义分析与切分的不同TTL。
- `ptb_worker/acoustic_result.py`核对真实核心配置及采样率，保留数值/时间/列顺序/标签；NaN和正负Infinity用null与不同mask表示，原因仅记legacy_nonfinite_unknown。未证实的异常文本不会进入协议。SOE两个额外参数仅在显式服务兼容模式出现。
- 纯目录移动到`phonetic_core/catalog.py`，旧路径重导出。最初直接导入acoustic/catalog会连带加载NumPy，实际轻量API环境因此无法启动；修正后该环境无需科学依赖即可生成全部OpenAPI。目录内容与上一提交原目录按LF归一后相同，旧导入返回同一映射对象，80标签/单位和14默认值逐项对照独立A审计。
- API版本1.1.0，科学schema为m01/1，算法版本legacy-numeric/1、适配版本m01-adapter/1。旧HTTP路径完全相同；旧components除Health/Capabilities的api_version默认值随版本更新外均相同。未增加科学任务HTTP入口或数据库迁移。

输入校验差异、缺失语义、TTL和兼容边界分别记录为[D01–D08](../../contracts/versions.md)。首次7项红测缺少模型失败；进一步测试发现Literal把bool转换成mask整数，改为转换前严格校验，未降低测试标准。

## 实际验证

科学环境为`.venv/m01-io`，CPython3.11.14；沿用C的40个第三方包，无新增第三方依赖。构建后重新安装core和API两个wheel，实际导入路径均为该环境site-packages。工程API检查使用原`.venv/v3-dev`，该环境没有NumPy，保持可运行。`uv pip check`确认42个安装包兼容。

| 验证 | 实际结果 |
| --- | --- |
| 安装wheel后的核心/格式/架构/契约测试 | **318 passed**：原C范围222项、原契约19项、新D协议68项及转换9项 |
| 工程API/契约/旧任务与存储边界 | **101 passed**，含与上述重叠的87项契约，不合计为419项独立测试 |
| API无需科学依赖导入 | 独立解释器断言NumPy和acoustic包未导入；两模式OpenAPI相同 |
| Schema/OpenAPI/TypeScript与版本生成 | 生成后检查无漂移；前端typecheck通过 |
| 架构、来源、文档、Git diff | 定向检查通过，来源仍323条 |
| 保存性 | v2相关7项前后相同，427份基线检查结果相同，36份golden hash不变 |

工程API测试出现现有Starlette/httpx和AnyIO别名两条弃用提示；未升级工程依赖。JSON Schema与TypeScript不执行跨字段约束，不能替代Pydantic或科学黄金对照。

主要命令（路径均相对项目根，解释器使用对应环境）：

```text
.venv/m01-io/Scripts/python.exe -m build --wheel --no-isolation --outdir output/validation/m01/d-wheels packages/phonetic_core
.venv/m01-io/Scripts/python.exe -m build --wheel --no-isolation --outdir output/validation/m01/d-wheels backend
uv pip install --python .venv/m01-io/Scripts/python.exe --no-deps --reinstall --link-mode copy output/validation/m01/d-wheels/phonetic_core-3.0.0a1-py3-none-any.whl output/validation/m01/d-wheels/ptb_api-3.0.0a1-py3-none-any.whl
.venv/m01-io/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini packages/phonetic_core/tests tests/parity backend/tests/test_m01_io.py backend/tests/test_m01_result.py tests/security/test_m01_formats.py tests/architecture tests/contracts -q --junitxml=output/validation/m01/d-wheel-final.xml
.venv/m01-io/Scripts/python.exe -X utf8 scripts/verify_m01_contract.py
.venv/v3-dev/Scripts/python.exe -m pytest -c tests/pytest.ini tests/contracts backend/tests/test_api.py backend/tests/test_job_boundary.py backend/tests/test_storage_boundary.py backend/tests/test_job_policy.py backend/tests/test_storage_policy.py -q --junitxml=output/validation/m01/d-api.xml
.venv/v3-dev/Scripts/python.exe scripts/generate_contracts.py --check
.venv/v3-dev/Scripts/python.exe scripts/sync_versions.py --check
npm --prefix frontend run contracts:check
npm --prefix frontend run typecheck
npm --prefix frontend run ui-data:check
.venv/v3-dev/Scripts/python.exe scripts/check_architecture.py
.venv/v3-dev/Scripts/python.exe scripts/validate_docs.py
git diff --check
```

### 安装wheel后的真实数值与产物

固定合成WAV实际解码为44100 Hz。使用现有哈希固定的REAPER，实际记录native_reaper及irapt1，然后序列化、JSON反序列化、恢复mask并逐数组核对原结果，再与独立M01-A冻结样例比较；每个双产物由C适配器完整回读。服务兼容例确实保留SOE_pF0/SOE_rF0。

| 例 | 行×列 | 冻结对照差异 | 紧凑JSON字节 | XLSX / SQLite字节 |
| --- | --- | --- | --- | --- |
| ASSOCIATED | 160×83 | 0 | 335,179 | 147,159 / 139,264 |
| FORMULA-EXPORT | 160×83 | 0 | 335,339 | 147,207 / 139,264 |
| SERVICE-NONE | 160×79 | 0 | 329,787 | 143,975 / 118,784 |

最终原始证据在忽略目录`output/validation/m01/contract-02451bec5c5c47659c86a58eda3b1279/`，各例含result.json、manifest.json与两种结果文件；构建包在`d-wheels/`，测试XML及`preservation-d.json`在同级。摘要和代码hash见[本阶段证据](../modules/evidence/M01-contract-migration.json)。历史B/C源码及依赖清单保留，不回写成D状态。

## 限制与下一步

当前批次是协议和计数校验，尚无持久批次实现。owner/期限校验的可信回调测试不是实际PG竞态验收；引用到期自动取消、租约/fencing、配额、结果原子发布和物理删除仍由F接入并实测。没有修改已有数据库或用户结果。JSON结果有行/单元格/文本预算，但全科学worker硬隔离仍待F。

下一步M01-E是桌面目录句柄与共同研究页：完整80参数、14设置、文件列表、关联与试听状态；全列表计算与持久结果需F/G后才能宣称完整M01完成。本轮未进行页面视觉验收、跨平台构建或公开发行。

Git根仍为D盘v3目录，未push或上传用户目录；未操作或关闭Codex内置浏览器。此次工作未观察到闪退，但未修复Codex应用本身，也不保证不会再次退出。
