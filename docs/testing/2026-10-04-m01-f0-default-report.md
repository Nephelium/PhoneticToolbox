# M01 基频默认范围调整

日期：2026-10-04。状态：verified，限定默认设置、配置传递与 Windows 开发态界面。

井井要求参数估计中 REAPER F0 默认范围改为 30–800 Hz。实际原默认值为 60–880 Hz。

## 修改

- `AcousticSettings` 的默认上下限改为 30 与 800 Hz，生成 OpenAPI、JSON Schema 和 TypeScript 契约。前端直接读取此契约，新草稿与省略配置的请求均使用新值。
- 现有最小/最大基频由 Praat、REAPER、WM 链与部分依赖基频的指标共用。没有新增独立算法范围或修改算法实现。
- 有效的已保存草稿、显式请求参数和历史任务保留原值。旧草稿需要在 REAPER 设置页调整后应用并保存。
- 原 V2 参数审计和 core 独立默认值保留。M01 正式任务通过 `to_core_config` 显式传递完整设置；本次默认值与原 V2 审计的两项差异在契约回归中明确列出。
- 更新[参数估计手册](../manual/parameter-estimation.md)。

## 验证

使用既有 Windows Node/npm 与 `.venv/m09-ui/Scripts/python.exe`，未安装环境或依赖。

| 检查 | 结果 |
| --- | --- |
| 前端 | 类型检查、266 项现有测试、生产构建通过，M01 状态测试 14 项包含在其中。构建保留既有大 bundle 提示。 |
| 后端与契约 | `pytest -o addopts='' tests/contracts/test_m01_contract.py backend/tests/test_m01_result.py backend/tests/test_review_reaper_policy.py -q`，84 项通过。 |
| 生成契约 | `scripts/generate_contracts.py --check` 无漂移，`npm --prefix frontend run contracts:check` 一致。 |
| 配置传递 | 实际创建省略 config 的请求，其设置与传入 core 的设置均为 30/800；显式历史 60/880 序列化后仍传入 60/880；原 core 独立默认值仍为 60/880。 |
| Chrome | 新草稿 REAPER 页实际显示 30/800，浅深色截图均读取新值；应用后提交全列表时状态快照保留 30/800；重新加载有效历史草稿恢复 60/880。三组通过，无页面异常。使用既有 M01-R3 合成文件/任务夹具，没有运行科学提取。 |
| WSL | NInfer 静态读取模型、请求 schema 与前端 state 三文件，SHA256 与 Windows 一致。未验证 Linux 科学任务或 GUI。 |
| 差异 | 定向 `git diff --check` 通过，同期工作树修改保留。 |

Chrome 证据：[report.json](../../output/validation/m01-f0-default/chrome-1791055801800/report.json)、[浅色](../../output/validation/m01-f0-default/chrome-1791055801800/settings-light.png)、[深色](../../output/validation/m01-f0-default/chrome-1791055801800/settings-dark.png)。首次临时检查脚本试图拦截 Vite 转换后的 HTML 源文本，匹配失败；改为直接核验实际页面及状态后通过，产品源码未因此改变。

本轮未用嘎裂自然语料评价检出率，不据此承诺检出效果。未重打 EXE、push、发布、DDL、修改系统环境或删除文件。
