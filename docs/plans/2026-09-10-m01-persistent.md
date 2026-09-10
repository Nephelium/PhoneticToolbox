# M01-F2 持久批次、执行和保存

状态：verified（限定报告列明的Windows持久任务和双端实际操作）；完整M01/P08仍in_progress，后续G见[报告](../testing/m01-persistent-report.md)。井井在005具体审阅和限定合成测试/清理请求后回复“好，继续”，本轮据此应用两份已固定hash的SQL；不再重复请求相同授权。

## 实施顺序与文件

1. `scripts/run_m01_validation.py`：只启动/停止自己持有的固定本机测试PG；调用已审阅迁移工具，在两库事务内验证旧行保留。记录每库成功/失败，不能把局部成功描述为两库成功。
2. `backend/src/ptb_worker/acoustic_batches.py`、`acoustic_executor.py`、存储/任务适配：持久顺序快照、提交幂等、取消和恢复；复用P06子任务/租约/代际校验，F1切分和C双格式导出。空间先预留，完成后原子发布；参数计算不能阻塞API和续租。
3. `backend/src/ptb_api/`相关模型/接口、`contracts/`生成物、`desktop/`目录发布和宿主接入、`frontend/`共享M01页：真实切分与批次操作、进度/取消/部分失败、结果刷新。继续使用同一宿主，不新增模块服务或科学算法实现。
4. 真实PG/SQLite与合成文件验证17项、有失败的批次、取消/重启/租约过期、来源变化、限额和旧文件保护；按实际实现逐项记录证据。单轮子步骤通过不冒充整个M01/F/G完成。

## 验证入口

- `scripts/run_m01_validation.py --approved-m01-schema-and-synthetic-tests --apply-reviewed-schema`只首次应用005；后续验证不得重复执行DDL。
- `.venv/m01-ui/Scripts/python.exe -X utf8 -m pytest -c tests/pytest.ini`配合新增定向测试和受影响P06/P07回归；受控真实库/文件结果另外回读。
- 按需重建并安装core/API/desktop wheel到项目内m01-ui；API在轻量v3-dev中验证不导入科学计算。
- 契约生成/检查、前端typecheck/test/build、架构/文档检查，新增实际UI路径由独立Chrome及Qt验证，避开Codex内置浏览器。

仅操作005审阅列明的现存测试库与新UUID合成资源。原有账号/项目/任务、v2目录和全局环境保持原样；按清单本地提交，不push。
