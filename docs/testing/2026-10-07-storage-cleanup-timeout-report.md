# 本机缓存清理请求取消超时

2026-10-07。井井授权取消本机缓存清理请求的时间限制。状态：源码及前端构建 verified；冻结 EXE 未重打、未运行验收。

## 改动

- `desktop/src/ptb_desktop/local_service.py` 对 `/api/v1/jobs/local-storage/cleanup` 显式使用 `timeout=None`，持续等待服务响应。其他请求保留现有超时。
- `frontend/src/platform/desktop.ts` 对 `local_storage_cleanup` 不创建超时计时器，串行队列等待原生完成或真实错误后继续。完成后的占用刷新沿用已有流程。
- 清理策略、期限、范围、文件保护和后端删除实现未修改。保留同期改动。

## 验证

- 通过 `.venv/m14/Scripts/python.exe` 与 `scripts/workbench_source.bind_sources()` 绑定本工程源码，使用 `tests/pytest.ini` 执行 `desktop/tests/test_local_service.py`：3 passed。新增回归使用临时 loopback HTTP 服务延迟 5.2 秒才返回，验证清理仍能接收完整响应；状态请求保留 5 秒超时。已有双实例隔离与正常退出通过。
- 前端定向清理与串行队列检查：4 passed。虚拟计时推进 24 小时后，清理仍接收完成或真实错误，随后执行排队的占用刷新；状态请求保留 5 分钟超时。24 小时为虚拟计时测试，不是实际运行时长。
- `npm --prefix frontend test`：337 passed。
- `npm --prefix frontend run typecheck`、`npm --prefix frontend run build` 与目标文件 `git diff --check` 通过。前端构建已重新生成，保留既有大 chunk 提示。

本轮未对用户缓存再次执行清理，未修改用户设置或数据库，未打包 EXE。未检验当前冻结应用、GUI 或跨平台实机；当前正在运行的旧 EXE 不包含本轮修改。
