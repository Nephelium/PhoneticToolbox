# 免安装版缓存位置与旧缓存提示验收

2026-10-18，Windows 本机当前源码，对应 `PORTABLE-CACHE-LOCATION`。设计见[计划](../plans/2026-10-18-portable-cache-location.md)。本轮未重建 EXE、未发布、未操作既有用户缓存或现存数据库。

## 交付行为

- 首次免安装启动在展开大组件前要求选择位置，取消即退出，失败不回落到隐藏默认目录。专用 `PhoneticToolbox-cache` 保存运行、更新下载/辅助文件和普通浏览器缓存；位置记录不绑定 EXE 名称、版本或目录。
- 更新偏好与下载目录分开。设置、草稿、研究工作区、录音、论文批注和 MFA 组件保持原位置；删除缓存后再打开仍可读取位置记录并重新准备。
- 按井井追加确认保留 Qt 绑定持久目录的 GPU/Dawn 图形缓存，设置单列实际路径。核对后只安排所选图形缓存，经过未保存/录音收尾和共享 profile 租约，在最后一个使用窗口退出后清理；失败保留请求重试，不清研究结果。
- 设置后台有界检查应用已知旧位置及已记录专用目录，显示路径、文件大小合计和打开核对。必须勾选亲自核对，执行前再查归属和占用；未知内容、联接和活动实例保留。没有全盘扫描或发现后自动删除。
- 安装版保持原目录，安装标识提前到预热之前写入。自动成品 QA 改用独立 profile 的真实位置偏好，未加生产绕过开关；本轮只检查其输入/语法，不冒用旧成品结果。

## 实际验证

Python 167项通过，定向覆盖 `test_cache_location`、`test_legacy_cache`、`test_graphics_cache`、`test_startup_cache`、`test_updates`、`test_update_apply`、`test_update_cache`、`test_update_coordinator`、`test_updates_bridge`、`test_review_platforms` 和 `test_cache_cleanup_entry`。包括选择/取消、中文空格路径、覆盖 EXE、删除缓存重建、跨进程首次选择、损坏/不可写/联接、活动实例、更新偏好保留、图形缓存退出保护，以及 Python/Node QA 偏好读取。最终结果见 `output/validation/portable-cache-location/python-tests.log`，26份受影响Python文件AST解析和三份Node脚本语法通过。

前端4项定向测试、typecheck、contracts:check、ui-data:check、生产构建通过。说明书严格校验/生成通过，21章/1059资产，错误和警告均为空。

`scripts/verify_portable_cache_qt.py` 实际运行 Win32 选择器取消/确定，真实 Qt 6.11/默认 GPU、原生桥、路径显示、打开请求、确认/取消、检查后新增租约拒绝、旧缓存清理及论文批注保留。浅深/1280与760逻辑宽度无横向越界，最终截图已查看。所选图形缓存经实际窗口安全退出清理，再打开同一 profile 成功读取真实 localStorage 草稿标记。

最终 Qt 回执为 `output/validation/portable-cache-location/qt-f3a847199681434dbfd6f49ff1682eec/report.json`，自有 scratch 清理回执为 deleted。早期原生 ctypes 参数错误已修复并重验；初轮页面选择器/截图等待问题属于验证脚本，失败记录保留，最终未降低断言。

统一入口 `Start-Research-Workbench.ps1 -CheckOnly` 确认当前源码绑定，并已启动本轮源码工作台供查看。日志与截图保留在受忽略的 `output/validation/portable-cache-location/`，包含前端构建、说明书、Qt、文档和架构回执。

## 未验及独立问题

- 当前 Windows 源码证据不代替最终 EXE。冻结首次选择、工程外运行、真实覆盖换版、安装/卸载须绑定新最终字节另验；其他平台、真实低空间/断电及长期运行未验。
- 打开目录验证到 QDesktopServices 受控路径和返回处理，Qt 集成测试用 URL 接收器记录目标，未逐一操作系统资源管理器窗口。
- 全仓架构检查观察到8项同期 M10 资源摘要/新资源登记差异，本轮未修改对应登记。文档检查错误为空，未改历史快照断链。
