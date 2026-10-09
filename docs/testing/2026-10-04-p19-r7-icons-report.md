# P19-R7 模块图标区分

日期：2026-10-04。源码定向 verified，限定 Windows 前端与 Chrome；EXE 仅构建。

录音 M16 使用麦克风，EGG M03 保留信号波形，发声类型合成 M07 使用成对声门/声带轮廓。修改仅涉及公共 `AppIcon.vue` 和模块 `registry.ts`，侧栏、模块卡片与标签页共用同一映射。原尺寸、线宽、主题色与模块身份保留。

类型检查通过，278 前端测试通过且无跳过，生产构建通过。实际 Chrome 主工作台浅/深两种模式中，三个图标路径各异，尺寸与原首页图标同为 17×17 CSS px；人工查看浅色侧栏确认清晰可辨。未追加镜像实现的单元用例。

首次预览随机分配端口 4045 被 Chrome 拒绝，改用已知可用的 31080。初始夹具误假定图标至少 18 px，读取现有 `.nav-item svg` 规范 17 px 后改为与原首页图标逐项比较，产品尺寸未变。

证据：`output/validation/sidebar-icons-report.json`、`sidebar-icons-light.png`、`sidebar-icons-dark.png`（后两项均在同一验证目录），`output/validation/sidebar-icons-tests.log`、`output/validation/sidebar-icons-build.log`。

本机最终包使用既有 `.venv/m14` 与 `--lean-qt` 构建成功，进程退出码为 0：

- PhoneticToolbox-v3-Latest-20261004-R4.exe（历史本地产物，当前工作区不存在；原路径 `../../dist/PhoneticToolbox-v3-Latest-20261004-R4/PhoneticToolbox-v3-Latest-20261004-R4.exe`）
- 311355280 字节。
- SHA-256：`12f87dbc72431a0d23c4dad64abd901958fac338ad5df9dd190508e52cbf4dab`。
- 构建日志：`output/validation/sidebar-icons-exe-build.log`。

基线提交 `0255daa8214f9d885d120d6562792cb8c9db421d`，冻结源码包含当前未提交修改。保留旧包，依赖原独立科学环境，portable:false，未运行成品检查。

无新 push、数据库 DDL、安装依赖、公开发布、删除用户文件或修改 V2，保留同期改动。
