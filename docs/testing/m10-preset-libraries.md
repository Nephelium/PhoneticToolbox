# 生理参数合成：多套构型与审查整改

## 2026-10-22 · 删除已保存构型

状态：verified，限定 Windows 源码原生／Qt 和独立 Chrome 实际 Wasm／IndexedDB；三端共用源码已同步。原界面只有保存、改名与常用设置，缺少删除入口。共用前端现已在“已存构型 → 管理已存构型”增加逐项删除、按名称确认与取消；保存构形弹层也使用同一管理区。确认后通过既有 `library/save` 同时移除当前套装的自存条目及其常用项，成功才更新界面；失败保留原列表供重试。当前编辑与关键帧保留，其他套装／模型隔离，内置构形始终可用。

依据同日截图反馈，套装选择／命名／操作改为紧凑工具栏，构型列表宽屏双列、窄屏单列；统一平面圆角输入框与 IPA 字体，标题旁显示实际条目数。删除确认附在对应条目下，使用主题危险色和清晰焦点轮廓。

- Windows 原生：配置当前 workbench 源码后执行 `pytest tests/parity/test_vocal_libraries.py -q -p no:cacheprovider --basetemp <scratch>/pytest`，13 项通过。新增覆盖 JD2/W02、磁盘／内存两种适配器、其他套装／模型和关键帧保持、末项删除、磁盘失败原字节保留，以及旧同名组存在时仍可删除其他条目。日志 `output/validation/m10-preset-delete-20261022/unit-8b8584b15dcc40a3bde70d653aa7cfcd/pytest.log`。
- 前端：`node --test tests/vocal-profile.test.ts tests/vocal-web.test.ts tests/vocal-wasm.test.ts`，19 项通过，包括删除与常用栏共同保存、配额失败回滚及新执行器重开恢复。此层存储／执行器使用测试替身，实际 IndexedDB 另见 Chrome 验收。
- Windows Qt：`scripts/verify_m10_shared_qt.py`，12 组通过。新增真实删除按钮确认／取消、受控写入失败与重试、磁盘字节保持、常用移除、末项空列表、编辑／关键帧／另一模型保持及 worker 重建后从磁盘恢复。初版完整构建报告 `output/validation/m10-shared-20261018/qt-f7e804f5bac04ce68a55ee694d598fb2/report.json`；美化后同一 M10 源码、既有成功宿主构建的隔离资源快照复验报告 `output/validation/m10-shared-20261018/qt-af321116473340309f14f4d8a799e2fd/report.json`。
- Chrome：`scripts/verify_m10_wasm.py --work output/validation/m10-smoothing-20261018 --science output/validation/browser-compute-20261017/browser-science --frontend frontend/dist --vocal-runtime output/validation/m10-review-20261022/wasm/vocal-wasm --libraries --port 18788`，最终完整工作台构建下 8 组通过。真实 IndexedDB 事务中止后原字节与计算端一致，重试删除成功；同名内置 a 恢复且加载 120 Hz、自定义音标和末项移除、原编辑／序列／其他套装保留，关闭原标签并重新登录后结果保持。报告 `output/validation/m10-wasm-20261018/browser-40f8fd5251f44755af101462f17a9581/libraries.json`。浅深主题各 1600／1000／800 像素共六组布局无控件溢出，输入框样式、取消后键盘焦点通过；实际确认按钮／标签对比度均达到 4.5。截图已目视核对。
- `npm run typecheck`、`contracts:check`、`ui-data:check` 与 `build:workbench` 通过；既有 Vite 大包提示保留。74 项 M10 资源登记已更新，公共算法与 Wasm Python 包未改动。

首次 Python 子进程没有继承父进程的源码 `sys.path`，收集失败；改为统一源码配置后的同进程 pytest 后通过。新增前端重开测试首次漏掉宿主必需的 `meta` 初始化，按实际宿主流程补齐后通过。没有改变产品断言或放宽标准。两次 Python 验证、Qt 和 Chrome 的 scratch 均已按回执清理，仅使用合成构型与独立配置，用户实际保存数据未操作。

美化时两次全工作台构建因同期 `GeolinguisticsPage.vue` 模板标签错误失败，未修改该模块。先用 scratch 内既有编译宿主与当前 M10 静态源码快照完成 Qt／Chrome，记录于 `output/validation/m10-preset-delete-20261022/layout-51453105c2944d0cbd6ddd52a4fa172a/`；阻塞解除后完整构建与 typecheck 通过，再以最终构建完成上述 Chrome 八组。最终 `presets.js`／`v3.css`／`app.js` 与受验源文件逐字节相同，隔离资源快照及所有合成配置已清理。

未部署公开网页或生成最终 EXE/Mac 应用，Mac 实机与 Safari 尚未验证，用户已有源码窗口未重启。本轮没有新增外部代码或来源，前期公开部署状态不代表此次删除功能已上线。

## 前期验收

2026-10-20追加：本页共享实现已同步公开网页，相关操作在实际HTTPS Chrome155重新验证，Linux声道回退为R21。当前网页部署以[同步验收](2026-10-20-web-module-sync.md)为准。以下保留原本机/各平台验收范围，最终Windows/macOS成品与实体设备边界不扩大。

日期：2026-10-19。任务 M10-PRESET-LIBRARIES。共用前端、Python核心、Windows/Mac桌面适配器与Web宿主；本轮实测Windows源码/Qt和本机独立Chrome真实Wasm。无公开部署或最终安装包更新。

## 审查整改

| 上轮问题 | 本轮结果 | 证据 |
| --- | --- | --- |
| 网页sessionStorage随标签关闭丢失 | 改为已有IndexedDB values，按账号/项目保存整个profile，男女模型分开，关闭不删；有耐久记录时不读取旧会话，损坏原文可导出恢复副本 | 真实Chrome关原标签、重登录新标签，4套构型、选中套装和序列一致；存储单测隔离账号/项目/模型 |
| 配额失败使后续保存永久失败 | 队列捕获前次拒绝后继续，当前错误仍返回；耐久写成功后提交profile，失败恢复计算端库/序列/音量 | 首次失败后下一次排队写成功；磁盘失败保留原文件；第一笔音量写失败恢复默认；过期窗口CAS不覆盖新记录 |
| 保存和导出名称规则不一致 | 核心统一单行名称，允许内部空格；规范化重名新写入拒绝，旧记录可读并逐项更名 | 原生、浏览器实际保存/导出a i；新增规范化重名与控制字符拒绝；旧同名检查点可恢复，多组重名可逐项更名 |
| 旧几何包无法导入 | v3写出，v1/v2/v3读取；R18–R20仅在模型、SHA和参数一致时预览，确认重算后导入 | 原生严格拒绝模型/SHA/参数/未知版本，Qt/Chrome实际R20预览不写入、未勾选禁用、勾选后新套装导入 |
| 桌面不读取legacy构型 | 当前套装文件优先，然后当前旧文件，再legacy；读取不迁移，编辑后写新文件，原文件保留；旧常用项中失效引用按原界面行为不显示 | 只旧目录、新旧都有、损坏旧文件、男女目录隔离及磁盘失败回归 |

导出包标明当前几何；旧包来源版本在导入预览显示，原包保持不变。旧音频不承诺逐样本复现。Web保存仅限本浏览器，换设备使用构型包交换；清除站点数据仍会清除该浏览器记录。

## 多套构型行为

- 新建空套装、复制、重命名、切换。不同套装可保存相同音标的不同参数，各有常用栏；上限100套、每套2000项、单模型库合计8MB。
- IPA输入使用当前套装，自存项优先内置同名项；切换不改写已有关键帧。真实Chrome中甲套s为130Hz、乙套s为220Hz，先后添加的两帧分别保留130/220Hz。
- 导入默认创建命名套装，重名时建议附加序号；可显式改为合并当前套装并选择保留/替换同名项。新名称保存成功后再刷新当前库。
- 核心LibraryProfile统一行为，桌面只处理原子文件读写，Web宿主只处理持久保存及计算端恢复。没有新增服务器数据库或schema迁移。

## 验证

- Windows：配置当前workbench源码后，全部 `test_vocal_*` / `test_m10_*` 定向parity、backend、desktop，共157项通过。之后增加检查点恢复与多组旧重名逐项修复回归，构型库8项、构型包与网页宿主合计23项定向复验通过；其中21项属于前述157项，共覆盖159个不同测试。两条既有Starlette/AnyIO弃用提示保留。
- 前端：`node --test tests/vocal-profile.test.ts tests/vocal-web.test.ts tests/vocal-wasm.test.ts tests/m10-model-pinch.test.ts`，20项通过。存储故障/CAS/隔离使用受控内存存储及计算端替身，不将其记为原生合成或真实IndexedDB证据。
- Windows Qt：`scripts/verify_m10_shared_qt.py`实际页面、IPC与文件对话框适配器4组通过，包含新建/复制/命名/切换、旧包确认导入、重建模型worker后从磁盘恢复。报告 `output/validation/m10-shared-20261018/qt-65c19f5f8b3b47ac96316277ee28e5cb/report.json`。测试模式不打开实体音频设备。
- Chrome新增验收：`scripts/verify_m10_wasm.py --libraries`，真实Wasm/IndexedDB四组通过，实际更名及浅深主题截图复核。报告 `output/validation/m10-wasm-20261018/browser-374cdb2b89fe4d0aa8d5bc683a1f6913/libraries.json`。
- 原有Chrome共享控制完整回归：19组通过，含60秒合成、取消恢复、全局jitter与区段shimmer、分享全选/全不选/自存、模型切换及文档往返。报告 `output/validation/m10-wasm-20261018/browser-6b4d2ffa4f4340b3820cdf0b76aa4003/shared-controls.json`。其后补充旧重名逐项恢复及失效常用ID兼容，已在最终23项定向、Qt4组和Chrome套装4组复验。
- 类型检查、契约及UI数据检查、保留旧哈希资源的workbench构建通过。Vite既有大包提示保留。资源登记72项幂等，架构检查零错误。文档2327文件/447来源/166任务零错误，历史快照断链单列；定向diff检查通过。

## 失败记录与范围

Chrome脚本初次访问隐藏的声源控制、添加首帧后自动折叠的IPA面板，均因测试未先打开相应面板而超时；已改为按实际界面打开。刷新Wasm源码包后漏重建前端锁，首轮验收确认回退服务器并失败；随后重建并严格断言browser-wasm。新标签需重新登录，脚本补齐同账号登录与跨页偏好测试路由。失败文件保留，没有通过放宽断言来达成验收。

所有验收使用自有进程、合成账号和scratch，不读写用户实际构型。Qt及Chrome最终运行的cleanup.json均确认scratch已删除，保留报告与截图。当前用户工作台未主动重启，需重开载入新前端。Mac本轮只同步共用源码，R21原生仍待构建，最终EXE/Mac应用、生产网站、Safari、实体音频设备未更新或新增验证。

上轮提出的完整几何参数空间量化、自适应采样评估与性能剖析仍属后续研究建议。本轮保持R21几何及声学默认值，集中修复五项已确认问题和多套构型需求。
