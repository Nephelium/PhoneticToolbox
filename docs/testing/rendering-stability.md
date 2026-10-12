# 3.0.1 界面绘制回归

状态：`built` / `verified` / `published`。2026-10-13，Windows 本机，保留默认 GPU 加速。

## 已确认的问题与证据边界

- 用户截图显示大片控件消失，用户补充滚动时按钮残留原位。与此前发声类型页滚动条引起的宽度反馈分开验证。
- R5 可见成品使用 Intel D3D11 ANGLE。音标页玻璃材质产生 443 个背景模糊元素、409 个合成层，去掉实时背景模糊的对照降为 13 层。本机 167 帧滚动对照尚未复现用户的控件消失，不能把对照结果写成完整复现。
- 通用玻璃、阴影和选中焦点规则覆盖侧栏规则，解释普通边框及间歇出现的紫色焦点框。
- 旧成品字体检查使用隐藏窗口和 `--disable-gpu`，只证明布局范围，未覆盖用户正常 GPU 绘制。这是前次验收遗漏。
- 外部同类线索：[Anki Qt WebEngine Windows 背景模糊报告](https://github.com/ankitects/anki/issues/4470)。该报告同样描述间歇闪烁、滚动相关及大量模糊元素，属于第三方问题报告，并非本机根因的独立证明。

## 修正与防止复发

玻璃按钮共用按背景、模糊、主题与窗口大小生成的位图，滚动不再实时采样其他合成图层。保留透明度、模糊强度、背景位置及生理参数页的图形加速。侧栏从通用按钮材质、阴影和焦点框中排除，保留选中底色和键盘焦点下划线。生理参数默认 n 仅将 VO 从 -0.1 改为 0.5，原生模型鼻端开口面积回读为 0.5。

最终发行必须单列默认 GPU、正常可见窗口的绘制证据，覆盖背景图、玻璃/普通材质、主题、字体、窗口尺寸、鼠标滚动及焦点切换。DOM 几何、隐藏窗口、禁用 GPU、源码/独立浏览器的通过均不能替代本次最终 EXE 证据。所有失败保留原日志并查明原因。不得通过全局关闭 GPU 影响声道 3D。

## 本轮记录

- 类型检查、392 项前端测试、UI 数据、20 章/1053 资源阅读构建检查通过。
- 可见源码 Qt：真实背景上传、浅深主题、三种按钮高亮、透明度/模糊、保存重开、原生取色及生理参数 iframe 背景对齐通过。首轮检查错误地用已选按钮断言普通按钮透明度，改为明确的三维视图按钮后重新执行；选中背景与普通背景分开验证，未改变产品选中语义。
- n 构型与 HEAD 比较只有 VO 一个数值变化，真实原生引擎回读鼻端开口面积 0.5。
- 可见源码 Qt：24 组侧栏样式、三字体×七字号×三实际窗口尺寸、双向真实滚动，151 帧 Qt 图像中的侧栏与音标文字像素检查通过。早期脚本滚动选择器写错、完整示例文字使用独立弱化色，两项分别修正并重跑，未放宽像素阈值。证据 `output/validation/rendering/visible-qt-29b79f67f2b04942ac2da8114bd46771/report.json`。
- 本轮源码 QA scratch 均在其进程退出后完成安全清理，保留原失败日志与图像。架构检查仍有 5 项此前已存在的 M10 资源摘要错误，本次只同步实际改动的 v3.css 和 default-presets.json，不扩改其他资源。
- R6 最终 EXE：默认 Intel D3D11 ANGLE、普通/最大化可见窗口，浅深主题×三字体×七字号×两窗口状态共 84 组、双向滚动 168 帧通过；文字像素检查最低 32 像素，高于 8 像素门槛。实时 backdrop-filter 为 0，合成层为 19；模糊 0/40 的位图边缘能量变化及侧栏样式通过。另用原生截图和实际滚轮/最大化确认控件随内容移动，n 腭咽口显示 0.50 cm²，3D 可用。
- 最终成品工程外分发、11 科学任务/18 帮助、真实隔离安装、预热复用、活动租约拒绝、含完成结果的卸载及研究设置摘要保留通过。免安装/安装两种真实 helper 换版、新窗口、组件复用、草稿与媒体通过。所有成功记录绑定 R6 字节，不能转用于旧线上 ZIP。
- 成品脚本先后遇 CSS URL 引号解码、背景上传提交时序及换版夹具缺相邻 application.json，修正实际测试输入并完整重跑，原失败日志保留，像素阈值与产品断言未放宽。
- 服务器六文件和 GitHub 四附件已按授权替换并回读大小/SHA，两个 EXE 分别 495841235、497757168 字节；版本、Release 文案/标签、既有 ZIP 及官网保持。用户后续要求停止公网整包下载，检查时已启动的校验已结束，此后未再启动下载；该下载不作为继续发布的条件。
- 用户追加要求每次打包回看既往故障，严格规则第 0 节新增历史故障检查表，发行 README 接入。按触发条件记录测试配方、最终 SHA、结果与未验范围，后续问题原位补充。源码同步范围为本轮应用修复、上轮尺寸/致谢/卸载修复及必要验证文档，排除同期官网工作。

本地配方、GPU/层数、帧和原始日志：`output/validation/rendering-20261013/`。源码外观报告：`output/validation/custom-appearance/qt-55926a29cb7c46d39aad30109fb5a4ad/report.json`。尚未验证其他电脑/驱动/跨平台，也不作永久不复发保证。

## 本轮最终身份与执行证据

- 冻结快照 SHA：`319e20f0e1f89dfca3290d0447dfc3078101a88c129a6c19be9c42a031bcc894`。
- 免安装 SHA：`27d0a664667dc9cb2c871aa1c29e4a4404564533d152c468cb130280659c5b08`。
- 安装 SHA：`d65b1c4d72880ab6e87c89d681229746f049e8914f9d5383430dca48fe5c3910`。
- 最终可见成品配方 `node scripts/verify_rendering_cdp.mjs` 与 `python scripts/verify_rendering_pixels.py`，实际参数、命令和结果保存在本地 `r6-visible-4.log`、`r6-visible/report.json`、`pixels.json`；原生检查见 `native-inspection.json`。源码 Qt 使用 `scripts/verify_rendering_qt.py`。
- 冻结/封装审计、分发、安装和换版入口分别为 `release/finalize_compact.py`、`release/package_compact.py`、`scripts/audit_release_contents.py`、`scripts/verify_distribution.py`、`scripts/verify_cached_install.py`、`scripts/verify_update_handoff.py`；完整调用参数和回执见本地 build/finalize/package/distribution 日志、`release-gate.json` 及 `qa-evidence/`。
- 发布与回读：`replace-final.json`、`replace-server-after.json`、`replace-github-after.json`、`github-asset-verification.json`。现有 ZIP/标签不动，故其内容仍属于首发 R2，不能当作 R6 源码/应用下载。
- 清理完成：保存独立报告后，逐项验证进程与路径，移除两个仅指向自有科学缓存的遗留联接入口，再按清单清理12处隔离副本/缓存（137568个文件条目，硬链接单独计数）。报告、脚本、截图与最终成品保留，见 `qa-cleanup/cleanup.json`。初次因联接及清理父进程引用被拒绝，解除已核实入口并等待自有进程退出后重跑。
