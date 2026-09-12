# P04-FONT 全局字体与整幅导出验收

2026-09-12。井井授权继续实施、全部IPA固定Doulos SIL，并明确已完成模块一起调整；本轮又指出整幅PNG上下字号失衡。上述要求均已纳入本次代码和规范。

**verified：限定Windows当前开发态的公共字体设置、M01/M02/M09/M10呈现与M02/M03图像、M10短视频。** P04-FONT全平台专项仍in_progress。M03页面、服务器字体预检、冻结EXE和跨平台不在该verified范围，完整M03仍in_progress。

## 实现及原因

- 公共状态按中文、英文与数字、代码/等宽、固定IPA分角色。中文字体自带的拉丁字形不会覆盖独立英文选择。前端字体名称限制长度及控制字符，不允许任意路径、CSS片段。字体必须实际加载后才能应用，缺少字体保留当前配置。偏好版本化并按网页账号在本机隔离，应用、取消、恢复默认和重开均有证据。
- Qt只提供字体名称。网页枚举只在用户主动操作后调用浏览器能力，不读取字体文件或上传系统路径。复用已有Doulos SIL 7.000字节和OFL，没有下载、安装字体或新增运行依赖。
- M01 TextGrid、M02参数/波形标注、M10预设及构形/姿势名称显式使用Doulos。普通混排的扩展IPA字符采用固定字体范围，字符序列不被改写。普通英文与数字、中文分别保留各自角色。
- M02原SVG转图链无法可靠访问页面已加载的字体，现先固定图形及文字位置，绘制几何后在同一浏览器字体环境绘字。可编辑SVG附Doulos和完整OFL，不提取系统字体。其他字体仍依赖阅读设备，不宣称所有SVG阅读器均兼容。
- 用户指出的字号失衡来自整幅导出读取SVG旧font-size属性，其优先级压过了页面已计算字号。修正为取计算样式，刻度、IPA、图例和分区标签共用基础字号，标题为14/12倍。24px边界下坐标留白、标注和图例占位同步扩大，窄窗通过图面横向滚动保留字号。
- M10通过所属父页面同步字体，Canvas和SVG标签都接入图表角色。视频导出冻结字体，期间更改先排队，导出结束后应用。只修改字体相关呈现，不扩展算法、关键帧或录制行为。
- M03请求新增可选font/1快照，历史无快照任务保留旧渲染。当前提交适配携带图表字体，独立MKL子进程按名称精确解析，缺少字体返回font_unavailable，不静默替代。元数据含请求/实际字体、字体哈希与既有运行时版本。科学核心不读取字体，单文件和批次CSV在两套字体下字节一致。

## 实际验证

环境保持原有隔离：Windows；M01/M02/M09与API使用`.venv/m09-ui`，M10使用`.venv/m10-ui`，M03使用`.venv/m03-compatible`的Conda/MKL。没有修改以上环境或v2。Python测试通过已有项目源码入口读取后端/桌面，科学核心仍使用对应安装包。

| 命令或入口 | 结果 |
| --- | --- |
| `npm --prefix frontend test` | 41 passed，包括字体偏好、角色固定和隔离键检查 |
| `npm --prefix frontend run typecheck`、`npm --prefix frontend run build` | 通过 |
| m09-ui：`-m pytest -c tests/pytest.ini backend/tests --ignore=backend/tests/test_m03_exports.py --ignore=backend/tests/test_fonts.py desktop/tests tests/contracts -q` | 258 passed；2条原有Starlette/AnyIO弃用提示 |
| `scripts/Invoke-M03-Python.ps1 -PythonArguments @('-X','utf8','-m','pytest','-c','tests/pytest.ini','-o','pythonpath=D:/PhoneticToolbox/PhoneticToolbox_v3/backend/src','backend/tests/test_fonts.py','backend/tests/test_m03_exports.py','-q')` | 29 passed；实际单文件/批次各三PNG、24px、两套字体的CSV不变、两个并行独立子进程字体快照隔离 |
| `node tests/e2e/fonts.cjs` | 实际Chrome设置、宋体/Times与楷体/Arial、固定IPA、无字体拒绝、重开、取消/恢复、图表独立字号、账号命名空间、真实PNG/SVG、Doulos独立Canvas像素对照、TextGrid组件、12/24px与390px窄窗 |
| `node tests/e2e/m02-png.cjs` | 既有整幅PNG导出回归，包含300dpi/CRC、双声道、主题及选区等原断言 |
| m09-ui：`scripts/verify_fonts_qt.py` | 真实Qt字体枚举和设置、M01/M09控件、M02标注与两套字体的实际原生PNG保存，原输入哈希不变，已开曲线保留 |
| m10-ui：`scripts/verify_fonts_qt.py --m10-only` | 真实原生引擎连接、预设Doulos、字体传播、导出中快照固定、短视频保存后应用排队的新字体 |
| FFprobe/FFmpeg对M10生成文件独立回读 | VP8/Opus、1280×720、0.4秒，实际解码画面检查；不替代完整R5科学或长视频验收 |

Python命令均使用对应`.venv/<环境>/Scripts/python.exe -X utf8`。m09/m10定向脚本的进程内PYTHONPATH限定为本项目`backend/src;desktop/src`，结束恢复，不写系统变量。M03使用专属包装器，不把Conda DLL带入Qt宿主。

本机证据目录（忽略文件，不包含用户语料）：

- `output/validation/fonts/chrome-1789209089713`：最新Chrome检查与字号修正后的同场景PNG。12px为`simsun-times.png`，24px窄窗为`large-font.png`；导出快照内刻度、IPA、图例均24px的断言通过。
- `output/validation/fonts/qt-research-6f5f19af0811415cbda5614190fe0175`：最终构建的Qt研究模块字体切换与12/24px两个真实PNG。
- `output/validation/fonts/qt-m10-89de0234db834fc1afe4859da36ecb47`：最终M10字体页面及实际短视频。此前`qt-m10-e516127894084c99b3d47ac0af4c55e8`保留首轮视频快照与独立解码证据。
- 原字号不一致的24px图保留在`output/validation/fonts/chrome-1789208803255`，没有覆盖失败现场。

初次混用m09-ui启动M10时，其历史安装核心缺少R5的envelope模块，未通过。改用原有m10-ui完成本项测试，未给m09-ui换包或声称该历史启动器已成为所有模块的统一科学运行环境。早期测试中Matplotlib环境不含FastAPI的错误通过按既有环境分离执行解决；批次字体用例错误携带ROI时按批次全文件协议修正测试输入，没有降低检查标准。

## F01–F12范围及剩余项

| 验收 | 本轮结论 |
| --- | --- |
| F01/F02 | verified，Windows所列字体及IPA样例；未安装的思源宋体/JetBrains Mono不冒充实测 |
| F03/F04/F05 | verified，所列真实设置、重开、账号命名空间、已打开页面及图表；没有重新运行声学任务 |
| F06 | verified，实际PNG、字体就绪、统一字号及Doulos像素参照；覆盖当前内部SVG结构，不能泛化为任意外部SVG转换器 |
| F07 | verified，可编辑SVG含Doulos和许可、其他字体依赖明确；跨设备/第三方阅读器便携兼容仍planned |
| F08 | verified，M03单文件/批次图及并行子进程；M10实际导出中字体固定 |
| F09 | 部分verified：前端账号命名空间切换、后台并行字体隔离；两个真实网页账号联合EGG提交仍planned，随M03页面验收 |
| F10 | 部分verified：Windows Chrome/Qt、浅深主题、390及1440宽、12/24px；1280/1920和各DPI的完整字体矩阵仍planned |
| F11 | verified：M03两字体的单文件/批次CSV完全相同，Qt输入未变，既有M02参数显示回归；其他未迁移模块继续各自核对 |
| F12 | verified：实际范围的设置、来源、说明书同步；冻结EXE与其他平台仍planned |

## 来源与下一项

ASSET-DOULOS登记已有字体在后端的同字节副本、SVG嵌入及M10等使用位置，字体和OFL一同进入后端包资源配置。REF-FONT-RENDERING记录官方CSS Font Loading、Qt字体目录、Matplotlib及Local Font Access文档参考。共328条来源，不将本机候选字体登记为随包字体。

说明书见[字体设置](../manual/settings.md)，设计见[UI_SPEC](../design/UI_SPEC.md)。`scripts/validate_docs.py`通过（512个文件、328条来源、32项任务，errors=[]，36条历史快照未决链接保留）；`scripts/check_architecture.py`通过（errors=[]）；`scripts/generate_contracts.py --check`及`npm --prefix frontend run contracts:check`无漂移；`git -c core.safecrlf=false diff --check`通过。M10受影响文本资源、字体副本和资源清单哈希已同步，原迁移来源哈希保留，并为本轮哈希涉及资源固定LF字节与Git换行规则，后端OFL只规范换行且正文与原件一致，避免Windows重新检出改变哈希。当前frontend/dist已构建，历史EXE未重新打包，没有DDL、push、全局依赖安装、公开发布或v2修改。下一项为M03-D页面接入公共字体与任务链，后续再做服务器预检、整批目录和各平台/发行专项。
