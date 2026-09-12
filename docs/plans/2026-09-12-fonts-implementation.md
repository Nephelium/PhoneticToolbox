# P04-FONT 字体实施计划

**Goal:** 在已完成模块中统一调整中英文和等宽字体，全部IPA固定Doulos SIL，图表与新导出图片同步。

**Architecture:** 公共角色与版本化偏好驱动CSS、Canvas和后台渲染适配。IPA为固定角色，不提供替换选项。后台任务保存字体快照，不影响科学核心。

**Tech Stack:** 现有Vue/TypeScript、Qt字体枚举、Canvas、M03隔离Matplotlib。复用Doulos资源，不安装字体或新增运行依赖。

2026-09-12井井授权继续，并明确已有模块字体一起调整。此授权包括M10的字体适配，M10算法、关键帧、录制流程和旧EXE保持原范围。P04-FONT本轮限定Windows开发态范围verified，全平台专项in_progress，见[实施报告](../testing/p04-fonts-report.md)。先前设计中IPA可选及M10字体待授权条款由本条覆盖。

## A 公共配置与设置

新增`frontend/src/design/fonts.ts`、`frontend/src/state/fonts.ts`、`frontend/src/components/FontSettings.vue`，修改`AppShell.vue`、`tokens.css`和平台桥接。先新增`frontend/tests/fonts.test.ts`验证字段校验、IPA固定、角色链和账号命名空间，运行失败后实现。字体使用用户已安装的字体，选择字体时检查可加载性，缺失明确提示。Qt枚举名称，不暴露系统路径。网页枚举需要主动授权，拒绝仍可使用候选名。偏好仅按账号本地保存。

## B 已完成模块与导出

统一M01/M02文字标注、M09界面与M10界面/视频叠字，语义音标用固定Doulos，图表继承公共角色。M02的PNG在当前浏览器字体环境绘字，独立SVG保留可编辑文字并携带Doulos资源，不复制/嵌入系统字体。`ParameterFigure.vue`、`export.ts`、TextGrid组件及视频渲染文件按实际位置适配。字体变化不能改变音频、CSV、选区或任务。

## C M03导出适配

新增后端字体快照模型及隔离渲染解析，修改`egg_models.py`、`egg_exports.py`、`egg_child.py`和前端任务提交适配。有效字体须精确解析，未知字体失败，不读取客户端任意路径。科学数值不受影响。先新增字体及CSV不变测试再实施，由生成脚本更新契约，无DDL。EGG页面仍按M03-D独立实施。

## D 验证与交付

基础命令：`npm --prefix frontend test`、`npm --prefix frontend run typecheck`、`npm --prefix frontend run build`、`python -X utf8 scripts/validate_docs.py`、`python -X utf8 scripts/check_architecture.py`。M03用`scripts/Invoke-M03-Python.ps1`运行定向pytest，协议按现有脚本生成及漂移检查。

新增独立Chrome字体测试与Qt字体/实际PNG保存验证，覆盖中文+Times New Roman、等宽、全部IPA固定Doulos、恢复/缺字错误、浅深/窄窗、SVG/PNG回读、已开模块及账号切换。沿用项目拥有的浏览器进程和现有测试库，不执行DDL。成功后补说明书、来源实际使用位置和报告，并作限定文件本地提交，不打包或push。

井井后续指出整幅PNG上下字号失衡。已修正SVG旧字号覆盖计算样式的问题，整幅图刻度、IPA、图例与分区标签同级同号，标题保持14/12倍，24px实际导出回归纳入退出门。
