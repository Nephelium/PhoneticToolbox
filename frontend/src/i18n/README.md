# 应用语言维护

Windows 与网页使用同一工作台、模块源码和语言状态，默认简体中文。唯一语言注册表为 [languages.json](languages.json)，当前含中文、英文、日文。语言选择只更新展示，不能改科研参数、文件名、任务身份或用户材料。

## 文案和提示

在 `locales/<语言>/<功能>.json` 维护稳定消息键，三个目录保持相同键和命名参数。Vue 模板调用 `tr(key, values)`，脚本中长期显示的标签使用 `computed` 或 `localized`，避免在初始化时将译文固定成普通字符串。

异步结果、状态和错误使用 `messageRef`、`uiMessage`、`uiErrorMessage` 保留描述及捕获的数据，展示时才翻译。原生和 Worker 的已知错误在受控边界解析；未知外部诊断保留原文。错误判断使用稳定码或类型，不能判断翻译后的片段。原始地名、论文、实验刺激、音标、参数键和既有工程名称不经过界面翻译。

`reviews.json` 记录当前中文与各译文摘要。修改任一侧会使该项重新进入待复核，不能通过缺译回退声称覆盖完成。生成结果位于 `generated/`，不手改。

```powershell
node frontend/scripts/generate-i18n.mjs
node frontend/scripts/generate-i18n.mjs --review <已审阅的功能命名空间>
node frontend/scripts/generate-i18n.mjs --check
npm run typecheck --prefix frontend
```

先实际审阅该功能三语内容，再执行 `--review`。该命令记录摘要，不自动检查译文语义。

## 原生窗口和独立页面

Qt 文案的唯一维护源位于 `desktop/src/ptb_desktop/locales/`。下列生成器同时输出冻结 Python 文案和 API 允许的语言代码，前端生成器从同一源派生 `native.*`，不另复制一套译文。

```powershell
python -B -X utf8 scripts/generate_native_i18n.py
python -B -X utf8 scripts/generate_native_i18n.py --review
python -B -X utf8 scripts/generate_native_i18n.py --check
```

原生选择绑定当前窗口，偏好原子写入用户目录。验证窗口显式使用自己的缓存、说明书和数据目录。M10 独立页面通过同源受控消息接收语言，使用显式文字/属性绑定和按需词典；切换不能重载页面或调用科学计算。其词典由同一前端生成器输出独立资源。

## 新语言与说明书

新增语言时，扩充注册表、新增各功能和原生目录中的完整译文，运行两套生成与复核检查。注册表派生前端类型、原生支持代码和 API 校验，避免分别维护固定语言枚举。最后验证对应字体、较长按钮、动态错误、账号保存、两端实际流程和导出，不能仅检查键数。

说明书的中文源工程保持一份。外语富文本、源摘要和对应语言实拍图位于 `manual/locales/`，维护流程见 [说明书说明](../../../manual/README.md)。默认软件仅附中文阅读资源，外语完整包主动安装后离线读取。正文、源图或功能变化后重新校验覆盖并按需重拍，最终包不能使用作者预览代替严格构建。
