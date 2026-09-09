# PhoneticToolbox v2 — Agent 规则

## 打包规则（重要）

**每次打包 exe 都必须使用 conda 环境 `phonetic_311`**，不要使用 base 环境
（base 环境缺少 pyqtgraph 等依赖，打出的 exe 会启动失败）。

打包命令（必须用 `python -m PyInstaller` 形式，该环境的 pyinstaller.exe
启动器已损坏，直接调用会失败）：

```powershell
conda run -n phonetic_311 python -m PyInstaller run.spec --noconfirm --log-level WARN
```

- 打包配置：`run.spec`（PyInstaller 单文件模式，输出到 `dist\`，
  文件名自动带 pyproject.toml 中的版本号，如 `PhoneticToolbox_v2.1.5.exe`）
- 大型构建，通常需要 10 分钟以上，建议后台运行并轮询日志
- 日志位置：`build\pyinstaller_build.log` / `build\pyinstaller_build.err`
- 改版本号需同步三处：`pyproject.toml`、`phonetic_toolbox/__init__.py`
  （`tests/test_version_consistency.py` 会校验）；写文件注意不要用带 BOM 的 UTF-8

## 语音标注对齐模块（v2.1.5 新增）

- 主页【语音标注对齐】按钮 → `main_window.on_web_praat_editor` →
  `services/web_praat_server.ensure_server()` 在进程内守护线程启动微型 HTTP 服务
  （仅 127.0.0.1，随机端口），再用默认浏览器打开；用户无需手动起服务器
- 网页前端静态资源：`phonetic_toolbox/gui/resources/web_praat_editor/`
  （源自 pitch_perception 项目 web_praat_editor，已普适化：词层/音素层名可在页面
  左侧自定义并经 localStorage 记忆；发音词典格式为每行“词 音素1 音素2 …”，
  内置普通话词典 default.dict，界面已中文化）
- 网页内“选择文件夹”对话框经 `_FolderPickerBridge`（pyqtSignal）转发到 Qt 主线程执行
- 唇形对齐保存写回语料目录 `audio_recording.pkl` 的 `metadata.lip_manual_offset`，
  与 `services/io/lip.py` 读取位置一致
- 测试：`tests/test_web_praat_server.py`（10 个用例）
- 说明书：`Phonetic_Export/index.html` 第 13 章；唇形新功能见第 7.3/7.4 节
- 说明书编辑器：`Phonetic_Export/PhoneticToolboxDoc.html`（PhoneticDoc）新增
  “导入网页”按钮：可读入导出的网页包 zip（index.html + images/audios）或单个
  index.html 并转为可编辑章节；导入的小节内容以原始 HTML 保存，SimpleMD.parse
  对以 `<` 开头的行原样透传。`Phonetic_Export/PhoneticDoc_网页导入包_v2.1.5.zip`
  是当前说明书的即载包
