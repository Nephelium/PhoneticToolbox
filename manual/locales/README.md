# 说明书翻译与可选语言包

中文正文仍由 `manual/chapters/` 和 `project.json` 维护。`en/`、`ja/` 保存英文、日文翻译目录，不复制中文工程、自然录音或视频。普通应用只附中文说明书；外语包通过应用下载或选择已下载的 ZIP 安装，之后可离线阅读。应用界面语言与说明书语言可以分别选择。

每个章节一个同名 JSON，素材说明与书名保存在 `project.json`。目录格式为 `ptb-manual-localization/1`，包含 `locale` 与 `entries`。每条记录使用原文及内联对象的 SHA-256 作为键，保存 `source` 和 `translation`。相同原文可以复用译文，插入其他段落不会改变已有键。修改原文、链接或格式对象会生成新键，旧译文不再计入覆盖率。

## 翻译完整段落

段落和标题作为完整翻译单位。无格式文字直接书写，有格式文字使用 `<s id="0">…</s>`，引用、公式和换行等对象使用 `<x id="1"/>`。译文可以调整标记的位置，以适应英语和日语语序。每个原有标记必须恰好保留一次，不得新增标签、属性或链接。代码、参数键和 IPA 标记中的文字保留原样。数学公式、引用条目、资产 ID 和路径不翻译。

例如原文 `点击<s id="0">打开</s>，参见<x id="1"/>`，可以译为 `See <x id="1"/>, then click <s id="0">Open</s>.`。不要把每个 text 节点单独拼接成译文。表格单元格、图注、替代文字、交叉引用标签和媒体来源类型也须翻译。数字、范围和默认值由检查器逐项比较，科学含义仍须人工核对。

| 中文 | English | 日本語 |
| --- | --- | --- |
| 基频 | fundamental frequency | 基本周波数 |
| 发声类型 | phonation type | 発声様式 |
| 声门 | glottis | 声門 |
| 语谱图 | spectrogram | スペクトログラム |
| 声道 | vocal tract | 声道 |
| TextGrid / IPA | 保留原名 | 原名を保持 |

## 校验与盘点

```powershell
python -B -X utf8 scripts/manual/localization.py --project manual --report output/validation/manual-localization-coverage.json
python -B -X utf8 scripts/manual/localization.py --project manual --locale en --require-complete
python -B -X utf8 scripts/manual/localization.py --project manual --inventory --report output/validation/manual-localization-assets.json
python -B -X utf8 scripts/manual/capture_plan.py --project manual
python -B -X utf8 -m unittest scripts.manual.test_localization
python -B -X utf8 scripts/manual/verify_localized_readers.py
```

覆盖报告列出每章缺译、失效、标记错误和已失去原文对应的记录。`--require-complete` 在任何必需译文缺失或无效时返回失败。覆盖率只衡量文本记录，不能代替语言、内容和界面截图的人工复核。素材盘点的分类依据现有元数据，仍需目视核对。只重拍当前正文实际引用的应用界面图；科学图、音频、视频和未引用旧图分别处理。

`verify_localized_readers.py` 在自有临时目录中构建全书，再通过应用实际使用的 TypeScript 解析器读回所有章节并检查全文搜索。同时比较稳定 ID、链接、公式、表格对象、媒体、代码和 IPA 的保真情况。它不生成缺少截图的可交付包，也不代替当前功能与步骤的人工核验。临时读本在结束后按项目测试产物规则清理，报告保留实际来源摘要。

## 本地化截图

每个语言目录的 `assets.json` 使用以下结构。正式截图按原有素材分发边界存入 `assets/public/` 或被忽略的 `assets/software-only/`，不得为了外语包改变权限。

```json
{
  "schemaVersion": "ptb-manual-localized-assets/1",
  "locale": "en",
  "overrides": {
    "existing-stable-asset-id": {
      "path": "assets/software-only/localized/en/example.png",
      "sha256": "文件的实际SHA-256",
      "sourceSha256": "对应中文源图当前字节的SHA-256",
      "locale": "en",
      "reviewed": true
    }
  }
}
```

覆盖保留原资产 ID。截图必须来自对应语言的实际应用，遵守统一截图尺寸、缩放与主题规则，局部标注图从该语言正式截图派生。构建会拒绝缺少当前引用界面图、摘要不符、未复核或分发权限不符的语言包。视频本轮不翻译。原音、处理例音和无需更换的科学图复用中文包中的资源，不在外语包重复携带。

`scientific-figures.json` 按中文源图实际字节记录科学图的目视复核：保留原科学图，或由相同数据真实重新导出应用生成的文字标签。当前有应用生成中文标题或轴标的图必须在每个外语包中提供覆盖；IPA、汉字例字、数据点、曲线和数值保持原义。覆盖登记与中文源图摘要绑定，源图变化后须重新核对。

`capture-plan.json` 只列当前正文引用的截图、派生图、科学图及共用媒体，包含源摘要、对应节点、旧拍摄记录和可复用脚本线索。线索不等于脚本已适配外语，也不表示已经拍摄。`scripts/manual/capture_locale.py` 可在独立截图实例中通过实际语言选择器切换并检查语言、DPR、缩放、浅色和字体状态。新采图脚本优先用稳定 DOM 标识或明确语言键；旧中文控件名仅在映射唯一时转换，不能把同一转换应用于文件名、调查数据、IPA或论文正文。局部图旧裁剪坐标仅供参考，应根据外语界面重新定位控件。

完整图目视通过后，使用 `register_localized_captures.py --locale en --chapter m02 --report <报告> --reviewed` 登记。`--capture NAME` 可重复指定本批已审过的部分，旧素材缺少捕获名时用 `--asset NAME=ID` 显式匹配。登记器逐项验证当前来源摘要，再按工程和语言加锁、重新读取并原子合并 `assets.json`，供独立采图任务并行使用。辅助全屏父图不会被直接登记为局部图。

论文阅读器的原生全屏截图 `m18-r3-reader-fullscreen-da8aa864` 单独记录真实窗口状态：Qt `fullscreen:true`、`maximized:false`、DOM `documentFullscreen:true`、进入前 `wasMaximized:true`，以及实际退出后核验的 `fullscreenExitRestoredMaximized:true`。仅该图接受完整全屏往返证据，尺寸、DPR、字体、缩放与主题要求不变。普通截图仍须实际最大化，不能为全屏截图伪写最大化状态。

局部图的父捕获条目须带 `regions`，例如 `{"editor":{"selector":".editor","box":[100,80,200,100],"units":"css-px"}}`。这些坐标来自该语言界面的实际 `getBoundingClientRect()`。使用 `derive_localized_capture.py --parent-report <报告> --parent-capture NAME --chapter m02 --asset-id ID --region editor`，可通过重复 `--include NAME` 合并区域，重复 `--mark NAME` 添加红色圆角框。派生报告记录父完整图与报告摘要、实际裁剪框、输出尺寸和工具来源；逐图复核后再登记。不得把完整图直接填到局部图 ID。

科学导出图单独使用 `--scientific-export` 登记。捕获条目须标明 `captureKind: scientific-export`，并提供 `ptb-manual-scientific-export/1` 证据：当前中文源图 ID/摘要、完整输入与参数、各自 canonical JSON 的 SHA-256、已核实旧配方的同一输入与参数及核对依据、实际应用导出动作。登记器只接受科学图复核清单中明确需要翻译生成标签的项。语言引起的文字布局与图像尺寸变化可以保留，数据、单位、曲线、IPA、汉字例字和科学参数须与原配方一致。

## 生成可安装包

```powershell
python -B -X utf8 scripts/manual/localization.py --project manual --locale en --pack output/manual-packs/PhoneticToolbox-manual-en-VERSION.zip --version VERSION
```

目标须为工程外的新文件。构建严格验证全文和截图覆盖，生成 `language-pack.json`、阅读工程索引、章节及对应截图。包内清单记录语言、版本、中文源摘要、项目 ID、共用媒体、截图覆盖和逐文件 SHA-256。图片按实际 SHA-256 复用唯一包内路径，原资产 ID 与中文源图摘要仍分别核对。阅读器按语言、版本、源摘要和覆盖路径缓存字节及显示 URL，切章回收 URL、返回时复用有界缓存；同字节双 ID 不重复读取。外语包共用的原音、科学图等按稳定资产 ID 读取当前宿主中文包的实际路径，兼容发行构建的 WebP/MP3 压缩改名，外语 caption 与 alt 保留。默认软件构建仅生成中文资源和下载清单，不包含外语包。公开构建仍遵守受限素材过滤规则，不得将含 software-only 截图的包上传公开 GitHub。

`manual/language-packs.json` 是应用读取的下载清单。未发布时保持 `published:false` 和空 `packs`。发布流程另行审核。生产地址只允许官方网站及本项目 GitHub Release，隔离验收仅在页面本身使用本机地址时允许本机 HTTP 下载。不能把合成验收包或部分译文登记成可用的完整说明书。

Web 在用户主动安装语言包时，将该包实际引用的共用媒体和当前中文资源索引另存入浏览器 CacheStorage。媒体按当前宿主实际文件的 SHA-256 复用，兼容 WebP/MP3 发行路径，未引用的作者旧素材不进入下载集合。缓存已有条目也须通过摘要核对，损坏条目可通过补存按钮重新下载；缓存不完整时仅说明正文已安装。默认中文启动不预取外语正文或共用媒体。桌面版继续读取随应用附带的中文资源。

Web 的离线范围是已经打开的网站内阅读已安装说明书，包括断网后关闭、重开说明标签。重新打开说明标签会优先读取在线中文索引，断网时可恢复与已安装包的项目和源摘要一致、且自身字节摘要正确的缓存索引。此机制不提供网站离线导航启动。译文包的版本与安装身份独立参与章节缓存身份，同一中文源的新译文包也会重新读取正文。跨语言顺序安装复用共用媒体，同一页面的重复图片 ID 共用一次读取与显示 URL。

可用 `python -B -X utf8 scripts/manual/verify_packs.py` 验证合成包的真实 HTTP/ZIP 安装、同源新译文版本、损坏缓存修复及断网切语。完整候选使用 `python -B -X utf8 scripts/manual/verify_complete_offline.py <candidate-manifest.json>`：实际安装两个完整 ZIP，断开整个浏览器上下文的网络，重开说明标签并读取全部章节和媒体。报告保留候选文件摘要、实际章节及媒体读回、来源摘要和 URL 回收数据，临时浏览器与测试输入按产物生命周期清理。
