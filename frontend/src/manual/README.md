# 共用说明书阅读器

阅读端不依赖 Tiptap 或作者服务。内容合同位于 `types.ts`、`schema-project.json`、`schema-chapter.json`。`content.ts` 在运行时校验版本、稳定标识、清单引用和路径。公式由 `formula.ts` 调用受限 KaTeX 渲染，禁用可信 HTML 扩展；无效或超限输入显示原文与明确提示。

## 产品入口

`app/HelpPage.vue` 从同一份生成清单加载说明书，`app/manual-project.ts` 只为浏览器保留最前面的网页版入门章，桌面目录与搜索排除此章。两端共用其余全部章节与素材，保留网页暂不可用模块的操作说明。全局入口按宿主打开入门章，模块帮助保持稳定章节和小节目标。

```vue
<ManualReader
  :project="project"
  :chapter-icons="chapterIcons"
  :target="{chapterId:'M08',targetId:'M08-input'}"
  :request-key="helpRequestSequence"
  asset-base-url="./manual/"
  :active="helpTabActive"
  :playback-allowed="examplePlaybackAllowed"
  return-label="返回变速变调"
  @navigate="onReaderNavigation"
  @location="saveReadingLocation"
  @return-tool="returnToOriginTool"
/>
```

`project` 必须由 `parseManualProject()` 验证，素材公开清单与软件专用清单可用 `mergeManualAssets()` 合并。默认从发布目录中的 `descriptor.path` 加载 JSON；自定义加载使用 `chapterLoader(descriptor, signal)`。只缓存最近三章的 JSON，当前章外没有媒体元素。

`target` 和递增的 `requestKey` 处理宿主发起定位；不修改工作台 URL hash。阅读器内部点击目录和引用时发出 `navigate`，滚动/切换时发出 `location:{chapterId,targetId?,scrollTop}`。宿主保存阅读位置时用 `initialLocation` 恢复。宿主负责复用唯一帮助标签页、返回来源标签和正式实验的运行保护。录音播放限制使用现有 capture lease。`active=false`、卸载或 KeepAlive 停用都会停止示例播放。

同章目录、搜索和引用定位保留现有正文与媒体元素，跨章及工程修订仍按缓存与取消规则加载。连续滚动的 `location` 通知最多每250毫秒一次，相同位置不重复通知；切章、定位、隐藏/关闭阅读器、页面隐藏和卸载时立即补存最新位置。加载中的空正文不覆盖原阅读位置。

目录初始只显示一级章节，点击章节展开或收起二级目录，同时只展开一个章节。当前二级小节显示其三级条目。滚动阅读仅为当前三级条目改变底色与文字颜色，保留 aria-current；直接点击一级或二级时标记相应条目，继续滚动后回到三级阅读标记。定位正文标题只短暂改变文字颜色，不绘制框线。字号随全局设置缩放，默认 14／14／12 px，一级仅加粗区分，三级采用 1.3 倍行高与上下各 3 px 内距。宿主可通过 `chapterIcons:Record<chapterId,iconName>` 提供一级章节的图标，工作台根据模块注册表复用侧栏图标，未提供时使用书本；二级小节不显示图标。

目录选中项不绘制外框或阴影；键盘焦点以文字下划线提示，共用按钮的实色/玻璃材质和高亮设置不得重新添加框线。

目录默认宽250 CSS px，边界复用 resizablePanels，范围190–420 px，中央阅读区保留420 px。保存独立阅读目录宽度，支持拖动、键盘微调和临时窄窗口收缩；小屏幕保留目录抽屉。

全书搜索使用生成的 `project.searchIndex`。没有该索引时明确显示搜索范围为目录和已打开章节，不假称搜索全书。

## 作者预览

```vue
<ManualDocument
  :chapter="chapter"
  :assets="project.assets"
  :references="project.references"
  :asset-resolver="asset => '/media/' + asset.path + '?session=' + session"
  @navigate="openPreviewTarget"
/>
```

支持标准段落/列表/表格及合并单元格、代码、图片放大、原生图音视频、图注、双列、提示框、脚注、引用和交叉引用。所有节点通过 Vue 转义渲染。自定义文字颜色在深色主题中使用专用 `colorDark/backgroundColorDark`，缺少深色值则继承主题文字/透明底色。IPA 使用内置PTB-Doulos（Doulos SIL），明确IPA标记和未标记的含音标/组合附加符号文字段均使用该字体，中文继续沿用界面字体；行内代码里的音标同样生效。

未知节点和文字标记仍保留在 JSON 中，阅读器显示具体类型提示。阅读端不做内容写入。媒体引用缺失或解码失败只影响该媒体；用户可继续阅读整章。

## 定向验证

在 `frontend` 中运行 `node --test tests/manual-reader*.test.ts`。测试样章仅存在于测试文件，不能作为正式手册正文。修改阅读交互时另验实际 Chrome/Qt，修改内容协议时另验作者保存往返。具体证据见 `docs/testing/` 对应报告，不将已有接线描述成待实现。

正文章标题与二级小节同为 1.52em，章标题加粗；窄窗口沿用相同字号。
