# 共用说明书阅读器

阅读端不依赖 Tiptap 或作者服务。内容合同位于 `types.ts`、`schema-project.json`、`schema-chapter.json`。`content.ts` 在运行时校验版本、稳定标识、清单引用和路径。公式由 `formula.ts` 调用受限 KaTeX 渲染，禁用可信 HTML 扩展；无效或超限输入显示原文与明确提示。

## 产品入口

```vue
<ManualReader
  :project="project"
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

支持标准段落/列表/表格及合并单元格、代码、图片放大、原生图音视频、图注、双列、提示框、脚注、引用和交叉引用。所有节点通过 Vue 转义渲染。自定义文字颜色在深色主题中使用专用 `colorDark/backgroundColorDark`，缺少深色值则继承主题文字/透明底色。IPA 使用固定 Doulos SIL。

未知节点和文字标记仍保留在 JSON 中，阅读器显示具体类型提示。阅读端不做内容写入。媒体引用缺失或解码失败只影响该媒体；用户可继续阅读整章。

## 定向验证

在 `frontend` 中运行 `node --test tests/manual-reader*.test.ts`。测试样章仅存在于测试文件，不能作为正式手册正文。修改阅读交互时另验实际 Chrome/Qt，修改内容协议时另验作者保存往返。具体证据见 `docs/testing/` 对应报告，不将已有接线描述成待实现。
