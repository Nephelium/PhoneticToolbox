# P04-UNIFY 公共布局接口

2026-09-26，最小接口已冻结。U2 现有主题、全局字体角色和页面缩放保持。组件只负责显示，不持有任务、文件、音频、选区、草稿或关闭状态。

| 组件 | props | slots | 约定 |
| --- | --- | --- | --- |
| `ModuleFrame` | `label: string` 必填 | `toolbar`、`status`、默认 | 单根 section 的可访问名称；class/style/aria-busy 等原生属性透传。内容始终挂载，不因状态销毁科研图窗。 |
| `ModuleToolbar` | `label?: string`，默认模块操作 | 默认、`actions` | 默认放文件/刷新/批量等操作，actions 固定放方法/帮助。窄窗自然换行，原生 Tab 顺序；使用 role=group，不冒充需要方向键导航的 ARIA toolbar。 |
| `ModuleSection` | `label: string`、`title?: string` | `actions`、默认 | 区块名称必填，视觉小标题可省略，图形布局由模块定义。 |
| `ModuleStatus` | `kind?: empty/loading/error/info`、`message: string` | 默认 | 错误 role=alert，加载/信息 role=status，空态非 live region。重试/重新选择由插槽提供，组件不伪造重试成功、不清除旧结果。 |

路径均为 `frontend/src/components/<组件名>.vue`。无新增 npm 依赖。`--module-padding`/`--module-gap`=12px，`--control-gap`=8px，`--control-size`=14px，`--support-size`=12px，`--control-height`=34px。字体继承 `--font`，IPA 显式使用既有 `.ipa-text` / `--font-ipa`，图表仍用 `--font-figure` 和 `--figure-size`。不在公共组件设置科研图高或改写图表事件。

```vue
<ModuleFrame label="普通话转 IPA 工作区">
  <template #toolbar>
    <ModuleToolbar>
      <button @click="convert">转换</button>
      <template #actions><button @click="emit('references')">方法与引用</button></template>
    </ModuleToolbar>
  </template>
  <template #status><ModuleStatus v-if="error" kind="error" :message="error"/></template>
  <ModuleSection label="转换输入" title="输入"><!-- 实际表单 --></ModuleSection>
</ModuleFrame>
```

## 注册与所有权

M08/M13 agent 独占各自模块代码。页面完成后交付准确 import 路径、props/emits、暴露的 save 方法、dirty 保存规则及实际验证证据。P04 负责人串行修改 AppShell 的 import、真实页面分派、pane-workspace、任务/播放器适用范围、关闭保存处理和状态文案。现有 registry.ts 已有 15 项导航元数据，元数据存在不等于模块已实现。未就绪时不接空页面或成功占位，不修改生成契约。

延续现有 `context: ResearchContext`、`stateKey: string`、`active: boolean`（按需要）和 `references` 事件。切换使用 v-show 保留已开页面；任务归属由原任务 adapter 维护。需要未保存保护的模块使用与 stateKey 一致的 workspace dirty，并暴露真实 save 成功/失败；不得让壳在保存失败时关闭。页面内部不放第二个关闭模块按钮。

`ModuleFrame` 是名为 `module` 的 inline-size CSS container。模块可用 `@container module (max-width:720px)` 按**实际内容宽度**折行。viewport media query 不随 CSS 页面 zoom 自动缩小，150% 时应同时检查内容最小列宽。

### 当前实际注册和共享依赖清单

本表由 P04 串行维护，导航元数据仍在 `frontend/src/app/registry.ts`，实际挂载和关闭逻辑在 `AppShell.vue`。未新增运行依赖，package.json/package-lock.json 未改。

| 页面 | 实际入口（frontend/src/modules 下） | 公共依赖与状态归属 | 本轮注册 |
| --- | --- | --- | --- |
| M01 | parameter-estimation/ParameterEstimationPage.vue | ModuleFrame/Toolbar/Status，WorkbenchColumns，公共波形/标注/播放器；既有 m01 store | 保留原注册，增量布局 |
| M02 | parameter-display/ParameterDisplayPage.vue | 公共波形、ScientificPlot、字体与导出；原页面 save | 原注册保持，导出/手势回归 |
| M03 | egg-analysis/EggAnalysisPage.vue | ModuleFrame/Toolbar/Status，四图、WaveformViewport/AudioTransport/TaskPanel；原 workspace dirty/save | 保留原注册，增量布局 |
| M04 | lpc-spectrum/LpcSpectrumPage.vue | ModuleFrame/Toolbar/Section/Status，原谱图/波形/播放/任务；原 dirty/save | 保留原注册，增量布局 |
| M09 | spectrogram-to-audio/Spec2WavPage.vue | ModuleFrame/Toolbar/Status；原文件/图像/结果与全局播放器 | 保留原注册，增量布局 |
| M10 | vocal-tract/VocalTractPage.vue | 原嵌入页和 dirty/close 事件 | 保持，未改模块内部 |
| M12 | annotation/AnnotationPage.vue | ModuleFrame/Toolbar/Status；原 editor、WaveformViewport、AnnotationTracks、异步 save | 保留原注册，R6 图窗优先 |
| M13 | mandarin-ipa/MandarinIpaPage.vue | 四个公共布局组件、既有 Doulos SIL、模块本地映射；`stateKey?`、`references`、`dirty:boolean` 事件、`save():boolean` | 已据模块真实 Chrome 证据接入；AppShell 独立 m13Dirty；无全局音频条 |
| M08 | pitch-manipulation/PitchManipulationPage.vue | 模块自有 M08Port 待生产宿主提供 | 未注册测试 adapter 或空 port 页，待模块/平台交付 |

M13 使用 `defineAsyncComponent` 按需载入，带公共加载/失败提示，其他页面不预取字表。当前字表独立 chunk 约 3.01 MB（gzip 149 KB），主壳约 471 KB。大包警告保留，不提高阈值。模块数据来源与科学含义由 M13 报告负责。

## 既有操作迁移

| 页面 | 旧页首操作 | 迁入位置 | 保留约束 |
| --- | --- | --- | --- |
| M01 | 方法与引用，M01/项目眉题和标题 | 引用进工具栏 actions；原目录/刷新/输出目录行进默认槽 | 全列表分析、参数草稿、切分仍在原参数/任务区，目录语义及回调不变 |
| M03 | 打开目录/导入、刷新、交换、批量、说明、来源 | 前五项在默认槽，说明/来源在 actions | 四图及总览保留，保存参数草稿移至参数区；关闭/任务/结果错误保护保留 |
| M04 | 方法、保存参数草稿、关闭 | 文件/刷新进工具栏默认槽，方法进 actions，保存移至谱图参数区 | 时间/频率分离、Shift 选区与原播放器不变 |
| M09 | 来源、关闭 | 图片目录/截屏/导入/刷新进默认槽，来源进 actions | 图像相位缺失警示属于科研限制，保留；四点/重建/结果动作不变 |
| M12 | 引用、关闭 | 引用进 actions，图窗顶部继续保留文件列表与保存 TextGrid | R6 图窗优先；层级/词典/词表/搜索/强度/唇偏、剪贴、中文输入和自动保存不变 |

标题、副说明与重复关闭逐项移除，标签名称和标签关闭仍由 AppShell 提供。未以全局隐藏 header/h1 的 CSS 处理。M02/M10 本轮作为公共组件回归对象，不擅自修改模块内部。

测试及精确修改文件最终记录在 `docs/testing/p04-unify-report.md`，未完成验证不标 verified。
