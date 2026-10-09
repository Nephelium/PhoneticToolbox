# 公共布局接口

视觉取值与操作位置统一见 [UI_SPEC](UI_SPEC.md)。本页只维护组件接口、状态归属及兼容规则，不记录历史注册批次或构建体积。

| 组件 | 输入与插槽 | 行为 |
| --- | --- | --- |
| ModuleFrame | label；toolbar/status/default；unified | 单根 section 的可访问名称，透传原生属性，内容保持挂载；名为 module 的 inline-size container。 |
| ModuleToolbar | 可选 label；default/actions | 工具组自然换行、保持 Tab 顺序，role=group，不冒充需要方向键导航的 ARIA toolbar。 |
| ModuleSection | label、可选 title；actions/default | 区域有可访问名称，可省视觉标题，图形布局由模块决定。 |
| ModuleStatus | kind、message；default | error 使用 alert，loading/info 使用 status，empty 非 live region；重试由调用者执行。 |
| ModuleWorkbench | 实际存在的左右插槽；unified | 只为存在的面板创建拖宽控制，不持有科学状态。 |

这些组件不拥有文件、任务、音频、选区、草稿或关闭状态，不改图表事件，不伪造重试成功，也不清空旧结果。字体与间距来自公共令牌，图表用 figure 角色，IPA 使用既有 Doulos 角色。

## 状态与注册

导航元数据在 frontend/src/app/registry.ts，实际挂载、dirty/save 和标签关闭在 AppShell.vue。新增页面提供准确 import 路径、props/emits、save 结果与验证证据；不能将测试 port、空页面或成功 mock 接成生产模块。

允许异步 `save(): Promise<boolean>`，失败时标签保留。模块内部不另放关闭按钮，重用统一确认框。页面按需加载时使用公共加载/失败状态，不能以资源分包成功声称模块行为通过。

响应式按实际 container 宽度折行，不能仅依赖不随页面 zoom 调整的 viewport 查询。窄窗口保留滚动、保存/取消和图窗可操作区域。

## 宽度与兼容

`resizablePanels` 提供 savedVariable 和 legacyKey，新写入只更新实际面板并保留同记录其他值。已有宽度与折叠优先，窗口临时变窄不覆盖偏好。

M03 左栏从旧右栏变量恢复，只有缺独立记录时读旧 M01 键，此后使用独立键。M13 上下布局参数在左，保留历史宽度与极窄窗横向滚动。M10/M17 的专用结构不自动启用 unified，其他已接入模块按 UI_SPEC 的 300 px 默认和当前公共令牌处理。

操作迁移保留文件目标、保存语义、禁用条件、科学手势和关闭保护。M12 保存 TextGrid、M16 波形下采集/编辑、M02 逐图导出等例外以对象关系为准，不通过全局隐藏 header/h1 的 CSS 删除功能。
