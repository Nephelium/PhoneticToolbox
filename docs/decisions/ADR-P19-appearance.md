# ADR-P19：配色方案与显示模式独立保存

2026-10-03，已按本轮外观请求实施。验证范围见[报告](../testing/2026-10-03-p19-appearance-report.md)。

旧 `theme` 偏好继续表示 system/light/dark，新 `palette` 只保存方案 ID。后续外观 R1 按井井要求移除 PhoneticToolbox 方案，首次使用、旧 `ptb` 和未知 ID 均回退 Everforest，其他有效选择保留。各方案由配对 seed 生成公共 CSS 语义令牌，模式由系统监听或用户选择解析；页面和模块状态不重建。实验聚焦期间保留原外观冻结行为。M10 仅通过既有父页面消息接收受限颜色变量，不新增服务或接口。

沿用公共令牌名称，配色与图表科学轨道色义分开。Codex 用于主题名称和视觉参考，本项目的适配值与补齐模式不标为官方逐像素复刻。颜色来源见[登记](../references/p19-appearance-sources.md)。

默认中文 SimSun、英文 Times New Roman、代码 JetBrains Mono。已有显式选择保留。前两者是系统字体，初始化时缺失则显示错误与兼容回退说明，保存的偏好不被覆盖；用户显式应用不可用字体仍拒绝。JetBrains Mono 2.304 Regular 作为有固定哈希与 OFL 的资源加载，不安装系统依赖；IPA 继续固定 Doulos SIL。

设置页采用限宽的左右两栏，窄容器转单栏，字体草稿与关闭保护沿用原实现。桌面图标只改变原 K2 的显示占用，生成 QIcon 与未来打包 ICO 使用同一函数。本轮不构建新 EXE。

外观 R1：代码字体由按输入筛选的 datalist 改为原生 select，内置 JetBrains Mono 置顶，保留旧保存值、本机字体列表和单独的自定义输入。应用成功后清除临时预览覆盖，避免先预览 Georgia 再应用 Mono 时预览仍显示旧字体。验收直接覆盖这一用户操作路径，见[R1 报告](../testing/2026-10-03-p19-appearance-r1-report.md)。
