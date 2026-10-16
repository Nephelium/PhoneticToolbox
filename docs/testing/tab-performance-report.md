# 多标签切换性能验收

2026-10-13，Windows 当前源码。井井报告打开新标签与切换已加载页面有时卡顿，总CPU与内存占用较低。

## 定位与修复

`AppShell.vue` 通过 `v-show` 保留多数工具。隐藏页面仍会收到 ResizeObserver 的零尺寸通知。共用波形此前将零宽度转为100 px，科学图窗转为180 px，并在恢复可见尺寸后再算一轮。语谱图分辨率随波形宽度变化，因而隐藏与显示都会提交请求。`ScientificPlot` 与波形模板还会在标记/光标更新时重新拼接静态 SVG 路径。参数估计原来随当前路由卸载，切回重挂载时重新扫描目录并读取关联标注。

修复仅调整页面与绘图生命周期：忽略隐藏零宽度，保留最后可见尺寸；图线路径按原始数据、时间窗和坐标依赖缓存；参数估计只在关闭标签时卸载。实际尺寸、视野与科学值变化仍刷新对应输出，原始数据、采样率、分析配置与导出数值保持原语义。

界面操作要等待主线程完成绘图与响应，短暂排队或单线程工作不要求整机CPU/内存占用很高。本轮未采集井井原进程的CPU/GPU轨迹，结论限于已复现的重复工作。

## 实际验证

夹具用六个同时保留的图窗，每份60秒/48kHz合成PCM与12000点曲线。音频只驻留浏览器内存，语谱图端口返回确定的合成像素，计数衡量真实组件提交的请求，未执行Praat计算。每个标签预先访问一次，再做浅深主题各六次切换。

| 检查 | 修改前 | 修改后 |
| --- | ---: | ---: |
| 12次切换额外语谱图请求 | 24 | 0 |
| 12次切换科学曲线路径重建 | 96 | 0 |
| 10次光标变化科学曲线路径重建 | 60 | 0 |

Chrome 154.0.8037.98 使用专有用户配置，可见窗口1440×1000。修改前证据 `output/validation/tab-performance/before-1791899307850/report.json`，最终Chrome证据 `output/validation/tab-performance/after-1791899588868/report.json`。真实缩放、窗口尺寸改变和新科学数组可使对应缓存更新，选区和静态图形保持；加载实际AppShell，以合成文件端口在参数估计与参数显示、EGG、LPC往返，确认原波形DOM保留且返回时不重扫目录。关闭参数估计再打开会释放已加载输入。页面错误0。

PyQt/Qt 6.11.0 的独立可见 QWebEngineView 使用默认GPU，同一源码夹具浅深主题12次切换不额外请求/重建，10次光标变化不重建曲线；实际AppShell三次往返也保留原波形且不重扫。证据 `output/validation/tab-performance/qt-9ebd5d58548444ca84850bbcb20c5b7a/report.json`，浅深截图已查看。Qt测试没有装配工作台原生科研端口，不能替代真实Praat/后台任务/设备验收。

运行命令：

```powershell
node tests/e2e/tab-performance.cjs --baseline # 仅在修改前运行
node tests/e2e/tab-performance.cjs --qt
# frontend 目录
node --test tests/m01-display.test.ts tests/annotation-r3.test.ts tests/m01-state.test.ts tests/lpc-display.test.ts tests/m02.test.ts
npm run typecheck
npm run contracts:check
npm run ui-data:check
npm run build
```

37项定向测试、类型、契约、UI数据及前端构建通过。文档检查2024文件、427来源、108任务零错误，定向差异检查通过。构建保留原有大chunk提示。初轮夹具运行时模板不能编译，改为render函数；目录计数原来混入其他页面首次加载，改为只比较返回动作；文件按钮包含大小文本，改用明确的文件列表入口。原失败报告保留。Chrome收尾原缺scratch摘要，已补逐文件大小/SHA清单并调用项目归属/联接/进程检查清理入口，七轮Chrome及一轮Qt的 `cleanup.json` 均为 `deleted`。Qt通过公共生命周期助手收尾。

## 范围与未验项

已验证Windows源码组件与实际AppShell、独立Chrome、独立Qt WebEngine。未重建最终EXE、未公开发布，井井正在使用的原窗口不会热更新。原文件/长时间完整模块负载、原生科学计算队列、真实采集与后台任务、多屏DPI及其他平台未验。此次消除了可重复证实的冗余链路，不能据合成端口测试断言所有偶发卡顿均已消失。
