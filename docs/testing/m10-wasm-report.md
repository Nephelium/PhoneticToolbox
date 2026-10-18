# 生理参数合成 WebAssembly 本地浏览器验收

日期：2026-10-18。任务：WEB-M10-WASM。范围：Windows 本机独立 Chrome、真实 VTL Wasm、原生 Windows 对照。生产网站、最终 EXE、macOS/Safari、实体移动设备及长期多用户负载不在本轮证据范围。

## 实现与计算位置

原版 VTL 2.4 API C++ 与现有 m10/r18 几何补丁编译为 Emscripten 4.0.9 / pyodide_2025_0 Wasm 动态库。编译适配只有 C 符号导出和内部符号隐藏，不改科研公式、speaker、随机数算法、轨迹、声源、采样时钟或文档格式。未使用的主机设备驱动 SoundLib 不参与链接。

独立同源 Worker 在既有 Pyodide 0.29.3 中运行原 Engine、Runtime、BrowserAudio、MemoryProfile 及原 NumPy/SciPy 计算。合成、分析、几何保持独立状态，已通过关闭分析实例后合成继续工作的实际检查。平台先选择本地运行库，初次装载不可用时回到原服务器；本地运算失败直接报告，不静默切换到丢失原状态的新会话。

浏览器继续负责播放、设备选择、文件及 2D/3D 显示。取消同步 Wasm 计算会立即终止该会话 Worker，新 Worker 恢复已确认的构形、构形库、常用、序列和音量；未完成编辑不冒充已保存。恢复前后播放代次保持区分，迟到的旧音频不能重新开始。增量恢复记录避免拖动时反复搬运整份构形库。

MP4 保持原视频功能：浏览器用 WebCodecs 产生原 VP8/Opus 包，原服务器编码器转换 H.264/AAC；该步骤仍需服务器。端到端抓包中声道 API 只有 video/begin、video/chunks、video/finish，几何、合成、监测、文档及动作画面没有服务器计算请求。

## 科学对照

固定合成 recipe：33/65 截面各自的 a/i/u/l/n/t，加局部舌形、独立喉高、唇宽，共 18 组完整网格、轮廓、横截面、声管、面积及传递函数。另测三类声源、F0 曲线、10 ms 送气边界和静音关键帧，采样率固定 48 kHz，所有时长、帧数、参数键和网格保持。

- 原始几何/声管和轨迹数组通过 rtol=1e-7、atol=1e-10 检查；共振峰频率网格相同。
- 传递函数显示值最大差 0.001 dB。未取整的幅度/相位微差和原始严格检查失败项保留，不改原生回归门槛。井井明确接受这类微小差异，验收将其与几何、时间轴或算法变化分开。
- 原 JD2 的 Jitter=10% 保持。跨平台 C rand() 的抖动轨迹不同，浊声 PCM 不作逐样本相同声明。首轮 0.5 秒静态浊声 RMS 差约 0.003 dB，含送气/静音的短序列约 0.020 dB，所测频带能量最大差约 0.39 dB；谱峰落在相同频率栅格。
- 仅供诊断的说话人副本将既有 Jitter 参数设为 0，正式 speaker 未修改。三个诊断元音最大浮点采样差约 6.54e-9，PCM16 量化差不超过 1 LSB。清声与耳语的首轮逐样本最大差约 1.5e-14。
- 独立实际 Chrome Worker 执行同一 recipe，再与原生 oracle 比较，未用浏览器结果生成 expected。大型原始结果只在测试 scratch，保留计算脚本、摘要、参数和比较报告。

## 功能与性能

完整真实页面覆盖：33/65 截面切换、器官编辑/撤销/重做、2D/3D/完整模型、构形库、三类声源、0.5 秒播放及编辑锁、输出监测、普通/送气/静音关键帧、版本 5 JSON 导出/清空/导入、5 秒预算、同步动作、真实 MP4、刷新恢复、浅深主题、取消后恢复及再次计算。现有网页的 0.5 秒测试发声、5 秒序列和禁用连续发声策略保持。

最终完整候选实测，硬件为 Intel Core Ultra 9 275HX（24 逻辑处理器）、Windows、独立 Headless Chrome 155。测试使用独立 profile、合成账号与设置夹具；账号/项目/视频走真实本机 HTTP，设置夹具不代表 PostgreSQL 持久化验收。

| 项目 | 最终候选实测 |
| --- | --- |
| 页面首次装载声道并完成首个姿态 | 5.67 s |
| Warm meta 请求，5 次 | 3.1–6.9 ms |
| 完整姿态快照，18组中位数 | 浏览器21.3 ms，本机原生24.8 ms |
| 0.5 秒静态发声，33/65截面及三声源 | 浏览器235–260 ms，本机原生228–302 ms |
| 5 秒带动作序列预计算 | 5.97 s，240000 样本 |
| 中断同步合成并确认停止 | 0.5 ms |
| 独立科学对照 Worker 的 Wasm 线性内存 | 288620544 B，约 275.2 MiB，非全标签总内存 |
| 对照期间主线程 16 ms 定时器最大间隔 | 22.8 ms |
| 首次所需运行资源，唯一 URL 字节合计 | 31581585 B，约 30.1 MiB，含共用 Python/NumPy/SciPy |
| VTL 与最小声道 Python/资源压缩包 | 约 0.40 MB，实际摘要以资源锁为准 |

0.5 秒静态发声的原生与浏览器原始计算时长、18 组姿态耗时、完整数组及各频带指标在比较 JSON；这些是本机观测，不能用于断言手机或所有桌面设备可实时合成。播放完成的预览和动作均无欠载。

## 命令和证据

```powershell
python scripts/build_m10_wasm.py --output-dir output/validation/m10-wasm-20261018/build --sdk output/validation/browser-compute-20261017/tools/emsdk-4.0.9 --write-lock resources/manifests/vocal-wasm-runtime.json
python scripts/build_m10_wasm.py --output-dir output/validation/m10-wasm-20261018/build --sdk output/validation/browser-compute-20261017/tools/emsdk-4.0.9 --refresh-runtime-only --write-lock resources/manifests/vocal-wasm-runtime.json
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/verify_m10_wasm.py --work output/validation/m10-wasm-20261018 --science output/validation/browser-compute-20261017/browser-science --frontend output/validation/m10-wasm-20261018/frontend-final
node --test frontend/tests/vocal-web.test.ts frontend/tests/vocal-wasm.test.ts
npm --prefix frontend run typecheck
npm --prefix frontend run contracts:check
npm --prefix frontend run ui-data:check
```

另以 `verify_m10_wasm.py ... --fallback-only` 人为拒绝运行包请求，真实原生服务器回退完整生成并播放24000采样/0.5秒，欠载0；报告在 output/validation/m10-wasm-20261018/browser-3af2c3de66854633b42b7231c15d610c/fallback.json。24项Python回归、9项前端定向、类型、契约、UI数据、最终构建、文档2260文件/446来源/157任务与架构检查通过，历史快照断链仍按原范围单列。构建保留既有chunk体积提示。未来验证器在自有scratch重新生成原生oracle，避免长期保存原始科学数组副本。

构建和对应源码/补丁/命令/二进制摘要：output/validation/m10-wasm-20261018/build/build-report.json。运行资源锁：resources/manifests/vocal-wasm-runtime.json。首轮完整浏览器报告：output/validation/m10-wasm-20261018/browser-ff308bc7d7be4f20bf38fe354357be36。最终候选：output/validation/m10-wasm-20261018/frontend-final、build/vocal-wasm；最终18组真实页面验收及独立科学对照：output/validation/m10-wasm-20261018/browser-913c4afc03de4916a82ccc9555128738。373862个几何/声管/轨迹数值最大绝对差1.13e-12；显示传递函数差0.001dB、未取整最大差0.000348dB、相位最大差0.0000372rad，共振峰网格相同。零抖动诊断的PCM16差为0LSB。预览/动作播放欠载均为0，页面和声道HTTP错误为0，浅深截图已人工检查。

保留的初次失败：①首次编译未显式指定 EXPORTED_FUNCTIONS，链接器删除未导出入口，修正导出清单后实际装载通过；②旧本机原生 DLL 无网页 33 截面入口，另从当前同源源码构建隔离 oracle，未替换桌面 DLL；③旧账号 double 无设置持久层，明确增加仅本机设置夹具，原登录/项目/视频 HTTP 保持；④性能 recipe 将单帧写成 5 秒，被原有 3 秒单帧上限正确拒绝，改为两帧各 2.5 秒；⑤ Node strip-only 不支持构造器参数属性，改用普通字段。上述问题均未绕过科研断言或修改产品限制。

## 未验与交付范围

本轮没有部署到网站、推送 GitHub、修改数据库/系统配置或重打 EXE。仍需后续独立验证其他浏览器、实体移动设备、首次公网下载速度、长期资源占用与多人环境。当前所有功能结论限定于本机浏览器候选；MP4 转码尚未下放。旧网页入口保持其原版本，只有提供新前端和对应锁定运行库的候选会启用 Wasm。

每次浏览器验证的自有scratch（含Chrome profile、合成JSON及MP4）已由产物生命周期工具清理，报告/日志/截图保留。额外按绝对路径、全目录无联接和无活动测试进程核对后，清理本任务首份过时前端、已消费原生/Node原始对照数组及诊断speaker共1270文件、481815647逻辑字节；最终前端、Wasm运行包、原生oracle组件和源码构建输入保留。回执为output/validation/m10-wasm-20261018/cleanup-owned.json，未读取或清理其他任务。诊断recipe已改为唯一文件并在finally删除，后续不遗留speaker副本。
