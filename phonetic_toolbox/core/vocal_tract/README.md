# 声道核心

`Engine(resource_dir=...)` 注入包含 VTL 2.4 DLL、独立分析 DLL、geometry_p2.dll 和 speaker 文件的只读目录。核心不读取用户配置，不启动服务器、浏览器或音频设备，不导入 GUI/Services。VTL 的静态全局状态要求每个工作进程只创建一套 Engine；三份原生状态分别用于声源合成、上游分析核对和几何计算。

`prepare_tube` / `snapshot` 使用相同构形与独立唇宽。宽度缩放原生唇表面的左右轴，再计算横截面和面积函数；不是仅在前端改变外观。宽度 1 与上游默认面积一致。`constrain_nasal_opening` 只约束腭咽开口 VO，不悄悄改舌位；它保护端点元音口腔通路，不是对整段动程的碰撞证明。

`trajectory.validate_frames` 校验 19 个控制参数、唇宽、F0、时长；播放为 2–12 帧、每帧 0.15–3 秒、总时长 ≤12 秒。存储可为 0–12 帧，也允许暂存超长序列供继续编辑。`sample_frame` 使用 smoothstep 连续插值参数、唇宽、F0，末帧保持至其停留时间结束。

软腭和鼻腔的二维/三维显示几何位于网页渲染模块；声学面积和合成始终来自本核心，显示重建不替代声学计算。鼻腔声学使用 VTL 分支管道，外部平均鼻腔网格不直接进行 CFD 或有限元声传播。

来源、修改和许可见 `../../resources/vocal_tract/THIRD_PARTY_NOTICES.md`。原生适配和派生桥接代码标明 GPL-3.0-or-later；主仓库其他代码的条款不因此被这里重新声明。

`validate_pitch_curve` 接受空曲线或 2–201 个 [归一化时间, Hz] 点，时间严格递增并覆盖 0–1，F0 为 60–350 Hz。`sample_trajectory` 先取器官姿势，再按总动程长度线性取样手绘 F0；不使用插值可能过冲的高阶曲线。

`monitor.OutputHistory` 是线程安全的 12 秒样本缓冲，无音频设备或文件访问。`analyze_output` 返回保峰值波形包络和有时间坐标的 Hann 窗幅度谱；5–80 ms 分析窗、5–40 ms 请求步长、1–12 s 可见时间范围。长范围受 320 时间列限制，返回实际步长；频谱幅度采用满幅正弦 0 dBFS 标定，显示动态范围 90 dB。窗口长度的倒数是频率细节尺度提示，不等同于 Hann 主瓣宽度或 FFT 频点间隔。

`source.validate_source` 校验 Pa、mm、mm² 及相对振动幅度；`interpolate_source` 为不同声源模式生成连续过渡状态。Engine.glottis 接受可选 source，不传时保持旧的中性声源参数；清声和耳语原生振动幅度为零，气流噪声由 VTL 的 TdsModel 生成。声门开度是内收的几何近似，不代表组织刚度或肌肉力。源初值集中在 Models，后续自振模型必须明确区分给定 F0 与自然产生的 F0。
