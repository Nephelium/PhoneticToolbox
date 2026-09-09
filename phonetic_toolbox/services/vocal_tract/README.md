# 声道工作台服务

入口是 API 的 `launch_vocal_tract()` / `shutdown_vocal_tract()`，返回 `VocalTractLaunchResult`。GUI 的 `VocalTractLaunchWorker` 仅将耗时启动移出 Qt 事件线程；浏览器由主线程使用返回的 URL 打开。

`../vocal_tract_service.py` 创建本实例专属子进程。开发态执行 `python run.py --vocal-tract-worker`，冻结态执行当前 EXE 的同一私有入口。端口由操作系统分配，只绑定 127.0.0.1。握手绑定实例随机值、父 PID、子 PID；不得读取原型 server.json 并接管旧进程。重复点击复用当前存活进程。

主窗口 closeEvent、QApplication.aboutToQuit 与 atexit 均走幂等关闭。先请求自己的 `/api/shutdown`，有界等待后只终止自己持有的 Popen 对象。Windows Job Object 使用 KILL_ON_JOB_CLOSE；工作进程另持有父进程内核句柄，父进程异常消失时退出。启动尚未完成时关闭应用也必须取消并回收子进程。原型的十分钟空闲关闭不适用于集成版。页面关闭发出停止音频请求；心跳中断超过五秒也结束音频。

`server.py` 处理同源 HTTP，Host/Origin 和 POST 会话令牌校验；网页目录之外的源文件、DLL、配置不可经静态路由读取。`animation.py` 在播放前生成 20 ms 音频块与 60 ms 显示帧，播放时按输出时钟读取缓存，保留末帧。`audio_output.py` 统一持续音、元音试听、测试音和关键帧音频的设备与音量路径。

用户状态位于 `%LOCALAPPDATA%/PhoneticToolbox/vocal_tract/`：`keyframes.json`、`audio-settings.json`、每主进程的握手与日志。关键帧经 `/api/keyframes` 原子保存，不依赖随机端口的 localStorage。输出设备以 host API 与名称保存，设备重排后重新定位；所选设备断开时报告错误，不暗中切换。多应用实例的进程独立，默认用户配置共享，最后保存的配置生效。

测试显式传入 `silent=True`，HTTP 拒绝物理播放；正常应用默认允许手动播放，从不自动发声。运行资源由 get_resource_path 定位，用户状态与 PyInstaller `_MEIPASS` 完全分离。

关键帧配置版本 2 包含 `frames` 和 `pitch_curve`，兼容版本 1；完整校验成功后才原子写回。动画 prepare 请求可附同一曲线；显示与音频沿统一时间轨迹采样。非法曲线不能覆盖既有姿势。

`POST /api/audio/monitor` 接受 `seconds`、`window_ms`、`hop_ms`，保持与其他 POST 相同的 Host/Origin/X-Session 检查。它仅分析最新输出历史，停止后保留画面，下一次播放清空并递增 generation。音频线程写入的是设备实际接收的浮点单声道通道数据（立体声两路相同），包含停止的短释放段；不获取系统其他应用声音或麦克风输入。分析不持有引擎计算锁。

声源设置通过 pose、关键帧与 animation 统一传递；旧关键帧无 source 时按中性浊声处理。监听设置另保存 `gain_db`，0–24 dB，旧设置缺省为零。放大在输出软限幅之前应用，记录到 OutputHistory 的仍是实际发送数据；不以声学参数变化代替监听增益。
