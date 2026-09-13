# M04-A V2 独立基准

2026-09-13，verified（限定Windows原V2数值/服务文件行为及导出函数）。完整M04为in_progress，V3 LPC页面、核心迁移和任务接入均尚未实现。

## 实测结果

`& .venv/m09-ui/Scripts/python.exe -X utf8 scripts/capture_m04_baseline.py --freeze-public`通过。原`phonetic_311`两个独立进程，NumPy2.2.6、SciPy1.16.3，原V2服务/核心实际导入路径已核对。12个计算场景、38组数组、44,497个值逐字节一致，元数据及导出PNG像素一致。6份源码和说明书共7文件哈希在运行前后不变。

证据：`output/validation/m04/baseline-fb2a43a99a8d4100b01f5f9ce128cafa/`。可公开冻结文件位于 [tests/fixtures/m04](../../tests/fixtures/m04)，仅确定性合成数组、参数、结果、哈希和错误信息，无自然语料或原PDF/手册媒体。

- 8个返回场景：默认、动态2k/48k上限、阶数1/200、96k采样率、常量、最短52样本。4个错误：静音、51样本过短、NaN、Inf。
- PCM16/PCM32/uint8/float32及双声道读取的原始值与float64单声道结果均留存，WAV未被读取流程改写。
- 6组TextGrid标签、4次层选择及缺少同名文件已检查。`(0,1)`和`(.15,.85)`均得到`ɑ̃˥+b`，保留旧末区间排除及去重规则。
- 直接调用原导出构图和保存函数，输出2400×1350、PNG标称300DPI（读取pHYs换算299.9994），白底黑线。中文/IPA用于文件名，图内原函数只有英文坐标轴；不能据此宣称V3公共字体已验。

原GUI类仅用于调用不依赖窗口实例的构图函数，未创建/展示Qt窗口，未测试原生播放设备或GUI交互。导出图已查看，是V2合成基准图。

首次运行 `baseline-72d75efc5fbc4400b37b19914a640965` 在解析频率网格的数学断言失败：`freqz`弧度转Hz末点为7992.187500000001，与解析值7992.1875相差一个ULP。仅该独立数学断言改用一个ULP，两个原进程的数组对照仍严格逐字节，无容差、无科学源码修改。失败记录保留。

## 当前决定与边界

完整功能/差异见 [源码映射](../modules/evidence/M04-source-map.md)，后续见 [实施计划](../plans/2026-09-13-m04-implementation.md)。已纠正旧规划将LPC等同Praat和暗示原FFT叠加的文字。V2默认算法原样保留；固定频率网格、显示上限、选区与时间轴、标签规则分别记录。

本轮不将旧模块标为已迁移，不改V2/第三方环境、不执行DDL、不启动其他模块、不打包或push。下一步M04-B：纯数组核心与兼容环境对照、计算预算验证，再接任务与页面。引用核查仍暂停。

## 收尾检查

脚本补全shape/dtype/bytes显式比较后，省略冻结开关再次执行，`baseline-ac6879de1b7e4e2b9017e8591dec9e08`双轮通过，已有冻结文件未覆盖。冻结arrays/result文件SHA-256及38数组44,497值独立回读通过。`scripts/validate_docs.py`检查583文件、330来源、32任务，errors=[]，历史快照失效链接单列；`scripts/check_architecture.py` errors=[]；`git -c core.safecrlf=false diff --check`通过。未改前端运行代码，无需为基准/文档修改重建当前EGG界面。
