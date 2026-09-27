# M05-A 原行为与独立基准

2026-09-27，verified **仅限以下 Windows 公开工程素材与纯函数范围**；完整 M05 in_progress。

- V2 说明书 7.1–7.4 和四份指定源码完成映射，见 [M05-source-map](../modules/evidence/M05-source-map.md)。原始来源 hash、模型 hash 和依赖版本在 `tests/fixtures/m05/v2.json.gz` 的 producer 中。
- 原环境 MediaPipe 0.10.14、NumPy 2.2.6、opencv-contrib-python 4.13.0.92。原环境仅以 `-B` 读取执行，没有安装或改动。
- NASA 肖像经 scikit-image v0.19.3 固定 URL 获取；使用条件/来源/hash 在 `resources/m05/resources.json`。局部裁切、遮挡、平移、二维旋转和两种分辨率共 3 个无损 FFV1 工程视频、90 帧，其中一组真实 VFR 编码。逐帧 PNG→RGB 与 FFV1 解码 RGB、PTS 已精确检查。
- 独立捕获执行原 V2 `metrics.py`、`LandmarkStabilizer` 和原 `_recognize_video_frames` 方法。只替换 Qt 进度显示为无界面 stub，不替换检测、指标、防抖或补全。filter off/on 共 6 场景、180 帧；两个新原环境进程的全部结果完全相同，冻结为测试 expected。额外 12 组解析坐标、3 截止频率冻结防抖边界。
- Python 原样迁移 13 项测试通过，逐值相等、NaN/索引/关键点 shape 一致。JS 指标/防抖/offset/预算 3 项通过测前阈值。JS 面积初次失败归因于原 strided float32 dot 的两项分组求和，按 OpenBLAS 0.3.29 对应步序修正，未改容差。此浮点适配只声明本基准平台，不扩展为所有 BLAS 构建一致。
- 离线 adapter 2 项测试通过：真实 VFR PTS、检测缺失、取消、分辨率/帧数上限、输出预算与短写失败。第一次 CLI 实测得到 30 帧、18 检测成功、12 缺失，原始缺失没有变成测量。

## 实际命令

```powershell
& '<原 phonetic_311>/python.exe' -B -X utf8 scripts/m05_prepare_fixtures.py
& '.venv/v3-dev/Scripts/python.exe' -X utf8 scripts/capture_m05_v2.py
& '.venv/m05/Scripts/python.exe' -m pytest -c tests/pytest.ini tests/parity/test_lip_extraction.py -q
node --test frontend/tests/m05.test.ts
& '.venv/m05/Scripts/python.exe' -m pytest -c tests/pytest.ini backend/tests/test_m05_video.py -q
& '.venv/m05/Scripts/python.exe' -X utf8 scripts/m05_analyze.py output/validation/m05/inputs/motion-occlusion-vfr/input.mkv --output output/validation/m05/offline-vfr
```

证据：`output/validation/m05/v2-1.json.gz`、`v2-2.json.gz`、`inputs/manifest.json`、`offline-vfr/`。脚本拒绝覆盖已有基准/结果目录。

## 未覆盖

本基线阶段使用公开素材。井井随后于同日明确授权真实设备和本次录制自然视频，其新证据见 [完整报告](m05-report.md)。不存在自然张口/真实侧脸/真实快速运动/声学同步真值证据。本基线阶段没有录制用户或上传研究语料。长片段/导出回读、正式 UI 与公共任务完整链路仍待后续阶段。本报告不宣称科研生理有效性。
