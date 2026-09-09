# 2026-09-06 代码审查问题修复

范围：修复本轮审查确认的五项问题，保留工作区原有改动，不修改版本号、不重新打包。

1. 网页扫描使用扫描 UUID 与绝对路径生成不可复用的文件 ID。切换到其他目录或重扫相同目录后，旧标签页无法将 TextGrid 或唇形偏移写给当前语料；返回错误，原文件保持不变。
2. 网页和参数导出共用 `resolve_lip_time_axis`。有绝对时间和音频起点时保留首帧延迟，优先采用元数据中的首个音频帧时间，再使用配套时间戳文件。仅有相对时间的旧录音仍沿用首帧归零；已有 `anchored_audio_start` 标记的相对时间保持原值。不迁移或改写原录音。
3. 网页加载以任务序号、取消信号和文件 ID 共同防止过期结果更新。音频、TextGrid、唇形加载和重新扫描均受保护；保存响应不清除请求期间的新编辑。
4. 每次连续统输出进入新批次。六组合成共用批次根目录，保存源/目标路径、分析及生成配置、实际使用的 F0 CSV 和输出清单。清单先标记 `incomplete`，全部完成后原子替换为 `complete`；取消或失败不会污染已有批次。
5. LPC 尾帧补零，残差与重合成写入范围限制在分析区间；保持输出原长度和区间外数据（响度匹配/峰值限制仍按原选项作用于整段输出）。

## 验证

- Python/Qt/HTTP 回归：`conda run -n phonetic_311 python -m pytest -q -o addopts='' -p no:cacheprovider`。
- 前端异步行为：`node --test phonetic_toolbox/tests/web_praat_async.test.cjs`，也由 pytest 调用；没有 Node.js 时该项会明确跳过。
- JavaScript 语法：`node --check phonetic_toolbox/gui/resources/web_praat_editor/app.js`。
- 重点覆盖：相同/不同目录重扫后的旧请求、五类时间轴的网页保存与参数导出往返、响应乱序及迟到失败、保存期间的新编辑、9 步后再生成 3 步、取消输出、不同长度尾帧与区间外样本。
- 未重新打包 EXE，未进行摄像头/麦克风硬件录制验证。

最终结果：pytest **110 passed in 11.56s**（包括调用 Node.js 的前端回归入口）；Node.js 异步行为用例 **6/6 通过**，JavaScript 语法检查通过，`git diff --check` 通过。另以临时生成的 120 Hz / 170 Hz 音频执行真实 Praat F0 提取、LPC 分析、连续统合成和 WAV 导出，连续生成 3 步、2 步进入不同批次，均保留每步 5513 点并生成有效完成清单；未接触用户录音。
