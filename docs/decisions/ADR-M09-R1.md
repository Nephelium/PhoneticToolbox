# ADR-M09-R1：基于原始相位的音频频谱绘制

状态：accepted（本轮用户明确授权），限定 Windows 源码与本地真实任务 verified。Linux 预览与实际 PostgreSQL 未验，见[报告](../testing/2026-10-05-m09-r1-report.md)。

在既有语谱图转音频任务中增加 image_draw / audio_draw 模式，默认 image 继续读取旧 m09/1 请求。音频源使用 audio 角色，图片源使用 image 角色。预览通过相同鉴权与源哈希检查后交给有界子进程，只返回灰度 PNG 和网格参数，不保存相位到浏览器。提交时从不可变源重新计算相位，并应用完整笔迹快照。

旧 Job 表的快照限制为 16 KiB。较大的笔迹经既有托管输入文件机制保存为带 SHA-256 的 spectral_drawing 引用，小笔迹仍可内嵌。入队前校验源归属，写入后再次校验，再原子绑定任务；执行与重试均校验笔迹文件，结果 JSON 展开保存完整笔迹。沿用已有存储预算与过期规则，不修改数据库 schema。创建/预览 HTTP 请求限定 1 MB，最多 256 笔、8192 个点。旧 image 请求快照字段与幂等哈希保持兼容。

音频使用 SciPy STFT/ISTFT（Hann，75% 重叠，边缘补零），幅值 dB 使用 20 log10，默认 60 dB 显示范围。只改笔迹覆盖的时频单元，未编辑单元保留原始幅值，不把显示量化灰度反向覆盖整个原谱。零幅值没有可用相位，约定其相位为 0 并记录。仅当编辑声道峰值超过 1 才以统一增益限制到 0.99，记录增益。WAV 使用 FLOAT，保留原时长、采样率、声道和未编辑声道。

图片保持旧 10 log10-times10 映射及 Griffin–Lim 路径，两者分别标记方法版本，不静默统一。图片先透视变换，再在校正网格上绘制。预览与任务复用同一校正函数，避免两套坐标解释。

修改后的复频谱不一定满足 STFT 一致性，逆变换产生的真实音频再次分析时可能偏离绘制目标。结果并列展示目标与输出频谱；本功能不承诺加密、隐蔽性或压缩后的信息保真。

依据：[SciPy ISTFT](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.signal.istft.html)、[OpenCV 透视变换](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html)。实际依赖沿用项目锁定的 SciPy 1.16.3 / OpenCV 4.13.0.92。
