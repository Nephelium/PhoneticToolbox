# M03-E3 方法与书目核查

2026-09-13 更新。V3 的直接迁移来源已确认是 V2 本地源码；书目和方法差异已核查，最初文献对应及完整代码授权链仍待核实。这些状态分别记录，本文不改变兼容计算参数。

## 已确认的 V2 代码来源

井井明确指向 V2 后，本轮重新检查相邻 V2 的原文件、函数及注释。路径均相对于 V2 根目录，未改写原文件：

| 行为 | V2 实现 | V3 迁入位置 |
| --- | --- | --- |
| GCI/GOI、峰谷与尺度/斜率分支 | `phonetic_toolbox/core/egg/analysis.py`，`find_gci_goi_peak_min_criterion`，6–186 行 | `packages/phonetic_core/src/phonetic_core/egg/events.py` |
| CQ/SQ | 同文件 `calculate_cq_sq`，188–279 行 | `packages/phonetic_core/src/phonetic_core/egg/metrics.py` |
| 自相关、LPC、简化逆滤波 | `phonetic_toolbox/core/egg/inverse_filtering.py`，三个函数，7–179 行 | `packages/phonetic_core/src/phonetic_core/egg/_inverse_legacy.py`；输入/取消校验由 `inverse.py` 包装 |
| 服务调用与参数传递 | `phonetic_toolbox/services/egg_service.py` | 纯数组服务与当前任务适配，详见迁移清单 |

[迁移清单](../../third_party/egg-migration.json)的八份 V2 源码 SHA-256 本轮全部一致。事件/CQ/SQ 文件为 `85d813ab64ae7e6c04b114346817569d5b3ade3bbdaa8325a4cd272af868af5d`，逆滤波文件为 `e4722762961445c4e0db8f2aeafa457a99b45ab240b876c32fb12446aeb83822`。

两份核心文件的 `git log --follow` 均只显示整仓导入提交 `f6108fff90788db0d1d2315ecba48e90b82913ec`（2026-03-20）。它证明当前可见历史中的导入节点，不能推出最初编写日期、作者或外部代码版本。对该目录及服务文件的作者、版权、许可、URL、DOI 和参考文献关键词检索，没有找到能补齐这些信息的声明。V2 代码已经找到，后续缺口是其方法文献和授权材料，不再把直接迁移来源列为未知。

## 简化逆滤波的实际步骤

以下依据上述 V2 文件及 V3 兼容层逐项读取，限定页面实际使用的默认调用：

1. 分析音频先按 `1 − 0.97 z⁻¹` 预加重。LP 自动阶数为 `int(fs / 1000) + 6`。
2. GCI 秒数乘以采样率后转整数索引。从该索引的下一个样本开始，取 `int(0.003 × fs)` 个样本，仅在 ROI 末端裁短。函数没有 GOI 输入，也没有按下一次 GCI 截断。因此 3 ms 是固定取窗假设，不能声称这些片段均已确认处于闭相。
3. 每段至少含 `LP 阶数 + 1` 个样本；用自相关和 Toeplitz 方程求 LPC，收集至少 3 个有效系数向量，再对系数逐项取算术平均。这里的有效是数值计算条件，未作独立生理判定。
4. 在整段预加重 ROI 上估计一阶 LPC 倾斜模型。以平均声道系数为分子、倾斜系数为分母滤波，异常或非有限输出时尝试 FIR 回退，之后去加重。公开包装层拒绝最终非有限结果。

这些步骤足以复核当前程序做了什么，不能仅由函数名推断它复现了某篇经典闭相逆滤波论文，也不把 IF 波形表述为经过生理真值验证的声门流。现阶段不因文献对照改成 GOI 自适应取窗、更换 LPC 求解或重新调参；这种科学行为变更应另建对照计划。

## 旧手册推荐论文

北京大学中文系图书馆的[学位论文目录](https://lib.chinese.pku.edu.cn/xueweilunwen?combine=&degree=All&degree_str=&field_xwlw_callno_value=&major=All&page=314&sort_by=field_xwlw_year_int_value&sort_order=DESC&year=&year_int_max=&year_int_min=)登记《汉语韵律的嗓音发声研究》，作者尹基德，专业语言学与应用语言学，导师孔江平，索书号 `020/D2010(71)`。单条记录地址 `/node/150402` 本次读取超时。已确认题名、作者和馆藏，未取得可核页码的完整论文。

旧手册 §3.3 图3-8 原图 `Phonetic_Export/images/img_7zxbhjwy4.jpg` 已在相邻 v2 中只读查看，SHA-256 `d3075669e5a6e6d38c0ea5b73516ac63582f9ae2bbc6365ceadea1a039d44c5b`。图中包含 Rothenberg、Henrich、Howard 等方法讨论，但截图没有页码和完整题名页。因此不能补写为论文某页已核准。截图不复制到 v3 或发行包。

截图所述混合方法用微分检测关闭点、42% 尺度检测开启点，随后作者选择尺度法，并讨论约20–25%的尺度。旧手册紧接截图作出的 GCI 斜率/GOI 尺度推荐，是软件手册自己的推荐。不能把它表述为已由该论文直接验证本程序的 0.25 混合算法。

## 可获得的原始论文

Henrich, N., d’Alessandro, C., Doval, B., & Castellengo, M. (2004). *On the use of the derivative of electroglottographic signals for characterization of nonpathological phonation*. JASA, 115(3), 1321–1332. DOI [10.1121/1.1646401](https://doi.org/10.1121/1.1646401)，[作者实验室 PDF](https://www.lam.jussieu.fr/Membres/Castellengo/publications/2004b-Voice%20DEGG.pdf)。已读取原文，以下页码为期刊印刷页：

- p.1323 区分 EGG 阈值方法、Howard 混合方法与本文的 DEGG 相关方法。
- pp.1327–1329 说明 DECOM 的相关估计流程。p.1329 的比较使用 Howard 阈值 3/7。

本项目 `events.py` 的导数极值分支不执行 DECOM 相关流程；尺度分支在每个局部谷峰幅度上固定取 0.25。保留原默认不意味着实现 Howard 的 3/7 方法。该论文登记为方法审阅参考，未移植其代码，也不将其列为本模块实现来源已闭合的证据。

## 代码关系与未决项

| 项目 | 当前可确认事实 | 尚未确认 |
| --- | --- | --- |
| GCI/GOI | 原 v2 文件哈希、四方法组合及 0.25 实际分支已保留 | 最初编写者引用链、代码再分发许可 |
| CQ/SQ | CQ 接触时长/周期；SQ 是两阶段时长差除以接触时长，独立缺失 mask | 此 SQ 定义的原始文献对应，不称常规速度商比值 |
| 简化 CP 逆滤波 | V2 固定 GCI 后 3 ms、自相关 LPC 系数平均流程、数值和双WAV语义已对照 | 原始方法与实现授权链，不能以 Henrich 论文替代 |
| 科学依赖 | NumPy/SciPy/Praat 等继续使用各自来源登记和锁定构建 | 对应发行候选的完整传递许可集合 |

`REF-YIN-EGG-THESIS` 为旧手册引用的书目，`REF-HENRICH-2004-DEGG` 为本轮方法审阅参考。两者均只提供引用与原站链接。`PENDING-EGG` 的代码许可继续是 `not-established`。不把论文可阅读等同于代码或全文可再分发，不向作者发送邮件，不对外发布。
