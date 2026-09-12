# M03-E3 方法与书目核查

2026-09-12。书目和方法差异已核查，完整代码授权链仍待核实。本文不改变兼容计算参数。

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
| 简化 CP 逆滤波 | 旧 LPC/闭相片段平均流程、数值和双WAV语义已对照 | 原始方法与实现授权链，不能以 Henrich 论文替代 |
| 科学依赖 | NumPy/SciPy/Praat 等继续使用各自来源登记和锁定构建 | 对应发行候选的完整传递许可集合 |

`REF-YIN-EGG-THESIS` 为旧手册引用的书目，`REF-HENRICH-2004-DEGG` 为本轮方法审阅参考。两者均只提供引用与原站链接。`PENDING-EGG` 的代码许可继续是 `not-established`。不把论文可阅读等同于代码或全文可再分发，不向作者发送邮件，不对外发布。
