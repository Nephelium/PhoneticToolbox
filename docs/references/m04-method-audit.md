# M04 LPC 方法与代码来源

2026-09-13，按井井最新要求补学术引用，代码出处查到哪里登记到哪里。未确定的更早出处不阻碍功能实现。

主要方法参考：John Makhoul (1975). *Linear Prediction: A Tutorial Review*. Proceedings of the IEEE, 63(4), 561–580. [DOI](https://doi.org/10.1109/PROC.1975.9792)，[大学站点原文](https://www.commsp.ee.ic.ac.uk/~xl404/papers/Linear%20prediction%20A%20tutorial%20review.pdf)。已读原文pp.563–566关于自相关正规方程、对称Toeplitz矩阵、增益与预测参数求解的说明。登记ID为 `REF-MAKHOUL-1975-LPC`，引用关系为方法参考，不据此推断V2代码作者。

本项目实际计算由V2直接迁入：预加重系数0.97、Hamming窗、直接全自相关、Toeplitz方程求解，之后计算1024点全极点频响。`a=[1,-coeff]`、频响分子固定1，纵轴为 `20*log10(abs(H)+1e-10)`。没有加入预测误差增益，也没有原音频FFT叠加；这里的dB不是校准声压级。默认阶数50、预加重参数和绘图范围来自V2实际设置，不能声称是该论文推荐的统一参数。

实际调用的 [SciPy 1.16.3 solve_toeplitz](https://github.com/scipy/scipy/blob/v1.16.3/scipy/linalg/_basic.py) 使用Levinson–Durbin递推；[freqz](https://github.com/scipy/scipy/blob/v1.16.3/scipy/signal/_filter_design.py) 计算数字滤波器频响，当前参数不包含Nyquist终点。NumPy与SciPy属于未改动的运行依赖，没有复制它们的源码到LPC模块。现有兼容环境为NumPy2.2.6/SciPy1.16.3的Conda/MKL构建，不能用相同版本号的另一构建替代逐位证据。

代码出处查找范围：V2相关文件、该文件本地Git历史、公开精确组合检索 `compute_lpc_spectrum solve_toeplitz`。本地历史最早可见 `f6108ff` 初始导入，源码没有更早作者/项目注记，公开组合检索没有命中。结论只限未找到可核准的更早出处，不宣称原创或不存在外部来源。直接迁移的7份来源文件及哈希见 [迁移记录](../../third_party/m04-migration.json)。

书目已进入统一登记、BibTeX与公共致谢生成数据。LPC功能页面仍待D阶段接入，届时复用公共方法与引用入口。没有收录论文全文，没有增加再分发许可承诺。
