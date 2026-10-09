# ADR-M11-R1：显式转写来源与 CPU 运行开销

2026-10-04，井井当前授权范围。实现/验收以 [报告](../testing/2026-10-04-m11-r1-report.md)为准。

新增转写格式选择在导入适配层决定资产，同一目录可以保留 LAB/TXT/TextGrid。任务仍为 m11/1，每条记录已有 transcript_format，新默认 100/400 不改写显式历史值。不变更现存数据库 schema。

MFA 3.3.8 的 run_kaldi_function 在 USE_MP=False 时仍启动 Worker，USE_THREADING=False 在 Windows 会启动新的 Python。保持 NUM_JOBS=1、USE_MP=False，使用官方 USE_THREADING=True，并直接设置子进程配置，移除被 PretrainedAligner 忽略的配置参数。算法、词典发音和搜索参数不为加速而缩减。

CheckedAligner 调用上游 normalize_text 后检查被排除发音与上游 save_oovs_found，错误先于词典图编译和 MFCC。成功任务仍完成原 setup、align、export、TextGrid 解析和原子结果发布。唯一 words 层重建整段转写是显式选择 TextGrid 时的输入适配，原 times 不作约束，转换记入 provenance。

运行环境文件仍逐一计算 SHA-256，4 个读取线程、64 文件有界批次，仅加快 I/O。按原相对路径排序组合，摘要算法与旧实现一致，不缓存文件摘要。Numba 库函数缓存允许跨任务复用，目录由宿主的已验证完整运行环境指纹决定，不接受网页路径。每次任务照常全量校验，不复用用户数据或 MFA 数据库，缓存与任务共同监测原 512,000,000 B 软预算。

当前主机安装的 Kaldi 包 build 为 cpu，普通话模型 metadata 为 GMM-HMM。MFA 3.3.8 当前对齐链没有可切换的 GPU 设备选项，保持 CPU，不提供无效 GPU 切换控件。更换对齐引擎须另行讨论科学行为和模型兼容性。

MFA 体积仅只读盘点现有已解包运行环境与之前已有 ZIP 中央目录。词典和声学模型分开，不生成新包、不修改打包配方。后续公开发行与全依赖许可要求仍沿用原审计边界。
