# 源码与测试基线

2026-09-09。从当前 v2 工作状态继承 427 文件（44,889,937 字节），并非仅旧 HEAD。源码基线 commit：ccf4ff73c355e8d955c2a6b9605b32ac2c7255a6；旧 HEAD：4eeff893eb6cdcddb40456072571dd5d30c51a76。独立 worktree 分支 codex/v3-rebuild。

[文件 hash 清单](source-manifest.json) 记录继承当时的字节；新根目录 README/ARCHITECTURE/AGENTS 的旧内容分别存为 [v2 README](v2-README.md)、[v2 架构](v2-ARCHITECTURE.md)、[v2 规则](v2-AGENTS.md)，仅为历史证据。

git checkout 曾按 EOL 设置转换部分换行；只在新工作区恢复与原目录等价的原始换行后，427 文件与原源码逐字节一致。后续编辑新文档不改变此时点源码清单；业务源码仍保持继承状态。原 v2 HEAD、暂存区 hash、状态和文件 hash 保留。

用户正在使用的桌面 EXE 与 v2 dist 同版本 EXE 的 SHA-256 均为 40c807cd9a58da11d1e87e805f9ea84cdb515b8998f99588e9ae788dDE218CDD（大小写无关）。这仅证明两个 EXE 相同，不证明当前所有未提交源码都在该 EXE 中。

## 环境和数据
用户确认开发环境为 conda phonetic_311，已核验存在、Python 3.11.14。当前安装 NumPy 2.2.6/OpenCV 4.13.0.92 与旧 pyproject 声明不一致；不要直接改旧环境。完整本机路径/状态/测试音频头信息只保存在被 gitignore 排除的 local-evidence.json，发布材料不包含它。

D0.3 当时只读取测试目录清单和 17 个 WAV 头，没有生成 v2 黄金结果。2026-09-09 后续 P03 已按授权完成 14 个固定输入的双轮独立捕获，含井井新替换并确认方向的双声道 EGG；没有上传或修改语料。详见 [捕获协议](capture-protocol.md) 与 [P03 报告](../testing/p03-baseline-report.md)。

## D0.3 当时没有做
未重写业务算法、运行新服务、安装新依赖、打包新 EXE、迁移数据库、删除旧网页或推送代码。继承 papers 和旧图标不构成 v3 发行分发许可。
