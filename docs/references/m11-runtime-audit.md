# M11 运行环境、模型与再分发审计

核对至 2026-10-11。使用已有环境做兼容基准，未安装或升级全局依赖。来源登记为 SRC-MFA／REF-MFA。下列实际环境与官方一般说明分开记录。

## 已观察的版本

既有 V2 `auto_alignment/env` 是 Windows x86_64 Conda 环境，MFA 3.3.8、Python 3.11.14、kalpy 0.8.2、Kaldi 5.5.1172、OpenFST 1.8.4、FFmpeg 8.0.0 GPL build、NumPy 2.2.6、SciPy 1.16.3、SoundFile 0.13.1、praatio 6.2.0。完整 193 个包的 build、来源、摘要、依赖、许可记录位于本机 `output/validation/m11/inventory.json`。这些 Conda 记录是已安装来源证据，单独不等于实际文件未改变；本轮另计算环境实际内容指纹及候选包逐文件 hash。

当前独立交付 ZIP 为 812,556,132 B，展开文件长度合计 2,805,971,894 B，包含运行环境、配套普通话资源与许可通知。运行环境排除缓存、字节码、旧重定位脚本及个人安装历史，Conda 元数据移除本机路径；科学库原字节保留。主 EXE 不收集此目录。

## 官方依据与边界

- [MFA 3.3.8 安装说明](https://montreal-forced-aligner.readthedocs.io/en/v3.3.8/installation.html)：环境需要原生依赖，不能将主程序的 Python 当成已具备 MFA 的证明。本轮直接复用已存在版本，不追随 latest。
- [3.3.8 全局配置](https://montreal-forced-aligner.readthedocs.io/en/v3.3.8/user_guide/configuration/global.html)：本轮通过独占 MFA_ROOT_DIR 和进程内配置隔离。所测方式使用每任务 SQLite，不启动 PostgreSQL，不复用 PhoneticToolbox 账号库。
- [MFA 3.3.8 LICENSE](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/v3.3.8/LICENSE)：程序为 MIT。不能将其直接套到整个 Conda 包、Kaldi 依赖、模型或词典。
- [conda-pack](https://conda.github.io/conda-pack/) 的可重定位方案要求相同操作系统，且解包修正后不能任意再次搬动。本轮没有安装 conda-pack，没有执行用户包中的任意安装钩子。候选包在最终短目录通过直接 Python／原生依赖／实际 MFA 自检，仍只作为所测 Windows 环境候选；它未证明所有命令行入口、长路径或其他 Windows 版本可重定位。
- [普通话模型 v2.0.0](https://mfa-models.readthedocs.io/en/latest/acoustic/Mandarin/Mandarin%20MFA%20acoustic%20model%20v2_0_0.html) 与 [普通话词典 v2.0.0](https://mfa-models.readthedocs.io/en/latest/dictionary/Mandarin/Mandarin%20MFA%20dictionary%20v2_0_0.html) 的官方页面标明 CC BY 4.0。本次使用 `2026-10-05-preview-source-preparation` 已准备的官方 v2.0.0a 身份材料，重新逐项核验本地文件与下载记录摘要，模型与词典的官方 README、许可随包提供。其他同名或旧版模型不据此认定许可。

## 本地模型身份

| 资源 | 大小 B | SHA-256 |
| --- | ---: | --- |
| mandarin_mfa.zip | 92,275,957 | bc16c3278295fad0d742399b849e5a8375a65bcbe5b7c7612ff1e3ffb81dcef7 |
| mandarin_mfa.dict | 8,709,372 | 6b71538ba4cd48d40f92561a129e4531ee256951fcf218ac1198da72ca13076e |

模型内部 metadata 的 MFA 构建版本为 `2.0.0rc4.dev19+ged818cb.d20220404`，架构为 GMM-HMM，采样特征设置 16 kHz。此字段不同于当前运行的 MFA 程序 3.3.8，也不能单独证明资源发行标签。本轮两者实际兼容任务通过。

## 发行状态

已生成 3.3.8-r1 独立组件及对应源码 ZIP，固定官网地址已写入源码，尚未上传或重打主 EXE。包内保留许可通知；配套源码含 193 个 Conda 包对应配方/补丁和 36 项已校验源码绑定，包括 FFmpeg、编解码库、Qt/PyQt。分发时两个 ZIP 的链接须同时提供。该材料不声称第三方 wheel 可逐字节重建；历史 `mandarin.zip` 不在此次包内。具体文件身份、验证和未验范围见[交付报告](../testing/2026-10-11-m11-component-delivery.md)。学术引用继续使用 McAuliffe et al. (2017)，[ISCA 原文记录](https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html)，它不代表实际使用的是 2017 年二进制。
