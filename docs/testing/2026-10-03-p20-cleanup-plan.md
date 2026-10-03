# P20 项目空间审计与待删除清单

日期：2026-10-03。状态：prepared，已完成只读盘点，尚未删除。

首次盘点 v3 目录 51.09 GiB、321,513 个文件。没有读取相邻V2内容或跟随链接，扫描错误为0。此数值在本轮新构建之前采集，后续新包/构建/测试会增加占用。GiB=1024³字节。这里只统计文件逻辑长度，实际磁盘释放还受分配粒度和压缩等影响。

建议第一批清理 18.48 GiB，即 19.84 GB。共95个具体目标，37,251个文件，候选硬链接数为0，Git已跟踪文件数为0。删除须等待井井明确批准，并在新版R1验收后重核目标。没有生成自动删除命令。

| 类别 | GiB | 具体目标数 |
| --- | ---: | ---: |
| 旧打包中间归档 | 9.016 | 84 |
| P01旧探针EXE | 0.193 | 1 |
| 旧试用EXE与本轮首个未通过验收候选 | 0.594 | 2 |
| MFA测试安装副本与候选压缩包 | 5.404 | 5 |
| EGG暂存运行时副本 | 0.698 | 1 |
| 两份合成60分钟录音工程 | 2.576 | 2 |

## 验证目录的含义与清理边界

`output/validation` 是历次验收输出，首次盘点约27.24 GiB，包含合成音频、自然音频的测试副本、结果、截图、日志、哈希清单、临时数据库与运行组件测试副本。可以按具体用途清理，不能整目录按垃圾处理。当前第一批只挑选用途已经核对的MFA测试副本、EGG暂存副本和合成录音工程。报告、日志、请求/回执、截图与旧结果对照保留。

M16两份大工程由 `scripts/verify_m16_long.py` 生成，两份 `report.json` 均声明 accelerated synthetic 60-minute PCM stream，保存了成功状态、帧数与SHA-256。它们合计2.5756 GiB，不属于实体设备录制。

MFA候选安装目录是历史测试对象，其中报告为失败候选，正式调用的注册表位于 `output/m11c-028b881d/registry.json`。该注册表指向自身 `versions` 和自身 `checks`，新版EXE明确绑定此根目录。第一批测试副本清理不会包含此实际运行根目录。

## 第一批的具体路径

以下路径均相对于 `D:\PhoneticToolbox\PhoneticToolbox_v3`。目标绝对路径及逐文件相对路径、大小、修改时间与类别见 [机器可读清单](../../output/validation/p20-pack-cleanup/cleanup-candidates.json)。后续删除前重新核对路径、mtime、体积、链接与任务进程。

| 目标 | GiB |
| --- | ---: |
| `output/p01-probe/PhoneticToolbox-P01.exe` | 0.193 |
| `dist/research-repair/PhoneticToolbox-v3-Research-Fix1.exe` | 0.304 |
| `dist/PhoneticToolbox-v3-Latest-20261003/PhoneticToolbox-v3-Latest-20261003.exe` | 0.290 |
| `output/validation/m11/component-a4159343dbc44af180e2f2e361666ef9/installed` | 2.920 |
| `output/validation/m11/component-a4159343dbc44af180e2f2e361666ef9/mfa-3.3.8-windows-x86_64-candidate.zip` | 0.754 |
| `output/validation/m11/qt-642b638d008c497abb576e5db63dcbe7/components` | 0.409 |
| `output/validation/m11/wiring-9fd09b905ee94a0898b100c956fc58b6/components` | 0.661 |
| `output/validation/m11/wiring-e534ae9509814f67b461f88c3f4d1877/components` | 0.661 |
| `output/validation/m03-runtime/probe-20260913-a/payload` | 0.698 |
| `output/validation/m16/long-2d09a9a45246454cae8fae4f8bdeb4ec/project` | 1.288 |
| `output/validation/m16/long-628bb058477d4025b620e95609e9131a/project` | 1.288 |

旧打包归档仅列入 `.pkg`、`.pyz` 和 `base_library.zip`，排除所有 `snapshot` 以及最终R1构建目录。保留源码快照、spec、TOC、日志及既有审计报告，不整删历史构建目录。下表汇总每个构建目录，逐文件清单在上述JSON中。

| 构建目录 | 待删归档 GiB |
| --- | ---: |
| `output/build` | 0.406 |
| `output/build-PhoneticToolbox-v3-Latest-20261003` | 0.311 |
| `output/build-PhoneticToolbox-v3-LocalPreview-20260927` | 0.355 |
| `output/build-PhoneticToolbox-v3-LocalPreview-20260927-R2` | 0.355 |
| `output/build-PhoneticToolbox-v3-LocalPreview-20260927-R3` | 0.355 |
| `output/build-PhoneticToolbox-v3-LocalPreview-20260927-R4` | 0.355 |
| `output/build-PhoneticToolbox-v3-LocalPreview-20261001` | 0.356 |
| `output/build-PhoneticToolbox-v3-M16-M17-20261002` | 0.356 |
| `output/build-PhoneticToolbox-v3-M16-M17-20261002-R1` | 0.356 |
| `output/build-PhoneticToolbox-v3-M16-M17-20261002-R2` | 0.356 |
| `output/build-PhoneticToolbox-v3-P17-20261001` | 0.356 |
| `output/build-PhoneticToolbox-v3-P17-M04-R2-20261001` | 0.356 |
| `output/build-PhoneticToolbox-v3-P17-M04-R2-Final-20261001` | 0.356 |
| `output/build-PhoneticToolbox-v3-P17-R1-20261001` | 0.356 |
| `output/build-PhoneticToolbox-v3-P18-20261003` | 0.311 |
| `output/build-PhoneticToolbox-v3-P19-20261003` | 0.311 |
| `output/build-PhoneticToolbox-v3-P19-20261003-R1` | 0.311 |
| `output/build-m10` | 0.829 |
| `output/build-m12-preview` | 0.324 |
| `output/build-m12-preview-r1` | 0.324 |
| `output/build-m12-preview-r2` | 0.324 |
| `output/build-m12-preview-r3` | 0.324 |
| `output/build-m12-preview-r4` | 0.324 |
| `output/build-m12-preview-r6` | 0.324 |
| `output/build-research-repair` | 0.324 |

## 保留与后续候选

- 最终 `dist/PhoneticToolbox-v3-Latest-20261003-R1` 及其整个构建目录。旧P19-R1首次检查存在，末次复查已不在dist，变化来源未确认，本轮没有执行删除或移动，不能将其列为现存回退包。
- `.venv/m03-compatible`、`.venv/m05`、`.venv/m09-ui`、`.venv/m14`、`.venv/runtimes` 和实际MFA注册目录明确保留。其他环境也未列入删除，需逐个核对开发/验证命令和解释器home。
- `output/validation/m05/native-device-*` 等真实设备录制、原始研究材料、旧算法比较基线、数据库和现存用户资料均保留。
- `.venv/m03-package-cache` 约0.85 GiB为Conda包缓存，运行时通常使用安装目录，但离线重建用途仍在，未纳入第一批。
- `output/validation/p07-policy` 约1.94 GiB包含多份测试PostgreSQL数据目录，当前未观察到postgres.exe运行，但仍未将数据库目录纳入第一批。
- `frontend/node_modules`、旧实验环境和浏览器缓存可作为第二批候选，目前没有为这些对象作删除结论。
- 根目录 `.git` 是指向相邻V2共用Git元数据的工作树入口，保留。`phonetic_toolbox`含迁移来源及打包器仍读取的REAPER二进制，保留。

构建与成品验证后的末次盘点为52.13 GiB、322,743文件，见 `output/validation/p20-pack-cleanup/storage-inventory-final.json`。清单再次核对37,251文件的长度与mtime无变化。本轮没有删除、移动、新增备份、改系统环境、执行DDL、push或公开发布。
