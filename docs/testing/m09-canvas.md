# 音频藏信息语谱图与画笔验收

日期：2026-10-18。范围：Windows当前源码Qt、独立Chrome候选及实际Pyodide本机计算。Mac、Windows、网页共享页面、绘图和Python核心已修改，最终成品及线上环境分别核验。

## 当前行为

音频选中后自动显示完整时间范围的可编辑语谱图，只有一条音频时自动选中。图窗宽度随面板，高度默认320px，可拖动或输入设为180–800px；切换音频、声道或FFT保持选定图高。显示上限默认5000Hz，可设至源Nyquist频率，带时间/频率刻度；改变频率上限和图高保留并重新投影笔迹。网页使用原账号设置接口保存这两项显示设置。

两种绘图模式的1–200px画笔使用相同像素中心距离规则。新增音频画布坐标按覆盖面积映射到原STFT，保留原相位、未覆盖单元和另一声道，旧快照缺字段时保持原路径。设计见[决定](../decisions/ADR-M09-audio-drawing-grid.md)。

## 输入与检查

- 自生成16000Hz、9217帧FLOAT双声道WAV，左声道400/1600Hz正弦叠加，右声道1700Hz正弦，时长0.5760625秒；401×401纯白PNG核查画笔。全程未读取用户原始音频。
- Python定向54项通过：核心笔迹/频带/面积覆盖、旧请求摘要、预览/四产物、边界、桌面桥及旧重建parity；前端M09/设置/契约24项通过。类型、生成契约双向检查、UI数据及`build:workbench`通过，保留活动窗口旧哈希资源。
- Windows真实Qt：自动38×513源谱显示、FFT512/1024/2048及声道自动更新、图高稳定、3000/8000/5000Hz上限及笔迹保留、图高调整、1/12/22/200px精确像素直径、缩放轮廓、撤销重做和独立模式笔迹。真实任务四产物保存回读验证原相位、16000Hz/9217帧及未编辑声道逐样本相同，浅深/1440×1000和1100×800截图。
- Chrome155真实Worker运行当前Pyodide/同源源码ZIP，覆盖自动预览、全部FFT尺寸稳定、频率轴刻度、显示设置不重复计算、原生右下角图高拖动、图片画笔精确直径及200px连续笔迹。真实重建四产物回读验证原相位和未编辑右声道FLOAT样本，四种浅深/宽窄布局，无API研究材料上传。

本机Chrome直接读取测试ZIP得到HTTP204及空正文，Node HTTP读取为完整字节。本轮Chrome验证使用Playwright API取得自有HTTP服务的原响应并传入实际Worker，未替换计算结果；Worker仍执行运行库SHA校验。该证据覆盖共享界面和Wasm计算，不代表线上下载/服务器账号端到端已验。

## 复现命令与证据

Python使用`.venv/m14/Scripts/python.exe`，PYTHONPATH含`backend/src`、`packages/phonetic_core/src`及`desktop/src`。

```powershell
.\.venv\m14\Scripts\python.exe -B -X utf8 -m pytest -c tests/pytest.ini -o addopts='' -q packages/phonetic_core/tests/test_m09_editing.py backend/tests/test_m09_r1.py backend/tests/test_m09_boundary.py desktop/tests/test_m09_bridge.py tests/parity/test_spec2wav.py
node --test frontend/tests/m09.test.ts frontend/tests/web-settings.test.ts frontend/tests/account-preferences.test.ts frontend/tests/contracts.test.ts
.\.venv\m14\Scripts\python.exe -B -X utf8 scripts/verify_m09_canvas_qt.py
node tests/e2e/m09-canvas.cjs
```

报告、日志和截图位于`output/validation/m09-canvas/`的各自运行目录。首轮Qt实测1px覆盖3×3px，已修复并追加精确断言；后续Qt缩放指针探针受滚动布局时序影响，脚本改为滚动完成后定位、按下后等事件处理，再核查真实指针。浏览器测试临时构建/配置、原生拖动和传输均记录在脚本及各轮报告中，失败未删除断言或放宽画笔直径标准。

最终Qt报告`qt-6611c2dfd2214f02bbacfb70fc965179/report.json`为10组、4种布局通过，系统DPR1.5；最终Chrome报告`chrome-1792318730295/report.json`为11组通过，两个最终scratch回执均为`deleted`。连续200px笔迹按增量栅格化当前段，整笔以同一不透明度混合，避免每个指针事件重画全部历史点。

生成音频、隔离库、浏览器配置与测试构建按scratch回执清理。新的网页运行库候选保留在`output/validation/m09-canvas/browser-runtime/`，由既有`prepare_browser_science.py`生成，编辑核心、接口模型、设置白名单三个修改源与ZIP及锁定清单逐字节一致。

文档检查2318文件、447来源、163任务零错误，架构检查零错误，定向diff/台账JSON/JS语法检查通过；未改动历史快照的旧断链单列在文档日志。

## 边界

Mac实机Qt/Safari、服务器账号设置持久化端到端、线上直接下载/服务回退、最终Windows EXE和Mac应用未重验。未打包、部署或推送。已打开的源码窗口需重启后载入当前构建。
