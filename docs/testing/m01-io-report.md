# M01-C 原生、格式与预算适配验收

2026-09-09，井井在 M01-B 后回复“继续”。**M01-C verified，限定 Windows 受控适配与合成双产物准备；完整 M01/P08 仍 in_progress。** 设计见 [C 实施记录](../plans/2026-09-09-m01-io.md)。下一项 M01-D 统一请求/结果契约与科研语义。

## 实现与实测

- 已有 REAPER 二进制 SHA-256 为 `279fecc82ed0a49b0277b114270771d7670299068e849058b392672825981824`，321,536 字节，PE AMD64。挂起创建进程、加入内存受限且关闭即终止的 Job Object 后才恢复。随机本机命名管道逐块强制输出字节限额，核对客户端所属 Job（包括虚拟环境解释器后代），拒绝外部客户端。只按所属进程句柄回收，不查杀端口、同名程序或 Codex。
- 实测正常 EST、128字节超限、取消、超时、非零/突然退出、后代终止和其他进程保留。64 MB Job 内申请512 MB的受控子进程得到MemoryError；16 MB导出进程不能完成并被回收。512 MB默认导出预算正常工作。
- Scratch先预留本操作磁盘预算，只生成随机物理名和规定后缀；拒绝调用者直接提供目录、未登记二进制与链接/reparse路径。REAPER没有任意输出目录入口；正常、取消和失败后自有暂存字节归零。
- WAV在解码前检查字节、RIFF/chunk长度、格式、采样布局、声道与样本数。保留实际采样率、原dtype和声道；支持小端RIFF PCM8/16/24/32、float32/64及有效位等于容器位的对应extensible格式。24-bit保持SciPy左对齐int32。切片保留`int(t*fs)`，明确拒绝负时间、越界和空片段；输出WAV也受字节限额约束。
- TextGrid支持UTF-8/BOM/UTF-16、长/短格式、IntervalTier，限定文本、层数、区间数和范围。中文、IPA、换行、双引号转义保留；Point/TextTier、畸形区间和重叠明确拒绝。
- 唇形`ptb.lip/1`采用`values + nonfinite`保存有限值/null及NaN/正负无穷mask；时间锚点、伴随起点、offset、稳定去重与四指标沿用B核心。本地旧PKL转换禁止全局、构造、REDUCE、persistent ID、扩展、超大memo和共享/循环容器，恶意构造未执行。未开放网页pickle入口。
- XLSX使用受限内存XML/ZIP写入，不走openpyxl隐式临时文件；SQLite只新建内存库，限定页数、单值长度和内存临时存储，随后序列化。两份内容及逐格回读全部成功才返回ExportPair。第二输出不足、内存不足、取消、暂存空间不足、单元格/文本超限均无部分结果返回。实际生成文件又经openpyxl/SQLite只读回读，列序、数值、NaN、文本与`idx_params_time`一致。

## 独立科研与实际文件证据

`scripts/verify_m01_native_io.py`执行WAV解码→安全关联格式→真实REAPER/科学核心→显示列重命名→XLSX/SQLite→独立回读。两例都是自有解析合成信号，不是自然录音。

| 用例 | 行×列 | 冻结数组差异 | REAPER EST | XLSX / SQLite |
| --- | --- | --- | --- | --- |
| ASSOCIATED | 160×83 | 0 | 3,521 B | 147,161 / 139,264 B |
| FORMULA-EXPORT | 160×83 | 0 | 3,521 B | 147,207 / 139,264 B |

安装wheel后的实际文件和hash位于忽略的`output/validation/m01/native-io-541d856a39bb4154b2ffc2911ee09238/`，各用例子目录包含两份结果；汇总为`report.json`。科学比较沿用原基准容差，导出逐格回读rtol/atol均为1e-12。XLSX包含生成时间，不以ZIP字节一致代替科学数组一致。

初始传输设计探针位于`output/validation/m01/pipe-probe-65b422570d974f919db99dda5881b660/report.json`，正常3,521字节与128字节拒绝均有实测。探针仅处理固定合成输入，正式适配用受限Job和Scratch能力，不将探针接入用户任务。

## 行为修正与验证结果

1. 原核心广义REAPER异常处理会吞掉取消/超限。新增轻量`BackendAborted`，在该边界重抛；不把中止改成NaN后继续计算。正常公式、数组和默认值未改。普通原生失败得到明确失败/核心unavailable元数据；Python后端按B的显式port选择，C不把Python结果冒充native。
2. 字符串单元格明确标为文本，`=literal`保持字面值，修复旧Excel公式解释问题。空表在构建产物前拒绝。TextGrid旧解析器的引号丢失和不完整结构静默接受也单列修正。
3. 初始试验中纯正弦没有REAPER有声点；正常频率测试改用明确谐波合成信号，完整数组另与独立冻结样例比较，没有扩大容差。32 MB足以运行最小导出，低预算测试使用16 MB；64 MB/512 MB分配实验另行直接验证OS限额。
4. venv启动器会生成解释器后代，单独比较启动器PID不正确。校验固定Job成员关系并缓存本次已认证连接；无须修改系统配置。

环境为`.venv/m01-io` / CPython3.11.14。科学8包保持B版本，增加A审计的openpyxl3.1.5与et-xmlfile2.0.0；worker当前装在ptb-api包中，亦补齐该包已声明的既有工程依赖。40个第三方包与2个本项目wheel，未安装Qt。工程锁的tzdata2026.3与科学2025.2冲突，因此逐项复用兼容工程版本，保留科学版本；`uv pip check`确认42个安装包兼容。

实际命令（解释器均为`.venv/m01-io/Scripts/python.exe -X utf8`）：

- `-m pytest -c tests/pytest.ini packages/phonetic_core/tests tests/parity backend/tests/test_m01_io.py tests/security/test_m01_formats.py tests/architecture -q --junitxml=output/validation/m01/c-wheel-final.xml`：**222 passed**。包括原核心/科研/转换/冻结保护149项、新适配69项、架构4项。最终来自安装wheel的site-packages。
- `scripts/verify_m01_native_io.py`：两组160×83实际双产物与冻结数组对照通过。
- `-m build --wheel --no-isolation --outdir output/validation/m01/c-wheels packages/phonetic_core`及同样命令的`backend`：两包构建通过；随后用项目uv `pip install --no-deps --reinstall --link-mode copy`将两wheel装入m01-io，核对实际导入路径。
- `scripts/check_architecture.py`、`scripts/validate_docs.py`、前端`ui-data:check`及`typecheck`、Git差异检查均通过。

来源登记323条与生成致谢一致。[C依赖清单](../../third_party/m01-io-inventory.json)保留新包安装文件hash及全部40包来源映射；[原生资源](../../resources/manifests/acoustic.json)保留未知commit/编译选项和发行许可缺口。B依赖/源码清单保留为历史快照；当前C文件差异见[M01-C证据](../modules/evidence/M01-io-migration.json)。

## 保存性与限制

`output/validation/m01/preservation-c.json`记录7项前后相同：v2 HEAD/index/status、427份基线检查、原包元数据hash和环境路径；P03的8份与M01-A的28份golden hash不变。Git根为`D:/PhoneticToolbox/PhoneticToolbox_v3`。未push、未上传用户目录、未操作Codex内置浏览器，未碰现存/服务数据库。

本轮未重现Codex闪退，也未执行之前可疑的内置浏览器关闭路径；不代表已修复Codex本身或保证永不闪退。

剩余：网页/桌面最终目录和manifest发布、PG配额/fencing/租约/7天保留、持久批次、完整UI、跨平台、自然录音及真实旧PKL全面兼容。含NumPy对象或自定义类的PKL、RIFF外格式、TextTier和特殊有效位packing明确不支持。C验证本操作预算、原生/导出进程和双产物准备；全科学worker硬隔离、持久发布与平台句柄由F集成，不能把本轮视为完整M01交付。
