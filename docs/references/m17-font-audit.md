# M17 固定字体审计

日期：2026-10-02。实际字体为 `frontend/src/assets/ipa-plus/PTBIPAPlus-Regular.ttf`，版本 PTB IPA Plus 1.000。这是模块内 OFL 派生资源，无系统安装，原 `DoulosSIL-Regular.ttf` 未改。

## 构建与许可

基础 Doulos SIL 7.000 SHA-256：`cc89b87c047bcc8dc00246398218bb6343e0f2372b153cf560c29a77f27068ef`。官方来源：[Doulos](https://software.sil.org/doulos/)、[版本历史](https://software.sil.org/doulos/history/)。原3944字形全部编译字节相同，仅追加23字形、组合定位及有限圈围连字；家族/PostScript名重命名，遵守 OFL 保留名要求。

Noto Sans Math 仅提供 U+27C5、27C6 两个文本替代分隔符轮廓，缩放合入派生字体。官方候选文件下载地址为 [NotoSansMath-Regular.ttf](https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSansMath/NotoSansMath-Regular.ttf)，SHA-256 `80b61fd613d3519197e64fff6f7e71fdc7f3e6526440ea4115b554ef7fd59af7`，[OFL 原文](https://raw.githubusercontent.com/notofonts/noto-fonts/main/LICENSE)。本地原始候选在 `output/validation/m17/source-fonts/`；未采用字体均移到此证据目录，仅派生 TTF 随产品打包。

许可证保存在 `OFL-PTBIPAPlus.txt` 和 `OFL-Noto.txt`，以 `?raw` 导入模块JS，帮助内可折叠查看，已实际检查生产bundle包含两份版权首行。项目新增轮廓注明 PhoneticToolbox contributors。字体许可不代表原论文/表图/中文译图取得同一许可。

可复现命令：

```powershell
python scripts/verify_m17_fonts.py --write
python scripts/verify_m17_fonts.py
```

脚本使用已有 Python fontTools，从原项目 Doulos 和上述候选 Noto 生成字体。缺少 Noto 输入时明确报路径和来源，不自动安装或静默改用系统字体。固定时间戳，生成字节可比较。

## 码位与塑形分别验收

机器报告 [`m17-font-coverage.json`](m17-font-coverage.json)记录当前字体哈希、23新增字形、264个目录所用字符及缺码位空列表。新增内容：六种组合括号 U+1ABB、1ABD、1AC1–1AC4，圈记号 U+20DD，空圈 U+25EF，两种分隔符，13个圈字母连字。

单字符圈形由原字形轮廓与独立圆/椭圆构成，13项分别检查 C、Ȼ、F、L、G、N、P、R、S、T、Ṽ、Ʞ、σ。部分清浊化括号用 mark-to-mark 定位到上下圈和下楔形记号，单侧与双侧分别显示；后续调整括号留白避免贴住小圈。原字形、组合框架与文本字符分开保存，不使用截图代替可复制Unicode。

Chrome DevTools `CSS.getPlatformFontsForNode` 对500个可见符号节点逐一检查，只允许 `familyName=PTB IPA Plus` 且 `isCustomFont=true`。因此目录符号不依靠系统fallback补齐。中文标签与手动输入的任意其他语言文字不在该字形覆盖声明内。

复杂字形接触表与实际小字号表分别检查。证据为 `output/playwright/m17/2026-10-02T07-36-14-260Z/font-shaping-{0,40,80}.png`、`font-platform-evidence.json`，以及后续最终E2E和Qt表截图。接触表检查了全部VoQS、上下/左右组合括号、SIL新增扩展字母及圈围，不能只依据 cmap 结果宣称正确。默认 extIPA 网格行高26px，符号22px，给下附加记号留空，较大页面30px行高。

多字符圈围的两项文本替代不声明为原图圆框；任意用户圈字串也不保证自动有跨度形态。源字体 Doulos 与组合字库的已有科学字形保留，不宣称这个字体覆盖所有Unicode或所有可能的发音记号组合。
