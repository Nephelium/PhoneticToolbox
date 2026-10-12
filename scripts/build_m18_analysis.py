"""Build the editorial analysis using bundled ReportLab, separate from app dependencies."""
from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, Rect, String, Line

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'; OUT.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont('ReaderSong', 'C:/Windows/Fonts/simsun.ttc', subfontIndex=0))
INK = colors.HexColor('#292722'); ACCENT = colors.HexColor('#AD5138'); MUTED = colors.HexColor('#706A63')
styles = {
    'title': ParagraphStyle('title', fontName='ReaderSong', fontSize=23, leading=34, textColor=INK, spaceAfter=18),
    'h': ParagraphStyle('h', fontName='ReaderSong', fontSize=15, leading=24, textColor=ACCENT, spaceBefore=12, spaceAfter=9),
    'p': ParagraphStyle('p', fontName='ReaderSong', fontSize=11, leading=20, textColor=INK, spaceAfter=11, wordWrap='CJK'),
    'small': ParagraphStyle('small', fontName='ReaderSong', fontSize=9, leading=16, textColor=MUTED, spaceAfter=8, wordWrap='CJK'),
}
story = []
def p(text, style='p'): story.append(Paragraph(text, styles[style]))
def page(title):
    if story: story.append(PageBreak())
    p(title, 'h')
def table(rows, widths):
    values = [[Paragraph(str(cell), styles['small']) for cell in row] for row in rows]
    t = Table(values, colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEABOVE',(0,0),(-1,0),1,INK),('LINEBELOW',(0,0),(-1,0),.6,INK),('LINEBELOW',(0,-1),(-1,-1),1,INK),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#F5EEE8'))]))
    story.extend([t, Spacer(1,14)])

p('让构音成为合成的控制入口', 'title')
p('Articulatory Source-Filter TTS<br/>论文解读与方法分析', 'h')
p('阅读对象：Bandekar、Watanabe 与 Ghosh 的 arXiv:2610.00735v1，2026-09-30。本文按固定版本核对。', 'small')
p('这篇论文值得读的地方，是把构音轨迹放进文本到语音的生成链。它提供了一种研究思路：先学习声音与构音的对应关系，再把预测轨迹作为可编辑的中间目标。精读时应分别判断合成质量、控制效果及构音表征的真实性。三类证据的验收条件各有侧重。')
p('先带着三个问题读', 'h')
p('第一，模型使用的轨迹从哪里来？第二，改变一条轨迹时，实际改变了哪些声学属性？第三，离开训练说话人后，轨迹与真实构音还有多大对应关系？带着这些问题，图 1、表 III、IV 和 VI 比摘要中的总体评价更有判断价值。')
p('本文的阅读立场', 'h')
p('本文为 PhoneticToolbox 编写的独立解读。标为原文证据的数字和结构可回到所列章节、表格核对。方法评论、普通话研究方案与实现建议属于本解读的分析，未在当前工具箱中实施，也没有作为该论文的实验结果。')
p('原作及中文译文按 CC BY 4.0 分发；本解读亦采用 CC BY 4.0。AI 辅助编写，未经原作者审校，不代表原作者认可。原作作者：Jesuraj Bandekar、Shinji Watanabe、Prasanta Kumar Ghosh。', 'small')
p('<link href="https://arxiv.org/abs/2610.00735v1" color="#AD5138">原文与固定版本</link>　<link href="https://creativecommons.org/licenses/by/4.0/" color="#AD5138">许可</link>　<link href="https://coding-phoenix-12.github.io/ArticulatorySFTTS/" color="#AD5138">作者演示</link>', 'small')

page('1　方法路线：真实监督、伪目标与可编辑预测')
p('原文证据：图 1、III-A 与 III-B。真实 EMA 用来训练声学到构音反演模型（AAI），模型随后给 TTS 语料生成伪轨迹。合成时，由文本预测构音、音高和能量，构音条件进入滤波器分支，声源经 OT-CFM 细化后与滤波器合成。', 'small')
d = Drawing(480,135)
for x,label in [(0,'真实 EMA + 声音'),(165,'AAI 生成伪轨迹'),(330,'构音条件滤波器')]:
    d.add(Rect(x,80,145,40,fillColor=colors.HexColor('#F5EEE8'),strokeColor=ACCENT,rx=5))
    d.add(String(x+72,95,label,fontName='ReaderSong',fontSize=10,textAnchor='middle',fillColor=INK))
    if x<330:d.add(Line(x+147,100,x+161,100,strokeColor=ACCENT))
d.add(Rect(165,10,145,40,fillColor=colors.HexColor('#EEEFE9'),strokeColor=MUTED,rx=5))
d.add(String(237,25,'文本 → 音高 / 能量 → 声源',fontName='ReaderSong',fontSize=9,textAnchor='middle',fillColor=INK))
d.add(Line(311,30,402,30,strokeColor=MUTED));d.add(Line(402,30,402,78,strokeColor=MUTED))
story.append(d)
p('上图是本解读绘制的概念示意，省略了对齐、说话人条件和训练损失。它帮助区分两条控制路径，不能代替原文图 1 的完整网络。', 'small')
p('为什么要追踪伪目标', 'h')
p('伪目标把反演模型的归纳偏差传给下游。如果第一阶段在某类发音上系统性偏离实测轨迹，TTS 即使很好地复现伪目标，也可能只是稳定复现该偏差。评估时应分别保存实测值、AAI 预测和 TTS 预测，避免把它们合并为一个构音真值。')
p('物理名称提供了什么', 'h')
p('舌尖、舌体和唇的名称能帮助设计干预，也能帮助发现不合理动作。不过，有名称的通道仍需要验证单位、归一化方式和传感器位置。声道截面积、三维舌形和二维传感器坐标之间，需要额外的几何模型和测量证据。')

page('2　跨说话人证据：相关系数之外还要看什么')
p('原文证据：IV-A / 表 II，EMA 合计 86 位说话人、35 小时 55 分钟。表 III 在 HPRC 测试集报告下列通道相关系数（CC）。TTS 行为零样本合成，两行数值的任务不同。', 'small')
table([['模型 / 任务','见过说话人','未见说话人'],['AAI · MMS-SSL','0.9450','0.7749'],['AAI · MMS-FT','0.9542','0.7711'],['TTS · Proposed (MMS-FT)','0.6673','0.6922']],[260,110,110])
p('怎样解读这些差异', 'h')
p('AAI 与 TTS 的输入条件不同，不能把它们的相关系数直接排成同一种算法的精度榜。对于反演模型，跨说话人下降提醒研究者检查个体形态、坐标规范与训练分布。对于文本驱动的轨迹预测，还需确认音素、时长和对齐如何影响逐帧比较。')
p('相关性不能覆盖空间误差', 'h')
p('如果预测轨迹的时间起伏与实测相似，但整体存在位置偏移或幅度压缩，相关系数仍可能较高。建议同时报告原坐标误差、归一化误差、幅度偏差和事件时刻差。对空间误差说明单位及坐标参考，对时间误差说明对齐方法。')
p('普通话实验的建议', 'h')
p('以说话人为外层划分单位，完整保留一组未见说话人。归一化参数和任何特征选择都只由训练集决定。分别查看声母类别、韵母、声调和语速，检查总体均值是否掩盖某些类别的失败。上述方案为后续研究建议，尚未执行。')

page('3　合成质量：给控制能力定价')
p('原文证据：表 IV，下表为 MCD，数值越小越好。参数与其他质量维度应回到原表核对。V-B 的 MOS 听辨由 20 名听者、每系统 20 句构成。', 'small')
table([['模型','test-clean','test-other'],['Source-Filter 基线','5.73','6.07'],['Proposed (MMS-FT)','5.89','6.36'],['去除 OT-CFM','6.72','7.49']],[260,110,110])
p('应该怎样提出比较问题', 'h')
p('如果研究目标是可控的构音干预，就要在可懂度和自然度达到可用水平后，继续检查控制是否稳定。若目标是尽可能忠实的重建，则应更重视失真。两种目标可以共享一个评估表，但需要各自明确主要终点，不能用一种指标的优势替另一种指标作结论。')
p('功能计数与科学解释分开', 'h')
p('原文 V-B 中可控性和可解释性评分按暴露的属性或中间表示累加。这样的计数适合说明接口丰富程度。研究者若要主张表征对应真实生理过程，还需独立构音测量、干预后效应和跨条件一致性，不能单凭通道名称或功能数量完成论证。')
p('后续听辨设计', 'h')
p('建议以匹配的说话人、句子和生成设置进行比较，把每位听者的重复判断纳入统计结构。提前确定主要比较及样本需求。自动质量预测可以辅助筛查，但需要保留人类判断作为独立结果，并报告刺激选取方法。')

page('4　控制证据：用干预检验分支分工')
p('原文证据：VI-D / 表 VI。缩放 0.7 时，所提模型 F0 CC 为 0.7498，HierSpeech++ 为 0.7584。逐条件核对表格可避免把概括性文字推广到所有指标、所有条件。VI-E / 表 VII 为跨说话人重组证据，VI-F 为构音编辑示例。', 'small')
p('为每次干预写一张效应表', 'h')
p('修改前先写出预期变化和需要保持稳定的量。例如，只调整声源音高时，检查实际测得的 F0，同时查看 F1、F2、F3、能量、时长及清浊状态。记录目标值与实测值，不能把输入控制参数直接当成输出成功的证据。')
p('怎样判断泄漏', 'h')
p('分支独立的程序结构不能保证结果完全独立。声学估计器本身可能对音高变化敏感。建议同时保存生成中间量和对最终音频的估计，使用固定配置及必要的替代估计器。若两者不一致，应先分辨合成变化与测量变化。')
p('怎样分析局部构音编辑', 'h')
p('用短片段编辑时，把过渡平滑、编辑范围和最大位移写成可复核参数。比较同一材料的未编辑版本，并检查相邻音段是否受到意外影响。可以加入等幅随机编辑和等维潜变量控制，判断收益来自构音约束还是额外自由度。')
p('怎样处理跨说话人交换', 'h')
p('声源与滤波器交换适合检验属性跟随哪一分支。它还需要检查说话人识别、句子内容、谱包络和韵律的关系。若某个属性仍随另一分支变化，应报告残留耦合，不把部分可分离性概括为完整解耦。')

page('5　面向工具箱与研究的可复核路线')
table([['研究环节','建议保留的证据'],['声学 → 构音','实测坐标、预测坐标、个体/类别误差与时间对齐规则'],['构音 → 声学','同材料未编辑对照、实际输出声学量、听辨判断'],['构音 → 几何','几何约束、测量参照、唯一性分析，声学重建误差单独报告'],['真实应用','未见说话人、自然语速/弱音/异常发声，失败案例逐类保留']],[120,360])
p('可作为第一步的实验', 'h')
p('先选范围小且有实测依据的任务，例如唇部动态与指定声学变化的对应关系。固定坐标、单位、时间网格及缺失值规则，建立跨说话人基线。在结果可复核之后，再研究更复杂的声道形态。此路线是本解读的建议，不构成新算法已集成或科学精度已验证的声明。')
p('当前模块中的使用方式', 'h')
p('论文原文、中文译文和本解读可分开阅读与批注。个人 PDF 可放入本地文件夹库，收藏夹对应子文件夹；它们不会成为服务器推荐或被自动上传。阅读管理功能与科研算法分别维护。')
p('来源与复现边界', 'h')
p('原论文固定版本：https://arxiv.org/abs/2610.00735v1。方法读图 1 与 III 节，数据读 IV-A / 表 II，反演读表 III，质量读表 IV–V，控制读 VI-D–F / 表 VI–VII。作者提供音频演示，本轮未确认完整训练代码及模型再分发许可。论文的 CC BY 4.0 不自动覆盖代码、数据集和检查点。', 'small')
p('编辑与许可：PhoneticToolbox，2026-10-10。本文引述数据均标出处，其余实验设计与方法评论为原创分析。完整原文及已有译文保留，未对原文的争议表述作静默修正。', 'small')

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#D8D1C9')); canvas.line(54,48,541,48)
    canvas.setFont('ReaderSong',8);canvas.setFillColor(MUTED)
    canvas.drawString(54,33,'PhoneticToolbox · 论文解读 · arXiv:2610.00735v1')
    canvas.drawRightString(541,33,str(doc.page))
file = OUT / '构音声源滤波器TTS_论文解读.pdf'
SimpleDocTemplate(str(file), pagesize=A4, leftMargin=54,rightMargin=54,topMargin=52,bottomMargin=64,
                  title='Articulatory Source-Filter TTS 论文解读与方法分析',author='PhoneticToolbox').build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps({'pdf':str(file)},ensure_ascii=False))
