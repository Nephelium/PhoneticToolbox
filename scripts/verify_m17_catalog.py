"""Build/audit the hand-transcribed M17 catalogue. User PDFs/CIN are read-only.

Source tables were inspected visually; extracted PDF text is not executable input.
Run --write to reproduce the checked-in JSON and source occurrence manifest.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'frontend/src/modules/ipa-plus/data/catalog.json'
VERSION = 'm17/1.0.0-20261002'
ENTRIES = []
CHARTS = {s: [] for s in ('ipa','extipa','voqs')}
SOURCE = {'ipa':'ipa-chart-2026','extipa':'extipa-chart-2025','voqs':'voqs-ball-2018'}
ALIASES = {}
cin = Path.home() / 'Documents/ipa.cin'
if cin.exists():
    body=cin.read_text('utf-8').split('%chardef begin',1)[1].split('%chardef end',1)[0]
    for line in body.splitlines():
        fields=line.split(maxsplit=1)
        if len(fields)==2 and fields[1].strip():
            value=fields[1].strip().replace('\uf267','\U0001df06').replace('\uf268','\U0001df04')
            ALIASES.setdefault(value, []).append(fields[0])
            # CIN precomposed circled capitals and today's combining-circle
            # sequences are searchable aliases; the original CIN is untouched.
            if len(value)==1 and 0x24B6<=ord(value)<=0x24CF:
                ALIASES.setdefault(chr(ord(value)-0x24B6+65)+'\u20dd',[]).append(fields[0])

def section(system, key, title, subtitle='', kind='list', columns=None):
    s={'id':key,'title':title,'subtitle':subtitle,'kind':kind,'ids':[]}
    if columns is not None: s.update(columns=columns, rows=[])
    CHARTS[system].append(s)
    return s

def add(system, sec, value, zh, en, meaning, *, mode='literal', display=None, usage=None,
        contrast=None, examples=(), example=False, prefix=None, suffix=None, representation=None, locator=None):
    ident=f'{system}-{sec["id"]}-{len(sec["ids"])+1:03d}'
    if usage is None:
        usage = ('在基底字母之后输入。按钮仅写入附加记号，显示用的虚线圆不进入文本。' if mode=='combining' else
                 '点击插入完整序列；有选区时替换选区。字符保存和复制均保持本条所列码位。')
    if contrast is None:
        contrast = ('这是音段层面的转写；需要描写一段话的整体音质时可参看 VoQS。' if system=='ipa' else
                    '此记号描述观察到的语音特征，单凭符号不能推出病因或临床诊断。')
    e=dict(id=ident,system=system,section=sec['id'],display=display or value,insertText=value,
           insertionMode=mode,codePoints=[f'U+{ord(c):04X}' for c in value],nameZh=zh,nameEn=en,
           descriptionZh=meaning,usageZh=usage,contrastZh=contrast,aliases=sorted(set(ALIASES.get(value,[]))),
           examples=[{'text':x,'noteZh':'原表例示' if not example else '本按钮输入的原表例示'} for x in examples],
           sourceRefs=[{'sourceId':SOURCE[system], 'locator':locator or f'{sec["title"]} / {en}'}],isExample=example)
    if mode=='paired-span': e.update(prefix=prefix,suffix=suffix)
    if representation: e['representation']=representation
    if system=='voqs' and sec['id']!='scope':
        e['aliases']=sorted(set(e['aliases']+zh.split('/')))
        e['sourceRefs'].append({'sourceId':'voqs-zhihu-203037479','locator':'2016 中英双语版 / '+en+'；中文名称'})
    if representation:e['sourceRefs'].append({'sourceId':'unicode-cartouche-2024','locator':'多字符圈围尚无专用编码；本项目明确标注文本替代'})
    ENTRIES.append(e); sec['ids'].append(ident)
    return ident

def examples(system,sec,values,zh,en,meaning,**kw):
    for v in values:
        add(system,sec,v,zh+'（示例）',en+' example',meaning,example=True,examples=[v],**kw)

def mark(system,sec,value,zh,en,meaning,ex,**kw):
    add(system,sec,value,zh,en,meaning,mode='combining' if unicodedata.category(value[0]).startswith('M') else 'literal',display='◌'+value,examples=ex,**kw)
    examples(system,sec,ex,zh,en,meaning,**kw)

# IPA pulmonic consonants. Merged dental/alveolar/postalveolar cells preserve the
# original chart scope; blanks differ from articulations judged impossible.
places=[('双唇','bilabial','两唇'),('唇齿','labiodental','下唇与上齿'),('齿','dental','舌与牙齿'),('齿龈','alveolar','舌尖或舌叶与齿龈'),('龈后','postalveolar','舌与齿龈后部'),('卷舌','retroflex','舌尖卷起接近硬腭前部'),('硬腭','palatal','舌面与硬腭'),('软腭','velar','舌后部与软腭'),('小舌','uvular','舌后部与小舌区域'),('咽','pharyngeal','舌根与咽壁区域'),('声门','glottal','声门')]
manners=[('塞音','plosive','完全阻塞气流后解除阻塞'),('鼻音','nasal','口腔形成阻塞，软腭下降使气流通过鼻腔'),('颤音','trill','活动发音器官在气流作用下发生重复接触'),('闪音／拍音','tap or flap','活动发音器官进行一次短促接触'),('擦音','fricative','构成狭窄通道，使气流产生摩擦噪声'),('边擦音','lateral fricative','气流从舌的一侧或两侧通过，并产生摩擦'),('近音','approximant','发音器官接近，但通常不产生持续的强摩擦'),('边近音','lateral approximant','中央受阻，气流沿舌侧通过而不产生强摩擦')]
matrix=[['p b','','','t d','','ʈ ɖ','c ɟ','k ɡ','q ɢ','','ʔ'],['m','ɱ','','n','','ɳ','ɲ','ŋ','ɴ','',''],['ʙ','','','r','','','','','ʀ','',''],['','ⱱ','','ɾ','','ɽ','','','','',''],['ɸ β','f v','θ ð','s z','ʃ ʒ','ʂ ʐ','ç ʝ','x ɣ','χ ʁ','ħ ʕ','h ɦ'],['','','','ɬ ɮ','','','','','','',''],['','ʋ','','ɹ','','ɻ','j','ɰ','','',''],['','','','l','','ɭ','ʎ','ʟ','','','']]
shaded={0:{},1:{9,10},2:{7,10},3:{7,10},4:set(),5:{0,1,9,10},6:{10},7:{0,1,9,10}}
ipa=section('ipa','pulmonic','肺部气流辅音','每格左清右浊；阴影表示原表判定不可能的构音。空白格不自动生成新符号。','matrix',[x[0] for x in places])
for ri, row in enumerate(matrix):
    cells=[]
    for ci, raw in enumerate(row):
        if ri!=4 and ci in (2,4): continue
        span=3 if ri!=4 and ci==3 else 1
        ids=[]
        for vi,ch in enumerate(raw.split()):
            voiced = len(raw.split())==1 and ch not in ('ʔ',) or vi==1
            voice='浊' if voiced else '清'
            pzh,pen,organ=places[ci]; mzh,men,action=manners[ri]
            if ch=='ʔ': voice=''
            ids.append(add('ipa',ipa,ch,voice+pzh+mzh,('voiced ' if voiced else 'voiceless ')+pen+' '+men,
                f'{voice}{pzh}{mzh}以{organ}为主要构音部位，{action}。'+('典型实现伴随声带振动。' if voiced else '典型实现不伴随声带周期振动。')+'表中的位置给出宽式分类，具体舌形、时长及协同发音需按材料补充。',
                examples=[ch],locator=f'Consonants (pulmonic) / {men} / {pen}'))
        cells.append(dict(ids=ids,span=span,shaded=ci in shaded.get(ri,set()),rightHalfShaded=ri==0 and ci in (9,10)))
    ipa['rows'].append(dict(label=manners[ri][0],cells=cells))

np=section('ipa','nonpulmonic','非肺部气流辅音','吸气音、浊内爆音、挤喉音分列；气流机制与清浊属性分别说明。','matrix',['吸气音 Clicks','浊内爆音 Implosives','挤喉音 Ejectives'])
clicks=[('ʘ','双唇'),('ǀ','齿'),('ǃ','龈后'),('ǂ','腭龈'),('ǁ','齿龈边')]
impl=[('ɓ','双唇'),('ɗ','齿／齿龈'),('ʄ','硬腭'),('ɠ','软腭'),('ʛ','小舌')]
for idx in range(5):
    c,p=clicks[idx]; a=add('ipa',np,c,p+'吸气音',p+' click',f'用舌前后（双唇音包括唇部）两处闭塞形成腔体，扩大腔体后释放前部闭塞，形成{p}吸气音。这里的符号标前部释放类型；需要时另记后部闭塞和发声类型。',contrast='吸气音的舌气流机制不同于全段肺部吸气；ǃ 是吸气音字母，不能用感叹号 ! 静默替换。')
    c,p=impl[idx]; b=add('ipa',np,c,p+'浊内爆音','voiced '+p+' implosive',f'表示{p}部位的浊内爆音。闭塞阶段喉部降低，形成喉部内向气流机制，常伴声带振动；实际口腔气流方向可能随语言和实现变化。',contrast='与喉部上升产生的挤喉音区分，也不等同于肺部吸气音。')
    c=['ʼ','pʼ','tʼ','kʼ','sʼ'][idx]
    d=add('ipa',np,c,['挤喉记号','双唇挤喉塞音','齿／齿龈挤喉塞音','软腭挤喉塞音','齿龈挤喉擦音'][idx],'ejective' if idx==0 else 'ejective example','喉部闭合并上升，压缩口腔内气体，形成外向喉气流。记号置于对应辅音之后；本区例示四种发音部位或方式。',example=idx>0,contrast='采用 U+02BC 修饰字母撇号，不以弯引号或普通 ASCII 撇号改变 catalog 编码。')
    np['rows'].append({'label':str(idx+1),'cells':[{'ids':[a]},{'ids':[b]},{'ids':[d]}]})

vw=section('ipa','vowels','元音','同一位置的左项不圆唇，右项圆唇。图形展示舌位关系，不是实测声道剖面。','vowels');vw['points']=[]
vowel_rows=[('i y','高前',20,9),('ɨ ʉ','高央',50,9),('ɯ u','高后',83,9),('ɪ ʏ','次高次前',31,23),('ʊ','次高次后',74,23),('e ø','半高前',27,37),('ɘ ɵ','半高央',54,37),('ɤ o','半高后',83,37),('ə','中央',57,51),('ɛ œ','半低前',34,65),('ɜ ɞ','半低央',60,65),('ʌ ɔ','半低后',83,65),('æ','次低前',40,78),('ɐ','次低央',63,78),('a ɶ','低前',42,91),('ɑ ɒ','低后',83,91)]
rounded=set('yʉuʏʊøɵoœɞɔɶɒ')
vowel_en={'i':'close front unrounded','y':'close front rounded','ɨ':'close central unrounded','ʉ':'close central rounded','ɯ':'close back unrounded','u':'close back rounded','ɪ':'near-close near-front unrounded','ʏ':'near-close near-front rounded','ʊ':'near-close near-back rounded','e':'close-mid front unrounded','ø':'close-mid front rounded','ɘ':'close-mid central unrounded','ɵ':'close-mid central rounded','ɤ':'close-mid back unrounded','o':'close-mid back rounded','ə':'mid central','ɛ':'open-mid front unrounded','œ':'open-mid front rounded','ɜ':'open-mid central unrounded','ɞ':'open-mid central rounded','ʌ':'open-mid back unrounded','ɔ':'open-mid back rounded','æ':'near-open front unrounded','ɐ':'near-open central','a':'open front unrounded','ɶ':'open front rounded','ɑ':'open back unrounded','ɒ':'open back rounded'}
for raw,pos,x,y in vowel_rows:
    ids=[]
    for ch in raw.split():
        lip='圆唇' if ch in rounded else '不圆唇'
        ids.append(add('ipa',vw,ch,pos+lip+'元音',vowel_en[ch]+' vowel',f'{ch} 位于元音图的{pos}位置，通常按{lip}描述。元音由舌位高低、前后及唇形共同区分，图中点位用于分类比较，不能直接解释为某个固定共振峰数值。',contrast='相邻元音符号代表可区分的音质类别；需要更精细转写可添加前移、后移、升高或降低等附加符号。'))
    vw['points'].append({'ids':ids,'x':x,'y':y})

other=section('ipa','other','其他符号与连音线')
for c,z,en,d in [('ʍ','清唇软腭擦音','voiceless labial-velar fricative','唇部与舌后部形成联合构音，通常为清擦音。'),('w','浊唇软腭近音','voiced labial-velar approximant','唇部圆起，舌后部接近软腭，形成联合近音。'),('ɥ','浊唇硬腭近音','voiced labial-palatal approximant','圆唇与舌面硬腭构音同时出现，近似高前圆唇元音的非音节性对应。'),('ʜ','清会厌擦音','voiceless epiglottal fricative','在会厌／杓会厌区域形成清擦音，实际实现可能包含颤动。'),('ʢ','浊会厌擦音','voiced epiglottal fricative','在会厌／杓会厌区域形成浊擦音，需按实际材料区分振动来源。'),('ʡ','会厌塞音','epiglottal plosive','在下咽部会厌相关区域形成完全闭塞，随后释放。'),('ɕ','清龈腭擦音','voiceless alveolo-palatal fricative','齿龈后部至硬腭前部的舌面构音，形成清咝擦音。'),('ʑ','浊龈腭擦音','voiced alveolo-palatal fricative','与 ɕ 对应的浊龈腭擦音，带声带振动。'),('ɺ','浊齿龈边闪音','voiced alveolar lateral flap','舌尖短促接触齿龈，气流从舌侧通过。'),('ɧ','同时的 ʃ 与 x','simultaneous ʃ and x','原表用此字母表示同时的后龈和软腭擦音构音；特定语言中的实现需另作描述。')]:
    add('ipa',other,c,z,en,d+'该项在主辅音表之外单列，保留原表分类与字符形式。')
for c,ex,z in [('͡','k͡p','上连音线'),('͜','t͜s','下连音线')]:
    add('ipa',other,c,z,'tie bar', '用于把两个符号关联为塞擦音或双重构音等一个构音单位。连线本身不能决定两端的音值。',mode='bridge',display=ex,examples=[ex],usage='选中恰好两个字素时，在其间插入连线；无选区时输入连线本身。复杂附加符号保留各自基底。')
    examples('ipa',other,[ex],z,'tie bar','原表的连音线示例，可作为一个完整序列插入。')

dia=section('ipa','diacritics','附加符号','每组先给独立附加记号，再给原表可点击示例；虚线圆仅作定位。')
diacritics=[
('̥','清化','voiceless','原本常为浊音的音段不伴或减少声带周期振动。',['n̥','d̥']),('̬','浊化','voiced','给通常为清音的音段标记声带振动。',['s̬','t̬']),('ʰ','送气','aspirated','辅音释放后具有可辨的气流噪声或较长声门开放期。',['tʰ','dʰ']),('̹','较圆唇','more rounded','相对基底音的典型唇形，增加圆唇程度。',['ɔ̹']),('̜','较不圆唇','less rounded','相对基底音减少圆唇程度。',['ɔ̜']),('̟','前移','advanced','主要舌部构音位置相对基底音向前。',['u̟']),('̠','后移','retracted','主要舌部构音位置相对基底音向后。',['e̠']),('̈','央化','centralized','元音舌位从前或后的位置向央部靠拢。',['ë']),('̽','中部央化','mid-centralized','元音向元音空间的中央区域靠拢，同时涉及高低和前后维度。',['e̽']),('̩','成音节','syllabic','使所标音段承担音节核功能。',['n̩']),('̯','非音节性','non-syllabic','使所标元音或响音不独立承担音节核功能。',['e̯']),('˞','卷舌音色','rhoticity','为元音标记 r 类音色，不指定唯一舌形。',['ə˞','a˞']),('̤','气声','breathy voiced','声带振动伴随较明显的漏气声。需与耳语以及 VoQS 耳语化嗓音区别。',['b̤','a̤']),('̰','嘎裂声','creaky voiced','标记较低频或不规则声带振动的嘎裂特征，不等于一般粗糙音质。',['b̰','a̰']),('̼','舌唇化','linguolabial','舌尖或舌叶与上唇接触或接近。',['t̼','d̼']),('ʷ','唇化','labialized','主构音伴随附加圆唇或唇部收拢。',['tʷ','dʷ']),('ʲ','腭化','palatalized','主构音伴随舌面朝硬腭抬起。',['tʲ','dʲ']),('ˠ','软腭化','velarized','主构音伴随舌后部向软腭接近。',['tˠ','dˠ']),('ˤ','咽化','pharyngealized','主构音伴随咽部收缩，具体机制应按语音材料说明。',['tˤ','dˤ']),('̴','软腭化或咽化','velarized or pharyngealized','横贯字母的附加记号，概括软腭化或咽化；需要明确机制时用专门记号。',['l̴']),('̝','升高','raised','使元音舌位更高，或使辅音狭窄程度增大。',['e̝','ɹ̝']),('̞','降低','lowered','使元音舌位更低，或使辅音狭窄程度减小。',['e̞','β̞']),('̘','舌根前移','advanced tongue root','描述舌根向前的构音设置；不等同于整个舌体的前移。',['e̘']),('̙','舌根后缩','retracted tongue root','描述舌根向后收缩的设置；不等同于笼统的后元音。',['e̙']),('̪','齿化','dental','构音接触向牙齿区域调整。',['t̪','d̪']),('̺','舌尖化','apical','明确活动构音部位为舌尖。',['t̺','d̺']),('̻','舌叶化','laminal','明确活动构音部位为舌叶，区别于舌尖构音。',['t̻','d̻']),('̃','鼻化','nasalized','口腔气流同时经鼻腔逸出，通常由软腭下降引起。',['ẽ']),('ⁿ','鼻音释放','nasal release','塞音的口腔闭塞经鼻腔开放而释放。',['dⁿ']),('ˡ','边音释放','lateral release','塞音释放时气流沿舌侧通过。',['dˡ']),('̚','无可闻释放','no audible release','闭塞结束时未听到独立释放爆破，不能由此断言闭塞从未解除。',['d̚'])]
for c,z,en,d,ex in diacritics: mark('ipa',dia,c,z,en,d,ex)
mark('ipa',dia,'̊','清化（上置异体）','voiceless above','有下伸笔画的基底音可把清化小圆放在上方，避免与字母笔画重叠。',['ŋ̊'])
sup=section('ipa','suprasegmentals','超音段')
for c,z,en,d,ex in [('ˈ','主重音','primary stress','放在承载主重音的音节前。',['ˌfoʊnəˈtɪʃən']),('ˌ','次重音','secondary stress','放在承载次重音的音节前。',[]),('ː','长音','long','放在音段后，标记相对较长时长。',['eː']),('ˑ','半长音','half-long','放在音段后，标记介于短音与长音之间的时长。',['eˑ']),('̆','超短音','extra-short','放在基底上方，标记相对很短的时长。',['ĕ']),('|','小韵律组边界','minor (foot) group','划分较小韵律单位的边界，不自动等于静音。',[]),('‖','大韵律组边界','major (intonation) group','划分较大语调单位的边界，不规定统一停顿时长。',[]),('.','音节界','syllable break','标明相邻音节的边界。',['ɹi.ækt']),('‿','连读（无断开）','linking (absence of a break)','表示相邻材料连读且没有明显断开。',[])]:
    mark('ipa',sup,c,z,en,d,ex) if unicodedata.category(c).startswith('M') else add('ipa',sup,c,z,en,d,examples=ex)
    if ex and not unicodedata.category(c).startswith('M'):examples('ipa',sup,ex,z,en,d)
tone=section('ipa','tones','声调与词调','调值用相对等级表示；调符与组合音调记号分别可点。')
for tonebar,comb,z,en in [('˥','̋','特高','extra high'),('˦','́','高','high'),('˧','̄','中','mid'),('˨','̀','低','low'),('˩','̏','特低','extra low'),('˩˥','̌','升','rising'),('˥˩','̂','降','falling'),('˦˥','᷄','高升','high rising'),('˩˨','᷅','低升','low rising'),('˧˦˧','᷈','升降','rising-falling')]:
    d=f'表示{z}调。调高等级是同一语言、说话人和语境中的相对比较，不直接给出赫兹值或跨说话人阈值。'
    add('ipa',tone,tonebar,z+'调（调符）',en+' tone letters',d)
    mark('ipa',tone,comb,z+'调（附加记号）',en+' tone diacritic',d,['e'+comb])
for c,z,en,d in [('ꜜ','降阶','downstep','相对于预期调域降低后续音调，须结合调系解释。'),('ꜛ','升阶','upstep','相对于预期调域升高后续音调，须结合调系解释。'),('↗','整体上升','global rise','表示较大语调范围的整体上升趋势。'),('↘','整体下降','global fall','表示较大语调范围的整体下降趋势。')]:add('ipa',tone,c,z,en,d,contrast='降阶／升阶针对调域关系，整体箭头针对更长语调范围，与一个音节上的升降调区别。')

# extIPA 2025, including every visually printed example.
ep=[('双唇','bilabial'),('唇齿','labiodental'),('唇龈','labio-alveolar'),('齿唇','dento-labial'),('双齿','bidental'),('舌唇','linguolabial'),('齿间','interdental'),('齿龈','alveolar'),('卷舌','retroflex'),('硬腭','palatal'),('软腭','velar'),('腭咽','velopharyngeal'),('上咽','upper pharyngeal')]
erows=[('塞音','plosive',['','p̪ b̪','p͇ b͇','p͆ b͆','','t̼ d̼','t̪͆ d̪͆','','','','','','ꞯ 𝼂']),('鼻音','nasal',['','','m͇̊ m͇','m̥͆ m͆','','n̼̊ n̼','n̪̥͆ n̪͆','','','','','','']),('颤音','trill',['','','','','','r̼','r̪͆','','','','','𝼀 𝼀̬','']),('中央擦音','median fricative',['','','f͇ v͇','f͆ v͆','h̪͆ ɦ̪͆','θ̼ ð̼','θ̪͆ ð̪͆','θ͇ ð͇','','','','ʩ ʩ̬','']),('边擦音','lateral fricative',['','','','','','ɬ̼ ɮ̼','ɬ̪͆ ɮ̪͆','','ꞎ 𝼅','𝼆 𝼆̬','𝼄 𝼄̬','','']),('边音＋中央擦音','lateral + median fricative',['','','','','','','','ʪ ʫ','','','','','']),('鼻擦音','nasal fricative',['m̥̾ m̾','ɱ̥̾ ɱ̾','','','','','','n̥̾ n̾','ɳ̥̾ ɳ̾','ɲ̥̾ ɲ̾','ŋ̥̾ ŋ̾','','']),('边近音','lateral approximant',['','','','','','l̼','l̪͆','','','','','','']),('叩击音','percussive',['ʬ','','','','ʭ','','','','','','','',''])]
eshade={0:{4,11},1:{4,11,12},2:{4,12},3:set(),4:{0,1,2,3,4,11,12},5:{0,1,2,3,4,11,12},6:{11,12},7:{0,1,2,3,4,11,12},8:{11}}
ec=section('extipa','consonants','额外辅音','2025 表中未列于 IPA 主表的辅音；每个清浊成员分别可输入。','matrix',[p[0] for p in ep])
for ri,(z,en,row) in enumerate(erows):
    cells=[]
    for ci,raw in enumerate(row):
        ids=[]
        for vi,c in enumerate(raw.split()):
            voice='' if en=='percussive' else ('清' if len(raw.split())==2 and vi==0 else '浊')
            detail={'plosive':'形成完全阻塞后释放。','nasal':'口腔闭塞而鼻腔开放。','trill':'气流驱动发音器官重复接触或振动。','median fricative':'气流主要沿中央狭窄通道产生摩擦。','lateral fricative':'气流沿舌侧产生摩擦。','lateral + median fricative':'中央与侧方气流同时形成摩擦。','nasal fricative':'口腔闭塞，鼻部逃逸气流形成可闻摩擦。','lateral approximant':'气流沿舌侧通过而不以强摩擦为主要特征。','percussive':'两发音器官相撞产生叩击声，不能把符号解释为普通肺部塞音。'}[en]
            ids.append(add('extipa',ec,c,voice+ep[ci][0]+z,('voiceless ' if voice=='清' else 'voiced ' if voice else '')+ep[ci][1]+' '+en,
                f'{voice}{ep[ci][0]}{z}。'+detail+'该组合按 extIPA 的发音部位和方式解释；组合附加记号属于整体转写的一部分，不应在复制时丢弃。',locator=f'2025 p1 Consonants / {en} / {ep[ci][1]}'))
        cells.append({'ids':ids,'shaded':ci in eshade.get(ri,set())})
    ec['rows'].append({'label':z,'cells':cells})
ed=section('extipa','diacritics','附加符号','定位、气流和构音程度；先点独立记号，或直接使用标为示例的完整组合。')
for c,z,en,d,ex in [('͍','唇展','labial spreading','双唇横向展开，用于区别同一基底下的唇形变化。',['s͍','u͍']),('͈','强构音','strong articulation','相对同类音的参照形式，构音作用较强。原表不给出统一声压或肌电阈值。',['f͈']),('͉','弱构音','weak articulation','相对同类音的参照形式，构音作用较弱，不直接等于较小音量。',['v͉']),('͊','部分去鼻化','partially denasal','鼻音的鼻腔共鸣或鼻气流相对典型形式减弱，保留“部分”的含义。',['m͊']),('̾','摩擦性鼻漏气','fricative nasal escape','伴随可闻摩擦的鼻部气流逃逸。鼻擦音以鼻部摩擦为主要成分，本附加记号可修饰其他基底。',['v̾']),('𐞐','腭咽摩擦','velopharyngeal friction','在腭咽区域形成的摩擦伴随基底音。该上标字母是 U+10790，不用私用区字符代替。',['s𐞐','ʒ𐞐']),('͔','主要构音偏右','main gesture offset right','主要构音位置相对正中向说话人右侧偏移。',['s͔']),('͕','主要构音偏左','main gesture offset left','主要构音位置相对正中向说话人左侧偏移。',['s͕']),('͎','吹哨式构音','whistled articulation','狭窄气流形成明显哨音性质，区别于一般擦音摩擦。',['s͎'])]:mark('extipa',ed,c,z,en,d,ex)
add('extipa',ed,'\\','重复构音','reiteration','标记同一闭塞或构音动作连续重复，置于反复出现的符号之间。',examples=['p\\p\\p'])
examples('extipa',ed,['p\\p\\p'],'重复构音','reiteration','三个反复出现的 p 以反斜线标示构音重复。')
add('extipa',ed,'↓','吸气气流','ingressive airflow','放在音段旁标记气流朝内。这里是音段气流方向标记，与 IPA 声调部分的降阶功能不同。',examples=['p↓'])
examples('extipa',ed,['p↓'],'吸气气流','ingressive airflow','原表的吸气 p 例示。')
add('extipa',ed,'͢','滑动构音','sliding articulation','从一个构音位置或音值连续滑向另一位置，箭头连在两基底之间。',mode='bridge',display='θ͢s',examples=['θ͢s'],usage='选中恰好两个字素时，在两者之间插入箭头；无选区输入附加箭头。')
examples('extipa',ed,['θ͢s'],'滑动构音','sliding articulation','由 θ 向 s 滑动的原表例示。')
ev=section('extipa','voicing','发声与送气','部分、起始部分和末尾部分分别保留；上置括号用于带下伸部件的基底。')
for c,z,en,d,ex,disp in [('ˬ','预浊化','pre-voicing','在目标音段之前出现声带振动。',['ˬz'],'ˬ◌'),('ˬ','后浊化','post-voicing','在目标音段之后出现声带振动。',['zˬ'],'◌ˬ'),('˷','后嘎裂','post-creak','在音段之后出现嘎裂声。',['a˷'],'◌˷')]:
    add('extipa',ev,c,z,en,d,display=disp,examples=ex,usage='按预／后位置置于目标音段之前或之后。同一字符的语义由相对位置决定。');examples('extipa',ev,ex,z,en,d)
for low,up,z,en in [('᪽','᪻','部分','partial'),('᫃','᫁','起始部分','initial partial'),('᫄','᫂','末尾部分','final partial')]:
    d=f'{z}清化：声带不振动涉及音段的部分时段，括号明确受影响的范围。此处不预设清化时长比例。'
    mark('extipa',ev,'̥'+low,z+'清化',en+' devoicing',d,['z̥'+low])
    mark('extipa',ev,'̊'+up,z+'清化（上置）',en+' devoicing above',d,['ʒ̊'+up])
    mark('extipa',ev,'̬'+low,z+'浊化',en+' voicing',f'{z}浊化：声带振动只出现在目标音段的一部分。括号方向与原表对应，不能用完整浊化记号替代。',['s̬'+low])
for c,z,en,d,ex,disp in [('ʰ','预送气','pre-aspiration','气流噪声发生在目标辅音之前。',['ʰp'],'ʰ◌'),('˭','无送气','unaspirated','明确标记辅音释放没有可辨的附加送气。',['p˭'],'◌˭'),('ʰʰ','长送气（双 h）','long aspiration','送气持续较长，原表给出重复上标 h 的写法。',['tʰʰ'],'◌ʰʰ'),('ʰ𐞁','长送气（时长）','long aspiration with length','送气持续较长，原表也给出上标 h 加上标长音号的写法。',['tʰ𐞁'],'◌ʰ𐞁')]:
    add('extipa',ev,c,z,en,d,display=disp,examples=ex);examples('extipa',ev,ex,z,en,d)
rh=section('extipa','rhythm','节奏、响度与速度','范围模板包住所选文字；无选区时把光标留在范围中间。')
for c,z in [('(.)','短停顿'),('(..)','中等停顿'),('(...)','长停顿'),('(1.3 sec)','定时停顿')]:add('extipa',rh,c,z,'pause', '在转写中标记无声间隔。点数形式提供相对长短，带秒数形式明确记录测量时长；三档不自动换算为固定秒数。')
for label,z,en,ex in [('f','较响','loud','laʊd'),('ff','更响','louder','laʊdə'),('p','较轻','quiet','kwaɪət'),('pp','更轻','quieter','kwaɪətə'),('allegro','快速','fast','fɑst'),('lento','缓慢','slow','sloʊ'),('crescendo','渐强','crescendo',None),('rallentando','渐慢','rallentando',None)]:
    d=f'在标记范围内表示{z}的言语。'+('强弱是相对于语境的响度描写，不规定声压阈值。' if label in ('f','ff','p','pp','crescendo') else '速度是相对于语境的节奏描写，不指定固定音节率。')
    add('extipa',rh,label,z+'标签',en,d)
    prefix='{'+label+' ';suffix=' '+label+'}'
    add('extipa',rh,prefix+suffix,z+'范围',en+' span',d,mode='paired-span',prefix=prefix,suffix=suffix,display=prefix+'…}',usage='按钮为紧凑范围预览，输入会在左右两端写入完整标签。包住选区，无选区则把光标放在中间。')
    if ex:examples('extipa',rh,[prefix+ex+suffix],z+'范围',en+' span',d)
un=section('extipa','uncertainty','不确定性、无声构音与外来噪声','圈围表不确定识别；括号无声构音与双括号外来噪声分开。')
add('extipa',un,'◯','不能确定的声音','indeterminate sound','已观察到声音，但无法可靠判定其语音性质。空圈保留不确定性，不应填入推测的确定音标。')
wildcards=[('C','辅音','consonant'),('Ȼ','阻碍音','obstruent'),('F','擦音','fricative'),('L','流音','liquid'),('G','滑音','glide'),('N','鼻音','nasal'),('P','塞音','plosive'),('R','r 类音／响音','rhotic or resonant'),('S','咝音','sibilant'),('T','声调／重音','tone or accent'),('Ṽ','鼻化元音','nasal vowel'),('Ʞ','吸气音','click'),('σ','音节','syllable')]
for base,z,en in wildcards:
    add('extipa',un,base+'⃝','不确定'+z,'indeterminate '+en,f'只能把声音可靠归入{z}这一类别，不能确定更细的音值。圈内字母是类别占位记号，不能把大写字母照普通字母音值读出。',contrast='圈围表示辨识不确定性；附加在小写基底上的清化小圆表示发声特征，二者功能不同。')
cartouche='原表连续圈围在纯文本中采用 ⟅…⟆ 作为明确标注的替代表示；这两个括号未被本模块宣称为 extIPA 正式新增符号。参见 Miller & Ball 2024 的 Unicode 圈围讨论。'
add('extipa',un,'⟅⟆','不确定片段（文本圈围）','indeterminate span (text substitute)','对一段只能暂定的转写保留不确定性。'+cartouche,mode='paired-span',prefix='⟅',suffix='⟆',display='⟅…⟆',representation=cartouche)
add('extipa',un,'⟅n̥ã⟆','可能是 n̥ã（例示）','probably n̥ã','原表把可能的 n̥ã 圈在同一个范围内。本例使用可复制的显式括号替代圈围。',example=True,representation=cartouche)
for prefix,suffix,z,en,d,ex in [('(',')','无声构音','silent articulation','观察到构音动作，却没有相应可闻语音。',['(ʃ)','(m)']),('⸨','⸩','外来噪声','extraneous noise','把咳嗽等额外声音或未能辨明的片段说明放入双括号。',['⸨2 sylls⸩','⸨2σ⸩','⸨cough⸩'])]:
    add('extipa',un,prefix+suffix,z,en,d,mode='paired-span',prefix=prefix,suffix=suffix,display=prefix+'…'+suffix,usage='包住选区，无选区插入成对括号并把光标留在中间。');examples('extipa',un,ex,z,en,d)
eo=section('extipa','other','其他声音','保留原表完整组合；现代 Unicode 替代旧 SIL 私用区字符。')
for c,z,en,d in [('ɹ̺','舌尖 r','apical r','以舌尖为主要构音器官的 r 类近音。'),('ɹ̈','团舌 r','bunched r (molar r)','舌体团起形成的 r 类音色，不能单凭 r 音色推断舌尖卷起。'),('s̻','舌叶清咝音','laminal fricative','舌叶构成主要狭窄，可包括舌尖降低的实现。'),('z̻','舌叶浊咝音','laminal fricative','带声带振动的舌叶咝音，可包括舌尖降低的实现。'),('d𐞞','d 的边擦释放','lateral fricated release','d 的释放具有浊边擦音成分。'),('k𐞜','k 的边擦释放','lateral fricated release','k 的释放具有清软腭边擦音成分。'),('t𐞙','t 的边＋中央释放','lateral and median release','t 释放时中央与舌侧同时出现摩擦。'),('d𐞚','d 的边＋中央释放','lateral and median release','d 释放时中央与舌侧同时出现浊摩擦。'),('t̼͡θ̼','清舌唇塞擦音','voiceless linguolabial affricate','舌唇塞音连续释放为舌唇擦音，连线标示一个组合构音单位。'),('d̼͡ð̼','浊舌唇塞擦音','voiced linguolabial affricate','舌唇塞音连续释放为舌唇浊擦音。'),('𝼃','清舌背软腭塞音','voiceless velodorsal plosive','反向舌背相关的特殊软腭闭塞，用专用扩展字母与普通 k 区分。'),('𝼁','浊舌背软腭塞音','voiced velodorsal plosive','对应特殊舌背软腭闭塞的浊音形式。'),('𝼇','舌背软腭鼻音','velodorsal nasal','相应特殊舌背软腭闭塞伴随鼻腔气流。'),('¡','舌下下齿龈叩击音','sublaminal lower-alveolar percussive','舌下表面与下齿龈区域碰击，区别于普通舌尖齿龈塞音。'),('ǃ¡','带舌下叩击释放的齿龈吸气音','alveolar click with sublaminal percussive release','齿龈吸气音释放结合舌下叩击成分。'),('ↀ͡r̪͆','颊气流齿间颤音','buccal interdental trill (raspberry)','由颊腔气体驱动齿间颤动，原表称 raspberry。颊气流与肺部气流需分辨。'),('tʰ̪͆','双齿送气 t','t with bidental aspiration','t 的送气经过上下齿形成的通道，双齿记号修饰送气成分。'),('*','暂无可用符号的声音','sound with no available symbol','已辨识的特殊声音缺少现成符号时使用星号，并应另附文字解释。不同于空圈所表达的无法辨识。')]:add('extipa',eo,c,z,en,d,example=c not in ('𝼃','𝼁','𝼇','¡','*'))

# VoQS 2016 revised chart, Figure 2 at journal p169 (2018 issue).
va=section('voqs','airstream','气流类型','VoQS 描写一段言语的设置；与同形的音段或调符按体系和范围区别。')
for c,z,en,d in [('ↀ','颊气流','buccal airstream','利用颊部压缩驱动气流。常见于短声音，也能覆盖短语流。'),('↓','肺部吸气言语','pulmonic ingressive speech','言语产生于肺部吸气过程中，气流方向向内。'),('Œ','食管气流','oesophageal airstream','食管内气体释放参与言语发声。字形是大写 OE 连字，在本体系不按元音 œ 解读。'),('Ю','气管–食管言语','tracheo-oesophageal speech','肺气流经气管食管通道参与发声，与单纯食管气流来源不同。')]:add('voqs',va,c,z,en,d+'记号与大括号结合时界定该设置的作用片段。')
vp=section('voqs','phonation','发声类型','常态浊声、假声、耳语、嘎裂及其组合；糙声、气声与耳语声分别说明。')
phonations=[('V','常态浊声','modal voice','以常态声带振动为基础的嗓音。是比较其他发声设置时的参照，不代表所有说话人共享一个固定频率。'),('F','假声','falsetto','假声发声设置。通常涉及声带边缘振动和较高音区，但不能仅凭高 F0 判断假声。'),('W','耳语','whisper','无常规周期性声带振动的耳语发声。与仍有声带振动的耳语声区分。'),('C','嘎裂','creak','嘎裂型发声，可具有低频或不规则脉冲。与在常态浊声中叠加嘎裂的 V̰ 区分。'),('Ṿ','耳语声','whispery voice','常态声带振动中带耳语性噪声或相应收缩设置。2016 修订回用下点记号。'),('V̰','嘎裂声','creaky voice','常态浊声带嘎裂特征，以下波浪标记；与一般粗糙或复音分开。'),('V̤','气声','breathy voice','声带振动伴气流漏泄形成的气声，以双下点与耳语声的单下点区分。'),('V!','糙声','harsh voice','在常态发声中加入粗糙性质。感叹号描写音质，不表示响度等级。'),('F̣','耳语假声','whispery falsetto','假声设置带耳语性质；基底 F 与单下点的意义均保留。'),('F̰','嘎裂假声','creaky falsetto','假声与嘎裂特征的组合，是原表列出的混合发声类型。'),('F!','糙假声','harsh falsetto','假声中带粗糙特征，不等同于所有高音。'),('C!','糙嘎裂','harsh creak','嘎裂设置伴粗糙成分，与 C 的单纯嘎裂区分。'),('Ṿ!','糙耳语声','harsh whispery voice','常态浊声同时带耳语与粗糙性质。'),('V̰!','糙嘎裂声','harsh creaky voice','常态浊声同时带嘎裂与粗糙性质。'),('V̰̣','耳语嘎裂声','whispery creaky voice','常态浊声同时带耳语和嘎裂性质；两个下置记号均须保留。'),('V̰̣!','糙耳语嘎裂声','harsh whispery creaky voice','耳语化、嘎裂化和粗糙三个性质的组合。符号允许组合，不意味着任意设置都可独立相加。'),('F̰̣','耳语嘎裂假声','whispery creaky falsetto','假声带耳语和嘎裂性质，保留 F 基底与两个附加符号。'),('F̰̣!','糙耳语嘎裂假声','harsh whispery creaky falsetto','假声、耳语、嘎裂及粗糙性质的原表组合。'),('V͉','弛/松声','slack/lax voice','以弱构音记号标喉部较松弛的发声设置，与响度较轻分开。'),('V͈','挤喉发声/紧声','pressed phonation/tight voice','以强构音记号表示较紧的喉部发声设置。2016 表取代旧的前移／挤压类记法。'),('V‼','室襞性发声','ventricular phonation','室带（假声带）振动作为声源参与发声，两个感叹号构成专门记号。'),('V̬‼','复音','diplophonia','以专用组合标记复音性质，涉及可区分的同时周期性成分；不能据符号直接给出病因。'),('Ṿ‼','耳语室襞性发声','whispery ventricular phonation','室襞性发声同时带耳语性质。下点不可省略为普通室襞性发声。'),('V𐞀','杓状会厌襞性发声','aryepiglottic phonation','杓会厌襞振动参与声源。使用 U+10780 上标小型大写 AA，区别于把全尺寸 Ꜳ 排在 V 后。'),('ꟿ','痉挛性发声障碍','spasmodic dysphonia','原表用于记录痉挛导致的音质波动。该记号本身不区分内收／外展类型，也不标明是否伴震颤，详细观察需另注。'),('И','电子喉发声','electrolarynx phonation','由电子喉提供振动声源。2016 修订将其从气流类型移到发声类型。')]
for c,z,en,d in phonations:add('voqs',vp,c,z,en,d,usage='可单独输入，也可用下方带标签大括号模板包住受影响片段。程度数字 1–3 可表达相对强弱。',contrast='VoQS 面向持续音质设置；临床诊断与具体声源机制仍须独立证据，组合记号不预设所有设置都兼容。')
vl=section('voqs','larynx','喉位高度')
for c,z,en,d in [('L̝','喉位偏高声','raised-larynx voice','相对中性参照喉位偏高声。喉位可以与发声、喉上设置组合，但与咽部紧缩存在相互制约。'),('L̞','喉位偏低声','lowered-larynx voice','相对中性参照喉位偏低声。不能仅根据音高变低就断言喉位偏低声。')]:add('voqs',vl,c,z,en,d)
for key,title,values in [
('labial','喉上形态 · 唇部形态',[('Vꟹ','唇化声（开圆唇）','labialized voice (open rounded)','嘴唇呈较开放的圆唇设置，修饰字母 oe 与窄圆唇的 w 区分。'),('Vʷ','唇化声（闭圆唇）','labialized voice (close rounded)','嘴唇呈较紧的圆唇设置，用上标 w 标记。'),('V͍','展唇声','spread-lip voice','嘴唇持续横向展开。与单个音段的唇展记号区分作用范围。'),('Vᶹ','唇齿化声','labio-dentalized voice','言语中持续带下唇接近上齿的设置。')]),
('lingual','喉上形态 · 舌部形态',[('V̺','舌尖化声','linguo-apicalized voice','舌尖在一段言语中成为较突出的活动构音部位。'),('V̻','舌叶化声','linguo-laminalized voice','舌叶在一段言语中成为较突出的活动构音部位。'),('V˞','卷舌声','retroflex voice','标记舌尖后卷相关的整体构音设置，不应将所有 r 音色一律解释为同一舌形。'),('V̪','齿化声','dentalized voice','舌部构音整体偏向牙齿区域。'),('V͇','龈化声','alveolarized voice','舌部构音整体偏向齿龈区域。'),('V͇ʲ','腭–龈化声','palato-alveolarized voice','齿龈区域构音同时带硬腭化设置，保留双下线和上标 j。'),('Vʲ','腭化声','palatalized voice','舌面持续朝硬腭抬起。'),('Vˠ','软腭化声','velarized voice','舌后部持续朝软腭接近。'),('Vʶ','小舌化声','uvularized voice','舌后部持续朝小舌区域接近。'),('Vˤ','咽化声','pharyngealized voice','咽部收缩相关的整体音质。2018 论文指出它与杓会厌收缩、喉位可能相互作用。'),('V̙ˤ','喉–咽化声','laryngo-pharyngealized voice','比一般咽化更强的喉咽部紧缩设置；附加符号表达组合和程度差异，不能假定完全独立。'),('Vꟸ','咽门化声','faucalized voice','咽门区域较扩张的音质设置。上标小型带横 H 使用 U+A7F8，其 petite-capital 字形问题见 Unicode 2020 专项澄清。')]),
('velum','喉上形态 · 软腭状态',[('Ṽ','鼻化声','nasalized voice','鼻腔耦合在一段言语中增强。以大写 V 加鼻化号表示持续设置。'),('V͊','去鼻化声','denasalized voice','鼻腔耦合在一段言语中减少。VoQS 原表标签 denasalized 与 extIPA 单音的 partially denasal 标签不同，不能混同解释。')]),
('jaw','喉上形态 · 颌和舌的形态',[('J̞','颌位偏开声','open-jaw voice','下颌相对中性位置下降，口腔开度增大。'),('J̝','颌位偏闭声','close-jaw voice','下颌相对中性位置抬高，口腔开度减小。'),('J͔','颌位偏右声','right offset-jaw voice','下颌相对正中向说话人右侧偏移。'),('J͕','颌位偏左声','left offset-jaw voice','下颌相对正中向说话人左侧偏移。'),('J̟','颌位突出声','protruded-jaw voice','下颌相对上颌向前突出，与舌伸出分开。'),('Θ','舌突出声','protruded-tongue voice','舌向口外伸出的持续设置。大写希腊 theta 在本体系作为音质记号，不按 IPA θ 的音值解释。')])]:
    s=section('voqs',key,title)
    for c,z,en,d in values:add('voqs',s,c,z,en,d,usage='将本记号作为范围标签使用，必要时配程度数字。设置是相对同一说话人的参照来描述。')
tools=section('voqs','scope','带标签的大括号和数字','大括号界定作用范围，数字 1、2、3 分别表示相对较弱、中等、较强。')
for n,z in [('1','较弱'),('2','中等'),('3','较强')]:add('voqs',tools,n,z+'程度','degree '+n,f'附在音质标签旁表示相对{z}的程度。原表不把这些数字规定为统一声学量或临床量表分数。')
for c,z in [('{','左大括号'),('}','右大括号')]:add('voqs',tools,c,z,'scope brace','带标签大括号界定音质影响的连续范围，左右范围标签应相互对应。')
add('voqs',tools,'{}','空范围','empty scope', '用来构造带音质标签的作用范围。',mode='paired-span',prefix='{',suffix='}',display='{…}',usage='包住选区；没有选区时，光标放在括号中间。')
add('voqs',tools,'{V!  V!}','糙声范围','harsh voice span','以糙声标签界定其作用片段；用户可自由修改标签和程度。',mode='paired-span',prefix='{V! ',suffix=' V!}',display='{V! … V!}')
add('voqs',tools,'{3V! ˈvɛɹi ˈhɑ˞ʃ ˈvɔɪs 3V!}','强糙声（原表例句片段）','degree 3 harsh voice example','从原表底部完整例句截取的强糙声范围；两端用数字 3 标记较强程度。',example=True)
add('voqs',tools,'[ˈnɔ˞məl ˈvɔɪs {3V! ˈvɛɹi ˈhɑ˞ʃ ˈvɔɪs 3V!} {L̝ 1V! ˈlɛs ˈhɑ˞ʃ ˈvɔɪs wɪð ˈɹeɪzd ˈlæɹɪŋks 1V! L̝}]','音质范围（原表完整例句）','complete labelled-brace example','原表底部例句先给常态浊声，再给较强糙声，最后组合升喉位和较弱糙声。不同维度可以组合，实际生理兼容性需依据材料。',example=True,display='[ˈnɔ˞məl …{3V!…}{L̝ 1V!…}]',usage='按钮显示紧凑预览。点击输入原表底部完整例句，包含方括号、重音和两个完整的带标签范围。')

def payload(): return {'version':VERSION, 'entries':ENTRIES, 'charts':CHARTS}
def manifest():
    buf=io.StringIO(newline='');w=csv.writer(buf,lineterminator='\n')
    w.writerow(['source','source_locator','page','section','display_item','symbol_id','insertion_sequence','codepoints','mode','tooltip','test_id','representation'])
    for e in ENTRIES:
        w.writerow([e['sourceRefs'][0]['sourceId'],e['sourceRefs'][0]['locator'],'169 Figure 2' if e['system']=='voqs' else '1',e['section'],e['display'],e['id'],e['insertText'],' '.join(e['codePoints']),e['insertionMode'],e['nameZh'],'click-'+e['id'],e.get('representation','')])
    return buf.getvalue()
def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    data=json.dumps(payload(),ensure_ascii=False,indent=2)+'\n';csvdata=manifest()
    coverage=ROOT/'docs/references/m17-symbol-coverage.csv'
    if args.write: DEST.write_text(data,'utf-8',newline='\n');coverage.write_text(csvdata,'utf-8',newline='\n')
    else:
        assert DEST.read_text('utf-8')==data,'catalog generation differs'
        assert coverage.read_text('utf-8')==csvdata,'coverage generation differs'
    ids=[x['id'] for x in ENTRIES]; assert len(ids)==len(set(ids))
    assert all(e['descriptionZh'] and e['usageZh'] and e['sourceRefs'] for e in ENTRIES)
    assert all(not(0xE000<=ord(c)<=0xF8FF) and not (ord(c)<32) for e in ENTRIES for c in e['insertText'])
    print(json.dumps({'version':VERSION,'entries':len(ENTRIES),'counts':{s:sum(e['system']==s for e in ENTRIES) for s in CHARTS},'sections':{s:len(v) for s,v in CHARTS.items()},'sha256':hashlib.sha256(data.encode()).hexdigest()},ensure_ascii=False))
if __name__=='__main__':main()
