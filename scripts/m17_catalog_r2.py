"""M17 R2: explicit terminology, source locators and reviewed matrix placement.

No source document is shipped or read at product runtime. Entry identities and
insertion sequences are preserved; charts are a view over the original inventory.
"""
from __future__ import annotations

# Independently transcribed from the user-supplied Chinese chart (2007,
# revised to 2005). User correction on 2026-10-03 takes precedence: click = 啧音.
# These names take precedence over the 2008 handbook prose.
DIACRITIC_NAMES = {
 '̥':'清化','̬':'浊化','ʰ':'送气','̹':'更圆','̜':'略展','̟':'偏前','̠':'偏后',
 '̈':'央化','̽':'中-央化','̩':'成音节','̯':'不成音节','˞':'r音色',
 '̤':'气声性','̰':'嘎裂声性','̼':'舌唇','ʷ':'唇化','ʲ':'腭化','ˠ':'软腭化',
 'ˤ':'咽化','̴':'软腭化或咽化','̝':'偏高','̞':'偏低','̘':'舌根偏前',
 '̙':'舌根偏后','̪':'齿化','̺':'舌尖性','̻':'舌叶性','̃':'鼻化',
 'ⁿ':'鼻除阻','ˡ':'边除阻','̚':'无闻除阻','̊':'清化（上置异体）',
}
SUPRA_NAMES = {'ˈ':'主重音','ˌ':'次重音','ː':'长','ˑ':'半长','̆':'超短',
 '|':'小(音步)组块','‖':'大(语调)组块','.':'音节间隔','‿':'连接(间隔不出现)'}
OTHER_NAMES = {'ʍ':'唇-软腭清擦音','w':'唇-软腭浊近音','ɥ':'唇-硬腭浊近音',
 'ʜ':'会厌清擦音','ʢ':'会厌浊擦音','ʡ':'会厌爆发音','ɕ':'龈-腭清擦音',
 'ʑ':'龈-腭浊擦音','ɺ':'龈边浊闪音','ɧ':'同时发ʃ和x'}
PLACE_NAMES = [('齿／齿龈','齿/龈'),('齿龈','龈'),('声门','喉'),('龈腭','龈-腭')]
MANNER_NAMES = [('闪音／拍音','拍音或闪音'),('挤喉','喷'),('吸气音','啧音'),('塞音','爆发音')]

def rename(e, name):
 old=e['nameZh']
 if old!=name: e['aliases']=sorted(set(e['aliases']+[old]));e['nameZh']=name

def ref(e, source, locator):
 item={'sourceId':source,'locator':locator}
 if item not in e['sourceRefs']:e['sourceRefs'].append(item)

def apply_r2(entries, charts):
 before=[(e['id'],e['insertText']) for e in entries]
 for e in entries:
  if e['system']!='ipa':continue
  s=e['section'];name=e['nameZh'];v=e['insertText']
  if s in ('pulmonic','nonpulmonic','combinations'):
   for old,new in PLACE_NAMES+MANNER_NAMES:name=name.replace(old,new)
  if s=='other' and v in OTHER_NAMES:name=OTHER_NAMES[v]
  if s=='nonpulmonic':
   if v=='ǃ':name='龈(后)啧音'
   if v=='ʼ':name='喷音记号'
  if s in ('diacritics','suprasegmentals'):
   mapping=DIACRITIC_NAMES if s=='diacritics' else SUPRA_NAMES
   if not e['isExample'] and v in mapping:name=mapping[v]
   elif e['isExample']:
    # Link examples to their immediately preceding independent mark, never
    # infer their semantic name from a substring inside the insertion value.
    previous=next(x for x in reversed(entries[:entries.index(e)]) if x['system']=='ipa' and x['section']==s and not x['isExample'])
    name=previous['nameZh']+'（示例）'
  if s=='tones':name=name.replace('特高','超高').replace('特低','超低')
  if s=='vowels':
   for old,new in [('次高','次闭'),('半高','半闭'),('次低','次开'),('半低','半开'),('高','闭'),('低','开')]:name=name.replace(old,new)
  rename(e,name)
  # Restrict terminology edits in prose to IPA entries. Anatomical descriptions
  # may still say 声门 or 齿龈: these name organs rather than chart headings.
  if s in ('pulmonic','nonpulmonic','combinations'):
   for old,new in [('闪音／拍音','拍音或闪音'),('挤喉音','喷音'),('挤喉号','喷音号'),('挤喉记号','喷音记号'),('吸气音','啧音')]:
    for key in ('descriptionZh','contrastZh'):e[key]=e[key].replace(old,new)
  ref(e,'ipa-chart-zh-2007','2007中文版（修订至2005年）／'+({'pulmonic':'辅音(肺部气流)','nonpulmonic':'辅音(非肺部气流)','vowels':'元音','other':'其他符号','diacritics':'附加符号','suprasegmentals':'超音段','tones':'声调与词重调','combinations':'构音术语与附加符号组合'}[s]))
  loc={'pulmonic':'第2.4节，pp.9–12（PDF pp.28–31）','nonpulmonic':'第2.5节，pp.12–13（PDF pp.31–32）','vowels':'第2.6节，pp.13–17（PDF pp.32–36）','other':'第2.9节，pp.23–24（PDF pp.42–43）','diacritics':'第2.8节，pp.20–23（PDF pp.39–42）','suprasegmentals':'第2.7节，pp.17–20（PDF pp.36–39）','tones':'第2.7节，pp.18–20（PDF pp.37–39）','combinations':'第2.4、2.8–2.9节，p.12、pp.20–24（PDF p.31、pp.39–43）'}[s]
  ref(e,'ipa-handbook-jiang-2008',loc)
  if s=='pulmonic' and v in ('t','d','n','r','ɾ','ɹ','l','ɬ','ɮ'):
   e['descriptionZh']+='中文版中除擦音外的齿、龈、龈后三栏共用相应基本符号；本扩展矩阵以龈栏为导航锚点，具体部位可用附加符号细化。' if v not in ('ɬ','ɮ') else ''
  if s=='nonpulmonic' and v in 'ʘǀǃǂǁ':
   ref(e,'ipa-chart-zh-2007','中文名称：啧音，按中文版图表及校正图核对')
  if s=='nonpulmonic' and v in 'ʘǀǃǂǁ':e['contrastZh']='啧音采用舌气流机制，前后闭塞间腔体扩大后释放前部闭塞；与肺部吸气言语区分。ǃ是音标字母，不能以普通感叹号替换。'
  if s=='nonpulmonic' and v in 'ɓɗʄɠʛ':e['contrastZh']='内爆音与喉部上升形成的喷音相区别；也不同于肺部吸气言语。'
  if s=='diacritics' and not e['isExample']:
   if v=='̚':e['contrastZh']='无闻除阻指未听到独立的爆破释放。闭塞可以经鼻腔、舌侧或后续音段解除，不能据此判定从未除阻。'
   if v=='˞':e['contrastZh']='r音色描述听觉性质。舌尖后卷和团舌等不同舌形都可能产生相关音色，不能从此符号唯一反推出舌形。'
   if v in ('̝','̞'):e['contrastZh']='偏高、偏低可改变元音舌位，也可改变辅音的狭窄程度；ɹ̝可用于更强摩擦的实现，β̞可用于近音实现。'
  if s=='vowels':e['descriptionZh']=e['descriptionZh'].replace('高前','闭前').replace('高央','闭央').replace('高后','闭后').replace('半高','半闭').replace('半低','半开').replace('次低','次开').replace('低前','开前').replace('低后','开后')
 # Repair an earlier dentolabial organ inversion using the clinical chart and
 # the Chinese 2013 paper: upper lip against lower teeth.
 for e in entries:
  if e['system']=='extipa' and e['section']=='articulatory' and e['insertText'] in ('͆','p͆','b͆'):
   ref(e,'extipa-ball-2018','p.156，第2.1节：dentolabial与labiodental的部位方向；PDF p.2')
   e['descriptionZh']='上唇与下齿接触或接近的齿唇构音，区别于下唇与上齿形成的唇齿构音。附加号须与具体基底一起解释。'
  if e['system']=='extipa' and any(x in e['nameEn'] for x in ('partially denasal','fricative nasal escape','velopharyngeal friction')):
   ref(e,'extipa-ball-2024','p.692 三项修订及p.693 Figure 2（PDF pp.2–3）；2024批准，现行2025表沿用')
 # Precise locations and independently authored explanations. Fixed Chinese
 # VoQS names are deliberately untouched.
 book_sections={'labial':'第1.5.4节，pp.26–28','lingual':'第1.5.2节，pp.20–24','velum':'第1.5.1节，pp.19–20','jaw':'第1.5.3节，pp.24–26','larynx':'第1.4.2节，pp.15–19'}
 phonation_pages={'V':'第2.3.3节，pp.44–47','F':'第2.3.10节，pp.60–63','W':'第2.3.7节，pp.53–56','C':'第2.3.11节，pp.63–67','V̤':'第2.3.8节，pp.56–58','Ṿ':'第2.3.9节，pp.58–60','V̰':'第2.3.11节，pp.63–67','V!':'第2.3.12节，pp.67–71','V‼':'第2.3.13节，pp.71–73','Ṿ‼':'第2.3.9、2.3.13节，pp.58–60、71–73','V𐞀':'第2.3.14节，pp.73–75','V͉':'第2.4节，pp.78–79','V͈':'第2.4节，pp.78–79'}
 descriptions={
 'V':'以声带周期振动为主要声源的参照发声设置。常态指未突出标记其他发声性质，并不规定某个固定基频，也不保证它是每个语言群体最常见的音质。',
 'F':'通常由声带纵向拉长、振动边缘变薄的设置形成，常位于较高音区。判断时结合发声机制与听觉性质，单独一个高F0值不足以确定假声。',
 'W':'气流经收窄的喉上管道产生耳语噪声，缺少常规声带周期振动。与加上声带振动的耳语声 Ṿ 区分。',
 'V̤':'声带振动伴随漏气，喉上管道相对开放。与耳语声 Ṿ 相比，喉部收缩较弱；两者都可有噪声，不能仅以是否带气流声分类。',
 'Ṿ':'在耳语设置中加入声带振动，通常伴杓会厌区域收缩和喉上管道变窄。单下点与气声 V̤ 的双下点分开；具体喉位和收缩程度可变化。',
 'C':'以嘎裂脉冲为显著特征的发声类型，常见低频或不规则振动。与常态浊声中呈现嘎裂性质的 V̰ 区别使用，不以统一基频阈值划分。',
 'V̰':'常态浊声带嘎裂性质，常伴较低或不规则的声带脉冲。嘎裂可有不同的喉部振动实现，不等同于所有粗糙音质。',
 'V!':'带粗糙听觉性质的发声。喉构音模型将其与喉部收缩、振动不规则及噪声联系；叹号不代表高响度，也不唯一指明室襞振动。',
 'V‼':'室襞参与持续振动的发声，可与下方声带振动并存。与一般糙声区分，若需确定振动来源，应结合喉部观察而非仅凭听感。',
 'V̬‼':'用于复音，即可感知两个同时音高成分的音质。不同振动源或振动模式可能产生类似表现，符号本身不确定具体病因，也不等同于一般粗糙。',
 'V𐞀':'杓会厌襞受气流驱动而振动，作为声源参与发声。上标小型大写AA标记这一来源，与室襞声源区别。实际振动可与声带声源相互作用。',
 'V͉':'相对参照较松弛的喉部发声设置，借用弱构音附加号表示。它不直接等于较轻响度或固定的低肌张力测量值。',
 'V͈':'相对参照较紧或受压的发声设置，借用强构音附加号表示。紧声可能涉及不同喉部结构的收缩，应补充具体观察，不能由符号唯一确定受力部位。',
 'Ṽ':'一段言语中较持续的鼻腔耦合设置，影响多个音段的共鸣。程度与腭咽通道的开放程度及持续时间有关，不能按单个元音鼻化直接概括整段音质。',
 'V͊':'一段言语中本来具有鼻音性的部分减弱或失去鼻音性的设置。它与extIPA用于单音的部分去鼻化分属不同描述层级，具体范围须用转写说明。',
 'Vꟸ':'咽门及相关咽腔较扩张的整体设置，可伴舌体前移和喉位降低。它描述共鸣通道设置，不等同于单纯低音高；上标带横H保留指定Unicode形式。',
 }
 for e in entries:
  if e['system']!='voqs':continue
  sec=e['section'];v=e['insertText']
  if v in descriptions:e['descriptionZh']=descriptions[v]
  # Every revised item has both chart evidence and prose location.
  loc={'airstream':'pp.166、168，第2.1、3.1节；p.169 Figure 2','phonation':'pp.166、168–170，第2.2、3.2节；p.169 Figure 2','larynx':'p.170 第3.3节；p.169 Figure 2','scope':'pp.168、170，第2.4、3.4节；p.169 Figure 2 底部例句'}.get(sec,'pp.166、168、170，第2.3、3.3节；p.169 Figure 2')
  e['sourceRefs'][0]['locator']=loc+'／'+e['nameEn']
  if v in phonation_pages:ref(e,'voice-quality-esling-2019',phonation_pages[v]+'；PDF页码为印刷页码加22')
  elif sec in book_sections:ref(e,'voice-quality-esling-2019',book_sections[sec]+'；PDF页码为印刷页码加22')
  elif sec=='phonation' and v not in ('ꟿ','И','V̬‼'):
   ref(e,'voice-quality-esling-2019','第1.4.1节，pp.13–15；第2章pp.53–75，组合发声的相互制约；PDF页码为印刷页码加22')
  if sec=='scope':e['contrastZh']='数字1–3是转写者对同一参照的相对程度判断，不是统一声压、基频、病理严重度或临床量表分数。多个标签可组合，生理上的兼容性须另行判断。'
 # Exact extended placements: original row/column positions + reviewed explicit
 # combination tokens. Names never drive classification at runtime.
 columns=['双唇','唇齿','齿唇','舌唇','齿','龈','龈后','卷舌','龈-腭','硬腭','软腭','小舌','咽/会厌','喉']
 labels=['爆发音','鼻音','颤音','拍音或闪音','擦音','边擦音','近音','边近音','塞擦音','边塞擦音','内爆音','喷音','啧音']
 rows=[{'label':name,'cells':[{'ids':[]} for _ in columns]} for name in labels]
 matrix={'id':'extended','title':'辅音','subtitle':'基本字母与组合输入按构音方式、部位合并。啧音、内爆音、喷音分别列出；空格不表示构音不可能。','kind':'matrix','columns':columns,'rows':rows,'ids':[]}
 old={s['id']:s for s in charts['ipa']}
 byid={e['id']:e for e in entries};placed=set()
 def put(ident,row,col):
  assert ident not in placed,ident
  rows[row]['cells'][col]['ids'].append(ident);matrix['ids'].append(ident);placed.add(ident)
 original_columns=[0,1,4,5,6,7,9,10,11,12,13]
 for ri,row in enumerate(old['pulmonic']['rows']):
  ci=0
  for cell in row['cells']:
   dest=5 if cell.get('span')==3 else original_columns[ci]
   for ident in cell['ids']:put(ident,ri,dest)
   ci+=cell.get('span',1)
 for ident in old['nonpulmonic']['ids']:
  e=byid[ident];v=e['insertText']
  if v=='ʼ':continue
  col={'ʘ':0,'ǀ':4,'ǃ':6,'ǂ':8,'ǁ':5,'ɓ':0,'ɗ':5,'ʄ':9,'ɠ':10,'ʛ':11,'pʼ':0,'tʼ':5,'kʼ':10,'sʼ':5}[v]
  put(ident,12 if v in 'ʘǀǃǂǁ' else 10 if v in 'ɓɗʄɠʛ' else 11,col)
 for ident in old['other']['ids']:
  v=byid[ident]['insertText']
  if v in ('ʜ','ʢ','ʡ','ɕ','ʑ','ɺ'):put(ident,0 if v=='ʡ' else 3 if v=='ɺ' else 4,12 if v in ('ʜ','ʢ','ʡ') else 5 if v=='ɺ' else 8)
 # Each combination is explicitly assigned by reviewed phonetic sequence.
 combo_map={}
 def bind(row,col,values):
  for value in values.split():
   assert value not in combo_map,value
   combo_map[value]=(row,col)
 bind(0,0,'pʰ');bind(0,4,'t̪ʰ t̪ d̪');bind(0,6,'t̠ʰ t̠ d̠');bind(0,7,'ʈʰ');bind(0,6,'t̠ʲ d̠ʲ');bind(0,9,'cʰ');bind(0,10,'kʰ');bind(0,11,'qʰ')
 for col,vals in [(0,'m̥'),(1,'ɱ̊'),(4,'n̪̊ n̪'),(6,'n̠̊ n̠'),(7,'ɳ̊'),(6,'n̠ʲ̊ n̠ʲ'),(9,'ɲ̊'),(11,'ɴ̥')]:bind(1,col,vals)
 for col,vals in [(0,'ʙ̥'),(4,'r̪̊ r̪'),(5,'r̥'),(6,'r̠̊ r̠'),(11,'ʀ̥')]:bind(2,col,vals)
 bind(3,1,'ⱱ̟');bind(3,5,'ɾ̥');bind(3,7,'ɽ̊')
 bind(4,4,'s̪ z̪');bind(4,6,'s̠ z̠');bind(5,4,'ɬ̪ ɮ̪');bind(5,6,'ɬ̠ ɮ̠')
 for col,vals in [(0,'β̞'),(4,'ð̞ ɹ̪'),(3,'ɹ̼'),(5,'ɹ̥'),(7,'ɻ̊'),(9,'j̊'),(10,'ɣ̞'),(11,'ʁ̞'),(12,'ʕ̞')]:bind(6,col,vals)
 bind(7,4,'l̪');bind(7,6,'l̠')
 for col,vals in [(0,'p͡ɸ b͡β'),(1,'p̪͡f b̪͡v'),(4,'t̪͡s̪ d̪͡z̪ t̪͡θ d̪͡ð'),(5,'t͡s d͡z'),(6,'t̠͡s̠ d̠͡z̠ t͡ʃ d͡ʒ'),(7,'ʈ͡ʂ ɖ͡ʐ'),(8,'t͡ɕ d͡ʑ'),(9,'c͡ç ɟ͡ʝ'),(10,'k͡x ɡ͡ɣ'),(11,'q͡χ ɢ͡ʁ'),(12,'ʡ͡ħ ʡ͡ʕ')]:bind(8,col,vals)
 bind(9,5,'t͡ɬ d͡ɮ')
 for col,vals in [(0,'ɓ̥'),(5,'ɗ̥'),(9,'ʄ̊'),(10,'ɠ̊'),(11,'ʛ̥')]:bind(10,col,vals)
 for col,vals in [(1,'fʼ'),(4,'θʼ t̪͡θʼ'),(5,'rʼ t͡sʼ t͡ɬʼ'),(6,'ʃʼ t͡ʃʼ'),(7,'ʈʼ ʂʼ ʈ͡ʂʼ'),(8,'t͡ɕʼ'),(9,'cʼ çʼ c͡çʼ'),(10,'xʼ k͡xʼ'),(11,'qʼ χʼ q͡χʼ')]:bind(11,col,vals)
 for ident in old['combinations']['ids']:
  row,col=combo_map[byid[ident]['insertText']];put(ident,row,col)
 remaining=[ident for key in ('pulmonic','nonpulmonic','other','combinations') for ident in old[key]['ids'] if ident not in placed]
 extras={'id':'extras','title':'其他符号与连音线','subtitle':'双重构音、同时构音与独立输入记号。','kind':'list','ids':remaining}
 charts['ipa']=[matrix,old['vowels'],extras,old['diacritics'],old['suprasegmentals'],old['tones']]
 old['tones']['title']='声调与词重调'
 assert before==[(e['id'],e['insertText']) for e in entries]
 assert len(matrix['ids'])+len(remaining)==sum(len(old[k]['ids']) for k in ('pulmonic','nonpulmonic','other','combinations'))
