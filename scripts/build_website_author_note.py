"""User-supplied author note and media; locally designed OFL-derived display font."""
from pathlib import Path
import hashlib,json,math,re,shutil
from fontTools.ttLib import TTFont
from fontTools import subset
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.transformPen import TransformPen
ROOT=Path(__file__).resolve().parents[1];assets=ROOT/'temporary-site/assets'
paragraphs=[
 '唔……想了想，好像也没什么要说的。只有关于源码分享和使用授权，韶子想再小声说两句。',
 '对于 PhoneticToolbox 中韶子有权授权的部分，大家可以自由修改，也可以用自己的项目名发布修改版。韶子不要求额外署名，也不要求附上原仓库链接。',
 '不过，项目用到的第三方代码、字体和其他素材，仍须遵守各自的许可。需要保留的版权声明、许可证和第三方声明，请不要删掉；论文、文章的方法来源与引用，也请一并保留。',
 '嗯……最重要的一点：本项目，以及基于本项目的任何修改版，均不得以盈利为目的，不得用于任何商业用途，也不得以任何形式向任何用户收取任何费用。',
 '希望语音学的门槛，更低一些\n也希望各位，科研之路顺利。',
 '以及，开源万岁！',
]
font=TTFont(assets/'fonts/ZCOOLKuaiLe-Regular.ttf')
s=subset.Subsetter();s.populate(text=''.join(paragraphs)+'井韶子小声说两句');s.subset(font)
glyphset=font.getGlyphSet();upm=font['head'].unitsPerEm;new={}
for i,name in enumerate(font.getGlyphOrder()):
    pen=TTGlyphPen(glyphset);angle=math.radians([-1.1,.6,-.4,1.2,0][i%5]);sx=[.97,1,.985][i%3];sy=[1.02,.98,1.01][i%3]
    a,b=math.cos(angle)*sx,math.sin(angle)*sx;c,d=-math.sin(angle)*sy,math.cos(angle)*sy
    glyphset[name].draw(TransformPen(pen,(a,b,c,d,upm*(1-a)/2,[0,7,-5,3][i%4])))
    new[name]=pen.glyph()
for name,glyph in new.items():font['glyf'][name]=glyph
for platform,encoding,language in [(3,1,0x409),(1,0,0)]:
    for nameid,value in [(1,'Phonetic Whisper'),(2,'Regular'),(3,'PhoneticWhisper-2026-10'),(4,'Phonetic Whisper Regular'),(6,'PhoneticWhisper-Regular')]:font['name'].setName(value,nameid,platform,encoding,language)
    font['name'].setName((assets/'licenses/ZCOOL-OFL.txt').read_text('utf-8'),13,platform,encoding,language)
    font['name'].setName('https://openfontlicense.org',14,platform,encoding,language)
font.save(assets/'fonts/PhoneticWhisper-Regular.ttf')
font_version=hashlib.sha256((assets/'fonts/PhoneticWhisper-Regular.ttf').read_bytes()).hexdigest()[:12]
css=ROOT/'temporary-site/site.css'
css.write_text(re.sub(r'assets/fonts/PhoneticWhisper-Regular\.ttf(?:\?v=[a-f0-9]+)?','assets/fonts/PhoneticWhisper-Regular.ttf?v='+font_version,css.read_text('utf-8')),'utf-8')
media=[('author-avatar.jpg',r'D:\xwechat_files\wxid_r75ts1cjaams22_5a74\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\88532fe8bad33844ed3a4ae9800fc2ce.jpg'),('tiny-whisper.jpg',r'D:\xwechat_files\wxid_r75ts1cjaams22_5a74\temp\RWTemp\2026-10\9e20f478899dc29eb19741386f9343c8\eab40319ae5566de4e35e1272188f6e6.jpg')]
for name,src in media:shutil.copyfile(src,assets/name)
note='<section class="author-note reveal" id="a-small-note" aria-labelledby="author-name"><div class="author-identity"><a class="author-avatar" href="https://www.zhihu.com/people/jingshaozi" target="_blank" rel="noopener noreferrer" aria-label="井韶子"><img src="assets/author-avatar.jpg" width="1280" height="1280" loading="lazy" alt="井韶子的头像"></a><span id="author-name">井韶子</span><span class="author-spark" aria-hidden="true">✧</span></div><div class="speech-bubble"><span class="bubble-spark" aria-hidden="true">✦</span>'
for i,p in enumerate(paragraphs):
    klass=['note-opening','note-body','note-body','note-promise','note-wish','note-cheer'][i]
    meme='<img class="tiny-whisper" src="assets/tiny-whisper.jpg" width="689" height="691" loading="lazy" alt="超小声 bb">' if i==0 else ''
    p='<span>'+p.replace('\n','</span><br><span>')+'</span>' if i==4 else p
    note+=f'<p class="{klass}">{p}{meme}</p>'
note+='</div></section>'
p=ROOT/'temporary-site/index.html';text=p.read_text('utf-8')
css_version=hashlib.sha256(css.read_bytes()).hexdigest()[:12]
text=re.sub(r'href="site\.css(?:\?v=[a-f0-9]+)?"','href="site.css?v='+css_version+'"',text)
if 'id="a-small-note"' in text:text=re.sub(r'<section class="author-note[\s\S]*?</section>',note,text,count=1)
else:text=text.replace('</main>',note+'\n</main>')
counter=[0]
def hint(m):
    counter[0]+=1
    return '<span class="expand-hint">点选任一张，查看原图 ↗</span>' if counter[0]==1 else ''
text=re.sub(r'<span class="expand-hint">[^<]*</span>',hint,text)
text=re.sub(r'(<img src="assets/screens/(?!lip-nord-dark)[^"]+" width=")\d+(" height=")\d+',r'\g<1>2560\g<2>1440',text)
p.write_text(text,'utf-8')
record={'font':{'file':'fonts/PhoneticWhisper-Regular.ttf','source':'ZCOOL KuaiLe 2.001, OFL 1.1; subset to note text, individually reshaped glyph proportions, tilt and baseline; renamed derived font','license':'licenses/ZCOOL-OFL.txt'},'media':[{'file':name,'source':'User-supplied image for this local website design','sha256':hashlib.sha256((assets/name).read_bytes()).hexdigest(),'modifications':'None; CSS rounded display only'} for name,_ in media],'profile':'https://www.zhihu.com/people/jingshaozi','link_policy':'Avatar only; name and text are not links','license_note':'Author statement for local preview; preserves separate third-party licenses; does not modify formal project or dependency licenses. Noncommercial restriction is not an OSI-approved open-source license; closing phrase is the author sentiment.'}
record['font']['sha256']=hashlib.sha256((assets/record['font']['file']).read_bytes()).hexdigest()
(assets/'author-note-provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n','utf-8')
