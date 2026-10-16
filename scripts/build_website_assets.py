"""Build static-site brand assets and designed IPA outlines from licensed sources.

No generated mockup is used as the web page. Font geometry is a reference for
redrawn monoline glyphs, with highlights and relief, not an altered font.
"""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
from PIL import Image
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'temporary-site/assets'
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'licenses').mkdir(exist_ok=True)
(OUT/'fonts').mkdir(exist_ok=True)
BASE='https://raw.githubusercontent.com/google/fonts/main/ofl/zcoolkuaile/'
for name,target in [('ZCOOLKuaiLe-Regular.ttf',OUT/'fonts/ZCOOLKuaiLe-Regular.ttf'),('OFL.txt',OUT/'licenses/ZCOOL-OFL.txt')]:
    if not target.exists():
        with urllib.request.urlopen(BASE+name,timeout=60) as response: target.write_bytes(response.read())
shutil.copyfile(ROOT/'frontend/src/assets/Doulos-OFL.txt',OUT/'licenses/Doulos-OFL.txt')
shutil.copyfile(ROOT/'frontend/src/assets/DoulosSIL-Regular.ttf',OUT/'fonts/DoulosSIL-Regular.ttf')
source=ROOT/'frontend/public/app-icons/sheets/w1-edited.png'
icon=Image.open(source).convert('RGBA').crop((53,53,1201,1201))
icon.resize((256,256),Image.Resampling.LANCZOS).save(OUT/'brand.png')
icon.resize((48,48),Image.Resampling.LANCZOS).save(OUT/'favicon.png')
font=TTFont(ROOT/'frontend/src/assets/DoulosSIL-Regular.ttf')
glyphs=font.getGlyphSet();cmap=font.getBestCmap()
records=[]
designed_paths={
    'ʂ':'M160 64 C150 49 117 46 105 61 C82 88 139 99 148 117 C162 144 123 153 105 137 M105 137 C102 150 98 165 103 176 C110 190 132 182 134 168',
    'ɻ':'M137 55 L132 144 C130 167 142 183 156 183 C169 183 177 173 177 163 M133 116 C123 135 107 147 95 139',
    'ɕ':'M169 76 C169 52 143 46 123 52 C93 60 82 93 95 123 C105 147 127 159 148 149 C161 142 173 131 169 123 C162 105 137 117 127 135 C119 152 117 170 119 186',
}
for char,name in [('ʂ','retroflex-fricative'),('ɻ','retroflex-approximant'),('ɕ','alveolopalatal-fricative')]:
    glyph=glyphs[cmap[ord(char)]];pen=SVGPathPen(glyphs);glyph.draw(pen)
    # Original font is reference only. Redraw a rounded monoline with diagnostic hooks.
    colors=('#50324d','#301934','#25102d') if char=='ɻ' else ('#fffef8','#fff5df','#eddfc6')
    stroke=17 if char=='ɕ' else 20
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 240" role="img" aria-label="[{char}]">
<title>[{char}] — Doulos-based designed relief</title>
<defs><linearGradient id="face" x1="0" y1="0" x2=".8" y2="1"><stop stop-color="{colors[0]}"/><stop offset=".65" stop-color="{colors[1]}"/><stop offset="1" stop-color="{colors[2]}"/></linearGradient>
<filter id="relief" x="-40%" y="-40%" width="180%" height="180%"><feDropShadow dx="2" dy="7" stdDeviation="2.8" flood-color="#251630" flood-opacity=".28"/></filter>
<path id="glyph" d="{designed_paths[char]}"/>
<path id="brackets" d="M65 43 H44 V192 H65 M196 43 H216 V192 H196"/></defs>
<g fill="none" stroke-linecap="round" stroke-linejoin="round">
<g filter="url(#relief)" stroke="url(#face)"><use href="#brackets" stroke-width="11"/><use href="#glyph" stroke-width="{stroke}"/></g>
<g transform="translate(-1,-2)" stroke="#fff" opacity=".18"><use href="#glyph" stroke-width="{stroke-5}"/><use href="#brackets" stroke-width="6"/></g>
</g></svg>'''
    target=OUT/(name+'.svg');target.write_text(svg,encoding='utf-8')
    records.append(dict(file=target.name,symbol=char,codepoint=f'U+{ord(char):04X}',glyph_name=cmap[ord(char)],source_font='Doulos SIL 7.000',reference_outline_sha256=hashlib.sha256(pen.getCommands().encode()).hexdigest(),design='custom redrawn rounded monoline, based on reference glyph anatomy; attached retroflex hooks retained; relief gradient, shadow and custom brackets; not font typesetting',sha256=hashlib.sha256(target.read_bytes()).hexdigest()))

# Six custom Chinese display characters, drawn stroke by stroke; no font outline.
lettering=[
 ('语',15,'M15 17L27 28 M12 48H27V104L43 91 M53 20H107 M68 21L59 61 M52 42H96V63 M47 64H111 M57 82H102V113H57Z'),
 ('音',15,'M57 12L65 23 M23 34H101 M38 38L45 55 M85 38L77 56 M16 61H108 M33 79H91V114H33Z M35 96H89'),
 ('学',16,'M27 14L35 25 M55 10L61 23 M94 12L80 29 M16 57V41H108V57 M36 63H85L62 81V105Q62 119 44 112 M18 86H106'),
 ('工',22,'M25 22Q60 23 101 21 M63 24V105 M15 108Q58 105 110 107'),
 # 具: two sides meet the extended baseline; three interior bars, four counters.
 ('具',14,'M35 95V10H88V95 M35 31H88 M35 52H88 M35 73H88 M18 95H109 M44 114L25 128 M81 114L99 128'),
 ('箱',13,'M25 10L15 31 M24 23H53 M38 24L44 34 M80 10L69 31 M79 23H110 M96 24L102 34 M34 50V118 M13 68H55 M33 75L12 103 M36 79L55 95 M73 52H109V116H73Z M75 74H107 M75 95H107'),
]
parts=[]
for index,(char,width,path_data) in enumerate(lettering):
    x=12+(index%3)*130;y=8+(index//3)*145
    angle=[-2,1.2,-1.5,.6,-1.2,1][index]
    parts.append(f'<g transform="translate({x},{y}) rotate({angle},62,62)"><path d="{path_data}" stroke-width="{width}"/></g>')
title='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 410 290" role="img" aria-label="语音学工具箱"><title>语音学工具箱 · 手绘标题字</title><g fill="none" stroke="#271331" stroke-linecap="round" stroke-linejoin="round">'+''.join(parts)+'</g><path d="M21 142Q57 134 111 137" fill="none" stroke="#ffe147" stroke-width="10" stroke-linecap="round"/></svg>'
(OUT/'hero-lettering.svg').write_text(title,encoding='utf-8')

# Display lockup: licensed character skeletons preserve Chinese orthography;
# individually reshaped proportions, bounce, tilt, rounded weight and underline.
display_font=TTFont(OUT/'fonts/ZCOOLKuaiLe-Regular.ttf')
display_glyphs=display_font.getGlyphSet();display_cmap=display_font.getBestCmap()
upm=display_font['head'].unitsPerEm
display_parts=[]
for row,(text,size,baseline) in enumerate([('更多功能，',114,125),('等你在应用中发现。',94,258)]):
    step=size*1.04;left=(1100-step*len(text))/2
    for i,ch in enumerate(text):
        pen=SVGPathPen(display_glyphs);display_glyphs[display_cmap[ord(ch)]].draw(pen)
        x=left+i*step;dy=[0,-3,2,-1,3][i%5];angle=[-3,2,-1,3,-2][i%5]
        scale=size/upm;fill='#271331' if row==0 else '#8c71b5'
        display_parts.append(f'<g transform="translate({x},{baseline+dy}) rotate({angle}) scale({scale*1.025},{-scale*.98})"><path d="{pen.getCommands()}" fill="{fill}" stroke="{fill}" stroke-width="{upm*.008}" stroke-linejoin="round"/></g>')
display_svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 290" role="img" aria-label="更多功能，等你在应用中发现。"><title>更多功能，等你在应用中发现。</title><path d="M330 143 Q480 126 720 143" fill="none" stroke="#ffe147" stroke-width="15" stroke-linecap="round"/>'+''.join(display_parts)+'</svg>'
(OUT/'more-lettering.svg').write_text(display_svg,encoding='utf-8')
data=dict(schema='ptb-website-assets/1',sources=[
    dict(id='ASSET-DOULOS',file='fonts/DoulosSIL-Regular.ttf',license='SIL OFL 1.1',usage='annotation text and reference geometry for independently redrawn decorative glyphs',upstream='https://software.sil.org/doulos/'),
    dict(id='WEBSITE-ZCOOL-KUAILE',file='fonts/ZCOOLKuaiLe-Regular.ttf',license='SIL OFL 1.1',authors='Liu Bingke, Yang Kang, Wu Shaojie; ZCOOL KuaiLe Project Authors',upstream=BASE+'ZCOOLKuaiLe-Regular.ttf',version=TTFont(OUT/'fonts/ZCOOLKuaiLe-Regular.ttf')['name'].getDebugName(5)),
    dict(id='ASSET-MASCOT',file='brand.png',source=str(source.relative_to(ROOT)).replace('\\','/'),usage='existing approved W1 cropped per catalog and downsampled, no redraw'),
],ipa_artwork=records,spectrogram='Deterministic procedural decorative texture, no measured-data claim',screenshots='screens/provenance.json')
data['hero_lettering']={'file':'hero-lettering.svg','source':'Six original hand-authored Chinese stroke drawings; no typeface contours','sha256':hashlib.sha256((OUT/'hero-lettering.svg').read_bytes()).hexdigest()}
data['more_lettering']={'file':'more-lettering.svg','source':'ZCOOL KuaiLe OFL 1.1 outlines, individualized proportions, rounded expansion, bounce, rotation and custom underline; display artwork, not a new font','sha256':hashlib.sha256((OUT/'more-lettering.svg').read_bytes()).hexdigest()}
for record in data['sources']:record['sha256']=hashlib.sha256((OUT/record['file']).read_bytes()).hexdigest()
(OUT/'provenance.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'output':str(OUT),'ipa':[r['symbol'] for r in records]},ensure_ascii=False))
