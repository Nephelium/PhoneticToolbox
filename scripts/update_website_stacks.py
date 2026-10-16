"""Integrate the additional screenshot stacks into the static landing page."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[1]
p=root/'temporary-site/index.html';text=p.read_text('utf-8')
groups={
 'egg':('coral','-8','GitHub','浅色','blue',[
 ('egg-inverse-github-light','EGG 逆滤波'),('egg-github-light','EGG 信号分析')]),
 'spectrum':('violet','8','Ayu','深色','rose',[
 ('spectrum-audio-draw-ayu-dark','语谱图转音频 · 音频藏信息'),('spectrum-image-draw-ayu-dark','语谱图转音频 · 图片涂鸦重建'),('spectrum-ayu-dark','语谱图转音频 · 图片重建')]),
 'pitch':('coral','-8','One','浅色','green',[
 ('pitch-batch-one-light','变速变调 · 三点全连接批量生成'),('pitch-one-light','变速变调 · 单音节基频调整')])}
for key,(backdrop,turn,palette,mode,dot,cards) in groups.items():
    figure=f'<figure class="screen-stage screen-stack stack-{len(cards)} reveal" id="{key}-stack" style="--turn:{turn}deg"><span class="stage-backdrop {backdrop}-backdrop"></span><div class="stack-deck">'
    for i,(file,label) in enumerate(cards):
        figure+=f'<a class="screen-link stack-card" style="--card:{i};--z:{i+1}" href="assets/screens/{file}.png" data-screen data-caption="{label} · {palette} {mode}" aria-label="查看{label}完整截图"><img src="assets/screens/{file}.png" width="2560" height="1440" loading="lazy" alt="{label}的中文界面，{palette} {mode}配色"></a>'
    figure+=f'</div><figcaption><span class="palette-dot {dot}"></span>{palette} <span>{mode}</span></figcaption></figure>'
    pattern=r'(<section[^>]+id="'+key+r'"[\s\S]*?)(<figure[\s\S]*?</figure>)'
    text,n=re.subn(pattern,lambda m:m[1]+figure,text,count=1);assert n==1
p.write_text(text,'utf-8')
v=root/'scripts/verify_website.cjs';text=v.read_text('utf-8').replace("['ipa-stack','transcription-stack']","['ipa-stack','transcription-stack','egg-stack','spectrum-stack','pitch-stack']").replace("id==='ipa-stack'?3:2","['ipa-stack','spectrum-stack'].includes(id)?3:2").replace('Each of five overlapping cards','Each of twelve overlapping cards').replace("count(),14)","count(),18)").replace('fourteen image links','eighteen image links')
v.write_text(text,'utf-8')
