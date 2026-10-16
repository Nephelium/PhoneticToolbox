"""Local 16:9 motion graphics for the spectrogram video; never publishes assets.

Uses the supplied PDFs, the website's own fonts/logo/palette, Pillow, NumPy,
SciPy and FFmpeg. No model inference, training, or external media generation.
Run with --preview first, then without arguments for the two MP4 deliverables.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import signal
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/creative-video-20261015'
ASSETS = OUT / 'assets'
WEB = ROOT / 'temporary-site/assets'
PAPER = Path('C:/Users/13680/Downloads/ABDM/Documents/The Sound Spectrograph.pdf')
AOKI = Path('C:/Users/13680/Downloads/ABDM/Documents/untitled.pdf')
W, H, FPS = 1920, 1080, 30
CREAM, INK, MUTED = '#fffaf0', '#271331', '#6b6074'
YELLOW, LAVENDER, CORAL, TEAL = '#ffe755', '#dfd2ff', '#ffb6a6', '#50a6a4'
cv2.setNumThreads(1)


def clamp(x):
    return max(0.0, min(1.0, x))


def ease(x):
    x = clamp(x)
    return x*x*(3-2*x)


@functools.lru_cache(None)
def font(size, face='body'):
    paths = {'title': WEB/'fonts/ZCOOLKuaiLe-Regular.ttf',
             'body': Path('C:/Windows/Fonts/msyh.ttc'),
             'bold': Path('C:/Windows/Fonts/msyhbd.ttc'),
             'caption': Path('C:/Windows/Fonts/simkai.ttf'),
             'brand': Path('C:/Windows/Fonts/trebucbd.ttf')}
    return ImageFont.truetype(str(paths[face]), size)


def text(im, xy, s, size=36, color=INK, face='body', center=False):
    d = ImageDraw.Draw(im)
    f = font(size, face)
    x, y = xy
    if center:
        x -= d.textlength(s, font=f)/2
    d.text((int(x), int(y)), s, font=f, fill=color, anchor='lt')


def box(im, rect, fill='white', radius=30, shadow=True, outline=INK, width=2):
    d = ImageDraw.Draw(im)
    x0,y0,x1,y1 = [int(v) for v in rect]
    if shadow:
        d.rounded_rectangle((x0+6,y0+10,x1+6,y1+10), radius, fill='#d9cfcb')
    d.rounded_rectangle((x0,y0,x1,y1), radius, fill=fill, outline=outline, width=width)


def arrow(im, a, b, color=INK, width=5, progress=1):
    d = ImageDraw.Draw(im)
    x0,y0=a; x1,y1=b
    x1=x0+(x1-x0)*clamp(progress); y1=y0+(y1-y0)*clamp(progress)
    d.line((x0,y0,x1,y1), fill=color, width=width)
    ang=math.atan2(y1-y0,x1-x0)
    pts=[(x1,y1),(x1-18*math.cos(ang-.5),y1-18*math.sin(ang-.5)),
         (x1-18*math.cos(ang+.5),y1-18*math.sin(ang+.5))]
    d.polygon(pts, fill=color)


def star(im, x,y,r=28,color=YELLOW):
    d=ImageDraw.Draw(im)
    p=[]
    for i in range(8):
        a=i*math.pi/4
        rr=r if i%2==0 else r*.27
        p.append((x+rr*math.cos(a),y+rr*math.sin(a)))
    d.polygon(p,fill=color)


def pill(im, x,y,s,fill=LAVENDER,size=26):
    w=int(ImageDraw.Draw(im).textlength(s,font=font(size)))+48
    box(im,(x,y,x+w,y+size+28),fill,100,False,outline=None)
    text(im,(x+24,y+12),s,size)


def caption(im,s):
    d=ImageDraw.Draw(im)
    d.line((110,981,1810,981),fill='#e6dccb',width=2)
    text(im,(960,1006),s,36,face='caption',center=True)


def heading(im,kicker,title):
    text(im,(100,61),kicker,22,MUTED,'brand')
    text(im,(95,105),title,65,face='title')
    d=ImageDraw.Draw(im)
    d.line((101,189,365,184),fill=YELLOW,width=12)


def wave(im,rect,t,color=TEAL,active=True):
    x0,y0,x1,y1=rect
    xs=np.linspace(x0,x1,440)
    u=np.linspace(0,1,440)
    amp=np.sin(np.pi*u)**.65
    ys=(y0+y1)/2+(y1-y0)*.38*amp*np.sin(u*85+t*4)*(.6+.4*np.cos(u*19-t))
    if not active: ys=ys*0+(y0+y1)/2
    ImageDraw.Draw(im).line(list(zip(xs,ys)),fill=color,width=4)


def play(im,x,y,r=36,fill=YELLOW):
    d=ImageDraw.Draw(im)
    d.ellipse((x-r,y-r,x+r,y+r),fill=fill,outline=INK,width=3)
    d.polygon([(x-r*.22,y-r*.45),(x-r*.22,y+r*.45),(x+r*.45,y)],fill=INK)


def background(t, section=''):
    im=BASE.copy()
    d=ImageDraw.Draw(im)
    y=50+10*math.sin(t*.7)
    d.ellipse((1640,y-210,2050,y+200),fill='#eee4fc')
    d.ellipse((-160,820+12*math.sin(t),200,1180),fill='#fff0b7')
    star(im,1795,850+math.sin(t)*14,25,CORAL)
    star(im,162,850+math.cos(t*.8)*13,18,LAVENDER)
    if section:
        text(im,(1815,65),section,22,MUTED,center=True)
    return im


def paper_image(n,w):
    return resized_page(n,w)


@functools.lru_cache(100)
def resized_page(n,w):
    if n>=len(PAGES):
        return Image.new('RGB',(w,round(w*760/572)), '#fffefb')
    return PAGES[n].resize((w,round(w*760/572)),Image.Resampling.LANCZOS)


def paste_page(im,n,x,y,w):
    p=paper_image(n,w)
    im.paste(p,(round(x),round(y)))


def turning_page_quad(flip, pw, ph, spine_x, top):
    """Perspective projection of a leaf rotating toward a camera at z=4800.

    The spine remains on z=0. The outer edge approaches the camera, so its
    vertical size grows by focal / (focal - depth), on both sides of the turn.
    """
    angle=math.pi*flip
    focal=4800.0
    depth=pw*math.sin(angle)
    scale=focal/(focal-depth)
    outer=spine_x+pw*math.cos(angle)*scale
    extension=ph*(scale-1)/2
    if flip<.5:
        return np.float32([[spine_x,top],[outer,top-extension],
                           [outer,top+ph+extension],[spine_x,top+ph]])
    return np.float32([[outer,top-extension],[spine_x,top],
                       [spine_x,top+ph],[outer,top+ph+extension]])


def book(im,index,flip=0):
    """Two-page spread, with a real front/back leaf crossing the spine."""
    pw=520; ph=round(pw*760/572); x=960-pw; y=231
    d=ImageDraw.Draw(im)
    box(im,(x-22,y-12,x+2*pw+22,y+ph+24),'#cab4e8',18,True)
    for k in range(5):
        d.line((x-7,y+ph+7+k*2,x+pw*2+7,y+ph+7+k*2), fill='#ad9eaf',width=1)
    paste_page(im,index,x,y,pw)
    paste_page(im,index+1+(2 if flip>0 else 0),x+pw,y,pw)
    if 0<flip<1:
        a=math.pi*flip
        src=paper_image(index+(1 if flip<.5 else 2),pw)
        quad=turning_page_quad(flip,pw,ph,960,y)
        if np.ptp(quad[:,0])>2:
            left,top=np.floor(quad.min(axis=0)).astype(int)-1
            right,bottom=np.ceil(quad.max(axis=0)).astype(int)+2
            q=quad-np.float32([left,top])
            mat=cv2.getPerspectiveTransform(np.float32([[0,0],[pw-1,0],[pw-1,ph-1],[0,ph-1]]),q)
            arr=np.asarray(src)
            # Lighting follows the turning leaf. The underlying PDF remains unedited.
            arr=np.uint8(arr*(1-.19*math.sin(a)))
            size=(int(right-left),int(bottom-top))
            dst=cv2.warpPerspective(arr,mat,size,flags=cv2.INTER_LINEAR,borderValue=(0,0,0))
            mask=cv2.warpPerspective(np.full((ph,pw),255,np.uint8),mat,size)
            im.paste(Image.fromarray(dst),(left,top),Image.fromarray(mask))
    d=ImageDraw.Draw(im)
    d.line((960,y,960,y+ph),fill='#b7aaba',width=3)


FLIPS=[(3.8,0,0.65),(5.0,2,.65)] + [(7.0+i*.68,4+i*2,.52) for i in range(13)]


def intro(t):
    im=background(t)
    if t<3.8:
        text(im,(117,145),'THE SOUND SPECTROGRAPH · 1946',25,MUTED,'brand')
        text(im,(110,238),'读着读着，',96,face='title')
        text(im,(110,363),'看见了声音。',96,face='title')
        pill(im,115,530,'语谱图，会发出什么声音？',YELLOW,30)
        w=560; x=1110-int(50*ease(t/2)); y=145-int(12*ease(t/2))
        box(im,(x-12,y-12,x+w+12,y+round(w*760/572)+12),'white',20)
        paste_page(im,0,x,y,w)
        star(im,960,340,42,LAVENDER)
        caption(im,'从一篇论文，和一点好奇心开始。')
    elif t<19.2:
        current=0; f=0
        for start,idx,duration in FLIPS:
            if t>=start+duration: current=idx+2
            elif t>=start: current=idx; f=ease((t-start)/duration); break
        title='这段声音，听起来会怎样？' if t<7 else ('附加材料呢？' if t<12 else ('再翻一页……' if t<16.2 else '翻到底了。'))
        heading(im,'A SMALL PROBLEM WHILE READING',title)
        book(im,current,f)
        if t>=8:
            pill(im,110,300,'音频？',YELLOW,31)
        if t>=11:
            pill(im,1580,470,'附件？',LAVENDER,31)
        if t>=14:
            pill(im,155,680,'补充材料？',CORAL,28)
        s='图就在纸上。' if t<7 else ('正文、末页，再往后翻……' if t<15.8 else '找了半天，还是没找到能播放的声音。')
        caption(im,s)
    elif t<22.3:
        heading(im,'STILL JUST A PICTURE','点一下……也没有声音。')
        book(im,4)
        play(im,1230,414,58,'#eee9ef')
        d=ImageDraw.Draw(im)
        for start in [19.8,20.7]:
            q=(t-start)/.6
            if 0<q<1:
                r=62+q*45
                d.ellipse((1230-r,414-r,1230+r,414+r),outline='#b7a3c7',width=3)
        xx=1295-30*ease((t-19.2)/.6); yy=493-34*ease((t-19.2)/.6)
        d.polygon([(xx,yy),(xx+7,yy+48),(xx+21,yy+34),(xx+40,yy+32)],fill=INK)
        if t>20.5: pill(im,120,450,'这是一张图片呀。',LAVENDER,29)
        caption(im,'那就，让图片开口说话。')
    else:
        q=t-22.3
        # A yellow circular wipe reveals the brand; no mock app results.
        scale=1+.12*math.exp(-q*3)*math.sin(q*12)
        d=ImageDraw.Draw(im)
        d.ellipse((240,210,1670,890),fill=LAVENDER)
        for i in range(9):
            a=i*2*math.pi/9+q*.1
            star(im,960+700*math.cos(a),545+330*math.sin(a),18+(i%3)*9,YELLOW if i%2 else CORAL)
        s=int(145*scale)
        logo=LOGO.resize((s,s),Image.Resampling.LANCZOS)
        im.paste(logo,(int(960-s/2),225),logo)
        text(im,(960,418),'PhoneticToolbox',112,face='brand',center=True)
        text(im,(960,585),'让纸上的声音，响起来。',78,face='title',center=True)
        wave(im,(620,700,1300,790),q)
        pill(im,752,835,'语谱图 → 音频',YELLOW,32)
        text(im,(960,1008),'phonetictoolbox.com',30,face='brand',center=True)
        if q<.30:
            # Fast color flash kept below full white.
            im=Image.blend(Image.new('RGB',(W,H),YELLOW),im,ease(q/.3))
    return im


def spec(im, rect, kind='long', scan=None):
    x0,y0,x1,y1=map(int,rect)
    src=SPECS[kind].resize((x1-x0,y1-y0),Image.Resampling.BILINEAR)
    im.paste(src,(x0,y0))
    d=ImageDraw.Draw(im)
    d.rectangle(rect,outline='#c9bdcc',width=2)
    if scan is not None:
        x=x0+int(clamp(scan)*(x1-x0))
        d.line((x,y0,x,y1),fill=TEAL,width=4)


def curve(im,rect,values,color=TEAL,progress=1):
    x0,y0,x1,y1=rect
    v=np.asarray(values,dtype=float)
    v=(v-v.min())/(np.ptp(v)+1e-10)
    n=max(2,int(len(v)*clamp(progress)))
    xs=np.linspace(x0,x1,len(v))[:n]
    ys=y1-v[:n]*(y1-y0)
    d=ImageDraw.Draw(im)
    d.line((x0,y1,x1,y1),fill='#c7bdcc',width=2)
    d.line(list(zip(xs,ys)),fill=color,width=4)


def phase(im,rect,t,organized=False):
    x0,y0,x1,y1=rect; d=ImageDraw.Draw(im)
    for j in range(5):
        for i in range(10):
            x=x0+(i+.5)*(x1-x0)/10; y=y0+(j+.5)*(y1-y0)/5
            a=(i*.75+j*.31 if organized else math.sin(i*79+j*37)*5)+t*.5
            d.ellipse((x-16,y-16,x+16,y+16),outline='#b2a2c4',width=2)
            d.line((x,y,x+14*math.cos(a),y+14*math.sin(a)),fill=INK,width=3)


def node(im,x,y,w,h,title,subtitle,fill='white',size=40):
    box(im,(x,y,x+w,y+h),fill)
    text(im,(x+w/2,y+29),title,size,face='bold',center=True)
    if subtitle: text(im,(x+w/2,y+h-54),subtitle,25,MUTED,center=True)


def section_bar(im,t):
    stages=[('Griffin–Lim',0,30),('短时窗',30,60),('NAFNet',60,78),('BigVGAN',78,94)]
    for i,(name,a,b) in enumerate(stages):
        x=125+i*445
        d=ImageDraw.Draw(im)
        d.rounded_rectangle((x,936,x+380,941),3,fill='#e9dfcf')
        if t>=a: d.rounded_rectangle((x,936,x+max(3,380*clamp((t-a)/(b-a))),941),3,fill=TEAL)
        text(im,(x+190,900),name,24,INK if a<=t<b else MUTED,center=True)


def algorithm(t):
    im=background(t,'流程示意')
    if t<7:
        heading(im,'01 / GRIFFIN–LIM','只有一张图，怎样变成声音？')
        node(im,170,280,660,490,'语谱图','先将灰度和坐标换算为幅度谱',LAVENDER)
        spec(im,(225,395,775,665),'long',t/7)
        arrow(im,(860,525),(1040,525),progress=ease(t/2))
        node(im,1080,280,660,490,'声音','还需要：相位',YELLOW)
        text(im,(1410,405),'?',160,face='title',center=True)
        wave(im,(1150,610,1670,675),t,active=False)
        caption(im,'语谱图告诉我们各处的强弱，却没有直接给出相位。')
    elif t<14:
        heading(im,'01 / GRIFFIN–LIM','先给相位一个初始猜测。')
        node(im,140,295,710,480,'幅度：来自目标谱','迭代时始终换回这个目标',LAVENDER)
        spec(im,(200,400,790,640),'long')
        node(im,1070,295,710,480,'相位：先随机初始化','接下来，让迭代逐步调整',YELLOW)
        phase(im,(1130,400,1720,650),t)
        text(im,(960,482),'+',90,face='brand',center=True)
        caption(im,'把目标幅度和初始相位组合起来，先试着合成一段声音。')
    elif t<23:
        heading(im,'01 / GRIFFIN–LIM','变成声音，再变回频谱。')
        for x,title,sub,fill in [(100,'当前复数谱','幅度 + 相位',LAVENDER),(730,'时域声音','逆变换并重叠相加',YELLOW),(1360,'重新分析','计算这段声音的 STFT','white')]:
            node(im,x,335,460,370,title,sub,fill,36)
        spec(im,(145,430,515,590),'long')
        wave(im,(775,430,1145,590),t)
        spec(im,(1405,430,1775,590),'long',((t-14)/3)%1)
        arrow(im,(590,510),(700,510)); arrow(im,(1220,510),(1330,510))
        text(im,(645,442),'iSTFT',23,face='brand',center=True)
        text(im,(1275,442),'STFT',23,face='brand',center=True)
        caption(im,'反复在时域与频域之间往返，让相邻帧能够拼成同一段声音。')
    elif t<30:
        heading(im,'01 / GRIFFIN–LIM','幅度换回目标，相位留下更新。')
        node(im,180,295,640,390,'目标幅度','固定',LAVENDER)
        spec(im,(235,395,765,575),'long')
        node(im,1100,295,640,390,'新相位','从刚刚重算的频谱中取得',YELLOW)
        phase(im,(1155,390,1685,565),t,True)
        arrow(im,(820,490),(1065,490))
        d=ImageDraw.Draw(im)
        d.line([(1420,725),(1420,800),(500,800),(500,725)],fill=TEAL,width=5)
        arrow(im,(500,785),(500,713),TEAL)
        pill(im,785,758,'再来一轮',YELLOW,31)
        caption(im,'重复这一过程，重建信号的幅度谱会逐步接近目标。')
    elif t<38:
        heading(im,'02 / SHORT WINDOWS','可时窗很短，怎么办？')
        node(im,140,280,740,485,'长时窗','频率结构较清楚',LAVENDER)
        spec(im,(195,390,825,650),'long',((t-30)/5)%1)
        node(im,1040,280,740,485,'短时窗','时间细节更清楚，频率结构更模糊',YELLOW)
        spec(im,(1095,390,1725,650),'short',((t-30)/5)%1)
        caption(im,'短时窗下，随机相位初始化可能让音高不稳。能换一个更好的起点吗？')
    elif t<45:
        heading(im,'02 / AOKI · 2023','从竖条纹里，寻找发声的节律。')
        # The full original paper page provides attribution without fabricated excerpts.
        page=AOKIPAGE.resize((455,644),Image.Resampling.LANCZOS)
        box(im,(111,220,590,880),'white',16)
        im.paste(page,(123,232))
        text(im,(700,294),'短时窗中的竖条纹',56,face='title')
        text(im,(700,379),'保留着细致的时间结构。',34)
        spec(im,(700,465,1730,695),'short',ease((t-38)/7))
        pill(im,700,759,'Aoki · 更有依据的 GL 初始相位',LAVENDER,28)
        caption(im,'Aoki 的思路：利用这些时间线索，粗略估计声源脉冲。')
    elif t<53:
        heading(im,'02 / A BETTER START','沿频率求和，再把峰值找出来。')
        # Computed from one synthetic pulse-excited signal; no claimed model result.
        for x,title,sub in [(115,'短时窗语谱图','每个时刻，沿频率方向求和'),(745,'垂直投影','得到随时间变化的强弱'),(1375,'峰值脉冲','提取峰值，组成近似声源')]:
            node(im,x,310,430,450,title,sub,'white',32)
        spec(im,(155,420,505,650),'short',clamp((t-45)/8))
        curve(im,(785,445,1135,645),PROJECTION,TEAL,ease((t-45)/3))
        d=ImageDraw.Draw(im)
        for px,pv in PEAKS:
            if px<=clamp((t-47)/4):
                x=1415+px*350
                d.line((x,645,x,645-pv*195),fill=TEAL,width=3)
        arrow(im,(570,535),(720,535)); arrow(im,(1200,535),(1350,535))
        caption(im,'对投影峰值做定位与插值，得到一个近似的脉冲序列。')
    elif t<60:
        heading(im,'02 / INITIAL PHASE','用脉冲的相位，启动 Griffin–Lim。')
        node(im,150,300,520,380,'近似声源脉冲','计算其短时傅里叶变换',LAVENDER,38)
        d=ImageDraw.Draw(im)
        for px,pv in PEAKS[::2]:
            x=200+px*420
            d.line((x,555,x,555-pv*130),fill=TEAL,width=4)
        arrow(im,(700,485),(980,485)); text(im,(840,416),'取相位',31,center=True)
        node(im,1020,300,700,380,'Griffin–Lim','从更有依据的相位出发迭代',YELLOW,52)
        phase(im,(1090,405,1650,565),t,True)
        pill(im,680,755,'目标幅度仍然来自原语谱图',LAVENDER,29)
        caption(im,'改进发生在初始化这一步，后面仍然由 GL 反复重建。')
    elif t<68:
        heading(im,'03 / MEL RESTORATION','接下来，把重建声音变成 Mel 谱。')
        node(im,145,300,650,450,'GL 输出声音','作为后续修复的输入',YELLOW)
        wave(im,(205,425,735,610),t)
        arrow(im,(835,520),(1080,520))
        text(im,(957,443),'Mel 分析',30,center=True)
        node(im,1120,300,650,450,'初始 Mel 谱','使用与声码器匹配的前端',LAVENDER)
        spec(im,(1180,415,1710,640),'mel',((t-60)/5)%1)
        caption(im,'当前流程先用 GL 得到声音，再计算与声码器匹配的 Mel 谱。')
    elif t<78:
        heading(im,'03 / CONDITIONAL NAFNET','再请训练过的网络，修一修。')
        node(im,100,350,450,360,'初始 Mel','80 带 Mel 输入','white',38)
        spec(im,(140,443,510,610),'mel')
        node(im,700,295,520,470,'NAFNet','条件化 Mel 谱修复网络',LAVENDER,66)
        d=ImageDraw.Draw(im)
        for k in range(4):
            x=770+k*83; yy=420+18*math.sin(t*2+k*.9)
            box(im,(x,yy,x+55,yy+130),YELLOW if k==2 else '#f4ecff',12,False)
        node(im,1370,350,450,360,'修复后 Mel','送入下一步声码器','white',38)
        # Schematic output uses the same pattern: no invented before/after result.
        spec(im,(1410,443,1780,610),'mel',((t-68)/4)%1)
        arrow(im,(575,520),(675,520)); arrow(im,(1245,520),(1345,520))
        pill(im,724,803,'韶子的语谱图修复训练',YELLOW,27)
        caption(im,'NAFNet 学习修正 Mel 谱中的失真，输出供声码器使用的频谱。')
    elif t<86:
        heading(im,'04 / BIGVGAN','最后，交给 BigVGAN 声码器。')
        node(im,100,345,450,370,'修复后 Mel','频谱条件',LAVENDER,38)
        spec(im,(140,445,510,625),'mel',((t-78)/4)%1)
        node(im,700,300,520,460,'BigVGAN','从 Mel 谱合成时域波形',YELLOW,62)
        wave(im,(770,440,1150,600),t,color=INK)
        node(im,1370,345,450,370,'可播放的音频','听见这张图的近似重建','white',36)
        play(im,1595,530,58)
        arrow(im,(575,525),(675,525)); arrow(im,(1245,525),(1345,525))
        caption(im,'BigVGAN 将修复后的 Mel 谱转换为可以播放的声音。')
    else:
        heading(im,'FROM PAPER TO SOUND','从纸上的图，到耳边的声音。')
        labels=[('语谱图','幅度与坐标'),('Griffin–Lim','相位重建'),('Mel 谱','声码器前端'),('NAFNet','修复 Mel'),('BigVGAN','合成声音')]
        for i,(title,sub) in enumerate(labels):
            x=100+i*356
            fill=[LAVENDER,YELLOW,'white',LAVENDER,YELLOW][i]
            node(im,x,385,295,250,title,sub,fill,34 if i!=1 else 30)
            if i<4: arrow(im,(x+309,510),(x+341,510),width=4)
        pill(im,347,720,'短时窗：脉冲相位初始化 → GL',LAVENDER,28)
        arrow(im,(644,714),(644,657),TEAL,width=4)
        wave(im,(1090,710,1735,805),t)
        caption(im,'这是一条近似重建的路径，输出不等同于原始录音。')
    section_bar(im,t)
    return im


def algorithm_transition(t):
    im=algorithm(t)
    for start in [7,14,23,30,38,45,53,60,68,78,86]:
        if start<=t<start+.27:
            return Image.blend(algorithm(start-.001),im,ease((t-start)/.27))
    return im


INTRO_CAPTIONS=[(0,3.8,'从一篇论文，和一点好奇心开始。'),(3.8,7,'图就在纸上。'),(7,15.8,'正文、末页，再往后翻……'),(15.8,19.2,'找了半天，还是没找到能播放的声音。'),(19.2,22.3,'那就，让图片开口说话。'),(22.3,28,'PhoneticToolbox：让纸上的声音，响起来。')]
METHOD_CAPTIONS=[(0,7,'语谱图告诉我们各处的强弱，却没有直接给出相位。'),(7,14,'把目标幅度和初始相位组合起来，先试着合成一段声音。'),(14,23,'反复在时域与频域之间往返，让相邻帧能够拼成同一段声音。'),(23,30,'重复这一过程，重建信号的幅度谱会逐步接近目标。'),(30,38,'短时窗下，随机相位初始化可能让音高不稳。能换一个更好的起点吗？'),(38,45,'Aoki 的思路：利用这些时间线索，粗略估计声源脉冲。'),(45,53,'对投影峰值做定位与插值，得到一个近似的脉冲序列。'),(53,60,'改进发生在初始化这一步，后面仍然由 GL 反复重建。'),(60,68,'当前流程先用 GL 得到声音，再计算与声码器匹配的 Mel 谱。'),(68,78,'NAFNet 学习修正 Mel 谱中的失真，输出供声码器使用的频谱。'),(78,86,'BigVGAN 将修复后的 Mel 谱转换为可以播放的声音。'),(86,94,'这是一条近似重建的路径，输出不等同于原始录音。')]


def synth_audio(duration,kind):
    """Original restrained plucked score and paper/click effects; no speech samples."""
    sr=48000; a=np.zeros((int(duration*sr),2),np.float32)
    rng=np.random.default_rng(20261015)
    def add(at,s,p=.5,gain=1):
        i=round(at*sr); n=min(len(s),len(a)-i)
        if n<=0:return
        a[i:i+n,0]+=s[:n]*math.sqrt(1-p)*gain
        a[i:i+n,1]+=s[:n]*math.sqrt(p)*gain
    notes=[60,67,71,74,67,64,69,72]
    for i,at in enumerate(np.arange(.35,duration-.3,.75)):
        if kind=='intro' and 19.2<at<22.3:continue
        u=np.arange(int(sr*1.4))/sr
        f=440*2**((notes[i%8]-69)/12)
        s=(np.sin(2*np.pi*f*u)+.20*np.sin(4*np.pi*f*u)+.09*np.sin(6*np.pi*f*u))
        env=(1-np.exp(-u*150))*np.exp(-u*4.5)
        add(at,s*env,.25+.5*(i%3)/2,.052)
    if kind=='intro':
        for at,_,dur in FLIPS:
            u=np.arange(int(sr*.22))/sr
            n=rng.normal(0,1,len(u)); n=signal.lfilter([1,-.8],[1],n)
            n=n*np.sin(np.pi*u/.22)**2*.034
            add(at+.06,n,.55)
        for at in [19.8,20.7]:
            u=np.arange(int(sr*.12))/sr
            add(at,np.sin(2*np.pi*430*u)*np.exp(-u*38),.5,.11)
        for k,n in enumerate([60,64,67,72]):
            u=np.arange(int(sr*2))/sr; f=440*2**((n-69)/12)
            add(22.3+k*.075,np.sin(2*np.pi*f*u)*np.exp(-u*2),.2+.2*k,.09)
    else:
        for at in [7,14,23,30,38,45,53,60,68,78,86]:
            u=np.arange(int(sr*.20))/sr
            add(at,np.sin(2*np.pi*(700*u-800*u*u))*np.exp(-u*20),.5,.055)
    ramp=int(sr*.6)
    a[:ramp]*=np.linspace(0,1,ramp)[:,None]
    a[-ramp:]*=np.linspace(1,0,ramp)[:,None]
    assert np.isfinite(a).all() and abs(a).max()<.98
    return sr,np.int16(a*32767)


def srt(path,rows):
    def ts(x):
        ms=round(x*1000); return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
    path.write_text('\n\n'.join(f'{i+1}\n{ts(a)} --> {ts(b)}\n{s}' for i,(a,b,s) in enumerate(rows))+'\n',encoding='utf-8')


def prepare():
    global BASE,PAGES,LOGO,AOKIPAGE,SPECS,PROJECTION,PEAKS
    ASSETS.mkdir(parents=True,exist_ok=True)
    BASE=Image.new('RGB',(W,H),CREAM)
    d=ImageDraw.Draw(BASE)
    for y in range(16,H,28):
        for x in range(16,W,28):d.ellipse((x,y,x+1,y+1),fill='#ece4d6')
    pdf=fitz.open(PAPER)
    PAGES=[]
    for page in pdf:
        pix=page.get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False)
        PAGES.append(Image.frombytes('RGB',(pix.width,pix.height),pix.samples))
    doc=fitz.open(AOKI); pix=doc[1].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
    AOKIPAGE=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
    LOGO=Image.open(WEB/'brand.png').convert('RGBA')
    # Analytic synthetic source: pulses through three resonators, 0.5 s at 16 kHz.
    sr=16000; n=sr//2; u=np.arange(n)/sr
    phase0=np.cumsum(145+24*np.sin(2*np.pi*u*1.8))/sr
    pulses=np.r_[0,np.diff(np.floor(phase0))]
    s=pulses.copy()
    for f,bw in [(650,90),(1450,120),(2600,180)]:
        rr=np.exp(-np.pi*bw/sr)
        s=signal.lfilter([1], [1,-2*rr*np.cos(2*np.pi*f/sr),rr*rr],s)
    s*=np.sin(np.pi*np.arange(n)/n)**.7
    s/=max(abs(s))
    SPECS={}
    for kind,win,hop in [('long',512,32),('short',64,16)]:
        f,tt,z=signal.stft(s,sr,nperseg=win,noverlap=win-hop,nfft=1024)
        mag=np.abs(z[f<=4000])
        db=np.clip(20*np.log10(mag/(mag.max()+1e-10)+1e-6),-65,0)
        a=(db+65)/65
        # Website ink on cream for spectra.
        bg=np.array([255,250,240]); fg=np.array([39,19,49])
        rgb=bg[None,None,:]*(1-a[::-1,:,None])+fg[None,None,:]*a[::-1,:,None]
        SPECS[kind]=Image.fromarray(np.uint8(rgb))
        if kind=='short':
            PROJECTION=mag.sum(axis=0)
            peaks,_=signal.find_peaks(PROJECTION,distance=4,prominence=PROJECTION.max()*.025)
            PEAKS=[(i/(len(PROJECTION)-1),float(PROJECTION[i]/PROJECTION.max())) for i in peaks]
    # A Mel-axis illustration, explicitly a schematic rather than a trained output.
    z=np.asarray(SPECS['long'])
    rows=np.exp(np.linspace(0,np.log(len(z)),80))-1
    rows=(len(z)-1-rows).astype(int)[::-1]
    SPECS['mel']=Image.fromarray(z[rows])
    srt(OUT/'01_片头_字幕.srt',INTRO_CAPTIONS)
    srt(OUT/'02_算法流程_字幕.srt',METHOD_CAPTIONS)
    manifest={'date':'2026-10-15','video':{'size':[W,H],'fps':FPS,'intro_seconds':28,'method_seconds':94},
      'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [PAPER,AOKI,WEB/'brand.png',WEB/'fonts/ZCOOLKuaiLe-Regular.ttf',Path('C:/Windows/Fonts/simkai.ttf'),ROOT/'temporary-site/site.css']],
      'revision':'Perspective-correct forward page turn; KaiTi only for bottom narrative captions. Other typography unchanged.',
      'references':['https://doi.org/10.1109/TASSP.1984.1164317','https://doi.org/10.1250/ast.44.186','https://github.com/megvii-research/NAFNet','https://github.com/NVIDIA/BigVGAN'],
      'method':'GL audio -> vocoder native Mel -> VisibleMelNAFNet (80 bands) -> BigVGAN. Aoki initialization precedes GL for short windows.',
      'scope':'Schematic explanation only, synthetic plot signal, no model-performance claim. Original procedural music and sound effects. No narration.',
      'paper_pages':len(PAGES),'paper_spreads':[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]}
    (OUT/'sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def previews():
    groups=[('01_片头',intro,[1.5,5.3,9.2,14.3,20.8,25]),('02_算法流程',algorithm,[3,10,18,26,34,41,49,56,64,73,82,90])]
    for name,fn,times in groups:
        sheet=Image.new('RGB',(1280,math.ceil(len(times)/2)*390),CREAM)
        for i,t in enumerate(times):
            im=fn(t)
            im.save(ASSETS/f'{name}_{t:g}s.png')
            thumb=im.resize((640,360),Image.Resampling.LANCZOS)
            xx=(i%2)*640; yy=(i//2)*390
            sheet.paste(thumb,(xx,yy))
            text(sheet,(xx+12,yy+365),f'{t:g} s',19,MUTED,'brand')
        sheet.save(OUT/f'{name}_分镜.jpg',quality=93)
    print('PREVIEWS_READY',flush=True)


def render(name,fn,duration,kind):
    dest=OUT/name
    with tempfile.TemporaryDirectory(prefix='ptb-video-') as scratch:
        audio=Path(scratch)/'soundtrack.wav'
        sr,a=synth_audio(duration,kind); wavfile.write(audio,sr,a)
        cmd=[shutil.which('ffmpeg'),'-hide_banner','-loglevel','error','-y',
          '-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0',
          '-i',str(audio),'-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
          '-threads','4','-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart',
          '-t',str(duration),str(dest)]
        with open(OUT/f'{kind}-encode.log','w',encoding='utf-8') as log:
            p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
            try:
                for frame in range(round(duration*FPS)):
                    im=fn(frame/FPS)
                    p.stdin.write(im.tobytes())
                    if frame%(FPS*5)==0: print(f'{kind}: {frame/FPS:g}/{duration} s',flush=True)
                p.stdin.close()
                code=p.wait()
                if code: raise RuntimeError(f'FFmpeg returned {code}: {kind}-encode.log')
            except BaseException:
                if p.poll() is None:p.terminate();p.wait()
                raise
    print('VIDEO_READY',str(dest),dest.stat().st_size,flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--preview',action='store_true'); parser.add_argument('--only',choices=['intro','method'])
    a=parser.parse_args();prepare();previews()
    if not a.preview:
        if a.only in [None,'intro']:render('01_论文翻页片头.mp4',intro,28,'intro')
        if a.only in [None,'method']:render('02_从GL到BigVGAN.mp4',algorithm_transition,94,'method')


if __name__=='__main__':main()
