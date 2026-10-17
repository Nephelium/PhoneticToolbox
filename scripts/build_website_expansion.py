"""Assemble the static, accessible listening lab and source-backed support sections."""
from pathlib import Path
import html,json,re,hashlib
from website_audio_baseline import verify as verify_basic_gl
from build_website_gl_comparison import verify as verify_three_way
ROOT=Path(__file__).resolve().parents[1];SITE=ROOT/'temporary-site'
def esc(s):return html.escape(str(s),quote=True)

CHAPTER_DIVIDERS = {
    'wave': '<svg viewBox="0 0 660 80"><path class="chapter-pencil" d="m14 43 29-1m13 0 39-2m15 1 23-1m13 1 35-1m14 1 26 1m13-1 18 1M408 39l27 1m13-1 33 2m14-1 23 1m14-1 38 2m13-1 25 1m13-1 25 1"/><path class="chapter-wash" d="M272 31q29-6 53-2t60-2l-6 13 8 12q-31 5-59 2t-59 5l7-15Z"/><path class="chapter-sound" d="m282 37 .5 9m12-18-.6 25m12-33 1 43m12-33-.8 22m12-37 .6 51m12-43-.8 35m12-27 .6 20m12-14-.4 9"/></svg>',
    'paper': '<svg viewBox="0 0 660 80"><path d="M12 31L24 29L36 33L48 30L60 32L72 28L84 31L96 30L108 34L120 31L132 29L144 32L156 31L168 29L180 33L192 30L204 32L216 28L228 31L240 30L252 34L264 31L276 29L288 32L300 31L312 29L324 33L336 30L348 32L360 28L372 31L384 30L396 34L408 31L420 29L432 32L444 31L456 29L468 33L480 30L492 32L504 28L516 31L528 30L540 34L552 31L564 29L576 32L588 31L600 29L612 33L624 30L636 32L648 28L648 51L636 49L624 53L612 50L600 52L588 49L576 48L564 50L552 53L540 50L528 52L516 48L504 51L492 49L480 53L468 50L456 52L444 49L432 48L420 50L408 53L396 50L384 52L372 48L360 51L348 49L336 53L324 50L312 52L300 49L288 48L276 50L264 53L252 50L240 52L228 48L216 51L204 49L192 53L180 50L168 52L156 49L144 48L132 50L120 53L108 50L96 52L84 48L72 51L60 49L48 53L36 50L24 52L12 49Z" fill="#ece2f4" stroke="none"/><path d="M12 31q110-5 212 0t220 0 192 0" fill="none" stroke="#c2add3" stroke-width="1.3" stroke-dasharray="2 7" opacity=".65"/><path d="m563 31 38-1-18 18Z" fill="#fff0ad" stroke="none"/></svg>',
    'tape': '<svg viewBox="0 0 660 80"><path d="M14 43q73-3 139 0t116-1M392 42q60-1 119 0t135-1" fill="none" stroke="#c5b3d4" stroke-width="1.5"/><g transform="rotate(-8 307 39)"><path d="M255 23 352 23l-3 6 3 6-3 6 2 7-96 1 3-7-3-6 3-7Z" fill="#ffe686" fill-opacity=".85" stroke="none"/></g><g transform="rotate(7 352 43)"><path d="M310 29 392 29l-3 7 3 7-3 7 2 8-81 0 3-8-3-7 3-7Z" fill="#d8c7eb" fill-opacity=".8" stroke="none"/><path d="m318 35 5 17m12-18 5 17m12-18 5 17m12-18 5 17" stroke="#b69acb" stroke-width="1.4" opacity=".45"/></g></svg>',
}
CHAPTER_STYLES = {
    "tutorials": "wave", "lab": "wave", "methods": "paper", "faq": "paper",
    "updates": "tape", "next": "tape", "feedback": "tape", "contact": "tape",
    "a-small-note": "tape",
}

def tutorials():
    modules = ['录音', '唇形提取', '参数估计', '参数显示', 'EGG 信号分析', 'LPC 谱图',
               '声学参数合成', '生理参数合成', '发声类型合成', '变速变调', '语谱图转音频',
               '国际音标表Plus', '汉字转国际音标', 'MFA 自动标注', 'TextGrid 标注', '音系归纳', '感知实验', '语音学论文精读']
    buttons = ''.join(
        f'<button type="button" class="tutorial-channel tutorial-available" id="tutorial-spectrogram" data-tutorial="spectrogram" aria-controls="tutorial-screen tutorial-episodes" aria-pressed="false">{esc(name)}</button>'
        if name == '语谱图转音频' else f'<button type="button" class="tutorial-channel" disabled>{esc(name)}</button>'
        for name in modules)
    return '''<section class="content-section tutorial-section" id="tutorials" aria-labelledby="tutorials-heading">
      <div class="tutorial-heading reveal"><div><span class="eyebrow">A LITTLE WATCH &amp; LEARN</span><h2 id="tutorials-heading">使用教程</h2></div><span class="tutorial-construction">正在施工中 <span aria-hidden="true">✦</span></span></div>
      <div class="tutorial-layout">
        <div class="tutorial-viewer reveal">
          <div class="tutorial-tv"><div class="tutorial-antenna" aria-hidden="true"></div>
            <div class="tutorial-screen" id="tutorial-screen" aria-label="使用教程视频">
              <div class="tutorial-idle"><span class="tutorial-play-mark" aria-hidden="true">▶</span><strong>先来认识一下工具箱</strong><p>点一下「介绍」，开播啦。</p><span class="tutorial-idle-spark" aria-hidden="true">✧</span></div>
            </div>
            <div class="tutorial-tv-chin" aria-hidden="true"><span><i></i> PHONETIC TOOLBOX</span><span class="tutorial-speaker"></span><span class="tutorial-dial"></span></div>
          </div>
          <div class="tutorial-actions"><button type="button" id="tutorial-fullscreen" hidden>全屏观看 ↗</button><a href="https://www.bilibili.com/video/BV1WWpr67EkX/" target="_blank" rel="noopener noreferrer">去 B 站播放 ↗</a></div>
          <p id="tutorial-status" class="tutorial-status" role="status" aria-live="polite"></p>
          <div class="tutorial-episodes" id="tutorial-episodes" hidden role="group" aria-label="语谱图转音频分集">
            <span>语谱图转音频</span><div><button type="button" data-tutorial-episode="spectrogram" aria-controls="tutorial-screen" aria-pressed="false">第一集</button><button type="button" disabled>第二集 · 待更新</button></div>
          </div>
          <noscript><p>前往 B 站观看：<a href="https://www.bilibili.com/video/BV1WWpr67EkX/" target="_blank" rel="noopener noreferrer">工具箱介绍 ↗</a> · <a href="https://www.bilibili.com/video/BV1RuJm6gEWE/" target="_blank" rel="noopener noreferrer">语谱图转音频 · 第一集 ↗</a></p></noscript>
        </div>
        <div class="tutorial-channels reveal" role="group" aria-label="教程节目">
          <div class="tutorial-channel-grid"><button type="button" class="tutorial-channel tutorial-intro" id="tutorial-intro" data-tutorial="intro" aria-controls="tutorial-screen tutorial-episodes" aria-pressed="false">介绍 <span aria-hidden="true">▶</span></button>''' + buttons + '''</div>
          <p class="tutorial-coming">更多教程正在准备中。</p>
        </div>
      </div>
    </section>'''
def reference_catalog():
    # Use the same public projection as the desktop component, never the raw registry.
    sources = json.loads((ROOT/'frontend/src/generated/sources.json').read_text('utf-8'))
    registry = (ROOT/'frontend/src/app/registry.ts').read_text('utf-8')
    modules = dict(re.findall(r"\{id:'([^']+)',title:'([^']+)'", registry))
    assert len(modules) == 18
    groups = [('academic', '学术引用'), ('software', '软件与代码库引用')]
    result = ''
    options = ''.join(f'<option value="{esc(key)}">{esc(name)}</option>' for key, name in modules.items())
    for group, title in groups:
        rows = [s for s in sources if s['acknowledgement_group'] == group]
        result += f'<details class="reference-catalog" data-reference-group="{group}"><summary>{title} · {len(rows)} 条<span>展开查看</span></summary><div class="reference-catalog-body">'
        result += f'<div class="reference-filters" hidden><label>搜索来源<input type="search" class="reference-query" id="reference-query-{group}" placeholder="搜索论文、项目、作者或模块"></label><label>查看范围<select class="reference-module" id="reference-module-{group}"><option value="">全部模块与公共组件</option>'+options+'</select></label></div>'
        result += f'<p class="reference-count" role="status" aria-live="polite">共 {len(rows)} 条来源</p><div class="reference-results" tabindex="0" role="region" aria-label="{title}列表">'
        for row in rows:
            names = '、'.join(name for key, name in modules.items() if key in row['modules']) or '公共组件'
            citation = row.get('citation') or ' · '.join(filter(None, [row['authors'], row['title']]))
            result += f'<article class="reference-source" data-source-id="{esc(row["id"])}" data-modules="{esc(" ".join(row["modules"]))}"><h4>{esc(row["title"])}</h4><small>用于：{esc(names)}</small><p class="reference-citation">{esc(citation)}</p>'
            if row.get('reference_note'):
                result += f'<p>{esc(row["reference_note"])}</p>'
            for key, prefix in [('version', '版本：'), ('license', '')]:
                if row.get(key): result += f'<small>{prefix}{esc(row[key])}</small>'
            if row.get('attribution'):
                date = f'（{row["permission_date"]} 作者邮件许可）' if row.get('permission_date') else ''
                result += f'<p>{esc(row["attribution"]+date)}</p>'
            if row.get('adaptation_note'):
                result += '<details class="reference-adaptation"><summary>改写说明</summary><p>'+esc(row['adaptation_note'])+'</p>'
                if row.get('adaptation_caution'): result += '<p>'+esc(row['adaptation_caution'])+'</p>'
                result += '</details>'
            result += '<div class="reference-source-links">'
            for label, url in row['urls'].items():
                assert url.startswith('https://'), f'Unexpected source URL: {row["id"]}'
                result += f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(label)} ↗</a>'
            result += '<button type="button" class="reference-copy" hidden>复制引用</button></div></article>'
        result += '</div></div></details>'
    return result

def upcoming(player,audio,examples):
    body='''<section class="content-section upcoming-section" id="next" aria-labelledby="next-heading">
  <div class="upcoming-paper reveal"><span class="upcoming-tab">下一步…</span>
    <div class="upcoming-intro"><div><span class="eyebrow">THE NEXT LITTLE STEP</span><h2 id="next-heading">短时窗里的声音，<br><em>还想听得更清楚。</em></h2></div><span class="upcoming-stamp">语谱图转音频<br><small>模型训练进行中</small><i aria-hidden="true">✦</i></span></div>
    <div class="upcoming-copy"><p>唔……最近在试着让语谱图转音频的效果更好一些。韶子还训练了一个小模型，让它结合图片和时间、频率标定，估计生成这张语谱图时可能使用的 STFT 窗宽与动态范围。有了这两个估计值，就可以先调整重建参数，再试着把声音找回来啦。</p><p>相位重建这一步，韶子参考了青木（Naofumi Aoki）关于短时窗语谱图重建的研究，先沿频率方向求和，从得到的时间投影里提取峰值脉冲，构造激励信号，再用它的相位初始化 Griffin–Lim。这样，迭代就有了一个可以参考的起点。迭代过程中，还借用了 Perraudin 等人的动量加速方法，把相邻两次迭代的变化也利用起来，帮助相位估计更快接近目标。工具箱目前采用 <code>0.99</code> 的动量系数，收敛通常能快一些。</p><p>有了初步重建的声音之后呢，接下来就是 Mel 谱修复啦。先把声音转成 Mel 谱，再训练一个条件化 NAFNet，让它以目标 Mel 谱为参照，学习修补重建过程中出现的频谱偏差，希望能把模糊、失真的频谱结构整理得更接近自然声音。修复后的 Mel 谱再交给 BigVGAN 声码器，由它生成最终的音频波形。这一步沿用预训练的 BigVGAN，让声码器利用已经学到的语音规律，把修复后的频谱合成为声音（其实和降噪、回声消除之类的算法差不多）。</p><p>目前还在继续折腾，希望能成功hhh 下面把原始 GL、动量加速、参数预测和 Mel 谱修复分成四路，一步一步听听变化叭。</p></div>
    <a class="upcoming-reference" href="https://doi.org/10.1250/ast.44.186" target="_blank" rel="noopener noreferrer"><span>相位起点：青木的短时窗重建研究 ↗</span><cite>Technique of improving speech restoration from spectrograms of short windows for the Griffin–Lim algorithm</cite><small>Naofumi Aoki · 2023 · Acoustical Science and Technology, 44(3), 186–188</small></a>
    <a class="upcoming-reference" href="https://perraudin.info/publications/perraudin-note-002.pdf" target="_blank" rel="noopener noreferrer"><span>迭代过程：动量加速 Griffin–Lim · 作者全文 ↗</span><cite>A fast Griffin–Lim algorithm</cite><small>Nathanaël Perraudin, Peter Balazs &amp; Peter L. Søndergaard · 2013 · WASPAA, 1–4</small></a>
    <ol class="upcoming-pipeline" aria-label="短时窗语谱图重建流程">
      <li><span class="upcoming-step">01 / 先读一读图片</span><h3>估计 STFT 参数</h3><span class="upcoming-model">窗宽与动态范围预测模型</span><p>结合语谱图与时间、频率标定，预测窗宽和动态范围，窗宽取整到毫秒。</p><span class="upcoming-arrow" aria-hidden="true">→</span></li>
      <li><span class="upcoming-step">02 / 再把声音找回来</span><h3>动量加速重建</h3><span class="upcoming-model">Griffin–Lim · 动量 0.99</span><p>按预测参数，从固定的随机相位起点出发，迭代 32 次，重建初步声音。</p><span class="upcoming-arrow" aria-hidden="true">→</span></li>
      <li><span class="upcoming-step">03 / 把频谱修一修</span><h3>Mel 谱修复</h3><span class="upcoming-model">条件化 NAFNet</span><p>把同一段预测参数动量 GL 的声音转成 Mel 谱，用训练好的模型修补频谱偏差。</p><span class="upcoming-arrow" aria-hidden="true">→</span></li>
      <li><span class="upcoming-step">04 / 再交给耳朵听</span><h3>合成音频</h3><span class="upcoming-model">BigVGAN</span><p>沿用预训练声码器，把修复后的 Mel 谱合成为最终的音频波形。</p></li>
    </ol>
    <div class="next-comparison-heading"><h3>同一张图，四种方式听一听。</h3><p>先听原始 GL，再听动量加速，接着让小模型估计窗宽和动态范围，最后把这段重建声音交给 <strong>NAFNet</strong> 修复 Mel 谱，再由 <strong>BigVGAN</strong> 合成。前两路固定 <code>10 ms</code> 窗宽和 <code>60 dB</code> 动态范围，后两路使用每张图各自的预测参数，Griffin–Lim 都迭代 <code>32</code> 次。</p></div><div class="upcoming-comparisons">'''
    pairs=json.loads((SITE/'assets/spectrogram-examples/comparison.json').read_text('utf-8'))['pairs']
    for pair in pairs:
        key=pair['id'];before=pair['before'].removesuffix('.wav');fast=pair['accelerated'].removesuffix('.wav');predicted=pair['predicted'].removesuffix('.wav');after=pair['after'].removesuffix('.wav')
        assert all(audio[pair[k]]['sha256']==pair[k+'_sha256'] for k in ['before','accelerated','predicted','after'])
        display=pair['display'];image=esc(display['image']);credit=esc(display['credit'])
        attribution=(f'<a href="{esc(pair["url"])}" target="_blank" rel="noopener noreferrer">{credit} ↗</a>' if pair.get('url') else credit)
        body+=f'<article class="next-example" data-case="{key}"><h4>{esc(display["title"])}</h4><figure><a href="{image}" target="_blank" rel="noopener noreferrer" aria-label="查看{credit}完整图像节选"><img src="{image}" width="720" height="580" loading="lazy" alt="{esc(pair["source"])} 语谱图原图节选"></a><figcaption>{attribution}</figcaption></figure>'
        prediction=pair['parameter_prediction']
        body+=f'<p class="next-parameters">前两路：<code>10 ms</code> Hann 窗 · <code>60 dB</code><br>模型估计：<code>{prediction["window_ms"]:.2f} ms</code> → 用 <code>{pair["applied_window_ms"]:g} ms</code><br>预测动态范围：<code>{prediction["dynamic_range_db"]:.2f} dB</code><br>随机相位起点 · <code>32</code> 次迭代</p>'
        body+=player('next',[(before,'原始 GL',f'{audio[pair["before"]]["duration"]:.2f} s · 动量 0'),(fast,'动量加速 GL',f'{audio[pair["accelerated"]]["duration"]:.2f} s · 动量 0.99'),(predicted,'预测参数 + 动量 GL',f'{audio[pair["predicted"]]["duration"]:.2f} s · 预测窗宽与动态范围'),(after,'预测参数 + 动量 GL + NAFNet + BigVGAN',f'{audio[pair["after"]]["duration"]:.2f} s · 接续第三路的同一段声音')],'next-sounds',instance='next-',volumes=pair['playback_volumes'],show_waveform=False)+'</article>'
    return body+'''</div><p class="next-figure-reference" id="next-lu-reference">Lu 两例图源：Lu, Y., Liang, C., &amp; Kong, J. (2025). <cite>Contribution of F0 and phonation to tone perception in the Zaiwa language.</cite> <i>Journal of Phonetics, 110</i>, 101413. <a href="https://doi.org/10.1016/j.wocn.2025.101413" target="_blank" rel="noopener noreferrer">查看论文 ↗</a></p><p class="audio-status" role="status" aria-live="polite">点一下，按顺序听听四种方式。</p><details class="next-comparison-settings"><summary>这组对比怎样生成？</summary><p>四路使用相同的图片、时间频率标定、Hann 窗、固定种子的随机相位、1.5 ms 帧移和 32 次迭代。前两路固定 10 ms 窗宽和 60 dB 动态范围，原始 GL 的动量为 0，第二路只把动量改为 0.99。第三路保持动量 0.99，改用小模型预测的窗宽与动态范围，窗宽四舍五入到整数毫秒。第四路直接读取第三路导出的同一段音频，接上 80 带 NAFNet 的 Mel 谱修复与 BigVGAN 声码器。</p><p>这组试听不使用青木脉冲初始化，方便分别比较动量、参数预测，以及 Mel 修复与声码器带来的变化。前两路沿用已固定的音频，播放时将四路衰减到相同均方根电平。窗宽和动态范围是模型的估计值，重建结果可以按顺序交替听听。</p></details><p class="upcoming-signoff">模型还在继续训练，先把这几组声音放在这里。<span aria-hidden="true">✧</span></p></div></section>'''

def feature_ribbon():
    # High-resolution 3D artwork is displayed as six cells of one transparent sheet.
    names = ['声学分析', '语音合成', '语音标注', '感知实验', '论文阅读', '国际音标']
    return '<div class="feature-ribbon" aria-label="主要功能">' + ''.join(
        f'<div><span class="ribbon-icon ribbon-icon-{i}" aria-hidden="true"></span><span>{name}</span></div>'
        for i, name in enumerate(names)) + '</div>'

def inline(s):
    s=esc(s)
    s=re.sub(r'\[([^\]]+)\]\((https://[^ )]+)\)',r'<a href="\2" target="_blank" rel="noopener noreferrer">\1 ↗</a>',s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
    return re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
def release_html(body):
    blocks=re.split(r'\n\s*\n',body.replace('\r','').strip());out=[]
    for block in blocks:
        if block.startswith('<details>'):out.append('<details class="release-extra" open>');block=block[len('<details>'):].strip()
        if block.startswith('<summary>'):
            title,_,block=block[len('<summary>'):].partition('</summary>');out.append('<summary>'+inline(title)+'</summary>');block=block.strip()
        if block=='</details>':out.append('</details>');continue
        if not block:continue
        if re.match(r'#{1,4} ',block):out.append('<h4>'+inline(block.lstrip('# '))+'</h4>')
        elif block.startswith('|') and '\n|' in block:
            rows=[line.strip().strip('|').split('|') for line in block.splitlines() if line.strip()]
            if len(rows)>1 and all(re.fullmatch(r'[:\s-]+',c) for c in rows[1]):
                out.append('<div class="release-table"><table><thead><tr>'+''.join('<th>'+inline(c.strip())+'</th>' for c in rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+inline(c.strip())+'</td>' for c in row)+'</tr>' for row in rows[2:])+'</tbody></table></div>')
            else:out.append('<p>'+inline(block).replace('\n','<br>')+'</p>')
        elif block.startswith('- '):out.append('<ul>'+''.join('<li>'+inline(x)+'</li>' for x in block.removeprefix('- ').split('\n- '))+'</ul>')
        else:out.append('<p>'+inline(block).replace('\n','<br>')+'</p>')
    return '\n'.join(out)
def showcase(id,title,copy,module,file,palette,mode,reverse=False):
    color='coral' if mode=='浅色' else 'violet'
    return f'''<section class="showcase split-showcase {'reverse-showcase' if reverse else ''}" id="{id}" aria-labelledby="{id}-heading"><div class="showcase-copy reveal"><span class="eyebrow">{id.upper()}</span><h2 id="{id}-heading">{title}</h2><p>{copy}</p><span class="section-label">{module}</span></div><figure class="screen-stage reveal" style="--tilt:4deg;--turn:{-7 if reverse else 7}deg"><span class="stage-backdrop {color}-backdrop"></span><a class="screen-link" href="assets/screens/{file}.png" data-screen data-caption="{module} · {palette} {mode}" aria-label="查看{module}完整截图"><img src="assets/screens/{file}.png" width="2560" height="1440" loading="lazy" alt="{module}的实际中文界面，{palette} {mode}配色"></a><figcaption><span class="palette-dot {'rose' if reverse else 'green'}"></span>{palette} <span>{mode}</span></figcaption></figure></section>'''
def main():
    verify_basic_gl()
    verify_three_way()
    p=SITE/'index.html';text=p.read_text('utf-8')
    text=re.sub(r'\s*<p class="web-release">[\s\S]*?</p>', '', text)
    text=text.replace('网页版预计于 2026 年 10 月 31 日发布。','')
    text,count=re.subn(r'<div class="feature-ribbon"[\s\S]*?</div>\s*</div>',lambda _:feature_ribbon(),text,count=1)
    assert count == 1, 'One generated feature ribbon'
    text=re.sub(r'<section class="more-features\b[\s\S]*?</section>', '', text)
    text=re.sub(r'<div class="chapter-divider"[\s\S]*?</div>\s*', '', text)
    nav=[('tutorials','使用教程','打开小电视，一起上手工具箱'),('lab','小实验','点一点，听听声音的变化'),('methods','方法与引用','找到方法背后的来源'),('faq','常见问题','下载、安装与使用的小疑问'),('updates','更新小记','这一版，添了哪些新东西'),('next','下一步…','短时窗语谱图重建，继续往前走'),('feedback','问题反馈','把遇到的问题告诉韶子'),('contact','联系我们','邮箱与 QQ 交流群')]
    text=re.sub(r'<nav aria-label="主导航">.*?</nav>','<nav aria-label="主导航">'+''.join(f'<a href="#{id}" class="nav-tip" aria-describedby="tip-{id}">{name}<span class="nav-tooltip" id="tip-{id}" role="tooltip">{tip}</span></a>' for id,name,tip in nav)+'<a class="header-github nav-tip" href="https://github.com/Nephelium/PhoneticToolbox" target="_blank" rel="noopener noreferrer" aria-describedby="tip-github"><svg aria-hidden="true"><use href="#i-github"/></svg>GitHub<span class="nav-tooltip" id="tip-github" role="tooltip">项目源码与使用说明</span></a></nav>',text)
    journey=[('top','回到开头'),('features','工具巡游'),('tutorials','使用教程'),('lab','小实验'),('methods','方法与引用'),('faq','常见问题'),('updates','更新小记'),('next','下一步…'),('feedback','问题反馈'),('contact','联系我们'),('a-small-note','韶子的小声话')]
    dock='<aside class="reading-dock" aria-label="阅读导航" hidden><div class="reading-menu" id="reading-menu" hidden><p class="reading-menu-title">翻到哪一页？<span aria-hidden="true">✦</span></p><nav aria-label="页面目录">'+''.join(f'<a href="#{key}"><span>{str(i+1).zfill(2)}</span>{label}</a>' for i,(key,label) in enumerate(journey))+'</nav></div><button type="button" class="reading-toggle" aria-expanded="false" aria-controls="reading-menu" aria-label="展开页面目录"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 6h14M5 12h14M5 18h9"/></svg><span>目录</span></button><button type="button" class="reading-top" aria-label="返回顶部"><svg class="reading-ring" viewBox="0 0 48 48" aria-hidden="true"><circle class="ring-track" cx="24" cy="24" r="21"/><circle class="ring-fill" cx="24" cy="24" r="21" pathLength="100"/></svg><span class="reading-face" aria-hidden="true">↑</span><span class="reading-top-label">顶一下</span></button></aside>'
    text=re.sub(r'<aside class="reading-dock"[\s\S]*?</aside>','',text)
    text=text.replace('</body>',dock+'\n</body>')

    if 'download-tip-portable' not in text:
        for klass,id,msg in [('primary','portable','免安装 · 下载后直接运行。约 496.8 MB，适合随身携带或先体验。'),('secondary','installer','安装版 · 按向导安装并创建快捷方式。约 498.7 MB，适合长期使用。')]:
            text=re.sub(r'(<a class="download '+klass+r'"[^>]*)(>)(.*?)(</a>)',lambda m:m[1].replace('aria-describedby="download-state"',f'aria-describedby="download-state download-tip-{id}"')+m[2]+m[3]+f'<span class="download-tooltip" id="download-tip-{id}" role="tooltip">{msg}<small>Windows 10 1809+ / 11 · x64</small></span>'+m[4],text)
    download=json.loads((SITE/'assets/downloads.json').read_text('utf-8'))
    for klass,kind,name in [('primary','portable','PhoneticToolbox-windows-x64.exe'),('secondary','installer',f'PhoneticToolbox-{download["version"]}-windows-x64-setup.exe')]:
        href=f'/releases/windows-x64/{download["version"]}/{name}'
        text,count=re.subn(r'(<a class="download '+klass+r'" href=")[^"]+',lambda m:m[1]+href,text,count=1)
        assert count==1
        size=f'{download["files"][name]["size"]/1_000_000:.1f}'
        text=re.sub(r'(<span class="download-tooltip" id="download-tip-'+kind+r'"[\s\S]*?约 )[0-9.]+ MB',lambda m:m[1]+size+' MB',text,count=1)
    text=re.sub(r'(<p class="download-note" id="download-state">[\s\S]*?<span>)[^<]+',lambda m:m[1]+download['version'],text,count=1)
    text=re.sub(r'\s*<(button|a)\b[^>]*class="web-entry"[\s\S]*?</\1>', '', text)
    web_entry='<button class="web-entry" type="button" disabled aria-label="网页版，灰测中，即将开放"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="M3 9h18M7 6.5h.01M10 6.5h.01"/></svg><span>网页版</span><small>灰测中，即将开放</small></button>'
    text=text.replace('<div class="download-actions"',web_entry+'\n   <div class="download-actions"',1)
    windows_links=''.join(re.search(r'<a class="download '+klass+r'"[\s\S]*?</a>',text)[0] for klass in ['primary','secondary'])
    windows_links=windows_links.replace('>下载免安装版<','>免安装版<').replace('>下载安装版<','>安装版<')
    mac=download.get('macos')
    if mac:
        assert mac['platform']=='macos-arm64' and mac['channel']=='preview'
        mac_button='<a class="download macos" href="'+esc(mac['url'])+'" download="'+esc(mac['name'])+'" aria-label="下载 macOS 测试版" aria-describedby="download-tip-macos">macOS<svg aria-hidden="true"><use href="#i-download"/></svg></a>'
        mac_note='macOS 14+<br>Apple 芯片 · 测试版'
        mac_help='<details class="macos-help"><summary>安装说明</summary><div class="macos-install-popover" id="download-tip-macos" role="region" aria-label="macOS 安装说明"><strong>PhoneticToolbox '+esc(mac['version'])+' macOS 测试版</strong><p>适用于 Apple 芯片（M 系列）、macOS 14 及以上。</p><ol><li>下载并解压 ZIP 文件。</li><li>将 <code>PhoneticToolbox.app</code> 拖入“应用程序”文件夹。</li><li>双击打开。</li></ol><p>当前测试版尚未通过 Apple 公证。首次打开若提示无法验证开发者，确认安装包来自本官网后，可进入 <b>系统设置 → 隐私与安全性 → 仍要打开</b>。</p><a href="https://support.apple.com/zh-cn/102445" target="_blank" rel="noopener">Apple 官方说明 ↗</a><footer>文件大小：'+f'{mac["size"]/1_000_000:.2f}'+' MB<br>SHA256：<code>'+esc(mac['sha256'])+'</code></footer></div></details>'
    else:
        mac_button='<button class="download macos" type="button" disabled aria-label="下载 macOS，尚未上线">macOS<svg aria-hidden="true"><use href="#i-download"/></svg></button>'
        mac_note='即将上线'
        mac_help=''
    download_actions='<div class="download-actions"><div class="platform-windows" role="group" aria-label="Windows 下载"><div class="platform-buttons">'+windows_links+'</div><p class="download-note" id="download-state">Windows 10 / 11 · x64 <span>'+download['version']+'</span></p></div><span class="download-divider" aria-hidden="true"></span><div class="platform-macos" id="macos-download" role="group" aria-label="macOS 下载">'+mac_button+'<p class="download-note">'+mac_note+'</p>'+mac_help+'</div></div>'
    text,count=re.subn(r'<div class="download-actions"[\s\S]*?(?=\n  </div>\n  <div class="hero-art")',lambda _:download_actions,text,count=1)
    assert count==1, 'One platform download group'
    if 'id="phonation"' not in text:
        appearance=re.search(r' <section class="showcase appearance-showcase"[\s\S]*?</section>',text).group()
        text=text.replace(appearance,showcase('phonation','一点点变化，<br><em>慢慢听出来。</em>','在两个声音之间制作连续统，分别控制发声类型与基频。','发声类型合成','phonation-gruvbox-dark','Gruvbox','深色'))
        new=showcase('alignment','从转写，<br><em>走向时间轴。</em>','把音频与已有转写对齐，生成 TextGrid，接着人工核对边界。','MFA 自动标注','mfa-monokai-dark','Monokai','深色')
        new+=showcase('annotation','每一个边界，<br><em>都可以细看。</em>','对照波形和语谱图，调整区间、编辑标签，让标注回到声音里。','TextGrid 标注','textgrid-material-light','Material','浅色',True)
        new+=showcase('lpc','在频谱里，<br><em>看见包络。</em>','选取短时片段，查看 LPC 谱包络，按标注区间分析并导出结果。','LPC 谱图','lpc-night-owl-dark','Night Owl','深色')
        appearance=appearance.replace('appearance-tokyo-night-dark','appearance-tokyo-night-light').replace('深色','浅色')
        text=text.replace(' <section class="showcase split-showcase" id="icons"',new+'\n'+appearance+'\n <section class="showcase split-showcase" id="icons"')
        text=re.sub(r'<div class="tool-divider[\s\S]*?</div>\s*','',text)
        n=[0]
        def div(m):
            n[0]+=1
            return m[0]+('' if n[0]==15 else f'<div class="tool-divider divider-{n[0]%3}" aria-hidden="true"><svg viewBox="0 0 330 50"><path d="M5 26Q30 8 54 25T103 25T152 25M188 25Q211 8 237 25T286 25T325 25"/><path class="divider-star" d="m170 8 5 12 13 5-13 5-5 12-5-12-13-5 13-5Z"/></svg><i></i><i></i></div>')
        text=re.sub(r'<section class="showcase\b[\s\S]*?</section>',div,text)
    if 'id="phonology"' not in text:
        section=showcase('phonology','从字音，<br><em>看见音系。</em>','整理调查字表，调整调类与顺序，查看声韵分组和二维表格。','音系归纳','phonology-matrix-gruvbox-light','Gruvbox','浅色',True)
        cards=[]
        for i,(file,label) in enumerate([('tones','声调排序归并'),('initials','声母到韵母'),('matrix','二维声韵表')]):
            src=f'assets/screens/phonology-{file}-gruvbox-light.png'
            caption=f'音系归纳 · {label}（声调仅为举例，并非真实调类）· Gruvbox 浅色'
            cards.append(f'<a class="screen-link stack-card" style="--card:{i};--z:{i+1}" href="{src}" data-screen data-caption="{caption}" aria-label="查看{label}完整截图"><img src="{src}" width="2560" height="1440" loading="lazy" alt="{caption}"></a>')
        figure='<figure class="screen-stage screen-stack stack-3 reveal" id="phonology-stack" style="--turn:-6deg"><span class="stage-backdrop coral-backdrop"></span><div class="stack-deck">'+''.join(cards)+'</div><figcaption><span class="palette-dot green"></span>Gruvbox <span>浅色</span></figcaption></figure>'
        section=re.sub(r'<figure[\s\S]*?</figure>',lambda _:figure,section)
        section=section.replace('<span class="section-label">音系归纳</span>','<p class="example-note">（声调仅为举例，并非真实调类）</p><span class="section-label">音系归纳</span>')
        text=text.replace(' <section class="showcase appearance-showcase"',section+'\n <section class="showcase appearance-showcase"',1)
    text=text.replace('从一个个字音，<br><em>理出声韵关系。</em>','从字音，<br><em>看见音系。</em>')
    for old,new in [('appearance-tokyo-night-light','appearance-tokyo-night-dark'),('icons-dracula-dark','icons-dracula-light'),('manual-temple-light','manual-temple-dark')]:
        pattern=r'<section class="showcase\b(?:(?!</section>)[\s\S])*?'+re.escape(old)+r'[\s\S]*?</section>'
        def recolor(m):
            light=old.endswith('-light')
            return m[0].replace(old,new).replace('浅色' if light else '深色','深色' if light else '浅色')
        text=re.sub(pattern,recolor,text)
    # Replace the retired palette overview with the user-selected background examples.
    text=re.sub(r'<section class="showcase appearance-showcase"[\s\S]*?</section>', '', text)
    text=re.sub(r'<section class="showcase\b[^>]*id="background"[\s\S]*?</section>', '', text)
    cards=[]
    # Back to front: the user requested image 2, 1, 3, 4 from front to back.
    for i,(file,label) in enumerate([('custom-background-experiment-dark','自定义背景 · 感知实验'),('custom-background-ipa-dark','自定义背景 · 国际音标表Plus'),('custom-background-settings-dark','自定义背景 · 外观设置'),('custom-background-home-dark','自定义背景 · 首页展示')]):
        src=f'assets/screens/{file}.png'
        cards.append(f'<a class="screen-link stack-card" style="--card:{i};--z:{i+1}" href="{src}" data-screen data-caption="{label} · 深色" aria-label="查看{label}完整截图"><img src="{src}" width="2559" height="1527" loading="lazy" alt="{label}的实际应用截图，背景图与界面共同显示"></a>')
    background='<section class="showcase split-showcase" id="background" aria-labelledby="background-heading"><div class="showcase-copy reveal"><span class="eyebrow">YOUR OWN BACKDROP</span><h2 id="background-heading">换张背景，<br><em>换个心情。</em></h2><p>选一张喜欢的背景图，调整模糊与不透明度，让工作台换个模样。</p><span class="section-label">自定义背景图</span></div><figure class="screen-stage screen-stack stack-4 reveal" id="background-stack" style="--turn:7deg"><span class="stage-backdrop violet-backdrop"></span><div class="stack-deck">'+''.join(cards)+'</div><figcaption><span class="palette-dot blue"></span>自定义配色 <span>自定义背景</span></figcaption></figure></section>'
    text,count=re.subn(r'(<section class="showcase\b[^>]*id="guide"[\s\S]*?</section>)',lambda m:m[0]+'\n'+background,text,count=1)
    assert count==1, 'Background showcase follows the manual, before the listening lab'
    text=re.sub(r'<div class="tool-divider[\s\S]*?</div>\s*','',text)
    sections=len(re.findall(r'<section class="showcase\b',text));n=[0]
    def divider(m):
        n[0]+=1
        return m[0]+('' if n[0]==sections else f'<div class="tool-divider divider-{n[0]%3}" aria-hidden="true"><svg viewBox="0 0 330 50"><path d="M5 26Q30 8 54 25T103 25T152 25M188 25Q211 8 237 25T286 25T325 25"/><path class="divider-star" d="m170 8 5 12 13 5-13 5-5 12-5-12-13-5 13-5Z"/></svg><i></i><i></i></div>')
    text=re.sub(r'<section class="showcase\b[\s\S]*?</section>',divider,text)
    text=re.sub(r'<a class="article-button"[\s\S]*?</a>', '', text)
    article_button='<a class="article-button" href="https://www.zhihu.com/question/793458414/answer/4607998600" target="_blank" rel="noopener noreferrer">读读韶子的知乎文章<span aria-hidden="true">↗</span></a>'
    text,count=re.subn(r'(<section[^>]+id="spectrum"[\s\S]*?<div class="showcase-copy[^>]*>[\s\S]*?)(</div>)',lambda m:m[1]+article_button+m[2],text,count=1)
    assert count==1
    audio=json.loads((SITE/'assets/audio/provenance.json').read_text('utf-8'));audio={r['file']:r for r in audio['files']}
    endpoint_f0=[round(audio[f'phonation-endpoint-{i}.wav']['median_f0']) for i in (1,2)]
    step_count=sum(bool(re.fullmatch(r'phonation-\d+\.wav',name)) for name in audio)
    assert step_count>=2
    step_word={7:'七',9:'九'}.get(step_count,str(step_count))
    step_f0=[round(audio[f'phonation-{i}.wav']['median_f0']) for i in range(1,step_count+1)]
    analysis=audio['phonation-1.wav']['parameters']['analysis']
    backend_name={'parselmouth':'Praat / Parselmouth','reaper':'REAPER'}[analysis['f0_backend']]
    phonation_recipe=f'这组{step_word}步使用应用导出的音频，同时改变发声类型与基频。基频分析采用 {backend_name}，范围 {analysis["min_f0_hz"]:g}–{analysis["max_f0_hz"]:g} Hz，开启周期能量与输出幅度匹配。两段原声中位 F0 分别约为 {endpoint_f0[0]} Hz 和 {endpoint_f0[1]} Hz，处理后各步中位 F0 依次约为 {"、".join(str(f) for f in step_f0)} Hz。'
    def player(prefix,choices,klass,instance='',volumes=None,show_waveform=True):
        result='<div class="sound-choices '+klass+'"'+(f' style="--continuum-steps:{step_count}"' if klass=='continuum' else '')+'>'
        for name,label,detail in choices:
            row=audio[name+'.wav'];bars=''.join(f'<i style="--bar:{max(8,round(v/max(row["waveform"])*100))}%"></i>' for v in row['waveform'][::2])
            level=f' data-playback-volume="{volumes[name]:.8f}"' if volumes else ''
            if volumes:assert 0<volumes[name]<=1
            waveform=f'<div class="mini-wave" aria-hidden="true">{bars}</div>' if show_waveform else ''
            result+=f'<div class="sound-option"><button type="button" class="sound-play" data-audio="audio-{instance}{name}" aria-label="播放{label}" aria-pressed="false"><span class="sound-symbol" aria-hidden="true">▶</span><span>{label}</span></button><span class="sound-detail">{detail}</span>{waveform}<audio id="audio-{instance}{name}" src="assets/audio/{name}.wav{("?v="+row["sha256"][:12]) if name.startswith("spectrum-") else ""}" preload="none" controls aria-label="{label}"{level}></audio></div>'
        return result+'</div>'
    lab='''<section class="content-section listening-lab" id="lab" aria-labelledby="lab-heading"><div class="section-heading reveal"><span class="eyebrow">A LITTLE SOUND PLAYGROUND</span><h2 id="lab-heading">先别急着走，<br><em>听几个小实验。</em></h2><p>把抽象的参数，交给耳朵感受。选一个声音，点一下播放。</p></div><div class="lab-grid">
    <article class="lab-card continuum-card reveal"><span class="paper-tab">01 / 从这里，到那里</span><div class="lab-title"><h3>声音，也有渐变色。</h3><span class="lab-doodle" aria-hidden="true">〰</span></div><p>同一个元音，一段 SCSC 录音和一段自有音频，基频不同。先听两个原声，再听由它们制作的七步发声类型与基频连续统。</p><div class="original-pair"><span class="original-caption">先听听，<br>两个原声。</span>'''+player('endpoints',[('phonation-endpoint-1','原声 A',f'中位 F0 ≈ {endpoint_f0[0]} Hz'),('phonation-endpoint-2','原声 B',f'中位 F0 ≈ {endpoint_f0[1]} Hz')],'endpoint-choices')+'''</div><p class="audio-origin">原声 A：SCSC 录音 · 原声 B：自有音频</p><div class="continuum-labels"><span>发声类型与基频更接近 A</span><span>一点点靠近 →</span><span>发声类型与基频更接近 B</span></div>'''+player('phonation',[(f'phonation-{i}',f'第 {i} 步',f'{(i-1)/(step_count-1)*100:g}% · {step_f0[i-1]} Hz') for i in range(1,step_count+1)],'continuum')+'''<details class="lab-recipe"><summary>这组声音是怎样做的？</summary><p>'''+phonation_recipe+'''</p><p>七步表示处理比例，听觉上的差别未必等距。第 1、7 步也是处理结果，可以与上面的两段原声对照。</p></details></article>
    <article class="lab-card pitch-card reveal"><span class="paper-tab">02 / 给声音换个步调</span><div class="lab-title"><h3>高一点，慢一点？</h3><span class="lab-doodle" aria-hidden="true">↗</span></div><p>从一段真人 /ma/ 录音出发，分别改变音高或语速，再和原声比较。</p>'''+player('pitch',[('pitch-original','原声','真实录音 · 0.58 s'),('pitch-lower','低一点','F0 −20 Hz'),('pitch-higher','高一点','F0 +20 Hz'),('pitch-slower','慢一点','0.75 倍 · 0.77 s'),('pitch-faster','快一点','1.25 倍 · 0.46 s')],'pitch-choices')+'''<p class="lab-footnote">原声中位 F0 约为 129 Hz。改变语速时保留原基频轮廓，其余示例由应用处理这段真实录音得到。</p></article>
    <article class="lab-card vowel-card reveal"><span class="paper-tab">03 / 声道变了，声音呢</span><div class="lab-title"><h3>让元音排排坐。</h3><span class="lab-doodle" aria-hidden="true">✧</span></div><p>相同的 160 Hz 声源，换三种声道构形。舌位和唇形的变化，会留下怎样的声音？</p>'''+player('vowel',[(f'vowel-{v}',f'/{v}/',desc) for v,desc in [('a','张开一点'),('i','舌位向前'),('u','圆唇向后')]],'vowel-choices')+'''<p class="lab-footnote">生理参数合成 · JD2 声道模型 · 每段 1.20 秒。示例由模型合成。</p></article></div><p class="audio-status" role="status" aria-live="polite">声音从点击开始。可以随时暂停，再选另一个。</p></section>'''
    lab=lab.replace('七步',step_word+'步').replace('第 1、7 步',f'第 1、{step_count} 步')
    labels=[audio[f'phonation-endpoint-{i}.wav']['parameters']['input']['source_label'] for i in (1,2)]
    lab=lab.replace('一段 SCSC 录音和一段自有音频','两段 SCSC 录音')
    lab=lab.replace('原声 A：SCSC 录音 · 原声 B：自有音频',f'原声 A：{labels[0]} · 原声 B：{labels[1]}')
    examples=json.loads((SITE/'assets/spectrogram-examples/provenance.json').read_text('utf-8'))['cases']
    spectrogram='<article class="lab-card reconstruction-card reveal"><span class="paper-tab">04 / 让图片出声</span><div class="lab-title"><h3>看得见，也能听一听。</h3><span class="lab-doodle" aria-hidden="true">♪</span></div><p>只剩一张语谱图，还能找回多少声音？选取图中的频谱区域，按坐标校正，再经过 GL、训练后的 80 带 Mel 修复与声码器重建。下面三段，都是从图片得到的近似声音。</p><div class="reconstruction-examples">'
    for i,example in enumerate(examples):
        key=example['id'];source=esc(example['source'])
        credit={'smith':'Smith (2007) · 图 8.10','levantan':'Chappell (2021) · 图 7.4','observo':'Chappell (2021) · 图 7.3'}[key]
        if example.get('processing'):
            processing=example['processing'];assumptions=processing['assumptions'];gl=processing['gl']
            settings=f"GL → 训练后 80 带 Mel 修复 → BigVGAN 声码器。脉冲初始化 GL · {gl['iterations']} 次迭代 · {assumptions['window_ms']:g} ms {assumptions['window']} 窗 · {assumptions['dynamic_range_db']:g} dB 幅度灰度映射 · 试听输出 {audio[f'spectrum-{key}.wav']['sample_rate']//1000} kHz。窗长与动态范围为重建假设。"
            if processing.get('parameter_prediction'):
                predicted=processing['parameter_prediction']
                settings=f"参数小模型 → GL → 训练后 80 带 Mel 修复 → BigVGAN。自动预估窗长 {predicted['window_ms']:.2f} ms（计算采用 {gl['applied_window_ms']:g} ms）、动态范围 {predicted['dynamic_range_db']:.2f} dB，规范化下限 −{predicted['dynamic_range_db']:.2f} dB、上限 0 dB。固定 Hann 窗、{gl['iterations']} 次脉冲初始化 GL；输出 24 kHz。原图参数没有真值。"
        else:
            settings=f"Griffin–Lim · {example['config']['n_iter']} 次迭代 · 重建窗长 {example['config']['win_length_ms']:g} ms · 输出 {example['config']['target_sr']//1000} kHz。灰度按 {example['config']['min_db']:g}–{example['config']['max_db']:g} dB 映射，固定初始随机种子。"
        spectrogram+=f'<section class="reconstruction-example" aria-labelledby="reconstruction-{key}"><span class="example-index">0{i+1}</span><h4 id="reconstruction-{key}">{esc(example["title"])}</h4><figure><a href="assets/spectrogram-examples/{key}.png" target="_blank" rel="noopener noreferrer" aria-label="查看{credit}完整图像节选"><img src="assets/spectrogram-examples/{key}.png" width="720" height="580" loading="lazy" alt="{source} 语谱图节选，保留坐标用于校正"></a><figcaption>原图节选 · 0–{example["duration"]} s / 0–{example["frequency"]//1000} kHz</figcaption></figure><a class="example-credit" href="{esc(example["url"])}" target="_blank" rel="noopener noreferrer">{credit} ↗</a><p>{esc(example["explanation"])}</p>'+player('spectrum',[(f'spectrum-{key}','听听重建结果',f'{audio[f"spectrum-{key}.wav"]["duration"]:.2f} s · 图片重建')],'reconstruction-player')+f'<details class="example-source"><summary>图源与重建设置 ↗</summary><p>{source}</p><a href="{esc(example["url"])}" target="_blank" rel="noopener noreferrer">查看原文 ↗</a><p>{esc(settings)}</p></details></section>'
    spectrogram+='</div><p class="lab-footnote">图片记录了能量分布，原始相位已经丢失。这里演示近似重建，声音会受截图分辨率、灰度映射与迭代设置影响，不能代替原始录音作语音测量。图像版权归原作者及出版方，出处见各例下方。</p></article>'
    lab=lab.replace('</div><p class="audio-status"',spectrogram+'</div><p class="audio-status"')
    faq=[('下载免安装版，还是安装版？','两者提供相同工具。免安装版下载后直接运行，安装版会引导安装并创建快捷方式。当前提供 Windows 10 1809 及以上版本、Windows 11 的 x64 版本。'),('网页版什么时候可以用？','网页版正在内测。现在可以浏览功能、试听小实验和下载 Windows 应用。'),('录音和研究材料会自动上传吗？','桌面版在本机处理材料，不会自动上传本地文件。发送问题反馈、下载组件或访问论文资源时会使用网络；分享附件前，可以先检查其中的个人信息。'),('MFA 自动标注为什么还要下载组件？','MFA 的运行环境、声学模型和发音词典是独立组件。首次使用时，在 MFA 自动标注中打开“组件安装与环境检查”，按提示下载安装，或选择已下载的组件包。'),('自动标注后，还需要人工检查吗？','需要。MFA 将声音与已有文本对齐，自动边界仍需核对。可以在 TextGrid 标注中对照波形与语谱图继续调整。'),('在哪里查看每个工具的操作和参数？','应用左下角有完整使用说明，每个工具右上角也有“帮助”和“方法与引用”。先找对应章节，再按自己的材料设置参数。')]
    support='<section class="content-section faq-section" id="faq" aria-labelledby="faq-heading"><div class="support-heading reveal"><span class="eyebrow">A FEW SMALL QUESTIONS</span><h2 id="faq-heading">有些小疑问，<br><em>先在这里碰个头。</em></h2><div class="question-doodle" aria-hidden="true">?</div></div><div class="faq-list reveal">'+''.join(f'<details><summary><span class="question-number">0{i}</span>{q}<span class="faq-plus" aria-hidden="true">+</span></summary><p>{a}</p></details>' for i,(q,a) in enumerate(faq,1))+'</div></section>'
    # Version snapshots are curated here; there is no automatic release fetching.
    releases=[json.loads((SITE/'assets/release.json').read_text('utf-8'))]
    releases.extend(json.loads(p.read_text('utf-8')) for p in sorted((SITE/'assets/releases').glob('*.json')))
    releases=sorted({r['tag_name']:r for r in releases}.values(),key=lambda r:tuple(int(n) for n in r['tag_name'].removeprefix('v').split('.')),reverse=True)
    support+='<section class="content-section updates-section" id="updates" aria-labelledby="updates-heading"><div class="section-heading reveal"><span class="eyebrow">NOTES FROM THE WORKBENCH</span><h2 id="updates-heading">工具箱的，<em>更新小记。</em></h2></div><div class="release-selector reveal"><span>翻到这一版</span><div class="release-tabs" role="tablist" aria-label="更新小记版本">'
    for i,r in enumerate(releases):
        support+=f'<button type="button" role="tab" id="release-tab-{i}" data-release-tab="release-panel-{i}" aria-controls="release-panel-{i}" aria-selected="{str(i==0).lower()}" tabindex="{0 if i==0 else -1}">{esc(r["tag_name"])}</button>'
    support+='</div></div>'
    for i,r in enumerate(releases):
        original_link=f'<a class="text-link" href="{esc(r["html_url"])}" target="_blank" rel="noopener noreferrer">在 GitHub 阅读原文 ↗</a>' if r.get("html_url") else ""
        support+=f'<article class="release-paper reveal" role="tabpanel" id="release-panel-{i}" aria-labelledby="release-tab-{i}" {"hidden" if i else ""}><div class="release-stamp"><span>{esc(r["tag_name"])}</span><time datetime="{esc(r["published_at"])}">{esc(r["published_at"][:10].replace("-","."))}</time><i aria-hidden="true">✦</i></div><div class="release-body"><h3>{esc(r["name"])}</h3><div class="release-content">{release_html(r["body"])}</div>{original_link}</div></article>'
    support+='</section>'

    sources=json.loads((ROOT/'third_party/source-registry.json').read_text('utf-8'))['sources'];sources={s['id']:s for s in sources}
    methods=[('SRC-PRAAT','分析与操控','Praat / Parselmouth','基频分析与语音操控等功能使用 Praat 及其 Python 接口。'),('REF-ZAIWA','发声类型合成','F0 与发声类型连续统','连续统方法参考 Lu、Liang 与 Kong 的研究，代码改写另获作者许可。'),('REF-MFA','自动标注','Montreal Forced Aligner','将已有转写与音频对齐，再在 TextGrid 工具中人工复核。')]
    support+='<section class="content-section methods-section" id="methods" aria-labelledby="methods-heading"><div class="section-heading reveal"><span class="eyebrow">FOLLOW THE IDEAS</span><h2 id="methods-heading">每一种方法，<br><em>都有来处。</em></h2><p>从几种常用方法开始，也可以展开下方列表查看全部论文与代码来源。</p></div><div class="reference-grid">'
    for i,(id,tag,title,desc) in enumerate(methods):
        row=sources[id];citation=row.get('citation',title);url=row['urls'].get('doi',row['urls'].get('paper',row['urls'].get('homepage')))
        support+=f'<article class="reference-card reveal"><span class="section-label">{tag}</span><h3>{title}</h3><p>{desc}</p><details><summary>查看引用</summary><p class="citation">{esc(citation)}</p></details><a class="text-link" href="{url}" target="_blank" rel="noopener noreferrer">查看来源 ↗</a></article>'
    support+='''<article class="reference-card reveal"><span class="section-label">生理参数合成</span><h3>VocalTractLab</h3><p>基于声道几何与声源控制生成声音。示例采用 JD2 模型与应用中的声道预设。</p><a class="text-link" href="https://www.vocaltractlab.de/download-vocaltractlab/VTL2.4-manual.pdf" target="_blank" rel="noopener noreferrer">查看官方手册 ↗</a></article></div>'''+reference_catalog()+'''</section>
    <section class="content-section feedback-section" id="feedback" aria-labelledby="feedback-heading">
      <div class="support-heading reveal"><span class="eyebrow">LEAVE A LITTLE NOTE</span><h2 id="feedback-heading">哪里卡住了？<br><em>一起把它理顺。</em></h2><p>问题、建议，或者一个新想法，都可以。留下邮箱，把这张便签直接寄给韶子。</p><div class="feedback-route"><span>01　验证邮箱</span><span>02　写下问题</span><span>03　等一封回信</span></div><p class="feedback-small">无需注册。网页和应用共用反馈会话，回复也可以在这里查看。</p><span class="feedback-doodle" aria-hidden="true">✎</span></div>
      <div class="feedback-paper reveal" id="feedback-paper">
        <div id="feedback-auth"><label for="feedback-email">回信寄到哪个邮箱？</label><div class="feedback-inline"><input id="feedback-email" type="email" autocomplete="email" maxlength="254" placeholder="你的邮箱地址" required><button type="button" class="note-button" id="feedback-code-send">发送验证码</button></div><div id="feedback-code-row" hidden><label for="feedback-code">邮箱里的 6 位验证码</label><div class="feedback-inline"><input id="feedback-code" inputmode="numeric" autocomplete="one-time-code" maxlength="6" pattern="[0-9]{6}" placeholder="6 位数字"><button type="button" class="note-button" id="feedback-code-verify">验证邮箱</button></div><p class="feedback-small">验证码 10 分钟内有效，请留意垃圾邮件文件夹。</p><button type="button" class="quiet-button" id="feedback-email-change">更换邮箱地址</button></div></div>
        <div id="feedback-identity" hidden><span><strong id="feedback-email-label"></strong><small>邮箱已验证</small></span><button type="button" class="note-button" id="feedback-switch">切换邮箱</button></div>
        <label for="feedback-module">在哪个工具里？</label><input id="feedback-module" maxlength="80" placeholder="例如：EGG 信号分析（选填）">
        <label for="feedback-text">发生了什么？</label><textarea id="feedback-text" rows="4" maxlength="10000" placeholder="操作步骤、预期结果、实际结果……"></textarea>
        <div class="feedback-attachments"><label for="feedback-images">附几张截图（选填）</label><input id="feedback-images" type="file" accept="image/png,image/jpeg,image/webp" multiple><small>最多 4 张，每张不超过 2 MB。</small><button type="button" class="quiet-button" id="feedback-images-clear" hidden>移除图片</button><div id="feedback-previews"></div></div>
        <label class="feedback-mail-choice"><input id="feedback-mail" type="checkbox" checked>收到回复时，用邮件提醒</label>
        <p class="feedback-privacy">服务器保存你填写的邮箱、反馈和所选图片。本页不自动上传录音或其他文件。</p><button class="pill-link" id="feedback-submit" type="button" disabled>把便签寄出去 <span aria-hidden="true">↗</span></button>
        <p id="feedback-status" class="feedback-status" role="status" aria-live="polite">先验证邮箱，就可以发送，也能找回之前的会话。</p>
        <details id="feedback-history" hidden><summary>我们的来信往返</summary><div class="feedback-history-tools"><button type="button" class="note-button" id="feedback-refresh">查看最新回复</button></div><div id="feedback-messages" role="log" aria-live="polite"></div><button type="button" class="note-button" id="feedback-more" hidden>继续查看</button></details>
        <noscript><p>启用 JavaScript 后可在这里反馈，或通过下方邮箱联系韶子。</p></noscript>
      </div>
    </section>
    <section class="content-section contact-section reveal" id="contact" aria-labelledby="contact-heading"><div class="contact-intro"><span class="eyebrow">KEEP IN TOUCH</span><h2 id="contact-heading">留一扇，<br><em>小小的门。</em></h2><p>一封信，或一场相遇。<br>选个顺手的方式，来聊聊吧。</p><span class="contact-trail" aria-hidden="true">〰 ✦</span></div><div class="contact-cards">
      <a class="contact-card email-card" href="mailto:jinglw3@mail2.sysu.edu.cn"><span class="contact-card-top"><span class="contact-icon" aria-hidden="true">✉</span><small>TO / 井韶子</small><span class="contact-arrow" aria-hidden="true">↗</span></span><h3>寄一封，慢慢说。</h3><p>想细聊的问题、建议，<br>都可以放进信里。</p><strong>jinglw3@mail2.sysu.edu.cn</strong><span class="contact-action">写邮件给韶子 ↗</span></a>
      <button type="button" class="contact-card qq-card" data-copy="698116028"><span class="contact-card-top"><span class="contact-icon" aria-hidden="true">♧</span><small>QQ / 交流群</small><span class="contact-arrow" aria-hidden="true">＋</span></span><h3>一起，碰个头。</h3><p>聊聊工具的用法，<br>也交换一点新想法。</p><strong>698116028</strong><span class="contact-action">点一下，复制群号</span></button>
      <p class="copy-status" role="status" aria-live="polite"></p></div></section>'''
    text=re.sub(r'<section class="guide-section\b[\s\S]*?</section>', '', text)
    text=re.sub(r'<!-- WEBSITE EXPANSION START -->[\s\S]*?<!-- WEBSITE EXPANSION END -->','',text)
    methods=re.search(r'<section class="content-section methods-section"[\s\S]*?</section>',support)[0]
    support=support.replace(methods,'',1)
    support=support.replace('<section class="content-section feedback-section"',upcoming(player,audio,examples)+'\n<section class="content-section feedback-section"',1)
    text=text.replace('<section class="author-note', '\n<!-- WEBSITE EXPANSION START -->\n'+tutorials()+'\n'+lab+'\n'+methods+'\n'+support+'\n<!-- WEBSITE EXPANSION END -->\n<section class="author-note',1)
    # Assign each chapter an approved motif; smaller tool dividers retain their stars.
    def chapter_boundary(match):
        style = CHAPTER_STYLES.get(match[1])
        if style is None:
            return match[0]
        return f'<div class="chapter-divider" data-style="{style}" aria-hidden="true">{CHAPTER_DIVIDERS[style]}</div>\n'+match[0]
    text = re.sub(r'<section\b[^>]*\bid="([^\"]+)"[^>]*>', chapter_boundary, text)
    # The optional design comparison uses the same SVG definitions as the homepage.
    preview_path = SITE/'divider-preview.html'
    if preview_path.exists():
        preview = preview_path.read_text('utf-8')
        for style, svg in CHAPTER_DIVIDERS.items():
            pattern = r'(<section class="preview-option" id="'+style+r'"[\s\S]*?<div class="chapter-divider"[^>]*>).*?(</div>)'
            preview = re.sub(pattern, lambda m: m[1]+svg+m[2], preview)
        preview_path.write_text(preview, 'utf-8')
    cards = ''.join(f'<article class="stats-chart" data-chart="{key}"><div class="stats-chart-heading"><h3>{title}</h3><span>{unit}</span></div><svg viewBox="0 0 480 180" role="img" aria-label="{title}历史曲线"></svg><p class="chart-empty">等待真实统计记录</p><div class="chart-dates"><span></span><span></span></div><label class="chart-scrub">查看记录<input type="range" min="0" max="0" value="0" aria-label="{title}采样时间"><output>暂无记录</output></label></article>' for key,title,unit in [('views','页面浏览','累计 / 次'),('downloads','下载发起','累计 / 次'),('web','网页版使用','累计 / 次'),('stars','GitHub Star','数量')])
    cards=cards.replace('aria-label="下载发起历史曲线"','aria-label="Windows 与 macOS 下载发起历史曲线"')
    mac_legend='<span class="legend-macos">macOS</span>'
    cards=cards.replace('<h3>下载发起</h3><span>累计 / 次</span></div>','<h3>下载发起</h3><span>累计 / 次</span></div><div class="download-platform-legend" aria-label="下载平台"><span class="legend-windows">Windows</span>'+mac_legend+'</div>')
    stats = '<section class="site-stats" aria-label="网站统计"><div class="stats-totals"><span class="stats-spark" aria-hidden="true">✦</span><div class="stat-number"><span>页面浏览</span><strong data-stat="views">—</strong><small>次</small></div><div class="stat-number"><span>下载发起</span><strong data-stat="downloads">—</strong><small>次</small></div><div class="stat-number"><span>网页版使用</span><strong data-stat="web">—</strong><small>次</small></div><div class="stat-number"><span>GitHub Star</span><strong data-stat="stars">—</strong><small>个</small></div></div><div class="stats-history-heading"><div><span class="eyebrow">LITTLE BY LITTLE</span><h2>工具箱的日常</h2><p data-stats-since>统计正在读取中…</p></div><label>查看时段<select data-stats-range aria-label="统计曲线时段"><option value="1">最近24小时</option><option value="7">最近7天</option><option value="30" selected>最近30天</option><option value="all">全部记录</option></select></label></div><div class="stats-charts">'+cards+'</div><div class="stats-caption"><p data-stats-updated>等待更新</p><details><summary>数字的小注脚</summary><p>页面浏览按服务器收到的页面请求统计，排除可识别的常见爬虫。下载总数为 Windows 与 macOS 合计，曲线分别展示两平台。整文件或首个分块请求计数，续传分块不重复，重新发起仍计数，数字不代表安装人数。网页版使用按成功新建临时工作区计数，同一工作区刷新或多开标签不重复。前三项每分钟更新，GitHub Star约每5分钟采样，可能增加或减少。历史使用服务器日志和实际采样，缺失时段保留空缺。</p></details><noscript>启用 JavaScript 后可查看最新统计与历史曲线。</noscript></div></section>'
    text = re.sub(r'<section class="site-stats"[\s\S]*?</section>', '', text)
    text = text.replace('<footer class="site-footer">', stats+'\n<footer class="site-footer">')
    # Retain the top GitHub entry and keep the second entry in the footer only.
    text = re.sub(r'<a class="footer-github"[\s\S]*?</a>', '', text)
    footer_link = '<a class="footer-github" href="https://github.com/Nephelium/PhoneticToolbox" target="_blank" rel="noopener noreferrer" aria-label="在 GitHub 查看使用说明与源码"><svg aria-hidden="true"><use href="#i-github"/></svg>GitHub <span aria-hidden="true">↗</span></a>'
    text = re.sub(r'(<footer class="site-footer">[\s\S]*?<div>)', lambda m: m[1]+footer_link, text, count=1)

    text=text.replace('site.css?v='+re.search(r'site.css\?v=([a-f0-9]+)',text)[1],'site.css?v='+hashlib.sha256((SITE/'site.css').read_bytes()).hexdigest()[:12])
    text=re.sub(r'site.js(?:\?v=[a-f0-9]+)?(?=")', 'site.js?v='+hashlib.sha256((SITE/'site.js').read_bytes()).hexdigest()[:12], text)
    text=re.sub(r'\n{3,}', '\n\n', '\n'.join(line.rstrip() for line in text.splitlines()))+'\n'
    p.write_text(text,'utf-8')
if __name__=='__main__':main()
