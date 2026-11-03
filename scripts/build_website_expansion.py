"""Assemble the static, accessible listening lab and source-backed support sections."""
from pathlib import Path
import html,json,re,hashlib
from website_audio_baseline import verify as verify_basic_gl
from build_website_gl_comparison import verify as verify_three_way
ROOT=Path(__file__).resolve().parents[1];SITE=ROOT/'temporary-site'
def esc(s):return html.escape(str(s),quote=True)

def learning_showcases(text):
    """Refresh only the two learning showcases, preserving all other sections."""
    import struct
    divider = re.search(r'<div class="tool-divider\b[\s\S]*?</div>', text).group()
    for section_id in ('paper-reading', 'language-map'):
        text = re.sub(r'<div class="tool-divider\b(?:(?!</div>)[\s\S])*?</div>\s*(?=<section class="showcase\b[^>]*id="' + section_id + '")', '', text)
        text = re.sub(r'<section class="showcase\b[^>]*id="' + section_id + r'"[\s\S]*?</section>', '', text)
    sections = []
    specs = [
        ('paper-reading', 'READ & ANNOTATE', '慢慢读，<br><em>随手记下来。</em>',
         '切换原文、中文译文与论文解读，标记重要段落，留下自己的批注。',
         '语音学论文精读', 'Lobster', '深色', False,
         [('paper-notice-lobster-dark', '请看这里~'),
          ('paper-annotated-lobster-dark', '全屏阅读、荧光标注与批注')]),
        ('language-map', 'LANGUAGE & PLACE', '把语言，<br><em>放回地图上。</em>',
         '浏览方言分布与语言现象，对照不同地图，比较各个调查点的记录。',
         '语言地图', 'Notion', '浅色', True,
         [('map-sites-notion-light', '苏州、广州、西安、哈尔滨与南京调查点对照'),
          ('map-feature-notion-light', '语言现象分布'),
          ('map-compare-notion-light', '双图对照'),
          ('map-similarity-notion-light', '方言相似性')]),
    ]
    for sid, eyebrow, title, copy, module, palette, mode, reverse, pictures in specs:
        cards = []
        for index, (file, label) in enumerate(pictures):
            raw = (SITE / 'assets/screens' / (file + '.png')).read_bytes()
            width, height = struct.unpack('>II', raw[16:24])
            caption = f'{module} · {label} · {palette} {mode}'
            src = f'assets/screens/{file}.png'
            cards.append(f'<a class="screen-link stack-card" style="--card:{index};--z:{index+1}" href="{src}" data-screen data-caption="{esc(caption)}" aria-label="查看{esc(label)}完整截图"><img src="{src}" width="{width}" height="{height}" loading="lazy" alt="{esc(caption)}，完整应用视图"></a>')
        attribution = ('<p class="example-note"><a href="https://arxiv.org/abs/2408.05641" target="_blank" rel="noopener noreferrer">示例论文 · Fan 等（2024）</a> · <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener noreferrer">CC BY 4.0</a><br>中文译文与阅读批注为演示内容。</p>' if sid == 'paper-reading' else '')
        sections.append(f'<section class="showcase split-showcase {"reverse-showcase" if reverse else ""}" id="{sid}" aria-labelledby="{sid}-heading"><div class="showcase-copy reveal"><span class="eyebrow">{eyebrow}</span><h2 id="{sid}-heading">{title}</h2><p>{copy}</p><span class="section-label">{module}</span>{attribution}</div><figure class="screen-stage screen-stack stack-{len(pictures)} reveal" id="{sid}-stack" style="--turn:{-7 if reverse else 7}deg"><span class="stage-backdrop {"coral" if reverse else "violet"}-backdrop"></span><div class="stack-deck">{"".join(cards)}</div><figcaption><span class="palette-dot {"rose" if reverse else "mauve"}"></span>{palette} <span>{mode}</span></figcaption></figure></section>')
    text, count = re.subn(r'(<section class="showcase\b[^>]*id="phonology"[\s\S]*?</section>)', lambda match: match[0] + ''.join(divider + section for section in sections), text, count=1)
    assert count == 1
    # The updated home is an unmodified native capture and retains its original palette.
    raw = (SITE / 'assets/screens/home-everforest-light.png').read_bytes()
    width, height = struct.unpack('>II', raw[16:24])
    text = re.sub(r'(<img src="(?:/site-releases/r[0-9a-f]{16}/)?assets/screens/home-everforest-light.png" width=")\d+(" height=")\d+', lambda m: m[1] + str(width) + m[2] + str(height), text)
    return text

def update_learning_only(index_file=None):
    path = index_file or SITE / 'content/home.zh-CN.html'
    before = path.read_text('utf-8')
    after = learning_showcases(before)
    assert learning_showcases(after) == after, 'Learning showcase generation must be idempotent'
    path.write_text(after, 'utf-8')
    if index_file is None:
        main()

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
               '国际音标表Plus', '汉字转国际音标', 'MFA 自动标注', 'TextGrid 标注', '音系归纳', '感知实验', '语音学论文精读', '语言地图']
    available = {'录音': 'recording', '唇形提取': 'lip-extraction', '参数估计': 'parameter-estimation',
                 '参数显示': 'parameter-display', 'EGG 信号分析': 'egg', 'LPC 谱图': 'lpc',
                 '发声类型合成': 'phonation-synthesis', '变速变调': 'pitch-manipulation',
                 '汉字转国际音标': 'mandarin-ipa', 'MFA 自动标注': 'mfa',
                 '语谱图转音频': 'spectrogram', '国际音标表Plus': 'ipa-plus',
                 'TextGrid 标注': 'annotation', '音系归纳': 'phonology',
                 '感知实验': 'perception', '语音学论文精读': 'paper-reading'}
    pending = {'声学参数合成': 'speech-synthesis', '生理参数合成': 'vocal-tract', '语言地图': 'geolinguistics'}
    buttons = ''.join(
        f'<button type="button" class="tutorial-channel tutorial-available" id="tutorial-{available[name]}" data-tutorial="{available[name]}" aria-controls="tutorial-screen tutorial-episodes" aria-pressed="false">{esc(name)}</button>'
        if name in available else f'<button type="button" class="tutorial-channel" id="tutorial-{pending[name]}" data-tutorial="{pending[name]}" aria-controls="tutorial-screen tutorial-episodes" aria-pressed="false" disabled>{esc(name)}</button>'
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
            <span>语谱图转音频</span><div><button type="button" data-tutorial-episode="spectrogram-special" aria-controls="tutorial-screen" aria-pressed="false">专题</button><button type="button" data-tutorial-episode="spectrogram" aria-controls="tutorial-screen" aria-pressed="false">本期教程</button></div>
          </div>
          <noscript><p>前往 B 站观看：<a href="https://www.bilibili.com/video/BV1WWpr67EkX/" target="_blank" rel="noopener noreferrer">工具箱介绍 ↗</a> · <a href="https://www.bilibili.com/video/BV1xsnJ6nES2/" target="_blank" rel="noopener noreferrer">01 录音篇 ↗</a> · <a href="https://www.bilibili.com/video/BV1NsnJ6nE2Z/" target="_blank" rel="noopener noreferrer">02 唇形提取篇 ↗</a> · <a href="https://www.bilibili.com/video/BV1NcJd6UEZv/" target="_blank" rel="noopener noreferrer">03 参数估计篇 ↗</a> · <a href="https://www.bilibili.com/video/BV11cJd6UESc/" target="_blank" rel="noopener noreferrer">04 参数显示篇 ↗</a> · <a href="https://www.bilibili.com/video/BV1KFJd6JEvg/" target="_blank" rel="noopener noreferrer">05 EGG信号分析篇 ↗</a> · <a href="https://www.bilibili.com/video/BV17mnc6QExx/" target="_blank" rel="noopener noreferrer">06 LPC谱篇 ↗</a> · <a href="https://www.bilibili.com/video/BV17mnc6QEYY/" target="_blank" rel="noopener noreferrer">11 语谱图转音频篇 ↗</a> · <a href="https://www.bilibili.com/video/BV17mnc6QEYL/" target="_blank" rel="noopener noreferrer">12 国际音标表Plus篇 ↗</a> · <a href="https://www.bilibili.com/video/BV17mnc6QEJN/" target="_blank" rel="noopener noreferrer">13 TextGrid标注篇 ↗</a> · <a href="https://www.bilibili.com/video/BV17mnc6QEJf/" target="_blank" rel="noopener noreferrer">14 音系归纳篇 ↗</a> · <a href="https://www.bilibili.com/video/BV1Bmnc6QEjn/" target="_blank" rel="noopener noreferrer">15 感知实验篇 ↗</a> · <a href="https://www.bilibili.com/video/BV1Bmnc6QELa/" target="_blank" rel="noopener noreferrer">16 语音学论文精读篇 ↗</a> · <a href="https://www.bilibili.com/video/BV1g1ns63EXQ/" target="_blank" rel="noopener noreferrer">发声类型合成篇 ↗</a> · <a href="https://www.bilibili.com/video/BV1QCns6kEqY/" target="_blank" rel="noopener noreferrer">变速变调篇 ↗</a> · <a href="https://www.bilibili.com/video/BV13Cns6kExb/" target="_blank" rel="noopener noreferrer">汉字转国际音标篇 ↗</a> · <a href="https://www.bilibili.com/video/BV1g1ns63EyT/" target="_blank" rel="noopener noreferrer">MFA自动标注篇 ↗</a> · <a href="https://www.bilibili.com/video/BV1RuJm6gEWE/" target="_blank" rel="noopener noreferrer">语谱图转音频 · 专题 ↗</a></p></noscript>
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
    assert modules and all(module in modules for row in sources for module in row['modules'] if re.fullmatch(r'M\d{2}', module)), 'References include an unregistered module'
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
    from build_website_five_comparison import KINDS, LABELS
    body='''<section class="content-section upcoming-section" id="next" aria-labelledby="next-heading">
  <div class="upcoming-paper reveal"><span class="upcoming-tab">下一步…</span>
    <div class="upcoming-intro"><div><span class="eyebrow">THE NEXT LITTLE STEP</span><h2 id="next-heading">短时窗里的声音，<br><em>还想听得更清楚。</em></h2></div><span class="upcoming-stamp">语谱图转音频<br><small>模型训练进行中</small><i aria-hidden="true">✦</i></span></div>
    <div class="upcoming-copy"><p>唔……最近在试着让语谱图转音频的效果更好一些。韶子还训练了一个小模型，让它结合图片和时间、频率标定，估计生成这张语谱图时可能使用的 STFT 窗宽与动态范围。有了这两个估计值，就可以先调整重建参数，再试着把声音找回来啦。</p><p>相位重建这一步，韶子参考了青木（Naofumi Aoki）关于短时窗语谱图重建的研究，先沿频率方向求和，从得到的时间投影里提取峰值脉冲，构造激励信号，再用它的相位初始化 Griffin–Lim。这样，迭代就有了一个可以参考的起点。迭代过程中，还借用了 Perraudin 等人的动量加速方法，把相邻两次迭代的变化也利用起来，帮助相位估计更快接近目标。工具箱目前采用 <code>0.99</code> 的动量系数，收敛通常能快一些。</p><p>有了初步重建的声音之后呢，接下来就是 Mel 谱修复啦。先把声音转成 Mel 谱，再训练一个条件化 NAFNet，让它以目标 Mel 谱为参照，学习修补重建过程中出现的频谱偏差，希望能把模糊、失真的频谱结构整理得更接近自然声音。修复后的 Mel 谱再交给 BigVGAN 声码器，由它生成最终的音频波形。这一步沿用预训练的 BigVGAN，让声码器利用已经学到的语音规律，把修复后的频谱合成为声音（其实和降噪、回声消除之类的算法差不多）。</p><p>目前还在继续折腾，希望能成功hhh 下面把原始 GL、动量加速、参数预测、青木初始化和 Mel 谱修复分成五路，一步一步听听变化叭。</p></div>
    <a class="upcoming-reference" href="https://doi.org/10.1250/ast.44.186" target="_blank" rel="noopener noreferrer"><span>相位起点：青木的短时窗重建研究 ↗</span><cite>Technique of improving speech restoration from spectrograms of short windows for the Griffin–Lim algorithm</cite><small>Naofumi Aoki · 2023 · Acoustical Science and Technology, 44(3), 186–188</small></a>
    <a class="upcoming-reference" href="https://perraudin.info/publications/perraudin-note-002.pdf" target="_blank" rel="noopener noreferrer"><span>迭代过程：动量加速 Griffin–Lim · 作者全文 ↗</span><cite>A fast Griffin–Lim algorithm</cite><small>Nathanaël Perraudin, Peter Balazs &amp; Peter L. Søndergaard · 2013 · WASPAA, 1–4</small></a>
    <ol class="upcoming-pipeline" aria-label="短时窗语谱图重建流程">
      <li><span class="upcoming-step">01 / 先读一读图片</span><h3>估计 STFT 参数</h3><span class="upcoming-model">窗宽与动态范围预测模型</span><p>结合语谱图与时间、频率标定，预测窗宽和动态范围，窗宽取整到毫秒。</p><span class="upcoming-arrow" aria-hidden="true">→</span></li>
      <li><span class="upcoming-step">02 / 再把声音找回来</span><h3>动量加速重建</h3><span class="upcoming-model">Griffin–Lim · 动量 0.99</span><p>按预测参数，用青木脉冲初始化相位，迭代 32 次，重建初步声音。</p><span class="upcoming-arrow" aria-hidden="true">→</span></li>
      <li><span class="upcoming-step">03 / 把频谱修一修</span><h3>Mel 谱修复</h3><span class="upcoming-model">条件化 NAFNet</span><p>把第四路青木初始化动量 GL 的同一段声音转成 Mel 谱，用训练好的 80 带模型修补频谱偏差。</p><span class="upcoming-arrow" aria-hidden="true">→</span></li>
      <li><span class="upcoming-step">04 / 再交给耳朵听</span><h3>合成音频</h3><span class="upcoming-model">BigVGAN</span><p>沿用预训练声码器，把修复后的 Mel 谱合成为最终的音频波形。</p></li>
    </ol>
    <div class="next-comparison-heading"><h3>同一张图，五种方式听一听。</h3><p>先听原始 GL，再听动量加速，第三路加入小模型预测的窗宽与动态范围，第四路再加上 <strong>青木脉冲初始化</strong>，第五路把第四路的同一段声音交给 <strong>80 带 NAFNet</strong> 修复 Mel 谱，再由 <strong>BigVGAN</strong> 合成。前两路固定 <code>10 ms</code> 窗宽和 <code>60 dB</code> 动态范围，后三路使用每张图各自的预测参数，Griffin–Lim 都迭代 <code>32</code> 次。</p><p class="next-hover-hint">把鼠标放在任一试听按钮上，看看五路语谱图的变化。键盘聚焦也可以查看。</p></div><div class="upcoming-comparisons">'''
    pairs=json.loads((SITE/'assets/spectrogram-examples/comparison.json').read_text('utf-8'))['pairs']
    for pair in pairs:
        key=pair['id']
        assert all(audio[pair[k]]['sha256']==pair[k+'_sha256'] for k in KINDS)
        display=pair['display'];image=esc(display['image']);credit=esc(display['credit'])
        attribution=(f'<a href="{esc(pair["url"])}" target="_blank" rel="noopener noreferrer">{credit} ↗</a>' if pair.get('url') else credit)
        body+=f'<article class="next-example" data-case="{key}"><h4>{esc(display["title"])}</h4><figure><a href="{image}" target="_blank" rel="noopener noreferrer" aria-label="查看{credit}完整图像节选"><img src="{image}" width="720" height="580" loading="lazy" alt="{esc(pair["source"])} 语谱图原图节选"></a><figcaption>{attribution}</figcaption></figure>'
        prediction=pair['parameter_prediction']
        body+=f'<p class="next-parameters">前两路：<code>10 ms</code> Hann 窗 · <code>60 dB</code><br>模型估计：<code>{prediction["window_ms"]:.2f} ms</code> → 用 <code>{pair["applied_window_ms"]:g} ms</code><br>预测动态范围：<code>{prediction["dynamic_range_db"]:.2f} dB</code><br>前三路：随机相位 · 后两路：<strong>青木初始化</strong><br>GL 均为 <code>32</code> 次迭代</p>'
        details=('动量 0','动量 0.99','预测窗宽与动态范围','动量 0.99 · 青木初始化','接续第四路的同一段声音')
        sounds=player('next',[(pair[kind][:-4],label,f'{audio[pair[kind]]["duration"]:.2f} s · {detail}') for kind,label,detail in zip(KINDS,LABELS,details)],'next-sounds',instance='next-',volumes=pair['playback_volumes'],show_waveform=False)
        sounds=re.sub(r'<span class="sound-detail">[^<]*</span>','',sounds)
        for kind in KINDS:
            marker=f'data-audio="audio-next-{pair[kind][:-4]}"'
            sounds=sounds.replace(marker,marker+f' data-spectrum-route="{kind}" aria-controls="spectrum-preview-{key}" aria-expanded="false"')
        body+=sounds+f'<aside class="next-spectrum-preview" id="spectrum-preview-{key}" role="region" aria-label="{esc(display["title"])}五路语谱图对比" hidden><div class="next-spectrum-title"><strong>{esc(display["title"])} · 五路语谱图</strong><button type="button" class="next-spectrum-close" aria-label="关闭语谱图对比">×</button></div><p class="next-spectrum-caption">实际输出 · 同一显示标尺 · 10 ms Hann 窗 / 60 dB<br>频率 {pair["frequency_range"][0]/1000:g}–{pair["frequency_range"][1]/1000:g} kHz · 时间 0–{pair["seconds"]:g} s</p><div class="next-spectrum-rows">'
        for number,(kind,label) in enumerate(zip(KINDS,LABELS),1):
            spectrum=pair['spectrograms'][kind]
            body+=f'<figure class="next-spectrum-row" data-spectrum-kind="{kind}"><figcaption><span>{number:02d} · {esc(label)}</span><b class="next-spectrum-current">当前</b></figcaption><img src="{esc(spectrum["image"])}?v={spectrum["sha256"][:12]}" width="768" height="104" loading="lazy" alt="第{number}路{esc(label)}的输出语谱图，横轴时间，纵轴频率"></figure>'
        body+='</div><p class="next-spectrum-scale"><span>0 s</span><span>'+f'{pair["seconds"]:g} s'+'</span></p></aside></article>'
    return body+'''</div><p class="next-figure-reference" id="next-lu-reference">Lu 两例图源：Lu, Y., Liang, C., &amp; Kong, J. (2025). <cite>Contribution of F0 and phonation to tone perception in the Zaiwa language.</cite> <i>Journal of Phonetics, 110</i>, 101413. <a href="https://doi.org/10.1016/j.wocn.2025.101413" target="_blank" rel="noopener noreferrer">查看论文 ↗</a></p><p class="audio-status" role="status" aria-live="polite">点一下，按顺序听听五种方式。</p><details class="next-comparison-settings" open><summary>这组对比怎样生成？</summary><p>五路使用相同的图片、时间频率标定与 Hann 窗，GL 均为 <strong>1.5 ms 帧移、32 次迭代</strong>。按下面的顺序，逐步加入新的处理：</p><ol><li><strong>原始 GL：</strong>固定 <code>10 ms</code> 窗宽、<code>60 dB</code> 动态范围，随机相位初始化，动量 <code>0</code>。</li><li><strong>动量加速 GL：</strong>沿用第一路的参数与随机相位起点，只将动量改为 <mark>0.99</mark>。</li><li><strong>预测参数 + 动量 GL：</strong>保持动量 <code>0.99</code> 和随机相位，改用小模型预测的窗宽与动态范围，<mark>窗宽四舍五入到整数毫秒</mark>。</li><li><strong>预测参数 + 青木初始化 + 动量 GL：</strong>沿用第三路的预测参数与动量 <code>0.99</code>，再加入 <mark>青木脉冲初始化</mark>。</li><li><strong>Mel 修复与声码器：</strong><mark>直接读取第四路导出的同一段音频</mark>，接上 <strong>80 带 NAFNet 的 Mel 谱修复</strong>与 <strong>BigVGAN 声码器</strong>。</li></ol><p>前三路沿用已固定的音频，播放时将五路衰减到相同均方根电平。浮窗展示五路实际输出音频的语谱图，统一使用 10 ms Hann 窗、1.5 ms 帧移和 60 dB 显示动态范围，同例共用频率范围与幅度参考。<strong>悬停在哪个按钮，就强调对应的语谱图。</strong>窗宽和动态范围是模型的估计值，重建结果可以按顺序交替听听。</p></details><p class="upcoming-signoff">模型还在继续训练，先把这几组声音放在这里。<span aria-hidden="true">✧</span></p></div></section>'''

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
def release_html(body,tag_name=None):
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
    content='\n'.join(out)
    if tag_name == 'v3.0.4':
        media='''<figure class="release-video"><video controls playsinline preload="none" width="1280" height="720" poster="assets/videos/v3.0.4-vocal-tract-chunjiang.jpg" aria-label="春江花月夜：生理参数合成演示" aria-describedby="release-vocal-video-caption"><source src="assets/videos/v3.0.4-vocal-tract-chunjiang.mp4" type="video/mp4">可<a href="assets/videos/v3.0.4-vocal-tract-chunjiang.mp4">打开演示视频</a>观看。</video><figcaption id="release-vocal-video-caption">《春江花月夜》· 生理参数合成演示</figcaption></figure>'''
        content,count=re.subn(r'(<h4>生理参数合成</h4>[\s\S]*?)(?=<h4>|$)',lambda m:m[1]+media+'\n',content,count=1)
        assert count == 1, 'v3.0.4 vocal-tract section must exist for its video'
    return content
def showcase(id,title,copy,module,file,palette,mode,reverse=False):
    color='coral' if mode=='浅色' else 'violet'
    return f'''<section class="showcase split-showcase {'reverse-showcase' if reverse else ''}" id="{id}" aria-labelledby="{id}-heading"><div class="showcase-copy reveal"><span class="eyebrow">{id.upper()}</span><h2 id="{id}-heading">{title}</h2><p>{copy}</p><span class="section-label">{module}</span></div><figure class="screen-stage reveal" style="--tilt:4deg;--turn:{-7 if reverse else 7}deg"><span class="stage-backdrop {color}-backdrop"></span><a class="screen-link" href="assets/screens/{file}.png" data-screen data-caption="{module} · {palette} {mode}" aria-label="查看{module}完整截图"><img src="assets/screens/{file}.png" width="2560" height="1440" loading="lazy" alt="{module}的实际中文界面，{palette} {mode}配色"></a><figcaption><span class="palette-dot {'rose' if reverse else 'green'}"></span>{palette} <span>{mode}</span></figcaption></figure></section>'''

def main():
    verify_basic_gl()
    verify_three_way()
    from build_website_i18n import build
    build()


def _reference_catalog_spans(data):
    """Locate the two outer catalogues, retaining any nested details."""
    starts=[match for match in re.finditer(rb'<details\b[^>]*\sclass=([\"\'])([^\"\']*)\1[^>]*>',data,re.I)
            if b'reference-catalog' in match[2].split()]
    if len(starts)!=2:
        raise ValueError('Expected exactly two reference catalogues')
    spans=[]
    for start in starts:
        depth=1
        for match in re.finditer(rb'</?details\b[^>]*>',data[start.end():],re.I):
            depth+=-1 if match[0].startswith(b'</') else 1
            if depth==0:
                spans.append((start.start(),start.end()+match.end()))
                break
        else:
            raise ValueError('Unclosed reference catalogue')
    if spans[0][1]>spans[1][0]:
        raise ValueError('Reference catalogues must not contain one another')
    return spans

def _replace_reference_catalogs(data,catalog):
    """Replace catalogue bytes only, preserving gaps, suffixes and line endings."""
    original=_reference_catalog_spans(data)
    replacement=_reference_catalog_spans(catalog)
    chunks=[];cursor=0
    for (start,end),(new_start,new_end) in zip(original,replacement):
        chunks.extend((data[cursor:start],catalog[new_start:new_end]))
        cursor=end
    chunks.append(data[cursor:])
    return b''.join(chunks)

def update_references_only(index_file=None):
    """Refresh the common bibliography without regenerating other homepage sections."""
    verify_basic_gl()
    verify_three_way()
    path=Path(index_file) if index_file is not None else SITE/'content/home.zh-CN.html'
    data=path.read_bytes()
    updated=_replace_reference_catalogs(data,reference_catalog().encode('utf-8'))
    if updated!=data:
        path.write_bytes(updated)
    if index_file is None:
        main()

def update_tutorials_only():
    """Refresh tutorial links without changing other homepage work."""
    verify_basic_gl()
    path=SITE/'content/home.zh-CN.html'
    text=path.read_text('utf-8')
    start=text.index('<section class="content-section tutorial-section"')
    end=text.index('</section>',start)+len('</section>')
    text=text[:start]+tutorials()+text[end:]
    text=re.sub(r'site.js(?:\?v=[a-f0-9]+)?(?=")', 'site.js?v='+hashlib.sha256((SITE/'site.js').read_bytes()).hexdigest()[:12], text)
    path.write_text(text,'utf-8')
    main()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--references-only',action='store_true',help='Refresh only the shared reference catalogues')
    group.add_argument('--tutorials-only',action='store_true',help='Refresh only tutorial links and the script content hash')
    group.add_argument('--learning-only',action='store_true',help='Refresh only paper reading, language map and current home screenshot dimensions')
    parser.add_argument('--index-file',type=Path,help='Owned candidate index.html for --references-only; other bytes and script hashes stay unchanged')
    args=parser.parse_args()
    if args.index_file is not None and not (args.references_only or args.learning_only):
        parser.error('--index-file requires --references-only or --learning-only')
    update_learning_only(args.index_file) if args.learning_only else update_references_only(args.index_file) if args.references_only else update_tutorials_only() if args.tutorials_only else main()
