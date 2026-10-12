"""Reflow the reviewed 2408.05641v1 translation in its original conference style.

Uses the previously prepared, private paper project. No application resources,
dependencies, translation strings or server files are modified by this build.
MiKTeX's automatic package installer and shell escape are disabled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


HEADINGS = {
    'S002': ('section', '引言'),
    'S010': ('section', '音素到构音模型'),
    'S011': ('subsection', '音素时间预测'),
    'S013': ('subsection', 'EMA 生成'),
    'S015': ('subsection', '模型超参数与训练'),
    'S016': ('section', '测量协同发音'),
    'S021': ('section', '结果'),
    'S024': ('subsection', '生成 EMA 的质量'),
    'S026': ('subsection', '协同发音实例'),
    'S027': ('subsection', '协同发音的时间范围与幅度'),
    'S030': ('subsection', '协同发音抗性'),
    'S031': ('section', '讨论'),
    'S034': ('section', '致谢'),
}
EQUATIONS = {
    'S011': r'\bm{M}=f_\theta(\{\bm{x}_1,\ldots,\bm{x}_n\})\quad\bm{M}\in\mathbb{R}^{n\times2}',
    'S013': r"\bm{\hat{Y}}=h_\lambda(g_\phi(\bm{M}'))\quad\bm{\hat{Y}}\in\mathbb{R}^{t\times s}",
    'S016': r'[\ldots P_{i-2}\;P_{i-1}\;\bm{P_i}\;P_{i+1}\;P_{i+2}\ldots]',
    'S017': r'\bm{\varphi}_k=\frac{1}{2\tau+1}\sum_{j=\mu_k-\tau}^{\mu_k+\tau}\hat{Y}_{j,:}\quad\bm{\varphi}_k\in\mathbb{R}^{s}',
    'S019': r'd_k=\|\bm{\varphi}^A_k-\bm{\varphi}^B_k\|',
}
FIGURES = {
    1: ('model_arch.pdf', 'P2A 模型结构。'),
    2: ('ema_pred_gt_compare.pdf', '比较句子“Pick a card and slip it under the pack”的模型生成与真实 EMA 声道变量。上部显示真实词对齐。生成的 EMA 接近真实 EMA。'),
    3: ('coarticulation_egs.pdf', '模型生成的 EMA 可复现 VCV 对中已知的协同发音。左：［ʌvi］与［ʌvʌ］的预测音素影响和声道变量；右：［ʌzi］与［ʌzʌ］。TTCL 与 TBCL 在最后一个元音开始前出现分离，表明预期协同发音。左侧分离早于右侧，体现［v］与［z］的协同发音抗性差异。'),
    4: ('coarticulation_extent_layer_3_tau_1.pdf', '量化协同发音的幅度和范围。各箱线图表示相应音素距离处的协同发音效应，须线标示 1.5 倍四分位距。虚线为 EMA 差异基线。效应在 $\\pm1$ 距离处最明显，在 $\\pm7$ 距离处仍显著高于基线。'),
    5: ('coarticulation_resistance.pdf', '比较辅音的协同发音抗性，误差棒为 bootstrap 重采样所得的 95% 置信区间。由于舌的参与，齿音、齿龈后音和齿龈音的抗性更高。'),
}


def escape_prose(text):
    """Preserve the existing checked inline math, escape prose for XeLaTeX."""
    def plain(part):
        part = re.sub(r'https://[^\s。]+', lambda m: '\x00' + m[0] + '\x00', part)
        pieces = part.split('\x00')
        for index in range(0, len(pieces), 2):
            pieces[index] = ''.join({
                '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
                '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
            }.get(char, char) for char in pieces[index])
        for index in range(1, len(pieces), 2):
            pieces[index] = r'\url{' + pieces[index] + '}'
        return ''.join(pieces)
    parts = re.split(r'(\$[^$]+\$)', text)
    return ''.join(part if part.startswith('$') else plain(part) for part in parts)


def figure(number, *, wide=False):
    filename, caption = FIGURES[number]
    environment = 'figure*' if wide else 'figure'
    width = r'0.65\textwidth' if wide else (r'0.8\linewidth' if number == 5 else r'\linewidth')
    placement = '!b' if wide else ('t' if number == 5 else 'ht')
    return '\n'.join([
        r'\setcounter{figure}{' + str(number - 1) + '}',
        r'\begin{' + environment + '}[' + placement + ']', r'\centering',
        r'\includegraphics[width=' + width + ']{figs/' + filename + '}',
        r'\caption{' + escape_prose(caption) + '}',
        r'\label{fig:' + str(number) + '}', r'\end{' + environment + '}',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', default='xelatex')
    args = parser.parse_args()
    project = args.project.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    units = json.loads((project / 'units.json').read_text('utf8'))
    translations_raw = (project / 'translations.json').read_bytes()
    translations = json.loads(translations_raw)
    assert len(units) == len(translations) == 34
    assert set(translations) == {unit['id'] for unit in units}
    source = project / 'source'
    class_raw = (source / 'Interspeech2024.cls').read_bytes()
    # The source's 8-bit IPA package is unused here: translated prose uses
    # Unicode IPA, and the original vector figures embed their own fonts.
    # Remove only this dependency from the private build copy. Keep the
    # original class and its layout definitions untouched in the source.
    class_text = class_raw.decode('utf8').replace(r'\RequirePackage{tipa}', '% Unicode IPA: tipa not required for this translated build')
    class_text = class_text.replace(r'\RequirePackage{enumitem}', '% No body lists: enumitem not required')
    class_text = class_text.replace(r'\AtBeginDocument{\setlist{noitemsep,topsep=2pt,parsep=2pt,partopsep=0pt,leftmargin=1em}}', '% No body lists; bibliography has its own original list spacing')
    (out / 'Interspeech2024.cls').write_text(class_text, 'utf8')
    (out / 'figs').mkdir(exist_ok=True)
    for filename, _ in FIGURES.values():
        shutil.copyfile(source / 'figs' / filename, out / 'figs' / filename)
    bbl = (source / 'main.bbl').read_text('utf8')
    assert len(re.findall(r'\\bibitem\{', bbl)) == 33
    (out / 'references.bbl').write_text(bbl, 'utf8')
    preamble = r'''\documentclass{Interspeech2024}
\usepackage{xeCJK}
\usepackage{stfloats}
\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Consolas}
\setCJKmainfont{SimSun}[AutoFakeBold,AutoFakeSlant]
\setCJKsansfont{SimHei}
\setCJKmonofont{SimSun}
\interspeechcameraready
\title{基于音素到构音模型的协同发音定量分析}
\name[affiliation={1}]{Chaofei}{Fan}
\name[affiliation={1}]{Jaimie M.}{Henderson}
\name[affiliation={1}]{Chris}{Manning}
\name[affiliation={1,2}]{Francis R.}{Willett}
\address{$^1$美国斯坦福大学\\
$^2$美国斯坦福大学 Howard Hughes Medical Institute}
\email{stfan@stanford.edu}
\keywords{协同发音，言语产生，音素到构音模型}
\renewenvironment{abstract}{\begin{center}{\bfseries\large 摘要}\end{center}}{\par\noindent\textbf{关键词：}协同发音，言语产生，音素到构音模型}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\makeatletter
\renewenvironment{thebibliography}[1]{\section{参考文献}\eightpt\list{[\arabic{enumi}]}{\settowidth\labelwidth{[#1]}\leftmargin\labelwidth\advance\leftmargin\labelsep\usecounter{enumi}}\def\newblock{\hskip .11em plus .33em minus .07em}\sloppy\clubpenalty4000\widowpenalty4000\sfcode`\.=1000\relax}{\endlist}
\def\ps@reader{\def\@oddhead{}\def\@evenhead{}\def\@oddfoot{\hfil\fontsize{7}{8}\selectfont arXiv:2408.05641v1 · 中文译文 · CC BY 4.0\quad\thepage\hfil}\let\@evenfoot\@oddfoot}
\makeatother
\setlength{\footskip}{16pt}
\AtBeginDocument{\fontsize{10}{12.4}\selectfont}
\begin{document}
\maketitle
\pagestyle{reader}\thispagestyle{reader}
\begin{abstract}
'''
    # Keep the original title block, flow, section order, figure dimensions and
    # double-column float placement. Chinese wrapping may change pagination.
    parts = [preamble, escape_prose(translations['S001']), r'\end{abstract}']
    for unit in units[1:]:
        sid = unit['id']
        parts.append('% Source block ' + sid)
        if sid in HEADINGS:
            kind, title = HEADINGS[sid]
            parts.append('\\' + kind + '{' + title + '}')
        if sid == 'S021':
            parts.append(r'\subsection{数据集}')
        if sid == 'S010':
            parts.append(figure(1))
        if sid == 'S024':
            parts.append(r'''\begin{table}[h]
\caption{比较 USC-TIMIT 与 HPRC 数据集中经动态时间规整的模型预测 EMA 与真实 EMA 的 PCC 和 RMSE。置信区间由 bootstrap 重采样计算。}
\centering\small
\begin{tabular}{ccc}\toprule
\textbf{数据集} & \textbf{PCC \tiny[95\% CI]} & \textbf{RMSE \tiny[95\% CI]}\\\midrule
USC-TIMIT & 0.83\,\tiny[0.82, 0.84] & 0.59\,\tiny[0.57, 0.62]\\
HPRC & 0.85\,\tiny[0.85, 0.85] & 0.53\,\tiny[0.53, 0.53]\\\bottomrule
\end{tabular}\end{table}''')
        if sid == 'S025':
            # Keep float numbering in reading order with the current LaTeX
            # output routine; retain the original spanning-bottom figure.
            parts.append(figure(2))
        if sid == 'S026':
            parts.append(figure(3, wide=True))
        text = escape_prose(translations[sid])
        if sid == 'S030':
            parts.append(r'\begin{minipage}{\columnwidth}')
        parts.append(text + '\n')
        if sid == 'S030':
            parts.append(r'\end{minipage}')
        if sid in EQUATIONS:
            env = 'equation*' if sid == 'S016' else 'equation'
            parts += [r'\begin{' + env + '}', EQUATIONS[sid], r'\end{' + env + '}']
        if sid == 'S029':
            parts.append(figure(4))
        if sid == 'S030':
            # Preserve the source's paired top-column result figures. In
            # two-column mode this closes the left column, not the page.
            parts.append(r'\newpage')
            parts.append(figure(5))
    parts.append(r'''\par\smallskip
{\fontsize{6.5}{8}\selectfont\noindent
译制说明：PhoneticToolbox，2026-10-12。AI 辅助中文翻译及排版，未经过原作者审校。沿用原稿双栏格式，图表取自原作，参考文献保留原文。原作及译文均为 CC BY 4.0。原文：\url{https://arxiv.org/abs/2408.05641v1}；许可：\url{https://creativecommons.org/licenses/by/4.0/}。原题名：Towards a Quantitative Analysis of Coarticulation with a Phoneme-to-Articulatory Model。原文 4.2 写 TCCL，图和后文写 TTCL，译文保留原缩写。}
\clearpage
\input{references.bbl}
\end{document}
''')
    tex = out / 'translation.tex'
    tex.write_text('\n'.join(parts), 'utf8')
    command = [args.engine, '-disable-installer', '-no-shell-escape',
               '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
               'translation.tex']
    for attempt in range(2):
        result = subprocess.run(command, cwd=out, capture_output=True)
        (out / ('compile-' + str(attempt + 1) + '.log')).write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError('XeLaTeX failed; inspect ' + str(out / ('compile-' + str(attempt + 1) + '.log')))
    log = (out / 'translation.log').read_text('utf8', errors='replace')
    issues = [line for line in log.splitlines() if 'Missing character:' in line or 'Overfull \\hbox' in line or 'Overfull \\vbox' in line]
    report = {
        'schema': 'ptb-paper-original-layout/1', 'source': 'arXiv:2408.05641v1',
        'bodyBlocks': list(translations), 'figures': 5, 'tables': 1,
        'numberedEquations': 4, 'unnumberedSequences': 1, 'references': 33,
        'translationStringsSha256': hashlib.sha256(translations_raw).hexdigest(),
        'classSha256': hashlib.sha256((out / 'Interspeech2024.cls').read_bytes()).hexdigest(),
        'sourceClassSha256': hashlib.sha256(class_raw).hexdigest(),
        'classAdaptation': 'Omit unused tipa and enumitem dependencies; Unicode IPA, embedded vector-figure fonts; no body lists',
        'pdf': str(out / 'translation.pdf'), 'typesettingIssues': issues,
    }
    (out / 'build-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf8')
    print(json.dumps(report, ensure_ascii=False))
    if issues:
        raise RuntimeError('Resolve typesetting issues before publication')


if __name__ == '__main__':
    main()
