"""Reconstruct user-supplied figure excerpts with the application's M09 core.

Only the spectral raster enters reconstruction. Captioned excerpts illustrate
the method; full source pages and their surrounding text are not republished.
"""
from pathlib import Path
import argparse,hashlib,json,sys
from website_audio_baseline import protect_pending
import numpy as np
from PIL import Image
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packages/phonetic_core/src'))
from phonetic_core.spec2wav.reconstruction import reconstruct

CASES=[
    dict(id='smith',clipboard='9c0e3851-f6c4-4a81-aa9f-e122839f1de8',crop=[94,388,752,907],display=[44,362,758,929],duration=1.48,frequency=4000,title='一张图，一句问候。',source='Julius O. Smith III (2007). Mathematics of the Discrete Fourier Transform (DFT), 2nd ed., Fig. 8.10.',url='https://www.dsprelated.com/freebooks/mdft/Spectrogram_Speech.html',explanation='Hi, this is Steve Jobs. 横向是一句话的时间，纵向是频率。图中深浅不同的条纹，记录着声音能量的变化。'),
    dict(id='levantan',clipboard='cb5ded10-8425-408b-a72e-0b56ed2ac4b8',crop=[212,472,1059,676],display=[154,201,1100,920],duration=.9334,frequency=5000,title='声音里的小停顿。',source='Whitney Chappell (2021). /s/ weakening in Nicaragua. In E. Núñez-Méndez (ed.), Sociolinguistic Approaches to Sibilant Variation in Spanish, pp. 217–245. Fig. 7.4, p. 228.',url='https://doi.org/10.4324/9781003153948-11',figure_url='https://www.researchgate.net/figure/Spectrogram-and-waveform-of-a-glottal-stop-in-Levantan-las-olas-the-waves-rise-up_fig4_349935398',explanation='Levantan las olas 这句话的原图展示了喉塞音。对照标注和语谱图，观察停顿附近能量的变化。'),
    dict(id='observo',clipboard='61741c76-a15c-4ed1-8673-9d26b8c6cbd0',crop=[182,442,1010,635],display=[99,165,1070,862],duration=.7278,frequency=5000,title='把摩擦，也听出来。',source='Whitney Chappell (2021). /s/ weakening in Nicaragua. In E. Núñez-Méndez (ed.), Sociolinguistic Approaches to Sibilant Variation in Spanish, pp. 217–245. Fig. 7.3, p. 226.',url='https://doi.org/10.4324/9781003153948-11',figure_url='https://www.researchgate.net/figure/Coda-sibilance-insertion-in-observo-from-Observo-un-par-de-islas-I-see-a-couple-of_fig3_349935398',explanation='原文用 observo 的这个片段说明咝音插入。看一看高频噪声，再听图片重建后的近似声音。'),
]

def import_pulse_study(study):
    """Copy the selected listening exports byte for byte, without rerunning models."""
    study=study.resolve()
    report_path=study/'auditions.json'
    report=json.loads(report_path.read_text('utf-8'))
    audio=ROOT/'temporary-site/assets/audio'
    figures=ROOT/'temporary-site/assets/spectrogram-examples/provenance.json'
    manifest=json.loads((audio/'provenance.json').read_text('utf-8'))
    examples=json.loads(figures.read_text('utf-8'))
    cases={r['id']:r for r in report['cases']}
    model={k:v for k,v in report['models']['base80'].items() if k!='path'}
    pending=[]
    for example in examples['cases']:
        key=example['id'];case=cases[key];selected=case['audio']['base80-repaired']
        source=(study/selected['path']).resolve()
        assert source.is_relative_to(study)
        payload=source.read_bytes();digest=hashlib.sha256(payload).hexdigest()
        assert digest==selected['sha256']
        assert case['source']['source']==example['source']
        assert case['seconds']==example['duration'] and case['source']['freq_end']==example['frequency']
        image=study/case['image']
        assert hashlib.sha256(image.read_bytes()).hexdigest()==case['source']['input_sha256']
        assert image.read_bytes()==(figures.parent/(key+'-input.png')).read_bytes()
        sr,pcm=wavfile.read(source)
        values=pcm.astype(np.float64)
        if np.issubdtype(pcm.dtype,np.integer):
            values/=max(abs(np.iinfo(pcm.dtype).min),np.iinfo(pcm.dtype).max)
        assert sr==24000 and pcm.ndim==1 and len(pcm)==round(case['seconds']*sr)
        assert np.isfinite(values).all() and 0<float(np.sqrt(np.mean(values**2))) and np.max(np.abs(values))<1
        assumptions=dict(report['assumptions']);gl=dict(report['gl'])
        prediction=case.get('parameter_prediction')
        if prediction:
            assumptions.update(window_ms=prediction['window_ms'],dynamic_range_db=prediction['dynamic_range_db'])
            gl.update(window_samples=case['window_samples'],applied_window_ms=case['applied_window_ms'])
        processing=dict(route='参数预测 → GL → 训练后 80 带修复 → 声码器' if prediction else 'GL → 训练后 80 带修复 → 声码器',engine='pulse GL / NAFNet 80 / BigVGAN',
                        assumptions=assumptions,gl=gl,model=model,
                        source_export=selected['path'],source_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                        source_output_sha256=digest,delivery='exact_listening_export')
        if prediction:processing['parameter_prediction']=prediction
        if 'config' in example:
            example['previous_gl_baseline']={k:example.pop(k) for k in ('config','metadata')}
        example['processing']=processing
        example['output_sha256']=digest
        name='spectrum-'+key+'.wav'
        row=next(r for r in manifest['files'] if r['file']==name)
        row.update(description='语谱图图片近似重建：'+key,parameters=dict(processing,source=example['source'],url=example['url']),
                   sample_rate=sr,duration=len(pcm)/sr,peak=float(np.max(np.abs(values))),
                   rms=float(np.sqrt(np.mean(values**2))),sha256=digest,
                   waveform=[round(float(np.max(np.abs(c))),4) for c in np.array_split(values,72)])
        pending.append((audio/name,payload))
    protect_pending(pending)
    for destination,payload in pending:
        destination.write_bytes(payload)
    (audio/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    figures.write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n','utf-8')
    comparison_path=figures.parent/'comparison.json'
    if report.get('parameter_model_sha256') and comparison_path.exists():
        comparison=json.loads(comparison_path.read_text('utf-8'))
        for pair in comparison['pairs']:
            if pair['id'] not in cases:continue
            case=cases[pair['id']]
            pair['after_sha256']=case['audio']['base80-repaired']['sha256']
            pair['parameter_prediction']=case['parameter_prediction']
            pair['applied_window_ms']=case['applied_window_ms']
            pair['parameter_model_sha256']=report['parameter_model_sha256']
            if pair.get('playback_volumes'):
                before_row=next(r for r in manifest['files'] if r['file']==pair['before'])
                after_row=next(r for r in manifest['files'] if r['file']==pair['after'])
                target=min(before_row['rms'],after_row['rms'])
                pair['playback_volumes']={pair['before'][:-4]:target/before_row['rms'],pair['after'][:-4]:target/after_row['rms']}
        comparison['processing']='处理前沿用各例明确登记的默认参数与相位方法；处理后使用各图固定参数模型的预测，经GL/80带NAFNet/BigVGAN合成；按对衰减播放音量。'
        comparison['after_source_report_sha256']=hashlib.sha256(report_path.read_bytes()).hexdigest()
        comparison['parameter_model_sha256']=report['parameter_model_sha256']
        comparison_path.write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(imported=[p.name for p,_ in pending],route='base80-repaired')))


def import_comparison(study,before_only=False):
    """Reuse the matched before/after listening exports without synthesis."""
    study=study.resolve();report_path=study/'auditions.json'
    report=json.loads(report_path.read_text('utf-8'))
    directory=ROOT/'temporary-site/assets';audio=directory/'audio'
    manifest=json.loads((audio/'provenance.json').read_text('utf-8'))
    examples=json.loads((directory/'spectrogram-examples/provenance.json').read_text('utf-8'))
    rows={r['file']:r for r in manifest['files']};cases={r['id']:r for r in report['cases']}
    comparison_path=directory/'spectrogram-examples/comparison.json'
    current=json.loads(comparison_path.read_text('utf-8')) if before_only else None
    current_pairs={p['id']:p for p in current['pairs']} if current else {}
    if before_only and len(current_pairs)!=len(examples['cases']):
        raise ValueError('The six-case comparison requires --six-defaults-study with --additional-comparisons; the legacy three-case importer cannot replace it.')
    pending=[];pairs=[]
    for example in examples['cases']:
        key=example['id'];case=cases[key]
        assert case['source']['source']==example['source'] and case['seconds']==example['duration']
        assert (study/case['image']).read_bytes()==(directory/'spectrogram-examples'/f'{key}-input.png').read_bytes()
        before=case['audio']['gl']
        after=({'sha256':current_pairs[key]['after_sha256']} if before_only else case['audio']['base80-repaired'])
        after_name=f'spectrum-{key}.wav';assert rows[after_name]['sha256']==after['sha256']
        assert hashlib.sha256((audio/after_name).read_bytes()).hexdigest()==after['sha256']
        source=(study/before['path']).resolve();assert source.is_relative_to(study)
        payload=source.read_bytes();digest=hashlib.sha256(payload).hexdigest();assert digest==before['sha256']
        sr,pcm=wavfile.read(source);assert sr==24000 and pcm.ndim==1 and len(pcm)==round(case['seconds']*sr)
        values=pcm.astype(np.float64)/32768;assert pcm.dtype==np.int16
        assert np.isfinite(values).all() and 0<float(np.sqrt(np.mean(values**2))) and np.max(np.abs(values))<1
        name=f'spectrum-before-{key}.wav'
        row=dict(file=name,description=f'同图处理前：{key} · 未知参数默认直接 GL' if before_only else f'同图处理前：{key} · 脉冲初始化 Griffin–Lim',sample_rate=sr,duration=len(pcm)/sr,
                 peak=float(np.max(np.abs(values))),rms=float(np.sqrt(np.mean(values**2))),sha256=digest,
                 waveform=[round(float(np.max(np.abs(c))),4) for c in np.array_split(values,72)],
                 parameters=dict(route=('随机相位初始化 Griffin–Lim' if report['gl'].get('phase_initialization')=='random' else '脉冲初始化 Griffin–Lim'),assumptions=report['assumptions'],gl=report['gl'],
                                 source=example['source'],url=example['url'],source_export=before['path'],
                                 source_output_sha256=digest,delivery='exact_listening_export'))
        if name in rows:manifest['files'][manifest['files'].index(rows[name])]=row
        else:manifest['files'].append(row)
        pending.append((audio/name,payload))
        pair=dict(current_pairs[key]) if before_only else dict(id=key,before=name,after=after_name,after_sha256=after['sha256'],source=example['source'],url=example['url'],seconds=case['seconds'])
        pair['before_sha256']=digest
        if before_only:
            pair['before_defaults']=dict(report['assumptions'],phase_initialization=report['gl'].get('phase_initialization','pulse'),iterations=report['gl']['iterations'])
            after_rms=rows[after_name]['rms'];target=min(row['rms'],after_rms)
            pair['playback_volumes']={name.removesuffix('.wav'):target/row['rms'],after_name.removesuffix('.wav'):target/after_rms}
        pairs.append(pair)
    protect_pending(pending)
    for path,payload in pending:path.write_bytes(payload)
    (audio/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    comparison=dict(paper=dict(title='Technique of improving speech restoration from spectrograms of short windows for the Griffin–Lim algorithm',
                               author='Naofumi Aoki',year=2023,doi='10.1250/ast.44.186',url='https://doi.org/10.1250/ast.44.186'),
                    processing='同一实验、同图输入、匹配试听响度；前为脉冲初始化GL，后为同GL接条件化NAFNet及BigVGAN',
                    source_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),pairs=pairs)
    if before_only:
        phase_label='随机相位初始化' if report['gl'].get('phase_initialization')=='random' else '脉冲初始化'
        current.update(pairs=pairs,before_source_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),processing=f'处理前采用未知参数默认20ms Hann/60dB{phase_label}直接GL，不预测参数或修复；处理后保留既有自动参数/NAFNet/BigVGAN音频。播放时按RMS降低较响一侧。')
        comparison=current
    comparison_path.write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(imported=[p.name for p,_ in pending],comparisons=len(pairs))))


def import_six_comparisons(defaults, additional):
    """Import six direct-GL baselines and the three explicitly selected v5 exports."""
    defaults=defaults.resolve();additional=additional.resolve()
    baseline=json.loads((defaults/'auditions.json').read_text('utf-8'))
    added=json.loads((additional/'auditions.json').read_text('utf-8'))
    assert baseline['assumptions']['window_ms']==10 and baseline['assumptions']['dynamic_range_db']==60
    assert baseline['gl']['phase_initialization']=='random' and not baseline['models'] and not baseline['training']
    assets=ROOT/'temporary-site/assets';audio=assets/'audio';path=assets/'spectrogram-examples/comparison.json'
    comparison=json.loads(path.read_text('utf-8'));manifest=json.loads((audio/'provenance.json').read_text('utf-8'))
    pairs={p['id']:p for p in comparison['pairs']};rows={r['file']:r for r in manifest['files']}
    added_cases={c['id']:c for c in added['cases']};pending=[];result=[]
    display={'smith':('Hi, this is Steve Jobs.','Smith (2007) · 图 8.10'),
             'levantan':('Levantan las olas','Chappell (2021) · 图 7.4'),
             'observo':('observo','Chappell (2021) · 图 7.3'),
             'lu2025-modal':('Lu 2025 · modal','Lu, Liang & Kong (2025) · modal'),
             'lu2025-pressed':('Lu 2025 · pressed','Lu, Liang & Kong (2025) · pressed'),
             'uploaded':('知乎例图 · 2 秒 / 6 kHz','知乎例图')}
    def export_row(name, source, expected, description, parameters):
        payload=source.read_bytes();digest=hashlib.sha256(payload).hexdigest();assert digest==expected
        sr,pcm=wavfile.read(source);assert sr==24000 and pcm.dtype==np.int16 and pcm.ndim==1
        values=pcm.astype(np.float64)/32768;assert np.isfinite(values).all() and np.std(values)>1e-7 and np.max(np.abs(values))<.951
        row=dict(file=name,description=description,sample_rate=sr,duration=len(pcm)/sr,
                 peak=float(np.max(np.abs(values))),rms=float(np.sqrt(np.mean(values**2))),sha256=digest,
                 waveform=[round(float(np.max(np.abs(c))),4) for c in np.array_split(values,72)],parameters=parameters)
        pending.append((audio/name,payload));rows[name]=row
        return row
    assert [c['id'] for c in baseline['cases']]==list(display)
    for case in baseline['cases']:
        key=case['id'];before_name=f'spectrum-before-{key}.wav';after_name=f'spectrum-{key}.wav'
        assert case['peak_count']==0 and case['parameter_prediction'] is None
        src=case['source'];image=defaults/case['image'];assert hashlib.sha256(image.read_bytes()).hexdigest()==src['input_sha256']
        if key in added_cases:
            new=added_cases[key];assert new['source']['input_sha256']==src['input_sha256'] and new['seconds']==case['seconds']
            selected=new['comparison_audio']['new-base80-repaired'];source=(additional/selected['path']).resolve();assert source.is_relative_to(additional)
            prediction=new['parameter_prediction'];assert prediction['checkpoint_sha256']==added['parameter_model_sha256']
            parameters=dict(route='参数预测 → 脉冲 GL → 80 带 NAFNet Mel 修复 → BigVGAN',assumptions=added['assumptions'],
                            gl=added['gl'],parameter_prediction=prediction,applied_window_ms=new['applied_window_ms'],
                            model={k:v for k,v in added['models']['base80'].items() if k!='path'},
                            source=src['source'],url=src.get('url'),source_export=selected['path'],
                            source_report_sha256=hashlib.sha256((additional/'auditions.json').read_bytes()).hexdigest(),
                            source_output_sha256=selected['sha256'],delivery='exact_listening_export')
            after=export_row(after_name,source,selected['sha256'],f'同图处理后：{key} · 预测参数 / NAFNet + BigVGAN',parameters)
            destination=assets/'spectrogram-examples'/f'{key}.png';pending.append((destination,image.read_bytes()))
            pair=dict(id=key,before=before_name,after=after_name,after_sha256=after['sha256'],source=src['source'],url=src.get('url'),seconds=case['seconds'],
                      parameter_prediction=prediction,parameter_model_sha256=added['parameter_model_sha256'],applied_window_ms=new['applied_window_ms'])
        else:
            pair=dict(pairs[key]);after=rows[after_name]
            assert hashlib.sha256((audio/after_name).read_bytes()).hexdigest()==pair['after_sha256']==after['sha256']
            assert image.read_bytes()==(assets/'spectrogram-examples'/f'{key}-input.png').read_bytes()
            pair['parameter_model_sha256']=comparison['parameter_model_sha256']
        selected=case['audio']['gl'];source=(defaults/selected['path']).resolve();assert source.is_relative_to(defaults)
        before=export_row(before_name,source,selected['sha256'],f'同图处理前：{key} · 默认 10 ms / 60 dB 随机相位 GL',
                          dict(route='随机相位初始化 Griffin–Lim',assumptions=baseline['assumptions'],gl=baseline['gl'],source=src['source'],url=src.get('url'),
                               source_output_sha256=selected['sha256'],delivery='exact_listening_export'))
        assert round(before['duration']*24000)==round(case['seconds']*24000)==round(after['duration']*24000)
        title,credit=display[key];shown=assets/'spectrogram-examples'/f'{key}.png'
        image_bytes=image.read_bytes() if key in added_cases else shown.read_bytes()
        target=min(before['rms'],after['rms'])
        pair.update(before_sha256=before['sha256'],before_defaults=dict(baseline['assumptions'],phase_initialization='random',iterations=baseline['gl']['iterations']),
                    display=dict(title=title,credit=credit,image=f'assets/spectrogram-examples/{key}.png',image_sha256=hashlib.sha256(image_bytes).hexdigest()),
                    input_sha256=src['input_sha256'],frequency_range=[src['freq_start'],src['freq_end']],
                    playback_volumes={before_name[:-4]:target/before['rms'],after_name[:-4]:target/after['rms']})
        result.append(pair)
    comparison.update(pairs=result,processing='六组对比：处理前统一10ms Hann/60dB随机相位GL；处理后为各图既有预测参数/80带NAFNet/BigVGAN输出，播放时仅衰减较响侧。',
                      before_source_report_sha256=hashlib.sha256((defaults/'auditions.json').read_bytes()).hexdigest(),
                      additional_source_report_sha256=hashlib.sha256((additional/'auditions.json').read_bytes()).hexdigest())
    protect_pending(pending)
    for destination,payload in pending:destination.write_bytes(payload)
    manifest['files']=list(rows.values())
    (audio/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    path.write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps(dict(comparisons=len(result),copied=[p.name for p,_ in pending])))


def import_six_best(study):
    """Update the six existing after tracks from a verified research export."""
    study=study.resolve();report_path=study/'auditions.json'
    report=json.loads(report_path.read_text('utf8'))
    assets=ROOT/'temporary-site/assets';audio=assets/'audio';figures=assets/'spectrogram-examples'
    comparison=json.loads((figures/'comparison.json').read_text('utf8'))
    manifest=json.loads((audio/'provenance.json').read_text('utf8'))
    examples=json.loads((figures/'provenance.json').read_text('utf8'))
    cases={r['id']:r for r in report['cases']}
    assert len(cases)==6 and set(cases)=={r['id'] for r in comparison['pairs']}
    model={k:v for k,v in report['models']['base80'].items() if k!='path'}
    pending=[]
    for pair in comparison['pairs']:
        case=cases[pair['id']];selected=case['audio']['base80-repaired'];source=(study/selected['path']).resolve()
        assert source.is_relative_to(study)
        assert case['source']['input_sha256']==pair['input_sha256']
        assert case['seconds']==pair['seconds']
        payload=source.read_bytes();digest=hashlib.sha256(payload).hexdigest();assert digest==selected['sha256']
        assert hashlib.sha256((audio/pair['before']).read_bytes()).hexdigest()==pair['before_sha256']
        sr,pcm=wavfile.read(source);assert sr==24000 and pcm.ndim==1 and len(pcm)==round(case['seconds']*sr)
        assert pcm.dtype==np.int16
        values=pcm.astype(np.float64)/32768;level=float(np.sqrt(np.mean(values**2)))
        assert np.isfinite(values).all() and level>0 and np.max(np.abs(values))<.951
        p=case['parameter_prediction']
        assert case['applied_window_ms']==int(p['window_ms']+.5)
        processing=dict(route='参数预测 → GL → 训练后 80 带修复 → 声码器',engine='pulse GL / NAFNet 80 / BigVGAN',
                        parameter_prediction=p,model=model,gl=dict(report['gl'],hop=case['hop_samples'],
                        hop_ms=case['hop_ms'],window_samples=case['window_samples'],applied_window_ms=case['applied_window_ms']),
                        assumptions=dict(report['assumptions'],window_ms=case['applied_window_ms'],dynamic_range_db=p['dynamic_range_db']),
                        source_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                        source_output_sha256=digest,source_export=selected['path'],delivery='exact_listening_export')
        row=next(r for r in manifest['files'] if r['file']==pair['after'])
        row.update(parameters=processing,sha256=digest,sample_rate=sr,duration=len(pcm)/sr,peak=float(np.abs(values).max()),rms=level,
                   waveform=[round(float(np.max(np.abs(c))),4) for c in np.array_split(values,72)])
        for ex in examples['cases']:
            if ex['id']==pair['id']:ex.update(processing=processing,output_sha256=digest)
        before=next(r for r in manifest['files'] if r['file']==pair['before']);target=min(before['rms'],level)
        pair.update(after_sha256=digest,parameter_prediction=p,applied_window_ms=case['applied_window_ms'],
                    hop_ms=case['hop_ms'],repair_model_sha256=model['sha256'],repair_model_step=model['step'],
                    parameter_model_sha256=report['parameter_model_sha256'],
                    playback_volumes={pair['before'][:-4]:target/before['rms'],pair['after'][:-4]:target/level})
        pending.append((audio/pair['after'],payload))
    protect_pending(pending)
    for destination,payload in pending:destination.write_bytes(payload)
    comparison.update(parameter_model_sha256=report['parameter_model_sha256'],after_source_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                      processing='处理后使用当前最佳参数预测与80带修复权重，窗长四舍五入到整数毫秒后重建；处理前10ms/60dB随机GL保持。')
    for path,data in [(audio/'provenance.json',manifest),(figures/'provenance.json',examples),(figures/'comparison.json',comparison)]:
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(imported=[p.name for p,_ in pending],repair_step=model['step'])))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--six-best-study',type=Path)
    parser.add_argument('--from-pulse-study',type=Path,help='Import the existing trained 80-band listening exports')
    parser.add_argument('--comparison-from-pulse-study',type=Path,help='Import matched before/after listening exports for the next-step preview')
    parser.add_argument('--before-from-pulse-study',type=Path,help='Replace only before audio using the unknown-parameter default GL study')
    parser.add_argument('--six-defaults-study',type=Path,help='Import the six 10 ms / 60 dB random-phase GL baselines')
    parser.add_argument('--additional-comparisons',type=Path,help='Previously generated Lu and uploaded-figure v5 comparison study')
    args=parser.parse_args()
    if args.six_best_study:
        import_six_best(args.six_best_study)
        return
    if args.six_defaults_study:
        if not args.additional_comparisons:parser.error('--additional-comparisons is required')
        import_six_comparisons(args.six_defaults_study,args.additional_comparisons)
        return
    if args.before_from_pulse_study:
        import_comparison(args.before_from_pulse_study,before_only=True)
        return
    if args.comparison_from_pulse_study:
        import_comparison(args.comparison_from_pulse_study)
        return
    if args.from_pulse_study:
        import_pulse_study(args.from_pulse_study)
        return
    existing=ROOT/'temporary-site/assets/spectrogram-examples/provenance.json'
    if existing.exists() and any(r.get('processing') for r in json.loads(existing.read_text('utf-8'))['cases']):
        parser.error('Confirmed trained exports are protected; use --from-pulse-study with an explicit study directory.')
    out=ROOT/'temporary-site/assets/spectrogram-examples';out.mkdir(exist_ok=True)
    audio=ROOT/'temporary-site/assets/audio';manifest=json.loads((audio/'provenance.json').read_text('utf-8'))
    manifest['files']=[r for r in manifest['files'] if not r['file'].startswith('spectrum-')]
    records=[]
    for case in CASES:
        src=Path('C:/Users/13680/AppData/Local/Temp')/f'codex-clipboard-{case["clipboard"]}.png'
        original=src.read_bytes();im=Image.open(src);x0,y0,x1,y1=case['crop']
        gray=np.asarray(im.convert('L'))[y0:y1,x0:x1]
        gray=np.asarray(Image.fromarray(gray).resize((420,257),Image.Resampling.LANCZOS))
        # Match v2/v3 defaults: legacy /10 mapping makes -60 dB suppress pale frication.
        config=dict(time_start=0.,time_end=case['duration'],freq_start=0.,freq_end=float(case['frequency']),min_db=-30.,max_db=0.,win_length_ms=10.,n_iter=64,target_sr=16000,seed=0)
        result=reconstruct(gray,**config);a=result['audio'];a=a/max(1.,np.max(np.abs(a))/.65)
        assert np.isfinite(a).all() and np.std(a)>.00001
        name='spectrum-'+case['id']+'.wav';path=audio/name
        wavfile.write(path,16000,np.int16(np.clip(a,-1,1)*32767))
        # Figure excerpts retain axes for readers; plot-only data exclude axes and labels.
        im.crop(case['display']).save(out/(case['id']+'.png'))
        Image.fromarray(gray).save(out/(case['id']+'-input.png'))
        Image.fromarray(result['image']).save(out/(case['id']+'-result.png'))
        assert src.read_bytes()==original
        record={k:v for k,v in case.items() if k!='clipboard'}
        record.update(config=config,metadata=result['metadata'],input_sha256=hashlib.sha256(original).hexdigest(),output_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        records.append(record)
        manifest['files'].append(dict(file=name,description='语谱图图片近似重建：'+case['id'],parameters=dict(engine='PhoneticToolbox M09 Griffin–Lim',source=case['source'],url=case['url'],config=config),sample_rate=16000,duration=len(a)/16000,peak=float(np.max(np.abs(a))),rms=float(np.sqrt(np.mean(a*a))),sha256=record['output_sha256'],waveform=[round(float(np.max(np.abs(c))),4) for c in np.array_split(a,72)]))
    (audio/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n','utf-8')
    (out/'provenance.json').write_text(json.dumps(dict(scope='User-requested attributed figure excerpts illustrating magnitude-only reconstruction. No original recording or phase recovered. Copyright retained by the respective authors/publishers; no open redistribution license claimed.',cases=records),ensure_ascii=False,indent=2)+'\n','utf-8')
    print(json.dumps([dict(id=r['id'],duration=r['metadata']['duration']) for r in records]))

if __name__=='__main__':main()
