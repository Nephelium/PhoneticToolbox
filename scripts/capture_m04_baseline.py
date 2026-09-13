"""M04-A: test-only original V2 capture. No product imports use this path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['core/acoustic/lpc.py', 'services/lpc_service.py', 'models/lpc_models.py',
           'gui/widgets/lpc_spectrum_widget.py', 'services/io/wav.py', 'services/io/textgrid.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def capture(source, output):
    assert sys.dont_write_bytecode
    sys.path.insert(0, str(source))
    import dataclasses
    import warnings
    import numpy as np
    import scipy
    from scipy.io import wavfile
    from PIL import Image
    from phonetic_toolbox.models.lpc_models import LPCSpectrumConfig
    from phonetic_toolbox.services.lpc_service import LPCSpectrumService
    from phonetic_toolbox.services.io.textgrid import TextGrid, Tier, Interval, write_textgrid
    from phonetic_toolbox.gui.widgets.lpc_spectrum_widget import LPCSpectrumWidget

    service = LPCSpectrumService()
    assert Path(sys.modules[LPCSpectrumService.__module__].__file__).resolve().is_relative_to(source)
    arrays, results = {}, {}
    rng = np.random.default_rng(404)
    noise = rng.normal(0, .15, 2400)
    t = np.arange(1600)/16000
    voice = np.sin(2*np.pi*220*t)*.4 + np.sin(2*np.pi*880*t)*.1 + noise[:1600]*.01
    cases = [
        ('default', voice, 16000, LPCSpectrumConfig()),
        ('dynamic_2k', voice, 16000, LPCSpectrumConfig(freq_max_hz=2000, dynamic_y=True)),
        ('dynamic_48k', voice, 16000, LPCSpectrumConfig(freq_max_hz=48000, dynamic_y=True)),
        ('noise_order1', noise[:800], 8000, LPCSpectrumConfig(order=1)),
        ('noise_order200', noise, 48000, LPCSpectrumConfig(order=200)),
        ('noise_96k', noise, 96000, LPCSpectrumConfig(order=50)),
        ('constant', np.ones(800)*.1, 16000, LPCSpectrumConfig()),
        ('silence', np.zeros(800), 16000, LPCSpectrumConfig()),
        ('short51', noise[:51], 16000, LPCSpectrumConfig()),
        ('minimum52', noise[:52], 16000, LPCSpectrumConfig()),
        ('nan', np.full(800, np.nan), 16000, LPCSpectrumConfig()),
        ('infinity', np.full(800, np.inf), 16000, LPCSpectrumConfig()),
    ]
    export_result = None
    for name, audio, rate, config in cases:
        arrays[name+'.input'] = audio.copy()
        entry = {'rate': rate, 'config': dataclasses.asdict(config)}
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter('always')
            try:
                result = service.compute_spectrum(audio, rate, config)
                arrays[name+'.frequency'] = result.frequencies_hz
                arrays[name+'.db'] = result.magnitude_db
                entry.update(status='returned', y_range=[result.amp_min_db, result.amp_max_db],
                             finite=bool(np.isfinite(result.magnitude_db).all()))
                if name == 'default':
                    export_result = result
            except Exception as exc:
                entry.update(status='error', error_type=type(exc).__name__, message=str(exc))
            entry['warnings'] = sorted(set(str(w.message) for w in observed))
        assert np.array_equal(audio, arrays[name+'.input'], equal_nan=True)
        results[name] = entry

    pcm = np.round(voice*32767).astype(np.int16)
    stereo = np.column_stack((pcm, -pcm//2))
    conversions = {'pcm16': pcm, 'stereo16': stereo, 'pcm32': pcm.astype(np.int32)*65536,
                   'uint8': np.array([0, 64, 128, 192, 255], dtype=np.uint8),
                   'float32': voice.astype(np.float32)}
    for name, samples in conversions.items():
        path = output/(name+'.wav')
        wavfile.write(path, 16000, samples)
        before = sha(path)
        rate, mono = service.load_audio(path)
        assert rate == 16000 and sha(path) == before
        arrays['wav.'+name+'.raw'] = samples
        arrays['wav.'+name+'.mono'] = mono
    tg = TextGrid(0., 1., [Tier('phones', 0., 1., [Interval(0., .2, 'ɑ̃˥'),
        Interval(.2, .4, 'b'), Interval(.4, .6, ''), Interval(.6, .8, 'b'),
        Interval(.8, 1., '末')]), Tier('words', 0., 1., [Interval(0., 1., '词')])])
    write_textgrid(tg, output/'pcm16.TextGrid')
    read = service.read_sibling_textgrid(output/'pcm16.wav')
    labels = {f'{a}:{b}': service.extract_label_in_range(read, 'phones', a, b)
              for a, b in [(0., 1.), (.15, .85), (.8, 1.), (.1, .2), (.2, .4), (.4, .6)]}
    tiers = [service.next_tier_name(read, name) for name in [None, 'phones', 'words', 'missing']]
    assert service.read_sibling_textgrid(output/'float32.wav') is None
    fig = LPCSpectrumWidget._create_export_figure(None, export_result, 8000)
    export = service.save_plot_figure(fig, output, '合成:声', 'ɑ̃˥+b')
    with Image.open(export) as image:
        png = {'size': list(image.size), 'dpi': list(image.info['dpi']), 'name': export.name,
               'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest()}
    assert png['size'] == [2400, 1350]
    assert np.array_equal(arrays['default.db'], arrays['dynamic_2k.db'])
    assert np.array_equal(arrays['default.db'], arrays['dynamic_48k.db'])
    # freqz converts radians to Hz: the analytic grid can round by one ULP.
    # The independent repeat comparison below still requires identical bytes.
    assert abs(arrays['default.frequency'][-1] - 8000*(1023/1024)) <= np.spacing(8000.)
    assert labels['0.15:0.85'] == 'ɑ̃˥+b' and labels['0.4:0.6'] == ''
    np.savez_compressed(output/'arrays.npz', **arrays)
    write_json(output/'result.json', {'producer': 'original-v2', 'numpy': np.__version__,
        'scipy': scipy.__version__, 'defaults': dataclasses.asdict(LPCSpectrumConfig()),
        'cases': results, 'labels': labels, 'tiers': tiers, 'png': png,
        'array_count': len(arrays), 'value_count': sum(x.size for x in arrays.values())})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=Path)
    parser.add_argument('--source', type=Path, default=ROOT.parent/'PhoneticToolbox_v2')
    parser.add_argument('--freeze-public', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    if args.child:
        folder = args.child.resolve()
        assert folder.is_relative_to(ROOT/'output/validation/m04') and not folder.is_relative_to(source)
        capture(source, folder)
        return
    context = json.loads((ROOT/'output/validation/p03/context-before.json').read_text('utf-8'))
    python = Path(context['v2_environment'])/'python.exe'
    assert python.is_file()
    files = [source/'phonetic_toolbox'/p for p in SOURCES] + [source/'Phonetic_Export/index.html']
    before = {p.relative_to(source).as_posix(): sha(p) for p in files}
    run = ROOT/'output/validation/m04'/('baseline-'+uuid.uuid4().hex)
    run.mkdir(parents=True)
    for number in [1, 2]:
        folder = run/str(number)
        folder.mkdir()
        prefix = python.parent
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', QT_QPA_PLATFORM='offscreen',
                   MPLCONFIGDIR=str(folder/'mplconfig'), TEMP=str(folder), TMP=str(folder),
                   PATH=os.pathsep.join([str(prefix), str(prefix/'Library/bin'), str(prefix/'Scripts'), os.environ['PATH']]))
        with (folder/'worker.log').open('w', encoding='utf-8') as log:
            result = subprocess.run([str(python), '-B', '-X', 'utf8', str(Path(__file__).resolve()),
                '--child', str(folder), '--source', str(source)], cwd=folder, env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=180, creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            raise RuntimeError(f'Original V2 capture failed: {folder / "worker.log"}')
    import numpy as np
    a, b = [json.loads((run/str(n)/'result.json').read_text('utf-8')) for n in [1, 2]]
    assert a == b, 'Repeat metadata/pixels differ'
    with np.load(run/'1/arrays.npz') as first, np.load(run/'2/arrays.npz') as second:
        assert first.files == second.files
        for key in first.files:
            assert first[key].shape == second[key].shape and first[key].dtype == second[key].dtype and first[key].tobytes() == second[key].tobytes(), key
    assert before == {p.relative_to(source).as_posix(): sha(p) for p in files}
    manifest = {'source_hashes': before, 'repeat_equal': True, 'source_unchanged': True,
                'privacy': 'public_synthetic_only', 'capture': run.relative_to(ROOT).as_posix(),
                'result_sha256': sha(run/'1/result.json'), 'arrays_sha256': sha(run/'1/arrays.npz')}
    write_json(run/'manifest.json', manifest)
    if args.freeze_public:
        frozen = ROOT/'tests/fixtures/m04'
        frozen.mkdir(parents=True, exist_ok=True)
        # Never overwrite an existing baseline silently.
        for name in ['arrays.npz', 'result.json', 'manifest.json']:
            source_file = run/'1'/name if name != 'manifest.json' else run/name
            target = frozen/name
            assert not target.exists(), f'Baseline already exists: {target}'
            target.write_bytes(source_file.read_bytes())
    print(json.dumps({'output': str(run), 'cases': len(a['cases']), 'arrays': a['array_count'],
                      'values': a['value_count'], 'repeat_equal': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
