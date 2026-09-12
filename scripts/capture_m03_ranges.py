"""Independent original-V2 micro range evidence; public synthetic data only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
CASES = [('min', 3.2, 5), ('max', 3.2, 5000), ('start', 0., 5),
         ('end', 6.4, 5), ('wide_start', 0., 5000), ('wide_end', 6.4, 5000)]


def worker(out):
    assert sys.dont_write_bytecode
    source = ROOT.parent/'PhoneticToolbox_v2'
    sys.path.insert(0, str(source))
    import numpy as np
    from scipy.io import wavfile
    from PyQt6.QtWidgets import QApplication
    from phonetic_toolbox.gui.widgets.egg_widget import EGGWidget
    from phonetic_toolbox.models.config import EGGConfig
    from phonetic_toolbox.services.egg_service import EGGAnalysisService
    with np.load(ROOT/'tests/fixtures/m03/EGG-SYN-PCM16.npz') as a:
        samples = np.tile(np.column_stack([a['load.egg_signal_raw'], a['load.audio_signal']]), (8, 1))
    wav = out/'input.wav'; wavfile.write(wav, 44100, samples)
    config = EGGConfig(); config.gci_method = 'slope'; config.goi_method = 'scale'
    service = EGGAnalysisService(); result = service.analyze_events(service.load_file(str(wav), config), config)
    app = QApplication(['M03-range-baseline']); widget = EGGWidget(); widget.result = result; widget.config = config
    arrays = {}; steps = {}
    for raw in (False, True):
        widget.show_filtered_egg = not raw
        for label, center, width in CASES:
            widget.zoom_duration_input.setText(str(width)); widget.update_zoom_plots(center)
            key = f'{label}.{raw}'
            for name, axis in [('egg', widget.egg_zoom_ax), ('audio', widget.audio_zoom_ax)]:
                arrays[key+'.'+name] = axis.lines[0].get_ydata()
                arrays[key+'.'+name+'_ms'] = axis.lines[0].get_xdata()
            first = max(0, int((center-width/2000)*44100)); last = min(len(samples), int((center+width/2000)*44100))
            steps[key] = widget._get_downsampling_step(last-first)
    np.savez_compressed(out/'ranges.npz', **arrays)
    evidence = dict(source='PENDING-EGG', source_sha256=hashlib.sha256((source/'phonetic_toolbox/gui/widgets/egg_widget.py').read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(wav.read_bytes()).hexdigest(), recipe='EGG-SYN-PCM16 normalized stereo repeated 8 times, FLOAT32 WAV 44100Hz', cases=CASES, strides=steps)
    (out/'ranges-source.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    widget.close(); app.processEvents()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker': return worker(Path(sys.argv[2]))
    import numpy as np
    import shutil
    out = ROOT/'output/validation/m03-e3'/('v2-ranges-'+uuid4().hex); out.mkdir(parents=True)
    prefix = Path(json.loads((ROOT/'output/validation/p03/context-before.json').read_text('utf-8'))['v2_environment'])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', QT_QPA_PLATFORM='offscreen', MPLCONFIGDIR=str(out/'mpl'))
    env['PATH'] = os.pathsep.join([str(prefix), str(prefix/'Library/bin'), str(prefix/'Scripts'), env.get('PATH','')])
    for i in (1,2):
        target = out/str(i); target.mkdir()
        with (target/'capture.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(prefix/'python.exe'),'-B','-X','utf8',str(Path(__file__).resolve()),'--worker',str(target)],
                cwd=target, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120, creationflags=subprocess.CREATE_NO_WINDOW)
    with np.load(out/'1/ranges.npz') as a, np.load(out/'2/ranges.npz') as b:
        assert a.files==b.files
        for key in a.files: np.testing.assert_array_equal(a[key], b[key])
    assert (out/'1/ranges-source.json').read_bytes()==(out/'2/ranges-source.json').read_bytes()
    for name in ('ranges.npz','ranges-source.json'):
        target = ROOT/'tests/fixtures/m03'/name
        if target.exists(): raise RuntimeError('Existing baseline preserved: '+str(target))
        shutil.copyfile(out/'1'/name, target)
    print(out)


if __name__=='__main__': main()
