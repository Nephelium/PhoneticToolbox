"""Compare cache behavior and exact outputs against the specified frozen EGG code."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from verification_artifacts import verification_run, write_json
from workbench_source import bind_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    bind_sources()
    prefix = Path(sys.prefix)
    dll = os.add_dll_directory(str(prefix/'Library/bin'))
    os.environ['PATH'] = str(prefix/'Library/bin')+os.pathsep+os.environ['PATH']
    import numpy as np
    from phonetic_core.egg import EGGConfig, prepare, analyze_events
    from ptb_api.egg_models import EggTaskConfig
    from ptb_worker import egg_preview
    spec = importlib.util.spec_from_file_location('ptb_worker.egg_preview_before', args.reference)
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    with verification_run('redundant-work', 'science', '1s 16kHz two-channel deterministic sine; ROI 0.1-0.4s; original frozen and current EGG preview at two display dB ranges; arrays/PNG bytes compared, no WAV files') as (out, scratch):
        fs = 16000
        time = np.arange(fs)/fs
        signal = np.column_stack((.6*np.sin(2*np.pi*120*time), .5*np.sin(2*np.pi*120*time+.3)))
        config = EGGConfig.for_workbench()
        result = analyze_events(prepare(signal, fs, config), config)
        settings = EggTaskConfig(mode='preview', roi_start=.1, roi_end=.4, micro_center=.2, keep_praat_f0=False, keep_reaper_f0=False)
        altered = settings.model_copy(update=dict(spec_vmin=-75., spec_vmax=-25.))
        report = dict(success=False, reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(), current_sha256=hashlib.sha256(Path(egg_preview.__file__).read_bytes()).hexdigest(), cases={})
        outputs = {}
        for name, module in [('frozen', before), ('current', egg_preview)]:
            counts = dict(spectrum=0, cq=0)
            original_spectrum, original_cq = module.spectral_series, module.cq_segment

            def spectrum(*args, **kwargs):
                counts['spectrum'] += 1
                return original_spectrum(*args, **kwargs)

            def cq(*args, **kwargs):
                counts['cq'] += 1
                return original_cq(*args, **kwargs)

            module.spectral_series, module.cq_segment = spectrum, cq
            cache = {}
            try:
                outputs[name] = [module.preview_files(result, config, s, 1600, 6400, cache=cache, include_audio=False) for s in (settings, altered)]
            finally:
                module.spectral_series, module.cq_segment = original_spectrum, original_cq
            report['cases'][name] = counts
        assert outputs['frozen'] == outputs['current'], 'display cache changed scientific data or PNG bytes'
        assert report['cases']['frozen'] == dict(spectrum=2, cq=2)
        assert report['cases']['current'] == dict(spectrum=1, cq=1)
        report['output_sha256'] = [dict(data=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(), png=hashlib.sha256(blobs['egg_PSD.png']).hexdigest()) for data, blobs in outputs['current']]
        report['success'] = True
        write_json(out/'report.json', report)
        print(json.dumps(dict(out=str(out), **report)), flush=True)


if __name__ == '__main__':
    main()
