"""Installed-wheel M03 parity from original PCM16/FLOAT inputs and frozen v2 evidence."""
import argparse
import dataclasses
import importlib.metadata
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from baseline_support import sha, write_json
from capture_m03_baseline import fixtures
import phonetic_core.egg as egg
from phonetic_core.egg.errors import EggError
from phonetic_core.egg.metrics import calculate_cq_sq
from phonetic_core.egg.f0 import praat_pitch
from phonetic_core.egg.inverse import inverse_filter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-private', action='store_true')
    args = parser.parse_args()
    module_path = Path(egg.__file__).resolve()
    assert module_path.is_relative_to(Path(sys.prefix).resolve()), 'Must verify installed wheel, not source tree'
    output = ROOT/'output/validation/m03-core'/('wheel-'+uuid.uuid4().hex)
    output.mkdir(parents=True)
    inputs = output/'inputs'; inputs.mkdir()
    cases = fixtures(inputs)
    manifest = json.loads((ROOT/'tests/fixtures/m03/manifest.json').read_text('utf-8'))
    if args.include_private:
        confirmed = json.loads((ROOT/'output/validation/p03/confirmed-egg.json').read_text('utf-8'))
        cases += [dict(c, privacy='private_local_only') for c in confirmed['cases']]
    records = []
    total = 0

    def compare(actual, old, arrays):
        nonlocal total
        if isinstance(old, dict) and 'array' in old:
            expected = arrays[old['array']]
            assert isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
            assert actual.tobytes() == expected.tobytes(), 'Array bytes differ, including signed zero or NaN payload'
            total += 1
        elif isinstance(old, (list, tuple)):
            assert len(actual) == len(old)
            for a,b in zip(actual,old): compare(a,b,arrays)
        else:
            assert actual == old
            total += 1

    for case in cases:
        record = next(c for c in manifest['public_cases']+manifest['private_cases'] if c['id']==case['case_id'])
        assert sha(case['input']) == record['input_sha256']
        private = case['privacy']=='private_local_only'
        meta_path = ROOT/record['capture']/'result.json' if private else ROOT/'tests/fixtures/m03'/record['metadata']
        array_path = ROOT/record['capture']/'arrays.npz' if private else ROOT/'tests/fixtures/m03'/record['arrays']
        assert sha(meta_path)==record['metadata_sha256'] and sha(array_path)==record['arrays_sha256']
        old = json.loads(meta_path.read_text('utf-8'))
        with np.load(array_path,allow_pickle=False) as archive:
            arrays=dict(archive)
        error = None
        if case['case_id'].endswith('BROKEN'):
            # File decoding remains the adapter's responsibility, not an array-core API.
            try: wavfile.read(case['input'])
            except ValueError: error='adapter_invalid_wav'
            assert error
        else:
            fs,samples=wavfile.read(case['input'])
            before=samples.copy()
            try: result=egg.prepare(samples,fs,egg.EGGConfig(),flip_channels=case.get('flip_channels',False))
            except EggError as exc: error=exc.code
            if error:
                assert old['load']['status']=='error' or case['case_id'].endswith('SHORT'), (case['case_id'],error)
            else:
                assert old['load']['status']=='returned'
                for key in ('time_vector','egg_signal_raw','egg_signal_processed','audio_signal'):
                    compare(getattr(result,key),{'array':'load.'+key},arrays)
                assert result.sample_duration_s == len(samples)/fs
                assert result.last_sample_time_s == (len(samples)-1)/fs
                for variant in old['variants']:
                    cfg=egg.EGGConfig(**variant['config'])
                    current=egg.analyze_events(result,cfg)
                    compare((current.gci_times,current.goi_times,current.peak_times),variant['events'],arrays)
                    compare((current.gci_f0_times,current.gci_f0_values),variant['gci_f0'],arrays)
                    compare(calculate_cq_sq(current.gci_times,current.goi_times,current.peak_times),variant['global_cq'],arrays)
                    for modes in variant['roi'].values():
                        for mode,item in modes.items():
                            for fn,key in ((egg.cq_segment,'cq'),(egg.events_segment,'events')):
                                compare(fn(current,*item['bounds_s'],cfg,use_raw_signal=mode=='raw'),item[key]['value'],arrays)
                pitch=praat_pitch(result.audio_signal,fs)
                compare(pitch.values,old['praat']['legacy']['values'],arrays)
                compare(pitch.times,old['praat']['actual']['value']['times'],arrays)
                compare(pitch.legacy_times,old['praat']['legacy']['times'],arrays)
                events=egg.analyze_events(result,egg.EGGConfig.for_workbench()).gci_times
                count=min(len(result.audio_signal),int(.12*fs))
                gci=np.asarray([t for t in events if t<count/fs])
                for label,order in [('auto',None),('explicit',12)]:
                    expected=old['inverse'][label]['value']
                    if expected is None:
                        try: inverse_filter(result.audio_signal[:count],fs,gci,lp_order=order)
                        except EggError as exc: assert exc.code=='inverse_unavailable'
                        else: raise AssertionError('Legacy unavailable IF changed to success')
                    else:
                        compare(inverse_filter(result.audio_signal[:count],fs,gci,lp_order=order),expected,arrays)
            np.testing.assert_array_equal(samples,before)
        assert sha(case['input'])==record['input_sha256']
        records.append({'id':case['case_id'],'privacy':case['privacy'],'input_unchanged':True,
                        'result':'explicit_error' if error else 'exact_parity','error':error})
    assert not any(n.startswith(('phonetic_toolbox','PyQt','fastapi','matplotlib','pywt')) for n in sys.modules)
    report={'status':'verified','scope':'Windows installed core wheel only','cases':records,
            'exact_comparisons':total,'core_module':module_path.relative_to(Path(sys.prefix)).as_posix(),
            'core_version':importlib.metadata.version('phonetic-core'),
            'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','scipy','praat-parselmouth']}}
    write_json(output/'report.json',report)
    print(json.dumps({'report':str(output/'report.json'),'cases':len(records),'exact_comparisons':total}))


if __name__=='__main__': main()
