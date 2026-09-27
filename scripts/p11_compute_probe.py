"""Host-owned short profiling child; run only through P11 run_bounded.

Times real core calls and export calls separately, without changing their inputs.
This diagnostic is separate from HTTP latency runs and is not a new capability.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time


def main():
    profile_path, phase, fixture_path, folder = sys.argv[1:]
    profile = json.loads(Path(profile_path).read_text())
    sys.path[:0] = profile['sys_paths']
    os.environ['PTB_LINUX_RUNTIME_PROFILE'] = profile_path
    os.environ['MPLCONFIGDIR'] = profile['cache']
    os.environ['MPLBACKEND'] = 'Agg'
    from ptb_worker.native.linux_runtime import fingerprint
    runtime = fingerprint()
    from ptb_worker.native.linux_reaper import require_group
    require_group()
    import numpy as np
    from scipy.io import wavfile
    from matplotlib import font_manager
    for font in profile['fonts']: font_manager.fontManager.addfont(font)
    with np.load(fixture_path) as data:
        samples = np.column_stack([data['load.egg_signal_raw'],data['load.audio_signal']])
    stream = io.BytesIO(); wavfile.write(stream,44100,samples[:,1] if phase=='acoustic' else samples)
    raw = stream.getvalue()
    timings = {'compute':[], 'export':[]}

    def timed(group, function, *args, **kwargs):
        start=time.perf_counter()
        try: return function(*args, **kwargs)
        finally: timings[group].append(dict(function=function.__module__+'.'+function.__name__,seconds=time.perf_counter()-start))

    def wrap(module, name, group):
        original=getattr(module,name)
        def measured(*args,**kwargs): return timed(group,original,*args,**kwargs)
        setattr(module,name,measured)

    font=dict(zh='Noto Sans SC',latin='DejaVu Sans')
    start=time.perf_counter()
    if phase=='lpc':
        import phonetic_core.lpc as core
        import ptb_worker.lpc_exports as exports
        wrap(core,'compute_spectrum','compute'); wrap(exports,'plot_bytes','export')
        from ptb_worker.lpc_child import prepare
        output=prepare(raw,dict(roi_end=.05,font=font),'public-stereo.wav')
    elif phase=='egg':
        import phonetic_core.egg as core
        import phonetic_core.egg.f0 as f0
        import ptb_worker.egg_exports as exports
        wrap(core,'prepare','compute'); wrap(core,'analyze_events','compute'); wrap(f0,'praat_pitch','compute')
        wrap(exports,'csv_bytes','export'); wrap(exports,'plot_bytes','export')
        from ptb_worker.egg_child import prepare
        output=prepare(raw,dict(mode='single',roi_end=.5,font=font),'public-stereo.wav')
    elif phase=='acoustic':
        from ptb_worker.io.audio import decode_wav
        from ptb_worker.io.limits import Limits
        from ptb_worker.managed_scratch import ReservedNativeScratch
        from ptb_worker.native.reaper import Reaper
        from ptb_worker.acoustic_result import to_core_config
        from ptb_worker.io.parameter_exports import table_from_frame,_build_pair
        from ptb_api.acoustic_models import AcousticConfigSnapshot
        from phonetic_core.models.associations import AcousticAssociations
        from phonetic_core.ports.acoustic import AcousticBackends
        from phonetic_core.services.acoustic import analyze_audio
        from phonetic_core.catalog import PARAMETER_MAPPING
        limits=Limits(input_bytes=64_000_000,output_bytes=64_000_000,process_bytes=1_000_000_000,timeout_seconds=240)
        audio=decode_wav(raw,limits)
        native_path=Path(folder)/'native.wav'
        with native_path.open('xb'): pass
        with ReservedNativeScratch(native_path,16_000_000) as scratch:
            native=Reaper(profile['reaper_binary'],scratch,limits)
            result=timed('compute',analyze_audio,audio,to_core_config(AcousticConfigSnapshot()),
                         AcousticAssociations(),AcousticBackends(reaper=native))
            assert any(e.get('actual')=='native_reaper' for e in result.backend_events)
            table=table_from_frame(result.to_dataframe().rename(columns=PARAMETER_MAPPING),limits)
            pair=timed('export',_build_pair,table,limits)
            output=pair.xlsx+pair.sqlite
    else:
        raise ValueError('Unsupported diagnostic phase')
    report=dict(phase=phase,runtime=runtime,input_sha256=hashlib.sha256(raw).hexdigest(),
                output_sha256=hashlib.sha256(output).hexdigest(),output_bytes=len(output),
                child_work_seconds=time.perf_counter()-start,timings=timings,
                pure_compute_seconds=sum(x['seconds'] for x in timings['compute']),
                export_seconds=sum(x['seconds'] for x in timings['export']))
    sys.stdout.buffer.write(json.dumps(report,allow_nan=False).encode())


if __name__=='__main__':
    # Core diagnostics can print; keep stdout's result protocol unambiguous.
    output=sys.stdout
    sys.stdout=sys.stderr
    try:
        # main's last write is redirected separately after work by this proxy.
        class Output:
            buffer=output.buffer
            def write(self,value): return sys.stderr.write(value)
            def flush(self): return sys.stderr.flush()
        sys.stdout=Output()
        main()
    finally:
        sys.stdout=output
