"""Fixed, standard-library bootstrap, executed by the OPTIONAL MFA interpreter.

Only this external process imports MFA. Source: SRC-MFA; aligner arguments match
the V2 pipeline. Paths/config belong to the host-created attempt, never web argv.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def write_json(path, value):
    target = Path(path)
    part = target.with_suffix('.part')
    part.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for attempt in range(40):
        try:
            os.replace(part,target)
            break
        except PermissionError:
            if attempt==39:raise
            # Windows readers/antivirus can briefly hold a sharing handle.
            time.sleep(.025)


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    root = Path(request['workspace'])
    runtime = Path(request['runtime'])
    root.mkdir(parents=True, exist_ok=True)
    for key in ('MFA_ROOT_DIR', 'TEMP', 'TMP', 'TMPDIR', 'JOBLIB_TEMP_FOLDER', 'NUMBA_CACHE_DIR'):
        folder = Path(request['kernel_cache']) if key=='NUMBA_CACHE_DIR' and request.get('kernel_cache') else root / ('mfa-root' if key == 'MFA_ROOT_DIR' else 'temp')
        folder.mkdir(exist_ok=True)
        os.environ[key] = str(folder)
    for key in ('BLAS_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'MFA_NUM_JOBS', 'CALC_JOBS', 'NUM_JOBS'):
        os.environ[key] = '1'
    # ADR-M11-001: V2's disabled JIT breaks current kalpy/librosa MFCC generation.
    # Enable it only here; the host may provide a cache of library kernels,
    # isolated by verified runtime content identity. Audio/state are never cached.
    os.environ['NUMBA_DISABLE_JIT'] = '0'
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'
    for key in ('PYTHONPATH', 'MFA_PROFILE', 'PGHOST', 'PGPORT', 'PGUSER', 'PGPASSWORD', 'PGDATABASE', 'DATABASE_URL', 'GITHUB_TOKEN', 'HF_TOKEN'):
        os.environ.pop(key, None)
    # This process's PATH only. Never activate/write the user's Conda environment.
    native = [runtime, runtime / 'Library/bin', runtime / 'Scripts', runtime / 'DLLs'] if os.name == 'nt' else [runtime / 'bin']
    system = [str(Path(os.environ['SystemRoot']) / 'System32')] if os.name == 'nt' else ['/usr/bin', '/bin']
    os.environ['PATH'] = os.pathsep.join([str(p) for p in native] + system)
    dll_handles = [os.add_dll_directory(str(p)) for p in native if p.is_dir()] if os.name == 'nt' else []
    log = (root / 'native.log').open('w', encoding='utf-8', buffering=1)
    sys.stdout = log
    sys.stderr = log
    started = time.monotonic()
    try:
        from importlib.metadata import version
        versions = {p: version(p) for p in ('montreal-forced-aligner', 'numpy', 'scipy', 'soundfile', 'praatio')}
        native_records = [json.loads(p.read_text(encoding='utf-8')) for p in (runtime / 'conda-meta').glob('*.json')]
        for name in ('kalpy', 'kaldi', 'ffmpeg', 'openfst'):
            match = [p for p in native_records if p['name'] == name]
            if len(match) != 1:
                raise ValueError('m11_dependency_missing')
            versions[name] = match[0]['version'] + '+' + match[0]['build']
        if versions['montreal-forced-aligner'] != '3.3.8':
            raise ValueError('m11_version_mismatch')
        from montreal_forced_aligner import config
        # Same default as observed V2 3.3.8. New root prevents reading old YAML,
        # shared caches or any account PostgreSQL credentials.
        config.USE_POSTGRES = False
        config.AUTO_SERVER = False
        config.NUM_JOBS = 1
        config.USE_MP = False
        # 3.3.8 run_kaldi_function still starts a worker with USE_MP=False.
        # Its default USE_THREADING=False spawns another Python per stage on
        # Windows. Keep one CPU job and use MFA's supported thread worker.
        config.USE_THREADING = True
        config.BLAS_NUM_THREADS = 1
        config.TEMPORARY_DIRECTORY = root / 'mfa_temp'
        if hasattr(config, 'GLOBAL_CONFIG'):
            config.GLOBAL_CONFIG.current_profile.use_postgres = False
            config.GLOBAL_CONFIG.current_profile.auto_server = False
            config.GLOBAL_CONFIG.current_profile.use_threading = True
            config.GLOBAL_CONFIG.current_profile.use_mp = False
            config.GLOBAL_CONFIG.current_profile.num_jobs = 1
            config.GLOBAL_CONFIG.current_profile.blas_num_threads = 1
        import _kalpy
        import soundfile
        from montreal_forced_aligner.alignment import PretrainedAligner
        write_json(root / 'status.json', dict(stage='runtime_checked', versions=versions))
        if request.get('action') == 'inspect':
            write_json(root / 'response.json', dict(success=True, versions=versions, python=sys.version, database='per-attempt-sqlite'))
            return
        corpus = root / 'corpus'
        output = root / 'output'
        for wav in corpus.rglob('*.wav'):
            info = soundfile.info(wav)
            if info.frames <= 0 or info.channels not in (1, 2) or info.duration > 120:
                raise ValueError('m11_audio_budget')
        from praatio import textgrid
        adaptations=[]
        for transcript in corpus.rglob('*.TextGrid'):
            tg=textgrid.openTextgrid(str(transcript),includeEmptyIntervals=True)
            usable=[tg.getTier(name) for name in tg.tierNames if name.lower()!='notes' and not name.endswith(('words','phones'))]
            if not any(isinstance(tier,textgrid.IntervalTier) and any(e.label.strip() for e in tier.entries) for tier in usable):
                words=[tg.getTier(name) for name in tg.tierNames if name.endswith('words') and isinstance(tg.getTier(name),textgrid.IntervalTier) and any(e.label.strip() for e in tg.getTier(name).entries)]
                if len(words)!=1:raise ValueError('m11_transcript_tiers')
                # Input is a host-owned copy. Preserve source files, and use
                # word labels as whole-recording text without old boundaries.
                labels=' '.join(e.label.strip() for e in words[0].entries if e.label.strip())
                converted=textgrid.Textgrid()
                converted.addTier(textgrid.IntervalTier('utterance',[(tg.minTimestamp,tg.maxTimestamp,labels)],minT=tg.minTimestamp,maxT=tg.maxTimestamp))
                converted.save(str(transcript),format='long_textgrid',includeBlankSpaces=True)
                adaptations.append(dict(file=transcript.relative_to(corpus).as_posix(),source_tier=words[0].name,mode='words-to-whole-recording-transcript'))
        timings={}
        diagnostics = root / 'diagnostics'
        diagnostics.mkdir(exist_ok=True)
        class CheckedAligner(PretrainedAligner):
            def dictionary_setup(self):
                begin=time.monotonic()
                try:return super().dictionary_setup()
                finally:timings['dictionary_seconds']=time.monotonic()-begin
            def normalize_text(self):
                begin=time.monotonic()
                try:
                    value=super().normalize_text()
                    # Native normalization is authoritative. Check before
                    # lexicon compilation and MFCC generation, not after setup.
                    if self.excluded_phones or self.excluded_pronunciation_count:raise ValueError('m11_model_mismatch')
                    self.save_oovs_found(str(diagnostics))
                    oov_files=[p for p in diagnostics.glob('oovs_found_*.txt') if p.stat().st_size]
                    if oov_files:
                        print('M11 未登录词（词典未收录）：',flush=True)
                        for p in oov_files:
                            print(p.read_text(encoding='utf8')[:4096],flush=True)
                        raise ValueError('m11_oov_words')
                    return value
                finally:timings['normalize_seconds']=time.monotonic()-begin
            def generate_features(self):
                begin=time.monotonic()
                try:return super().generate_features()
                finally:timings['features_seconds']=time.monotonic()-begin
        write_json(root / 'status.json', dict(stage='aligning', versions=versions))
        aligner = CheckedAligner(
            corpus_directory=str(corpus), dictionary_path=request['dictionary'],
            acoustic_model_path=request['model'], beam=request['config']['beam'], retry_beam=request['config']['retry_beam'])
        begin=time.monotonic()
        aligner.setup()
        timings['setup_seconds']=time.monotonic()-begin
        if aligner.excluded_phones or aligner.excluded_pronunciation_count:
            raise ValueError('m11_model_mismatch')
        aligner.save_oovs_found(str(diagnostics))
        if any(p.stat().st_size for p in diagnostics.glob('oovs_found_*.txt')):
            raise ValueError('m11_oov_words')
        begin=time.monotonic()
        aligner.align()
        timings['alignment_seconds']=time.monotonic()-begin
        write_json(root / 'status.json', dict(stage='exporting', versions=versions))
        aligner.export_files(str(output))
        # Parse actual output; a zero exit or empty output directory is insufficient.
        from praatio import textgrid
        summaries = []
        for path in sorted(output.rglob('*.TextGrid')):
            tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=True)
            tiers = []
            labels = 0
            for name in tg.tierNames:
                tier = tg.getTier(name)
                entries = []
                for entry in tier.entries:
                    if len(entry) != 3 or entry[0] < 0 or entry[1] < entry[0] or entry[1] > tg.maxTimestamp + 1e-6:
                        raise ValueError('m11_invalid_textgrid')
                    entries.append(list(entry))
                    labels += bool(entry[2].strip())
                tiers.append(dict(name=name, entries=entries))
            if not tiers or not labels:
                raise ValueError('m11_empty_alignment')
            summaries.append(dict(name=path.relative_to(output).as_posix(), end=tg.maxTimestamp, tiers=tiers))
        if len(summaries) != request['expected_files']:
            raise ValueError('m11_incomplete_output')
        write_json(root / 'response.json', dict(success=True, versions=versions, textgrids=summaries,
                   elapsed_seconds=time.monotonic()-started, database='per-attempt-sqlite',timings=timings,
                   execution=dict(device='cpu',num_jobs=1,use_threading=True,use_mp=False,kernel_cache=bool(request.get('kernel_cache'))),transcript_adaptations=adaptations))
        write_json(root / 'status.json', dict(stage='complete', versions=versions))
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        kind = type(exc).__name__
        code = str(exc) if str(exc).startswith('m11_') and len(str(exc)) < 80 else {
            'ReadError': 'm11_model_mismatch',
            'BadZipFile': 'm11_model_mismatch',
            'DictionaryError': 'm11_dictionary_mismatch', 'PronunciationAcousticMismatchError': 'm11_model_mismatch',
            'KaldiProcessingError': 'm11_alignment_failed', 'NoAlignmentsError': 'm11_no_alignments',
            'CorpusError': 'm11_corpus_invalid', 'AcousticModelError': 'm11_model_mismatch',
            'ModuleNotFoundError': 'm11_dependency_missing', 'ImportError': 'm11_dependency_missing',
            'FeatureGenerationError': 'm11_feature_failed',
        }.get(kind, 'm11_execution_failed')
        write_json(root / 'response.json', dict(success=False, error=code, exception=kind,timings=locals().get('timings',{})))
    finally:
        log.close()


if __name__ == '__main__':
    main()
