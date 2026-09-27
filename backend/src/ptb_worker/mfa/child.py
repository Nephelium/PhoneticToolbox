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
        folder = root / ('mfa-root' if key == 'MFA_ROOT_DIR' else 'temp')
        folder.mkdir(exist_ok=True)
        os.environ[key] = str(folder)
    for key in ('BLAS_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'MFA_NUM_JOBS', 'CALC_JOBS', 'NUM_JOBS'):
        os.environ[key] = '1'
    # ADR-M11-001: V2's disabled JIT breaks current kalpy/librosa MFCC generation.
    # Enable it only here; the cache is confined to this attempt.
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
        if hasattr(config, 'GLOBAL_CONFIG'):
            config.GLOBAL_CONFIG.current_profile.use_postgres = False
            config.GLOBAL_CONFIG.current_profile.auto_server = False
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
        for transcript in corpus.rglob('*.TextGrid'):
            tg=textgrid.openTextgrid(str(transcript),includeEmptyIntervals=True)
            usable=[tg.getTier(name) for name in tg.tierNames if name.lower()!='notes' and not name.endswith(('words','phones'))]
            if not any(isinstance(tier,textgrid.IntervalTier) and any(e.label.strip() for e in tier.entries) for tier in usable):
                raise ValueError('m11_transcript_tiers')
        write_json(root / 'status.json', dict(stage='aligning', versions=versions))
        aligner = PretrainedAligner(
            corpus_directory=str(corpus), dictionary_path=request['dictionary'],
            acoustic_model_path=request['model'], output_directory=str(output),
            temporary_directory=str(root / 'mfa_temp'), clean=True, verbose=True,
            num_jobs=1, use_mp=False, beam=request['config']['beam'], retry_beam=request['config']['retry_beam'])
        aligner.setup()
        if aligner.excluded_phones or aligner.excluded_pronunciation_count:
            raise ValueError('m11_model_mismatch')
        diagnostics = root / 'diagnostics'
        diagnostics.mkdir(exist_ok=True)
        aligner.save_oovs_found(str(diagnostics))
        if any(p.stat().st_size for p in diagnostics.glob('oovs_found_*.txt')):
            raise ValueError('m11_oov_words')
        aligner.align()
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
                   elapsed_seconds=time.monotonic()-started, database='per-attempt-sqlite'))
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
        write_json(root / 'response.json', dict(success=False, error=code, exception=kind))
    finally:
        log.close()


if __name__ == '__main__':
    main()
