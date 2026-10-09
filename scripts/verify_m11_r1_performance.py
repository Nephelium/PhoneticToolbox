"""M11-R1 timings and size inventory. Existing environment, no build/install."""
import argparse
import json
import time
from pathlib import Path
from uuid import uuid4
import shutil
import zipfile

from ptb_worker.mfa.runtime import fingerprint, run
from ptb_worker.mfa.probe import generate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--runtime', default=str(ROOT / 'output/m11c-028b881d/versions/mfa338-abbae3a9da0c46c7'))
    parser.add_argument('--oov', action='store_true')
    parser.add_argument('--fingerprint', action='store_true')
    parser.add_argument('--size-only', action='store_true')
    parser.add_argument('--kernel-cache', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'output/validation/m11-r1' / (args.label + '-' + uuid4().hex)
    out.mkdir(parents=True)
    report = {'label': args.label, 'scope': 'public 2.8 s generated audio; existing Windows MFA 3.3.8', 'stages': []}
    print(str(out), flush=True)
    if args.size_only:
        report['scope']='read-only existing Windows MFA 3.3.8 runtime and existing archive; excludes dictionaries/acoustic models; no packaging'
        runtime=Path(args.runtime)
        files=[p for p in runtime.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ptb-component.json']
        archive=ROOT/'output/validation/m11/component-a4159343dbc44af180e2f2e361666ef9/mfa-3.3.8-windows-x86_64-candidate.zip'
        with zipfile.ZipFile(archive) as z:
            entries=[i for i in z.infolist() if not i.is_dir()]
            archive_installed_bytes=sum(i.file_size for i in entries)
            cached_resources=[i.filename for i in entries if 'pretrained_models' in i.filename]
        report.update(installed_bytes=sum(p.stat().st_size for p in files),files=len(files),existing_archive_bytes=archive.stat().st_size,
                      archive_installed_bytes=archive_installed_bytes,archive_files=len(entries),archive_pretrained_resources=cached_resources,
                      method='read-only current inventory and existing 2026-09-27 ZIP central directory; no archive created')
        records=[json.loads(p.read_text('utf8')) for p in (runtime/'conda-meta').glob('*.json')]
        report['packages']=len(records)
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(report,ensure_ascii=False),flush=True)
        return
    if args.fingerprint or args.kernel_cache:
        started = time.perf_counter()
        report['fingerprint'] = fingerprint(args.runtime)
        report['fingerprint_seconds'] = time.perf_counter() - started
        print(json.dumps({'fingerprint_seconds': report['fingerprint_seconds']}), flush=True)
    generate(out / 'corpus', word='not_a_known_word' if args.oov else '啊')
    model = out / 'model.zip'
    dictionary = out / 'dictionary.dict'
    shutil.copyfile(Path.home() / 'Documents/MFA/pretrained_models/acoustic/mandarin_mfa.zip', model)
    shutil.copyfile(Path.home() / 'Documents/MFA/pretrained_models/dictionary/mandarin_mfa.dict', dictionary)
    resources = {}
    started = time.perf_counter()
    def progress(stage):
        row = {'stage': stage, 'seconds': time.perf_counter() - started}
        report['stages'].append(row)
        print(json.dumps(row), flush=True)
    try:
        request=dict(action='align', model=str(model), dictionary=str(dictionary), config=dict(beam=100, retry_beam=400), expected_files=1)
        if args.kernel_cache:request['kernel_cache_key']=report['fingerprint']
        report['result'] = run(args.runtime, out, request, evidence=resources, progress=progress)
    except Exception as exc:
        report['result'] = dict(success=False, error=str(exc))
    report['resources'] = resources
    report['elapsed_seconds'] = time.perf_counter() - started
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'out': str(out), 'success': report['result']['success'], 'error': report['result'].get('error'), 'elapsed_seconds': report['elapsed_seconds'], 'resources': resources}), flush=True)


if __name__ == '__main__':
    main()
