"""Build P01 using an explicit process-local search path; audit every native input."""
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if __name__ == '__main__':
    environment = os.environ.copy()
    system = Path(os.environ['SystemRoot'])
    # This does not change the machine or user PATH. Third-party tools such as Poppler
    # contain conflicting icuuc.dll names and must never enter this probe's analysis.
    environment['PATH'] = os.pathsep.join([str(Path(sys.executable).parent), str(system / 'System32'), str(system)])
    command = [sys.executable, '-X', 'utf8', '-m', 'PyInstaller', str(ROOT / 'desktop/experiments/p01-probe.spec'),
               '--distpath', str(ROOT / 'output/p01-probe'), '--workpath', str(ROOT / 'output/build/p01-clean'),
               '--noconfirm', '--log-level', 'WARN']
    output = ROOT / 'output/validation/p01'
    output.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / 'desktop/experiments/host_probe.py', ROOT / 'desktop/experiments/probe_paths.py',
              ROOT / 'desktop/experiments/p01-probe.spec',
              *[p for p in (ROOT / 'frontend/experiments/audio-viewport/dist').rglob('*') if p.is_file()]]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    with (output / 'build-clean.log').open('wb') as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, 'See build-clean.log'
    toc_path = ROOT / 'output/build/p01-clean/p01-probe/Analysis-00.toc'
    toc = ast.literal_eval(toc_path.read_text('utf-8'))
    binaries = []

    def walk(value):
        if isinstance(value, (list, tuple)):
            if len(value) == 3 and value[2] in ('BINARY', 'EXTENSION') and isinstance(value[1], str):
                binaries.append({'destination': value[0], 'source': value[1]})
            else:
                for item in value:
                    walk(item)
    walk(toc)
    forbidden = [b for b in binaries if not (Path(b['source']).resolve().is_relative_to(ROOT / '.venv')
                                             or Path(b['source']).resolve().is_relative_to(system))]
    artifact = ROOT / 'output/p01-probe/PhoneticToolbox-P01.exe'
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == value for p, value in hashes.items()), 'Build inputs changed during packaging'
    report = {'command': command, 'process_path': environment['PATH'], 'binary_inputs': binaries, 'input_sha256': hashes,
              'unexpected_binary_inputs': forbidden, 'exe_bytes': artifact.stat().st_size,
              'exe_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
    (output / 'build-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    assert not forbidden, forbidden
    print(json.dumps({key: report[key] for key in ['unexpected_binary_inputs', 'exe_bytes', 'exe_sha256']}))
