"""P02 reproducible acceptance with independent wheel installation and owned processes."""
import datetime
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
OUTPUT = ROOT / 'output/validation/p02' / STAMP


def run(name, args, cwd=ROOT, timeout=180):
    result = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=timeout,
        env={**os.environ, 'PYTHONUTF8': '1', 'UV_LINK_MODE': 'copy'})
    (OUTPUT / (name + '.log')).write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    print(f'{name}: exit {result.returncode}', flush=True)
    if result.returncode:
        raise RuntimeError(name + ' failed; see ' + str(OUTPUT / (name + '.log')))
    return result.stdout


def server_probe(python):
    process = subprocess.Popen([str(python), '-m', 'ptb_api.cli', '--mode', 'server', '--managed'],
        cwd=OUTPUT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        lines = queue.Queue()
        threading.Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True).start()
        url = json.loads(lines.get(timeout=15))['url']
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + '/api/v1/health', timeout=5) as response:
            health = json.load(response)
        assert health['mode'] == 'server' and health['status'] == 'ok'
        process.stdin.write('shutdown\n')
        process.stdin.flush()
        process.wait(timeout=10)
        assert process.returncode == 0
        return {'health': health, 'exit_code': process.returncode}
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()
        (OUTPUT / 'server-stderr.log').write_text(process.stderr.read(), encoding='utf-8')
        process.stderr.close()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    python = Path(sys.executable)
    npm = 'npm.cmd' if os.name == 'nt' else 'npm'
    run('versions', [python, 'scripts/sync_versions.py', '--check'])
    run('schemas', [python, 'scripts/generate_contracts.py', '--check'])
    run('architecture', [python, 'scripts/check_architecture.py'])
    run('docs', [python, 'scripts/validate_docs.py'])
    for command in ['contracts:check', 'typecheck', 'test', 'build']:
        run('frontend-' + command.replace(':', '-'), [npm, '--prefix', 'frontend', 'run', command])
    run('npm-audit', [npm, '--prefix', 'frontend', 'audit', '--json'])
    run('pytest-dev', [python, '-m', 'pytest', '-c', 'tests/pytest.ini', 'tests/contracts',
                       'tests/architecture', 'backend/tests', 'desktop/tests', '-q'])
    wheels = OUTPUT / 'wheels'
    for folder in ['packages/phonetic_core', 'backend', 'desktop']:
        run('wheel-' + Path(folder).name, [python, '-m', 'build', '--wheel', '--no-isolation', '--outdir', wheels, folder])
    wheel_files = list(wheels.glob('*.whl'))
    assert len(wheel_files) == 3
    for path in wheel_files:
        with zipfile.ZipFile(path) as archive:
            assert not any('phonetic_toolbox/' in name or '/experiments/' in name for name in archive.namelist())
    clean = ROOT / '.venv' / ('p02-clean-' + STAMP)
    runtime = Path(sys.base_prefix) / 'python.exe'
    run('clean-venv', ['uv', 'venv', '--python', runtime, clean])
    clean_python = clean / 'Scripts/python.exe'
    version = json.loads((ROOT / 'release/version.json').read_text('utf-8'))['app_version']
    run('core-install', ['uv', 'pip', 'install', '--python', clean_python, '--no-index', '--find-links', wheels, 'phonetic-core==' + version])
    core = run('core-isolated', [clean_python, '-I', '-c',
        'import phonetic_core,importlib.util,json; assert importlib.util.find_spec("fastapi") is None; '
        'assert importlib.util.find_spec("PyQt6") is None; '
        'print(json.dumps({"version":phonetic_core.__version__,"file":phonetic_core.__file__}))'], cwd=OUTPUT)
    run('locked-install', ['uv', 'pip', 'install', '--python', clean_python, '--require-hashes', '-r', ROOT / 'requirements-v3-dev.lock'])
    run('api-desktop-install', ['uv', 'pip', 'install', '--python', clean_python, '--no-index', '--find-links', wheels,
                              'ptb-api==' + version, 'ptb-desktop==' + version])
    run('pip-check', ['uv', 'pip', 'check', '--python', clean_python])
    # Editable setuptools builds expose both dist-info and source egg-info. Verify
    # unique versions per name before comparing distributions, not metadata rows.
    distribution_script = ('import importlib.metadata as m,json; '
        'rows=[(d.metadata["Name"].lower().replace("_","-"),d.version) for d in m.distributions()]; '
        'assert all(len({v for n,v in rows if n==name})==1 for name,_ in rows); '
        'print(json.dumps(dict(sorted(rows))))')
    dev_packages = run('dev-packages', [python, '-I', '-c', distribution_script], cwd=OUTPUT)
    clean_packages = run('clean-packages', [clean_python, '-I', '-c', distribution_script], cwd=OUTPUT)
    assert json.loads(dev_packages) == json.loads(clean_packages), 'Lock differs from development packages'
    run('qt-import', [clean_python, '-I', '-c',
        'from PyQt6 import QtCore,QtWebEngineCore; '
        'print(QtCore.PYQT_VERSION_STR,QtCore.qVersion(),QtWebEngineCore.qWebEngineVersion())'], cwd=OUTPUT)
    local = run('desktop-handshake', [clean_python, '-I', '-m', 'ptb_desktop.main'], cwd=OUTPUT)
    server = server_probe(clean_python)
    run('pytest-clean', [clean_python, '-I', '-m', 'pytest', '-c', ROOT / 'tests/pytest.ini', ROOT / 'tests/contracts',
                        ROOT / 'tests/architecture', ROOT / 'backend/tests', ROOT / 'desktop/tests', '-q'], cwd=OUTPUT)
    # Preserve P01 evidence: call the read-only capture function, writing only P02 evidence.
    sys.path.insert(0, str(ROOT / 'desktop/experiments'))
    from capture_context import capture
    after = capture()
    before = json.loads((ROOT / 'output/validation/p02/context-before.json').read_text('utf-8'))
    preservation = {key: before[key] == after[key] for key in before if key.startswith('v2_')}
    assert all(preservation.values()) and not after['baseline_files_changed_since_D03']
    (OUTPUT / 'context-after.json').write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'status': 'verified', 'scope': 'P02 Windows package and contract scaffold',
        'python': sys.version, 'clean_core': json.loads(core), 'local': json.loads(local), 'server': server,
        'package_versions_match_lock_install': True, 'v2_preservation': preservation,
        'baseline_files_checked': after['baseline_files_checked'],
        'wheels': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheel_files},
        'evidence_directory': OUTPUT.relative_to(ROOT).as_posix(),
        'not_verified': ['Scientific algorithms P03/P08', 'Full desktop UI P04', 'Accounts/jobs P05/P06',
                         'Full distribution licenses and cross-platform native hardware acceptance']}
    (OUTPUT / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'output/validation/p02/latest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
