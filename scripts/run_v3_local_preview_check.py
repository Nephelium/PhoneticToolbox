"""Run the owned preview outside the checkout and observe its child processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from uuid import uuid4

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--natural-manifest', type=Path)
    args = parser.parse_args()
    exe = args.exe.resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    out = root / 'output/validation' / ('v3-preview-exe-' + uuid4().hex)
    out.mkdir(parents=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('PYTHON', 'PTB_', 'QT_')):
            env.pop(key)
    windows = Path(env.get('SystemRoot', 'C:/Windows'))
    env['PATH'] = os.pathsep.join(map(str, (windows / 'System32', windows)))
    print(out, flush=True)
    seen = {}
    with (out / 'stdout.log').open('wb') as stdout, (out / 'stderr.log').open('wb') as stderr:
        arguments=[str(exe), '--verify-preview', str(out / 'results')]
        if args.natural_manifest:
            arguments=[str(exe),'--verify-natural-preview',str(out/'results'),str(args.natural_manifest.resolve(strict=True))]
        process = subprocess.Popen(arguments,
            cwd=tempfile.gettempdir(), env=env, stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW)
        parent = psutil.Process(process.pid)
        deadline = time.monotonic() + 480
        while process.poll() is None:
            try:
                for child in parent.children(recursive=True):
                    seen[(child.pid, child.create_time())] = child
            except psutil.NoSuchProcess:
                pass
            if time.monotonic() > deadline:
                for child in reversed(list(seen.values())):
                    try:
                        child.kill()  # psutil verifies PID identity before killing.
                    except psutil.NoSuchProcess:
                        pass
                process.kill(); process.wait()
                raise TimeoutError(str(out))
            time.sleep(.3)
    _, alive = psutil.wait_procs(list(seen.values()), timeout=5)
    report = dict(exit_code=process.returncode, remaining_owned_pids=[p.pid for p in alive],
                  observed_children=len(seen), cwd=tempfile.gettempdir(), sanitized_environment=True)
    report_path = out / 'process-report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    assert process.returncode == 0 and not alive, str(out)
    for arguments in (['--ptb-worker', 'os'], ['--ptb-worker'], ['--unknown-worker']):
        result = subprocess.run([str(exe), *arguments], cwd=tempfile.gettempdir(), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45,
            creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode == 2, (arguments, result.returncode)
    report['unknown_dispatch_rejected'] = 3
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Verified preview, process cleanup, and rejected dispatches', flush=True)


if __name__ == '__main__':
    main()
