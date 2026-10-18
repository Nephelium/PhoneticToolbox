"""Owned loopback acceptance using real browser Wasm and native MP4 fallback."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
from workbench_source import ROOT, configure
from verification_artifacts import verification_run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--science', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18783)
    parser.add_argument('--frontend', type=Path)
    parser.add_argument('--fallback-only', action='store_true')
    parser.add_argument('--controls', action='store_true')
    parser.add_argument('--shared-controls', action='store_true')
    parser.add_argument('--libraries', action='store_true')
    parser.add_argument('--stop-timing', action='store_true')
    parser.add_argument('--vocal-runtime', type=Path)
    args = parser.parse_args()
    work, science = args.work.resolve(), args.science.resolve()
    configure()
    sys.path.insert(0, str(ROOT / 'tests/support'))
    import uvicorn
    from starlette.staticfiles import StaticFiles
    from ptb_api.main import create_app
    from ptb_api.auth import AuthSettings
    from account_double import MemoryAccountStore
    store = MemoryAccountStore()
    store.create_user('vocal-qa', 'vocal-qa-local-password')
    origin = f'http://127.0.0.1:{args.port}'
    app = create_app(account_store=store, auth_settings=AuthSettings(
        origin=origin, signing_key='owned-m10-wasm-loopback-' * 4, allow_insecure_loopback=True),
        vocal_resources=work / 'native')
    app.mount('/app/vocal-wasm', StaticFiles(directory=args.vocal_runtime.resolve() if args.vocal_runtime else work / 'build/vocal-wasm'))
    app.mount('/app/browser-science', StaticFiles(directory=science))
    app.mount('/app/wasm-qa', StaticFiles(directory=ROOT / 'tests/support'))
    app.mount('/app', StaticFiles(directory=args.frontend.resolve() if args.frontend else work / 'frontend', html=True))
    with verification_run('m10-wasm-20261018', 'fallback' if args.fallback_only else 'browser',
        'Actual isolated Chrome, fixed synthetic MemoryAccountStore, current VTL Wasm and original server video encoder; no production service or database') as (out, scratch):
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=args.port,
                                                access_log=False, log_level='warning'))
        thread = threading.Thread(target=server.run)
        thread.start()
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(origin + '/api/v1/health', timeout=.5).close()
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise RuntimeError('Owned HTTP server did not start')
            from ptb_api.preferences import FIELDS
            write_json(scratch / 'preference-fields.json', FIELDS)
            if not args.fallback_only and not args.shared_controls and not args.libraries and not args.stop_timing:
                from vocal_wasm_cases import run as native_oracle
                write_json(scratch / 'native.json', native_oracle(work / 'native'))
            env = {**os.environ, 'PTB_VOCAL_WASM': '1', 'PTB_VOCAL_QA_FIELDS': str(scratch / 'preference-fields.json')}
            script = 'vocal-stop-timing.cjs' if args.stop_timing else 'vocal-libraries.cjs' if args.libraries else 'vocal-shared-controls.cjs' if args.shared_controls else 'vocal-wasm-fallback.cjs' if args.fallback_only else 'vocal-browser-controls.cjs' if args.controls else 'web-vocal.cjs'
            result = subprocess.run(['node', str(ROOT / 'tests/e2e' / script), origin, str(out), str(scratch)], cwd=ROOT, env=env)
            if result.returncode:
                raise RuntimeError('Browser Wasm acceptance failed; see retained report')
            if not args.fallback_only and not args.shared_controls and not args.libraries and not args.stop_timing:
                from compare_m10_wasm import compare
                import json
                comparison = compare(
                    json.loads((scratch / 'native.json').read_text('utf-8')),
                    json.loads((scratch / 'wasm-browser.json').read_text('utf-8')))
                write_json(out / 'scientific-comparison.json', comparison)
                if not all(value for value in comparison['acceptance'].values() if isinstance(value, bool)):
                    raise RuntimeError('Scientific acceptance failed; raw strict audit retained')
            write_json(out / 'scope.json', dict(origin=origin, browser='Chrome isolated profile',
                scientific_execution='native server fallback' if args.fallback_only else 'browser Wasm', video='original native H.264/AAC encoder',
                accounts='synthetic MemoryAccountStore; no production/PostgreSQL claim'))
            print(str(out), flush=True)
        finally:
            server.should_exit = True
            thread.join(timeout=15)
            if thread.is_alive():
                raise RuntimeError('Owned HTTP server failed to stop')


if __name__ == '__main__':
    main()
