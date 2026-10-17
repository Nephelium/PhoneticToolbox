"""Exercise the actual installed nginx map and collector on an owned loopback."""
from pathlib import Path
import json
import socket
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen
from collect_stats import collect


def main():
    with tempfile.TemporaryDirectory(prefix='ptb-stats-nginx-') as tmp:
        root = Path(tmp)
        root.chmod(0o755)
        mac = '/releases/macos-arm64/3.0.3/PhoneticToolbox-3.0.3-macos14-arm64-preview-r6.zip'
        win = '/releases/windows-x64/3.0.3/PhoneticToolbox-windows-x64.exe'
        for name in [mac, win, '/index.html']:
            target = root / name.lstrip('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'0123456789abcdef')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        maps = (Path(__file__).parent / 'website-stats.nginx.conf').read_text('utf-8')
        config = f'''pid {root}/nginx.pid;
error_log {root}/error.log;
events {{}}
http {{
{maps}
server {{ listen 127.0.0.1:{port}; root {root};
access_log {root}/ptb-website-stats.log ptb_website_stats if=$ptb_stats_loggable;
}}
}}'''
        path = root / 'nginx.conf'
        path.write_text(config)
        process = subprocess.Popen(['nginx', '-c', str(path), '-p', str(root), '-g', 'daemon off;'])
        try:
            for attempt in range(100):
                if process.poll() is not None:
                    raise RuntimeError((root / 'error.log').read_text())
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.1):
                        break
                except OSError:
                    time.sleep(.02)
            def get(name, *, method='GET', agent='Mozilla/5.0', byte_range=None):
                headers = {'User-Agent': agent}
                if byte_range:
                    headers['Range'] = byte_range
                request = Request(f'http://127.0.0.1:{port}{name}', method=method, headers=headers)
                with urlopen(request, timeout=5) as response:
                    response.read()
                    return response.status
            assert get(mac, byte_range='bytes=0-3') == 206
            assert get(mac, byte_range='bytes=4-7') == 206
            assert get(mac, method='HEAD') == 200
            assert get(mac, agent='PTB-Website-Verify') == 200
            assert get(mac, agent='ExampleCrawler') == 200
            assert get(mac) == 200
            assert get(win) == 200
            assert get('/index.html') == 200
            process.terminate()
            process.wait(timeout=5)
            value = collect(root, root / 'state.json', root / 'site-stats.json', backfill_access_logs=True)
            assert value['downloads']['macos'] == 2, value
            assert value['downloads']['windows'] == 1, value
            assert value['downloads']['total'] == 3, value
            assert value['pageViews'] == 1, value
            assert collect(root, root / 'state.json', root / 'site-stats.json')['downloads']['total'] == 3
            print(json.dumps({'success': True, 'macos': 2, 'windows': 1, 'total': 3,
                              'checks': ['full GET', 'first Range', 'resume excluded', 'HEAD excluded',
                                         'verification and crawler excluded', 'repeat collection unchanged'],
                              'productionLogUntouched': True}))
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


if __name__ == '__main__':
    main()
