"""P11-ENV synthetic Linux API/static/font smoke, with no database or scientific job.

Starts the existing shared FastAPI app on an owned loopback socket, mounts the
existing frontend, checks it, and stops it. Never starts ptb_api.server's storage
recovery/cleanup. Run from a Linux venv with installed project wheels.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import ProxyHandler, build_opener


class AssetLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        key = 'src' if tag == 'script' else 'href' if tag == 'link' else None
        if key and key in values:
            self.links.append(values[key])


def font_check(chinese: Path, ipa: Path, output: Path) -> list[dict]:
    from fontTools.ttLib import TTFont
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new('RGB', (1100, 200), 'white')
    draw = ImageDraw.Draw(canvas)
    records = []
    for row, (path, text) in enumerate(((chinese, '语音研究工作台：中文显示测试'),
                                       (ipa, 'IPA: ə ɐ ɪ ʊ ŋ ɕ ʑ ɻ tʰ aː'))):
        with TTFont(path) as font:
            cmap = font.getBestCmap()
            missing = [c for c in text if not c.isspace() and
                       (ord(c) not in cmap or font.getGlyphID(cmap[ord(c)]) == 0)]
            if missing:
                raise ValueError('Missing requested font glyphs')
        raster = ImageFont.truetype(str(path), 40)
        if raster.getmask(text).getbbox() is None:
            raise ValueError('Empty font raster')
        draw.text((24, 20 + 85 * row), text, font=raster, fill='black')
        records.append({'role': 'chinese' if row == 0 else 'ipa',
                        'font_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'glyph_coverage': True, 'nonempty_raster': True})
    canvas.save(output / 'linux-font-smoke.png')
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend', type=Path, required=True)
    parser.add_argument('--chinese-font', type=Path, required=True)
    parser.add_argument('--ipa-font', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory only')
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or sys.executable.lower().endswith('.exe'):
        parser.error('Native Linux Python required')
    import phonetic_core
    import ptb_api
    import uvicorn
    from ptb_api.main import create_app
    from starlette.staticfiles import StaticFiles
    for module in (phonetic_core, ptb_api):
        if not Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError('Installed wheel in this venv required')
    args.output.mkdir(parents=False, exist_ok=False)
    report = {'scope': 'Linux synthetic API/static and explicit font raster only',
              'python': sys.executable, 'imports': {m.__name__: m.__file__ for m in (ptb_api, phonetic_core)},
              'database_configured': False, 'scientific_tasks_executed': False}
    app = create_app(mode='server')
    app.mount('/server', StaticFiles(directory=args.frontend, html=True), name='p11-static')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    origin = 'http://127.0.0.1:' + str(sock.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_level='warning', proxy_headers=False))
    runner = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    runner.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not runner.is_alive() or time.monotonic() > deadline:
                raise RuntimeError('API startup failed')
            time.sleep(.02)
        opener = build_opener(ProxyHandler({}))

        def get(path):
            with opener.open(origin + path, timeout=5) as response:
                payload = response.read(32_000_001)
                if response.status != 200 or len(payload) > 32_000_000:
                    raise ValueError('HTTP status or size')
                return payload

        health = json.loads(get('/api/v1/health'))
        if health.get('status') != 'ok' or health.get('mode') != 'server':
            raise ValueError('Health payload')
        capability = json.loads(get('/api/v1/capabilities'))
        if capability.get('algorithms') != [] or capability.get('task_operations') != []:
            raise ValueError('Unconfigured host must not advertise scientific tasks')
        page = get('/server/')
        links = AssetLinks()
        links.feed(page.decode('utf-8'))
        if '语音研究工作台' not in page.decode('utf-8') or not links.links:
            raise ValueError('Static HTML entry')
        assets = []
        for href in links.links:
            parsed = urlsplit(urljoin(origin + '/server/', href))
            if parsed.netloc != urlsplit(origin).netloc or not parsed.path.startswith('/server/'):
                raise ValueError('Unexpected static asset origin')
            data = get(parsed.path)
            local = (args.frontend / parsed.path.removeprefix('/server/')).resolve()
            if not local.is_relative_to(args.frontend.resolve()) or data != local.read_bytes():
                raise ValueError('Static asset content mismatch')
            assets.append({'path': parsed.path, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
        report.update(health=health, algorithms=capability['algorithms'], static_assets=assets,
                      fonts=font_check(args.chinese_font, args.ipa_font, args.output))
    finally:
        server.should_exit = True
        runner.join(timeout=10)
        sock.close()
        report['server_stopped'] = not runner.is_alive()
        (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if runner.is_alive():
        raise RuntimeError('Owned API did not stop')
    print(json.dumps({'result': 'passed', 'report': str(args.output / 'report.json')}))


if __name__ == '__main__':
    main()
