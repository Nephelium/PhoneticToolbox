"""Owned M19 browser acceptance against the already-built production frontend.

Copies only the 16 private runtime files through the packaging entry point.
The original book and source data are never adopted as test artifacts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from package_geolinguistics_data import package_data
from verification_artifacts import verification_run

ROOT = Path(__file__).resolve().parents[1]


def prepare(source, scratch):
    good = scratch / 'geolinguistics-data'
    bad = scratch / 'damaged-data'
    result = package_data(source, good)
    package_data(source, bad)
    if (source/'data-manifest.json').is_file():
        assert (good/'data-manifest.json').read_bytes()==(source/'data-manifest.json').read_bytes(), 'Reviewed manifest identity changed while packaging'
    # Deliberately unlisted synthetic file. No original book is copied.
    (good / 'unlisted-private-book.pdf').write_bytes(b'%PDF-1.4\nM19 synthetic unread sentinel\n%%EOF\n')
    damaged = bad / 'catalog.json'
    raw = bytearray(damaged.read_bytes())
    raw[0] = ord('[') if raw[0] != ord('[') else ord('{')
    damaged.write_bytes(raw)  # Same length, wrong SHA-256; manifest is unchanged.
    catalog = json.loads((good / 'catalog.json').read_text('utf-8'))
    sites = json.loads((good / 'sites.json').read_text('utf-8'))
    assert len(sites) == 930 and len(catalog['features']) == 510
    return {
        'runtime': result,
        'manifestSha256': hashlib.sha256((good / 'data-manifest.json').read_bytes()).hexdigest(),
        'catalogSha256': hashlib.sha256((good / 'catalog.json').read_bytes()).hexdigest(),
        'source': str(source.resolve()),
        'fixtures': '16 packaged runtime files + manifest; unlisted synthetic PDF; second package with same-size catalog hash corruption',
        'expected': {'sites': 930, 'features': 510, 'F120Groups': 6,
                     'F120NoNasalSites': ['歙县', '绩溪', '辰溪湘_熊首山', '汤溪旧_岩下']},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'local-data/geolinguistics')
    parser.add_argument('--headed', action='store_true')
    parser.add_argument('--layout-only', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--geography-audit', action='store_true',
                        help='Explicitly include historical map-content probes; omitted from normal UI acceptance.')
    parser.add_argument('--frontend-dist', type=Path, default=ROOT/'frontend/dist')
    args = parser.parse_args()
    if not (args.frontend_dist / 'index.html').is_file():
        raise RuntimeError('Build frontend/dist before running this acceptance; this runner never builds product code.')
    with verification_run('geolinguistics-browser', 'chrome',
                          'Production Vite preview; private runtime package; independent Chrome profile; real directory input, exports, reload and rejection') as (out, scratch):
        (out / 'inputs.json').write_text(json.dumps(prepare(args.source, scratch), ensure_ascii=False, indent=2), encoding='utf-8')
        env = dict(os.environ, PTB_GEO_HEADED='1' if args.headed else '0', PTB_GEO_LAYOUT_ONLY='1' if args.layout_only else '0',
                   PTB_GEO_DOWNLOAD_ONLY='1' if args.download_only else '0',PTB_GEO_DIST=str(args.frontend_dist.resolve()))
        env['PTB_GEO_GEOGRAPHY_AUDIT'] = '1' if args.geography_audit else '0'
        result = subprocess.run(['node', str(ROOT / 'tests/e2e/geolinguistics.cjs'), str(out), str(scratch)],
                                cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        (out / 'browser.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        print(out, flush=True)
        if result.returncode:
            print(result.stderr, flush=True)
            raise RuntimeError('M19 production browser acceptance failed; inspect report.json and browser.log')


if __name__ == '__main__':
    main()
