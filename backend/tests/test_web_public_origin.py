import httpx
import pytest
from fastapi import FastAPI
from ptb_api.server import PublicOriginGuard, account_settings


def test_public_settings_require_https_and_secure_cookies():
    settings = account_settings({'public_origin': 'https://www.phonetictoolbox.com',
                                 'signing_key': 'x' * 48}, 18775)
    assert settings.secure and settings.cookie == '__Host-ptb-session'
    assert not settings.allow_insecure_loopback
    with pytest.raises(ValueError):
        account_settings({'public_origin': 'http://www.phonetictoolbox.com',
                          'signing_key': 'x' * 48}, 18775)
    assert not account_settings({'signing_key': 'x' * 48}, 18765).secure


def test_public_guard_rejects_foreign_and_duplicate_host_or_origin():
    import asyncio
    async def run():
        app = FastAPI()
        @app.get('/')
        def index(): return {'ok': True}
        transport = httpx.ASGITransport(app=PublicOriginGuard(app, 'https://www.phonetictoolbox.com'))
        async with httpx.AsyncClient(transport=transport, base_url='https://www.phonetictoolbox.com') as c:
            assert (await c.get('/')).status_code == 200
            assert (await c.get('/', headers={'Origin': 'https://www.phonetictoolbox.com'})).status_code == 200
            assert (await c.get('/', headers={'Origin': 'https://example.org'})).status_code == 403
            assert (await c.get('/', headers={'Host': 'example.org'})).status_code == 403
            assert (await c.get('/', headers=[('Origin', 'https://www.phonetictoolbox.com'),
                                              ('Origin', 'https://www.phonetictoolbox.com')])).status_code == 403
    asyncio.run(run())


def test_public_policy_pins_the_actual_vocal_import_map():
    import base64
    import hashlib
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    html = (root/'frontend/public/vocal-tract/index.html').read_text('utf-8')
    inline = re.search(r'<script type="importmap">(.*?)</script>', html, re.S).group(1)
    digest = base64.b64encode(hashlib.sha256(inline.encode()).digest()).decode()
    policy = (root/'deployment/p15-staging/existing-site.conf.in').read_text('utf-8')
    assert "'sha256-" + digest + "'" in policy
    assert "frame-ancestors 'self'" in policy
