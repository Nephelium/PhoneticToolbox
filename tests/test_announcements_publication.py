"""Public-feed boundary checks, using small owned data and no remote connection."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('announcement_publisher', ROOT / 'deployment/announcements/publish.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def template():
    return json.loads((ROOT / 'deployment/announcements/first-announcement.example.json').read_text('utf-8'))


def raw(value):
    return json.dumps(value, ensure_ascii=False).encode('utf-8')


def test_production_first_notice_matches_confirmed_input_method_launch():
    feed = publisher.validate_feed((ROOT / 'deployment/announcements/public.json').read_bytes())
    assert len(feed['items']) == 1
    assert feed['items'][0]['id'] == 'ipa-ime-0-1-5-launched'
    assert feed['items'][0]['publishedAt'] == '2026-10-24T15:30:45Z'
    assert feed['items'][0]['minVersion'] == '3.1.0'
    assert feed['items'][0]['action']['url'] == 'https://www.phonetictoolbox.com/#ipa-ime-download'
    sample = publisher.validate_feed(raw(template()))
    assert sample['items'][0]['id'] == 'ipa-ime-0-1-5-launched'
    assert sample['items'][0]['action']['url'] == 'https://www.phonetictoolbox.com/#ipa-ime-download'


@pytest.mark.parametrize('url', [
    'javascript:alert(1)', 'http://www.phonetictoolbox.com/',
    'https://www.phonetictoolbox.com.evil.test/', 'https://user@www.phonetictoolbox.com/',
    'https://www.phonetictoolbox.com:443/', 'https://www.phonetictoolbox.com/\\evil.test',
    'https://www.phonetictoolbox.com/\nprivate',
    'https://www.phonetictoolbox.com/?redirect=other',
    'https://www.phonetictoolbox.com/a.%7Aip',
])
def test_action_does_not_expand_official_origin(url):
    feed = template(); feed['items'][0]['action']['url'] = url
    with pytest.raises(ValueError):
        publisher.validate_feed(raw(feed))


def test_feed_size_identity_and_validity_boundaries():
    feed = template()
    bad = deepcopy(feed); bad['items'] *= 2
    with pytest.raises(ValueError): publisher.validate_feed(raw(bad))
    bad = deepcopy(feed); bad['items'][0]['expiresAt'] = '2026-10-23T00:00:00Z'
    with pytest.raises(ValueError): publisher.validate_feed(raw(bad))
    bad = deepcopy(feed); bad['items'][0]['publishedAt'] = '2026-02-31T00:00:00Z'
    with pytest.raises(ValueError): publisher.validate_feed(raw(bad))
    bad = deepcopy(feed); bad['items'][0]['minVersion'] = '3.1.0'; bad['items'][0]['maxVersion'] = '3.0.5'
    with pytest.raises(ValueError): publisher.validate_feed(raw(bad))
    bad = deepcopy(feed); bad['items'][0]['id'] = '_bad'
    with pytest.raises(ValueError): publisher.validate_feed(raw(bad))
    bad = deepcopy(feed); bad['items'][0]['body'] = '\U0001f338' * 4001
    with pytest.raises(ValueError): publisher.validate_feed(raw(bad))
    bad = deepcopy(feed); bad['items'][0]['minVersion'] = '3.1\u0660.0'
    with pytest.raises(ValueError): publisher.validate_feed(raw(bad))
    with pytest.raises(ValueError): publisher.validate_feed(b' ' * (publisher.LIMIT + 1))


def test_default_preview_does_not_need_credentials_or_connect():
    result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(ROOT / 'deployment/announcements/publish.py'),
        '--feed', str(ROOT / 'deployment/announcements/public.json')], capture_output=True, text=True, encoding='utf-8', check=True)
    value = json.loads(result.stdout)
    assert value['applied'] is False and value['items'] == 1


def test_apply_without_compare_exchange_identity_is_refused():
    result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(ROOT / 'deployment/announcements/publish.py'),
        '--feed', str(ROOT / 'deployment/announcements/public.json'), '--apply'], capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 2
    assert '--expected-sha256' in result.stderr
