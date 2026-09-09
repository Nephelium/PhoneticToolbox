import urllib.error
import urllib.request

import pytest
from ptb_desktop.local_service import LocalService


def test_two_owned_instances_have_separate_sessions_and_clean_exit():
    first, second = LocalService(), LocalService()
    try:
        assert first.start()['status'] == 'ok'
        assert second.start()['status'] == 'ok'
        assert first.url != second.url
        assert first.token != second.token
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for headers in [{}, {'Authorization': 'Bearer ' + second.token},
                        {'Authorization': 'Bearer ' + first.token, 'Origin': 'https://external.invalid'},
                        {'Authorization': 'Bearer ' + first.token, 'Host': 'external.invalid'}]:
            request = urllib.request.Request(first.url + '/api/v1/health', headers=headers)
            with pytest.raises(urllib.error.HTTPError) as error:
                opener.open(request, timeout=3)
            assert error.value.code == 403
        assert first.get('/api/v1/capabilities')['algorithms'] == []
        first.close()
        assert first.exit_code == 0
        assert second.get('/api/v1/health')['status'] == 'ok'
    finally:
        first.close()
        second.close()
    assert second.exit_code == 0
