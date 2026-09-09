from fastapi.testclient import TestClient
from ptb_api.main import create_app
import subprocess
import sys


def test_modes_share_contract_and_do_not_claim_algorithms():
    schemas = []
    for mode in ['local', 'server']:
        with TestClient(create_app(mode)) as client:
            health = client.get('/api/v1/health').json()
            assert health['mode'] == mode
            assert health['app_version'] == health['core_version'] == '3.0.0a1'
            capabilities = client.get('/api/v1/capabilities').json()
            assert capabilities['algorithms'] == []
            assert capabilities['stage'] == 'P02'
            response = client.post('/api/v1/jobs', json={})
            assert response.status_code == 503
            assert response.json() == {'detail':'task_service_unavailable'}
            assert capabilities['task_operations'] == []
            schemas.append(client.get('/openapi.json').json())
    assert schemas[0] == schemas[1]


def test_m01_schema_does_not_import_scientific_runtime():
    code = '''
import sys
from ptb_api.main import create_app
schemas = create_app().openapi()['components']['schemas']
assert {'AcousticRequest','AcousticResult','AcousticBatchSummary','ResultManifestEnvelope'} <= schemas.keys()
assert 'numpy' not in sys.modules
assert 'phonetic_core.acoustic' not in sys.modules
'''
    subprocess.run([sys.executable,'-c',code],check=True,timeout=30)
