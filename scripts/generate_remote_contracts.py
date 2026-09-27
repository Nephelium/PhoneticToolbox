"""Independent remote/1 snapshot until A releases the common generator."""
import argparse
import json
from pathlib import Path
from fastapi import FastAPI
from ptb_api.remote import create_remote_router
from ptb_api.remote_models import REMOTE_MODELS, Poll, Heartbeat, Output, Failure, LeaseResponse

ROOT = Path(__file__).resolve().parents[1]


def snapshots():
    app = FastAPI()
    # Schema construction does not call a coordinator or connect to a database.
    app.include_router(create_remote_router(None))
    values = {'openapi': app.openapi(), 'protocol': 'remote/1',
              'models': {m.__name__: m.model_json_schema() for m in REMOTE_MODELS},
              'examples': {
                  'poll': Poll(request_id='00000000-0000-4000-8000-000000000010', runtime_hash='a'*64).model_dump(mode='json'),
                  'heartbeat': Heartbeat(generation=1, phase='upload', node_bytes=4096).model_dump(),
                  'lease_response': LeaseResponse(server_time=1000, lease_until=1060, deadline=1300,
                      lease_remaining_seconds=60, deadline_remaining_seconds=300).model_dump(),
                  'output': Output(generation=1, key='lpc_json', name='lpc.ptb.json', size_bytes=1024, sha256='b'*64).model_dump(),
                  'normal_exit': Failure(generation=1, code='node_shutdown').model_dump(),
              }}
    return json.dumps(values, ensure_ascii=False, sort_keys=True, indent=2)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); target = ROOT/'contracts/remote-v1.json'
    content = snapshots()
    if args.check:
        raise SystemExit(0 if target.exists() and target.read_text('utf-8') == content else 1)
    target.write_text(content, encoding='utf-8', newline='\n')
