"""Exercise the real native feedback transport against the isolated service only."""
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'output/validation/web-launch/isolated/client'
config=json.loads((PRIVATE/'client.json').read_text('utf-8'))
assert config['manager_url']=='https://127.0.0.1:18768/api/v1/feedback'
os.environ['SSL_CERT_FILE']=str(PRIVATE/'ca.pem')
sys.path.insert(0,str(ROOT/'scripts'))
from workbench_source import bind_sources
bind_sources(ROOT)
from ptb_desktop.feedback import FeedbackClient

request=json.loads(sys.stdin.buffer.read().decode('utf-8'))
assert request['op'] in ('webStatus','webRequests','webDecision','webQuota','conversations')
client=FeedbackClient(PRIVATE/'verification-profile',admin=True,base_url=config['manager_url'])
client.state={'token':config['manager_token'],'email':'','cursor':0}
try:
    print(json.dumps(client.request(request['op'],request.get('args',{})),ensure_ascii=True))
finally:
    client.close()
