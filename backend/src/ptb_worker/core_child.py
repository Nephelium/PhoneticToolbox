"""One trusted core operation per child; EOF of owner stdin ends this child."""
import json
import os
import sys
import threading
import time
from phonetic_core import __version__
from phonetic_core.pipeline_check import blocks


def main():
    config=json.loads(sys.stdin.readline())
    def owner_closed():
        sys.stdin.readline()
        os._exit(75)
    threading.Thread(target=owner_closed,daemon=True).start()
    snapshot=config['snapshot']
    if snapshot['operation']!='pipeline_check' or snapshot['core_version']!=__version__:
        raise ValueError('Unsupported operation or core version')
    delay=float(config.get('step_delay',0))
    if not 0 <= delay <= 0.2:raise ValueError('Invalid probe delay')
    for message in blocks(**snapshot['config']):
        if 'result' in message:
            message['result'].update(complete=True,kind='pipeline_check_metadata',core_version=__version__)
        print(json.dumps(message),flush=True)
        if delay:time.sleep(delay)


if __name__=='__main__':
    try:main()
    except Exception:
        print('Core probe failed',file=sys.stderr)
        raise SystemExit(1) from None
