"""Fixed synthetic P11 process failures; no user content or scientific imports."""
import os
import json
from pathlib import Path
import subprocess
import sys
import time

action = sys.argv[1]
if action == 'echo':
    sys.stdout.buffer.write(sys.stdin.buffer.read())
elif action == 'limits':
    group = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    root = Path('/sys/fs/cgroup')/group.lstrip('/')
    print(json.dumps({name: (root/name).read_text().strip() for name in
                      ('memory.max', 'memory.swap.max', 'cpu.max', 'pids.max')}))
elif action == 'output':
    sys.stdout.buffer.write(b'x' * 100_000)
elif action == 'leaf':
    Path(sys.argv[2]).write_text(str(os.getpid()), encoding='ascii')
    if len(sys.argv) > 3:
        allocation = bytearray(45_000_000)
    time.sleep(30)
elif action in ('descendant', 'crash', 'memory'):
    child = subprocess.Popen([sys.executable, __file__, 'leaf', sys.argv[2]] +
                             (['allocate'] if action == 'memory' else []))
    while not Path(sys.argv[2]).exists():
        time.sleep(.01)
    if action == 'crash':
        raise SystemExit(7)
    if action == 'memory':
        allocation = bytearray(45_000_000)
    time.sleep(30)
else:
    raise SystemExit(9)
