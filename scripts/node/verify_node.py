"""Installed-client Linux smoke; owns all child processes and temporary files."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise SystemExit('Linux required')
    from ptb_node.config import Config, NodeError
    from ptb_node.lease import Identity
    from ptb_node.storage import Attempts
    from ptb_node.service import control
    from ptb_node.inventory import inspect

    with tempfile.TemporaryDirectory(prefix='ptb-node-smoke-') as temp:
        root = Path(temp)
        config_path = root / 'config.json'
        config_path.write_text(json.dumps({'schema': 'ptb-node-config/1',
            'server_origin': 'https://example.invalid', 'state_dir': str(root / 'state'),
            'credential_file': str(root / 'credential')}), encoding='utf-8')
        config = Config.load(config_path)
        commands = [sys.executable, '-B', '-m', 'ptb_node', '--config', str(config_path)]
        children = []
        def start():
            child = subprocess.Popen(commands + ['start'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            children.append(child)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    state = control(config, 'status')
                    if state['pid'] == child.pid:
                        return child
                except (OSError, NodeError):
                    pass
                if child.poll() is not None:
                    raise RuntimeError('smoke_start_failed')
                time.sleep(0.02)
            raise RuntimeError('smoke_start_timeout')
        try:
            node = start()
            paused = control(config, 'pause')['paused']
            resumed = not control(config, 'resume')['paused']
            duplicate = subprocess.run(commands + ['start'], capture_output=True, timeout=5)
            # Kill only the exact Popen child created above. Do not pkill/python-name scan.
            node.kill()
            node.wait(timeout=3)
            store = Attempts(config.state_dir, 100, 1)
            residual = store.create(Identity('synthetic-crashed', 1), time.time() - 1, 50)
            (residual / 'synthetic-data').write_bytes(b'public synthetic data')
            restarted = start()
            cleaned = not residual.exists()
            state = control(config, 'status')
            control(config, 'stop')
            restarted.wait(timeout=4)
            gate = subprocess.run(commands + ['check'], capture_output=True, timeout=5)
            result = {'platform': 'personal-wsl-linux' if 'microsoft' in os.uname().release.lower() else 'linux-local',
                      'paused': paused, 'resumed': resumed, 'duplicate_rejected': duplicate.returncode == 2,
                      'crash_restart_cleaned': cleaned, 'normal_exit': restarted.returncode == 0,
                      'resource_gate_exit': gate.returncode, 'resource_gate': json.loads(gate.stdout),
                      'status': state, 'inventory': inspect(root)}
            if not all((paused, resumed, duplicate.returncode == 2, cleaned, restarted.returncode == 0, gate.returncode == 2)):
                raise RuntimeError('smoke_assertion_failed')
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps({'passed': True, 'resource_ready': False}))
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                child.communicate(timeout=5)


if __name__ == '__main__':
    main()
