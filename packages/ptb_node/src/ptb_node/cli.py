import argparse
import json
from pathlib import Path
import sys
from .config import Config, NodeError
from .inventory import inspect


def main():
    parser = argparse.ArgumentParser(description='P06-REMOTE node components (B binding pending)')
    parser.add_argument('--config', type=Path)
    parser.add_argument('command', choices=('inspect', 'check', 'credential', 'start', 'pause', 'resume', 'status', 'stop', 'abort', 'cleanup'))
    args = parser.parse_args()
    try:
        if args.command == 'inspect':
            print(json.dumps(inspect(Path.cwd()), ensure_ascii=False, indent=2))
            return 0
        if sys.platform != 'linux':
            raise NodeError('linux_required')
        if not args.config:
            raise NodeError('config_required')
        cfg = Config.load(args.config)
        if args.command == 'credential':
            import getpass
            from .config import save_credential
            if not sys.stdin.isatty():
                raise NodeError('credential_requires_private_terminal')
            save_credential(cfg.credential_file, getpass.getpass('Node credential (hidden): '))
            print(json.dumps({'stored': True, 'enrolled': False}))
            return 0
        if args.command == 'check':
            from .runtime import P11Runtime
            info = inspect(cfg.state_dir)
            print(json.dumps({'ready': False, 'hard_limits': info['hard_limits'],
                              'protocol': 'binding_pending', 'runtime': P11Runtime().capabilities()}))
            return 2
        from .service import Service, control
        from .storage import Attempts, Instance
        if args.command == 'start':
            Service(cfg).run()
        elif args.command == 'cleanup':
            with Instance(cfg.state_dir):
                Attempts(cfg.state_dir, cfg.temporary_bytes, cfg.reserve_disk_bytes).recover()
            print(json.dumps({'cleaned': True}))
        else:
            print(json.dumps(control(cfg, args.command)))
        return 0
    except NodeError as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        return 2
    except (OSError, ValueError):
        print(json.dumps({'error': 'local_operation_failed'}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
