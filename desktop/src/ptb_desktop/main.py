import json

from .local_service import LocalService


def main():
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',action='store_true')
    parser.add_argument('--dist',type=Path)
    parser.add_argument('--jobs-path',type=Path)
    parser.add_argument('--local-files-root',type=Path)
    parser.add_argument('--reaper-binary',type=Path)
    args=parser.parse_args()
    if args.workspace:
        from .host import run
        if args.dist is None:parser.error('--workspace requires --dist pointing to the built frontend')
        raise SystemExit(run(args.dist.resolve(),jobs_path=args.jobs_path,local_files_root=args.local_files_root,reaper_binary=args.reaper_binary))
    with LocalService() as service:
        health = service.get('/api/v1/health')
    if service.exit_code != 0:
        raise RuntimeError('Local service did not exit cleanly')
    print(json.dumps({'health': health, 'service_exit_code': service.exit_code}))


if __name__ == '__main__':
    main()
