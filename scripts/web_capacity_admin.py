"""Private operator CLI; no browser approval route or credentials on command line.

Requires the owned 0600 private configuration: dsn and storage_root. Actions are
explicit and never delete user files or bypass the global disk reserve.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import stat
from ptb_api.storage import Storage, no_links
from ptb_api.workspaces import WorkspaceService
from ptb_api.web_capacity import decide, set_quota


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('action',choices=['status','requests','default','account','approve','decline'])
    parser.add_argument('--bytes',type=int)
    parser.add_argument('--id')
    parser.add_argument('--note',default='')
    parser.add_argument('--duration',choices=['today','7d','30d'])
    args=parser.parse_args()
    no_links(args.config)
    info=args.config.stat()
    if os.name=='posix' and (info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600):
        parser.error('Private configuration must be owned by this user with mode 0600')
    config=json.loads(args.config.read_text('utf-8'))
    storage=Storage(config['dsn'],config['storage_root'],min_free_bytes=config.get('min_free_bytes',10_000_000_000))
    service=WorkspaceService(storage);service.activate()
    if args.action in ('default','account'):
        if args.bytes is None or (args.action=='account' and not args.id):parser.error('Provide --bytes and account --id when applicable')
        result=set_quota(service,args.bytes,args.id if args.action=='account' else None)
    elif args.action in ('approve','decline'):
        if not args.id or (args.action=='approve' and args.bytes is None):parser.error('Provide request --id and approved --bytes')
        result=decide(service,args.id,granted_bytes=args.bytes if args.action=='approve' else None,note=args.note,duration=args.duration).model_dump(mode='json')
    else:
        with storage._locked() as c:
            if args.action=='requests':
                result=c.execute("SELECT id,owner_id,requested_bytes,duration,reason,created_at FROM ptb_web.capacity_requests WHERE state='pending' ORDER BY created_at LIMIT 100").fetchall()
            else:
                totals=c.execute('SELECT coalesce(sum(used_bytes),0) AS used_bytes,coalesce(sum(reserved_bytes),0) AS reserved_bytes FROM ptb_storage.quota_accounts').fetchone()
                result=dict(default_quota_bytes=service.default_quota(c),disk_free_bytes=shutil.disk_usage(storage.root).free,disk_margin_bytes=storage.min_free_bytes,**totals)
    print(json.dumps(result,ensure_ascii=False,default=str))


if __name__=='__main__':main()
