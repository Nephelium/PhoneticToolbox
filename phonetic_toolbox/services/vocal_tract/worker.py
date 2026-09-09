"""Private entry point, dispatched before Qt/OpenCV imports in run.py."""
import argparse
import json
import os
from pathlib import Path
from .process_guard import watch_parent


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent-pid',type=int,required=True)
    parser.add_argument('--handshake',type=Path,required=True)
    parser.add_argument('--instance',required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--silent',action='store_true')
    args=parser.parse_args(argv)
    watch_parent(args.parent_pid)
    from phonetic_toolbox.utils import get_resource_path
    from .server import create_server
    server=create_server(resource_dir=get_resource_path('phonetic_toolbox/resources/vocal_tract/native'),
                         web_dir=get_resource_path('phonetic_toolbox/gui/resources/vocal_tract'),
                         profile_dir=args.profile,playback_allowed=not args.silent)
    ready={'pid':os.getpid(),'parent_pid':args.parent_pid,'instance':args.instance,
           'url':f'http://127.0.0.1:{server.server_port}','token':server.app.token}
    temporary=args.handshake.with_suffix('.tmp')
    temporary.write_text(json.dumps(ready),encoding='utf-8');temporary.replace(args.handshake)
    try:server.serve_forever(poll_interval=.1)
    finally:
        server.server_close()
        server.app.live.stop()
        server.app.engine.close()
    return 0
