"""Loopback development entry; no data endpoints or production deployment."""
import argparse
import asyncio
import hmac
import json
import socket
import sys
import threading

import uvicorn
from starlette.responses import JSONResponse

from .main import create_app


class LoopbackGuard:
    def __init__(self, app, authority: str, token: str | None):
        self.app, self.authority, self.token = app, authority, token

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            headers = scope['headers']
            hosts = [v.decode('latin1') for k, v in headers if k == b'host']
            origins = [v.decode('latin1') for k, v in headers if k == b'origin']
            auth = [v.decode('latin1') for k, v in headers if k == b'authorization']
            permitted = hosts == [self.authority] and (not origins or origins == ['http://' + self.authority])
            if self.token is not None:
                permitted = permitted and len(auth) == 1 and hmac.compare_digest(
                    auth[0].encode('latin1'), ('Bearer ' + self.token).encode('ascii'))
            if not permitted:
                await JSONResponse({'code': 'permission_denied'}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['server', 'local'], default='server')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--managed', action='store_true', help='Receive shutdown over stdin')
    args = parser.parse_args()
    token = None
    if args.mode == 'local':
        config = json.loads(sys.stdin.readline())
        token = config.get('token')
        if not isinstance(token, str) or len(token) < 32:
            raise ValueError('Local service requires a per-session token through stdin')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind before reporting the port; never find-then-rebind an ephemeral port.
    sock.bind(('127.0.0.1', args.port))
    authority = '127.0.0.1:' + str(sock.getsockname()[1])
    app = LoopbackGuard(create_app(args.mode), authority, token)
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False))

    if args.managed or args.mode == 'local':
        def watch_owner():
            # EOF also ends the owned child if its launcher disappears.
            sys.stdin.readline()
            server.should_exit = True
        threading.Thread(target=watch_owner, daemon=True).start()

    async def serve():
        task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started and not task.done():
            await asyncio.sleep(0.01)
        if server.started:
            print(json.dumps({'url': 'http://' + authority, 'mode': args.mode}), flush=True)
        await task

    try:
        asyncio.run(serve())
    finally:
        sock.close()


if __name__ == '__main__':
    main()
