"""Bound account JSON request size before framework parsing, without logging bodies."""
from urllib.parse import urlsplit
import re
from starlette.responses import JSONResponse
from .quota import CHUNK_BYTES

class AccountBoundary:
    def __init__(self, app, origin=None, max_bytes=16384):
        self.app,self.origin,self.max_bytes=app,origin,max_bytes

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or not scope['path'].startswith(('/api/v1/auth/','/api/v1/projects','/api/v1/jobs','/api/v1/storage','/api/v1/assets','/api/v1/uploads')):
            return await self.app(scope,receive,send)
        async def reject(code,status):
            await JSONResponse({'detail':code},status_code=status,headers={'Cache-Control':'no-store'})(scope,receive,send)
        if self.origin:
            expected=urlsplit(self.origin).netloc.encode('ascii')
            if [v for k,v in scope['headers'] if k==b'host'] != [expected]:
                return await reject('host_rejected',403)
        # This binary route authenticates before reading and bounds its stream
        # per input role. Do not buffer audio as small account JSON here.
        if scope['method']=='POST' and scope['path'] in ('/api/v1/jobs/local-inputs','/api/v1/jobs/local-lip-conversion'):
            return await self.app(scope,receive,send)
        if scope['method'] in ('POST','PATCH','PUT'):
            limit = CHUNK_BYTES if scope['method']=='PUT' and re.fullmatch(r'/api/v1/uploads/[0-9a-fA-F-]{36}/blocks',scope['path']) else self.max_bytes
            if scope['method']=='POST' and scope['path']=='/api/v1/jobs/batches/create':limit=1_000_000
            if scope['method']=='PUT' and re.fullmatch(r'/api/v1/jobs/m05/uploads/[0-9a-fA-F-]{36}',scope['path']):limit=350_000
            size=0
            chunks=[]
            while True:
                message=await receive()
                if message['type']=='http.disconnect':return
                block=message.get('body',b'')
                size+=len(block)
                if size>limit:
                    return await reject('request_too_large',413)
                chunks.append(block)
                if not message.get('more_body',False):break
            consumed=False
            async def replay():
                nonlocal consumed
                if consumed:return await receive()
                consumed=True
                return {'type':'http.request','body':b''.join(chunks),'more_body':False}
            return await self.app(scope,replay,send)
        await self.app(scope,receive,send)
