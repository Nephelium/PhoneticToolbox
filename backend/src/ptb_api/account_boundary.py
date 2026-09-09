"""Bound account JSON request size before framework parsing, without logging bodies."""
from urllib.parse import urlsplit
from starlette.responses import JSONResponse

class AccountBoundary:
    def __init__(self, app, origin=None, max_bytes=16384):
        self.app,self.origin,self.max_bytes=app,origin,max_bytes

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or not scope['path'].startswith(('/api/v1/auth/','/api/v1/projects','/api/v1/jobs')):
            return await self.app(scope,receive,send)
        async def reject(code,status):
            await JSONResponse({'detail':code},status_code=status,headers={'Cache-Control':'no-store'})(scope,receive,send)
        if self.origin:
            expected=urlsplit(self.origin).netloc.encode('ascii')
            if [v for k,v in scope['headers'] if k==b'host'] != [expected]:
                return await reject('host_rejected',403)
        if scope['method'] in ('POST','PATCH','PUT'):
            size=0
            chunks=[]
            while True:
                message=await receive()
                if message['type']=='http.disconnect':return
                block=message.get('body',b'')
                size+=len(block)
                if size>self.max_bytes:
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
