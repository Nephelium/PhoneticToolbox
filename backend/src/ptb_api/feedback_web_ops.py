"""Small fixed-route loopback bridge; no manager-supplied URL or credentials."""
import json
from urllib.parse import urlsplit,urlencode
from urllib.request import Request,build_opener,ProxyHandler,HTTPRedirectHandler
from urllib.error import HTTPError,URLError
from .feedback import FeedbackError


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


class WebOpsGateway:
    def __init__(self,config):
        u=urlsplit(config['url'])
        if u.scheme!='http' or u.hostname!='127.0.0.1' or not u.port or u.path!='/api/v1/_ops' or u.query or u.fragment or u.username:
            raise ValueError('Operations bridge must use the fixed loopback API')
        origin=urlsplit(config['origin'])
        if origin.scheme!='https' or not origin.netloc or origin.path or origin.username or origin.query or origin.fragment:
            raise ValueError('Operations origin must be HTTPS')
        if not 32<=len(config['key'])<=128:raise ValueError('Private operations key required')
        self.url,self.key,self.host=config['url'],config['key'],origin.netloc
        self.opener=build_opener(ProxyHandler({}),NoRedirect())

    def call(self,path,body=None):
        request=Request(self.url+path,data=None if body is None else json.dumps(body).encode(),headers={
            'Authorization':'Bearer '+self.key,'Content-Type':'application/json','Host':self.host,'X-Forwarded-Proto':'https'})
        try:
            with self.opener.open(request,timeout=8) as r:
                raw=r.read(131073)
                if len(raw)>131072:raise ValueError('Response limit')
                return json.loads(raw)
        except HTTPError as e:
            try:code=json.loads(e.read(4096)).get('detail')
            except ValueError:code=None
            safe={'capacity_request_closed','capacity_not_increase','disk_space_low','storage_frozen','invalid_capacity_grant','account_not_found','web_server_busy'}
            raise FeedbackError(code if code in safe else 'web_operations_unavailable',e.code if code in safe else 503) from None
        except (OSError,URLError,ValueError):raise FeedbackError('web_operations_unavailable',503) from None

    def snapshot(self):return self.call('/status')
    def requests(self,state='pending',before=None):return self.call('/requests?'+urlencode({'state':state,**({'before':before} if before else {})}))
    def decision(self,body):return self.call('/decision',body.model_dump(mode='json'))
    def quota(self,body):return self.call('/quota',body.model_dump(mode='json'))
