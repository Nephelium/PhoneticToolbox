"""Account-bound feedback. Only the server supplies the verified contact email."""
import json
from typing import Literal
from urllib.request import Request,build_opener,ProxyHandler,HTTPRedirectHandler
from urllib.error import HTTPError,URLError
from fastapi import APIRouter,Depends,HTTPException
from pydantic import Field
from .models import WireModel

class WebFeedbackRequest(WireModel):
    operation:Literal['status','health','unread','messages','send','read','setMailNotifications']
    args:dict=Field(default_factory=dict,max_length=12)

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

class WebFeedbackGateway:
    def __init__(self,key):
        if not isinstance(key,str) or not 32<=len(key)<=128:raise ValueError('Private feedback bridge key required')
        self.key=key;self.opener=build_opener(ProxyHandler({}),NoRedirect())
    def call(self,email,operation,args):
        raw=json.dumps(dict(email=email,operation=operation,args=args)).encode()
        if len(raw)>14_000_000:raise HTTPException(413,'request_too_large')
        request=Request('http://127.0.0.1:18766/api/v1/feedback/internal/web-operation',data=raw,
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
        try:
            with self.opener.open(request,timeout=20) as response:
                encoded=response.read(18_000_001)
                if len(encoded)>18_000_000:raise ValueError('Response budget')
                return json.loads(encoded)
        except HTTPError as exc:
            try:code=json.loads(exc.read(4096)).get('detail')
            except ValueError:code=None
            safe={'invalid_image','invalid_message','message_key_conflict','message_rate_limited','conversation_budget','invalid_mail_preference'}
            raise HTTPException(exc.code if code in safe else 503,code if code in safe else 'feedback_service_unavailable') from None
        except (OSError,URLError,ValueError):raise HTTPException(503,'feedback_service_unavailable') from None

def create_web_feedback_router(ctx,gateway):
    router=APIRouter(prefix='/api/v1/web-feedback',tags=['feedback'])
    @router.post('',operation_id='account_feedback_operation')
    def operation(body:WebFeedbackRequest,session=Depends(ctx.mutation)):
        if gateway is None:raise HTTPException(503,'feedback_service_unavailable')
        email=ctx.store.verified_email(session['id'])
        if not email:raise HTTPException(403,'verified_email_required')
        # Neither a browser-supplied email nor a feedback owner/token is trusted.
        allowed={'messages':{'after'},'send':{'text','images','requestId'},'read':{'through'},'setMailNotifications':{'mailNotifications'}}.get(body.operation,set())
        if set(body.args)-allowed:raise HTTPException(422,'invalid_request')
        return gateway.call(email,body.operation,body.args)
    return router
