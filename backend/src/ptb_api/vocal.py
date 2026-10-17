"""Same-origin, account/project-scoped VTL sessions. Science runs in a worker."""
from typing import Any
from uuid import UUID
from fastapi import APIRouter,Depends,HTTPException,Request
from pydantic import Field
from .models import WireModel
from .account_store import hash_token
from ptb_worker.vocal_sessions import VocalError


class VocalCall(WireModel):
    op:str=Field(min_length=1,max_length=80)
    body:dict[str,Any]=Field(default_factory=dict)


class VocalSessionView(WireModel):
    id:str
    idle_seconds:int


def create_vocal_router(ctx,service):
    router=APIRouter(prefix='/api/v1/vocal',tags=['vocal-tract'])
    def identity(request,session,project):
        if service is None:raise HTTPException(503,'vocal_unavailable')
        ctx.project(session,str(project))
        if not ctx.store.get_project(session['id'],str(project)):raise HTTPException(404,'project_not_found')
        return (str(session['id']),hash_token(request.cookies.get(ctx.settings.cookie,'')),str(project))

    @router.post('/sessions',response_model=VocalSessionView,operation_id='open_vocal_session')
    def open_session(project_id:UUID,request:Request,session=Depends(ctx.mutation)):
        try:return service.open(identity(request,session,project_id))
        except VocalError as e:raise HTTPException(e.status,e.code) from None

    @router.post('/sessions/{session_id}',operation_id='invoke_vocal_session')
    def invoke(session_id:UUID,project_id:UUID,body:VocalCall,request:Request,session=Depends(ctx.mutation)):
        try:return service.invoke(str(session_id),identity(request,session,project_id),body.op,body.body)
        except VocalError as e:raise HTTPException(e.status,e.code) from None

    @router.delete('/sessions/{session_id}',operation_id='close_vocal_session')
    def close(session_id:UUID,project_id:UUID,request:Request,session=Depends(ctx.mutation)):
        try:return service.release(str(session_id),identity(request,session,project_id))
        except VocalError as e:raise HTTPException(e.status,e.code) from None
    return router
