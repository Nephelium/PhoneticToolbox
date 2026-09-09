"""Every project lookup/mutation receives the authenticated owner explicitly."""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from .account_models import ProjectInput, ProjectList, ProjectView
from .auth import AccountContext


def create_project_router(ctx: AccountContext):
    router = APIRouter(prefix='/api/v1/projects',tags=['projects'])

    @router.get('',response_model=ProjectList,operation_id='list_projects')
    def listing(session=Depends(ctx.session)):
        return ProjectList(projects=ctx.store.list_projects(session['id']))

    @router.post('',response_model=ProjectView,status_code=201,operation_id='create_project')
    def create(body: ProjectInput, session=Depends(ctx.mutation)):
        result=ctx.store.create_project(session['id'],body.name)
        if result is None:
            raise HTTPException(409,'project_limit_reached')
        return result

    @router.get('/{project_id}',response_model=ProjectView,operation_id='get_project')
    def get(project_id: UUID, session=Depends(ctx.session)):
        result=ctx.store.get_project(session['id'],str(project_id))
        if not result:
            raise HTTPException(404,'project_not_found')
        return result

    @router.patch('/{project_id}',response_model=ProjectView,operation_id='rename_project')
    def rename(project_id: UUID, body: ProjectInput, session=Depends(ctx.mutation)):
        result=ctx.store.rename_project(session['id'],str(project_id),body.name)
        if not result:
            raise HTTPException(404,'project_not_found')
        return result

    return router
