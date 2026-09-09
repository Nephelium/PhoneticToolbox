from typing import Literal

from fastapi import FastAPI
from phonetic_core import __version__ as core_version

from . import __version__
from .models import Capabilities, Health, Viewport
from .protocol_version import API_VERSION


def create_app(mode: Literal['local', 'server'] = 'server') -> FastAPI:
    if mode not in ('local', 'server'):
        raise ValueError('Unsupported service mode')
    app = FastAPI(title='PhoneticToolbox API', version=API_VERSION,
                  description='P02 read-only scaffold; no research processing or user storage.')

    @app.get('/api/v1/health', response_model=Health, operation_id='get_health')
    def health() -> Health:
        return Health(app_version=__version__, core_version=core_version, mode=mode)

    @app.get('/api/v1/capabilities', response_model=Capabilities, operation_id='get_capabilities')
    def capabilities() -> Capabilities:
        return Capabilities(algorithms=[], limitations=[
            'Scientific modules pending P03/P08', 'Accounts and tasks pending P05/P06'])

    base_openapi = app.openapi

    def openapi():
        schema = base_openapi()
        # Shared process models are published without adding a fake analysis endpoint.
        shared = Viewport.model_json_schema(ref_template='#/components/schemas/{model}')
        definitions = shared.pop('$defs', {})
        schema.setdefault('components', {}).setdefault('schemas', {}).update(definitions)
        schema['components']['schemas']['Viewport'] = shared
        return schema

    app.openapi = openapi
    return app
