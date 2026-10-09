"""Local-only, owned-process view queries over immutable managed inputs."""
import json
import time
import os
from uuid import uuid4
from .acoustic_stream_child import atomic
from .spectrogram_preview import PreviewError


def render(files,asset_id,view):
    if os.name!='nt':raise PreviewError('preview_platform_unverified',503)
    from .native.windows import OwnedProcess
    from .process_entry import command
    from .io.scratch import no_links
    with files.locked():
        asset=dict(files._asset(asset_id))
        if asset['kind']!='input' or asset['role']!='parameter_bundle' or asset['state']!='ready':raise PreviewError('invalid_parameter_table')
        source=files._path(asset_id)
    parent=files.scratch_root/'parameter-views';no_links(parent);parent.mkdir(exist_ok=True)
    cache=parent/(asset['sha256']+'.sqlite');no_links(cache)
    root=parent/uuid4().hex;root.mkdir()
    atomic(root/'request.json',dict(source=str(source),cache=str(cache),name=asset['name'],sha256=asset['sha256'],view=view))
    process=OwnedProcess(command('ptb_worker.parameter_bundle_child',str(root)),root,1_073_741_824)
    try:
        deadline=time.monotonic()+300
        while process.poll() is None:
            if time.monotonic()>deadline:raise PreviewError('parameter_read_timeout',503)
            time.sleep(.05)
        response=root/'response.json'
        if not response.exists() or response.stat().st_size>64_000_000:raise PreviewError('parameter_read_failed')
        value=json.loads(response.read_text('utf-8'))
        if 'error' in value:raise PreviewError(value['error'])
        return value['result']
    finally:process.close()
