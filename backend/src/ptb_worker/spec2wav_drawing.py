"""Large drawing snapshots use existing immutable managed inputs, never DB DDL."""
import hashlib
import json
from .store import JobError,canonical


def persist(store, owner, project, strokes):
    raw=canonical(strokes).encode();sha=hashlib.sha256(raw).hexdigest()
    if not 0<len(raw)<=1_000_000:raise JobError('spectrogram_budget',422)
    name=sha+'.m09-drawing.json'
    if not store.postgres:return store.files.import_input(raw,name,'spectral_drawing')
    from ptb_api.storage_models import UploadInput
    from ptb_api.quota import CHUNK_BYTES
    storage=store.files.storage
    # Unique attempt key avoids racing partial uploads. Normal job idempotency is
    # checked before this function and again when the complete asset is linked.
    from uuid import uuid4
    asset=storage.create(owner,UploadInput(project_id=project,name=name,expected_bytes=len(raw),idempotency_key=uuid4().hex))
    for offset in range(0,len(raw),CHUNK_BYTES):storage.append(owner,asset['id'],offset,raw[offset:offset+CHUNK_BYTES])
    result=storage.finalize(owner,asset['id'],sha)
    return dict(asset_id=result['id'],sha256=result['sha256'])


def expand(config, raw):
    from ptb_api.spec2wav_models import Spec2WavConfig
    if not 0<len(raw)<=1_000_000:raise ValueError('spectrogram_budget')
    return Spec2WavConfig.model_validate({**config,'strokes':json.loads(raw)}).snapshot()
