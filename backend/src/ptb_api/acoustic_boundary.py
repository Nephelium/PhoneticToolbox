"""M01-D trusted metadata boundary; F must call it within owner/fencing locks.

No public request may supply TrustedAcousticAsset. Lookup must be provided by the
authenticated host/store. No filesystem paths, database access or fake job success.
"""
from dataclasses import dataclass
import math
from .acoustic_models import AcousticInputSnapshot, AcousticRequest
from .storage_policy import RETENTION_SECONDS


class AcousticBoundaryError(ValueError):
    def __init__(self,code):
        self.code=code
        super().__init__(code)


@dataclass(frozen=True)
class TrustedAcousticAsset:
    asset_id: str
    owner_id: str
    project_id: str
    sha256: str
    role: str
    name: str
    state: str
    expires_at: float | None


def resolve_inputs(request: AcousticRequest,owner_id,lookup,now,*,mode='server'):
    if mode not in ('server','local') or not math.isfinite(now) or now<0:raise ValueError('Invalid host context')
    snapshots=[]
    for role in ('audio','textgrid','lip'):
        reference=getattr(request.inputs,role)
        if reference is None:continue
        asset=lookup(reference.asset_id)
        if (not isinstance(asset,TrustedAcousticAsset) or asset.asset_id!=reference.asset_id or
            asset.owner_id!=owner_id or asset.project_id!=request.project_id or asset.state!='ready'):
            raise AcousticBoundaryError('asset_unavailable')
        if mode=='server' and (asset.expires_at is None or not math.isfinite(asset.expires_at) or asset.expires_at<=now):
            raise AcousticBoundaryError('input_expired')
        if asset.sha256!=reference.sha256:raise AcousticBoundaryError('input_changed')
        if asset.role!=role:raise AcousticBoundaryError('input_kind_mismatch')
        snapshots.append(AcousticInputSnapshot(asset_id=asset.asset_id,sha256=asset.sha256,role=role,
            expires_at=asset.expires_at if mode=='server' else None))
    return snapshots


def choose_named_association(candidates,*,owner_id,project_id,name,role):
    """Optional auto-association over already queried metadata; ambiguity is an error."""
    matches=[a for a in candidates if isinstance(a,TrustedAcousticAsset) and a.owner_id==owner_id and
             a.project_id==project_id and a.name.casefold()==name.casefold() and a.role==role and a.state=='ready']
    if len(matches)>1:raise AcousticBoundaryError('ambiguous_association')
    return matches[0] if matches else None


def output_expiry(*,mode,operation,completed_at,inputs):
    """Policy only. F must re-resolve expiry/cancellation immediately before commit."""
    if mode not in ('server','local') or operation not in ('analysis','segment'):
        raise ValueError('Unknown retention policy')
    if not math.isfinite(completed_at) or completed_at<0:raise ValueError('Invalid completion time')
    if mode=='local':return None
    if not inputs or any(i.expires_at is None or i.expires_at<=completed_at for i in inputs):
        raise AcousticBoundaryError('input_expired')
    if operation=='analysis':return completed_at+RETENTION_SECONDS
    return min(completed_at+RETENTION_SECONDS,*(i.expires_at for i in inputs))
