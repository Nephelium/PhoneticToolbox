"""M01-D value-preserving serialization of actual core results, outside the API.

No inverse inference from NaN to physiology/failure, and no algorithm execution.
Source relationships remain defined by the registry and the recorded core version.
"""
import math
from phonetic_core import __version__ as core_version
from phonetic_core.models.acoustic import AcousticConfig
from phonetic_core.catalog import PARAMETER_MAPPING
from ptb_api.acoustic_models import (AcousticRequest,AcousticResult,AcousticMetadata,
    AcousticDecodedAudio,AcousticNumericColumn,AcousticTextColumn,AcousticBackendObservation,
    config_digest,parameter_unit,MAX_CELLS)

SOURCE_IDS=('SRC-PRAAT','SRC-REAPER','SRC-IRAPT','SRC-WMPC','SRC-VOICESAUCE','SRC-OPENSAUCE',
            'REF-CPP','REF-HNR','REF-SHR','REF-ISELI','REF-HAWKS','REF-SOE')
KNOWN_REASONS={'native_failed','empty_audio','praat_missing','praat_failed','praat_insufficient',
               'irapt_unavailable_or_insufficient','native_exit_1','native_exit_7','native_exit_9'}


def to_core_config(snapshot):
    return AcousticConfig(**snapshot.settings.model_dump(),
        selected_parameter_keys=None if snapshot.selection.mode=='legacy_service' else tuple(snapshot.selection.keys),
        use_reaper=snapshot.backend_policy.reaper!='disabled')


def build_acoustic_result(result,audio,request: AcousticRequest,inputs,*,native_sha256=None):
    request=AcousticRequest.model_validate_json(request.model_dump_json())
    expected=to_core_config(request.config)
    from dataclasses import asdict
    if result.config_snapshot!=asdict(expected):raise ValueError('Actual core configuration disagrees with requested snapshot')
    if result.sampling_rate!=audio.sample_rate_hz:raise ValueError('Actual decoded sample rate mismatch')
    references={role:getattr(request.inputs,role) for role in ('audio','textgrid','lip') if getattr(request.inputs,role) is not None}
    if len(inputs)!=len(references) or any(i.role not in references or i.asset_id!=references[i.role].asset_id or i.sha256!=references[i.role].sha256 for i in inputs):
        raise ValueError('Resolved inputs disagree with request')
    decoded=AcousticDecodedAudio(sample_rate_hz=audio.sample_rate_hz,sample_count=len(audio.samples),
        channels=1 if audio.samples.ndim==1 else audio.samples.shape[1],sample_dtype=str(audio.samples.dtype))
    backends=[]
    for event in result.backend_events:
        reason=event.get('reason')
        if reason is not None and reason not in KNOWN_REASONS:reason='backend_failure'
        backends.append(AcousticBackendObservation(stage=event['stage'],actual=event['actual'],reason=reason,
            resource_sha256=native_sha256 if event['actual']=='native_reaper' else None))
    if request.config.backend_policy.reaper=='disabled':
        backends.append(AcousticBackendObservation(stage='reaper',actual='disabled'))
    metadata=AcousticMetadata(core_version=core_version,computation_revision=result.computation_revision,project_id=request.project_id,inputs=inputs,decoded=decoded,
        config=request.config,config_sha256=config_digest(request.config),source_ids=list(SOURCE_IDS),backends=backends)
    frame=result.to_dataframe()
    if (len(frame)+1)*len(frame.columns)>MAX_CELLS:raise ValueError('Result cell budget exceeded')
    numeric=[];text=[]
    for key in frame:
        if key=='Time_s':continue
        if key.startswith('text_'):
            if any(not isinstance(v,str) for v in frame[key]):raise ValueError('Annotation values must be strings')
            text.append(AcousticTextColumn(key=key,values=frame[key].tolist()))
            continue
        values=[];masks=[];reasons=[]
        for value in frame[key]:
            value=float(value)
            mask=0 if math.isfinite(value) else 1 if math.isnan(value) else 2 if value>0 else 3
            values.append(value if mask==0 else None);masks.append(mask)
            reasons.append(None if mask==0 else 'legacy_nonfinite_unknown')
        numeric.append(AcousticNumericColumn(key=key,label=PARAMETER_MAPPING.get(key,key),unit=parameter_unit(key),
            scope='catalog' if key in PARAMETER_MAPPING else 'legacy_service_extension',values=values,nonfinite=masks,reason=reasons))
    return AcousticResult(metadata=metadata,times_s=result.time_axis.tolist(),numeric=numeric,text=text,column_order=list(frame.columns))
