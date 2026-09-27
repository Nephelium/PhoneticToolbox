"""M01-D invalid public configuration must fail before scientific execution."""
import pytest
from pydantic import ValidationError
from ptb_api.acoustic_models import AcousticSettings
from ptb_api.acoustic_models import (
    AcousticRequest, AcousticConfigSnapshot, AcousticSelection, AcousticMetadata,
    AcousticResult, AcousticNumericColumn, AcousticFileManifest, AcousticBatchSummary,
    config_digest, parameter_unit, RETENTION_SECONDS,
)
from ptb_api.acoustic_boundary import (
    TrustedAcousticAsset, AcousticBoundaryError, resolve_inputs,
    choose_named_association, output_expiry,
)
from ptb_api.job_models import ResultManifestEnvelope, JobInput, FileJobInput
from phonetic_core.catalog import PARAMETER_MAPPING
from dataclasses import replace
import json
from pathlib import Path


def uid(n):return f'00000000-0000-4000-8000-{n:012d}'


def request():
    return AcousticRequest(project_id=uid(1),idempotency_key='contract-test-01',
        inputs={'audio':{'asset_id':uid(2),'sha256':'a'*64}})


def asset():
    return TrustedAcousticAsset(uid(2),uid(9),uid(1),'a'*64,'audio','sample.wav','ready',200.)


def metadata():
    config=AcousticConfigSnapshot()
    return AcousticMetadata(core_version='3.0.0a1',project_id=uid(1),
        inputs=resolve_inputs(request(),uid(9),lambda _:asset(),100.),
        decoded={'sample_rate_hz':44100,'channels':2,'sample_count':44100,'sample_dtype':'int16'},
        config=config,config_sha256=config_digest(config),source_ids=['SRC-PRAAT'],backends=[])


def column():
    return dict(key='pF0',label=PARAMETER_MAPPING['pF0'],unit='Hz',scope='catalog',
        values=[0.,None,None,None],nonfinite=[0,1,2,3],
        reason=[None,*(['legacy_nonfinite_unknown']*3)])


def result():
    return AcousticResult(metadata=metadata(),times_s=[0.,.005,.01,.015],
        numeric=[column()],text=[{'key':'text_声调','values':['阴平','ɑ̃','', '=1+1']}],
        column_order=['Time_s','pF0','text_声调'])


def manifest():
    expiry=100.+RETENTION_SECONDS
    return AcousticFileManifest(policy_version=2,job_id=uid(3),metadata=metadata(),completed_at=100.,
        retention='server',expires_at=expiry,row_count=4,
        files=[{'asset_id':uid(n),'format':fmt,'size_bytes':200,'sha256':'b'*64,'expires_at':expiry}
               for n,fmt in ((4,'xlsx'),(5,'sqlite'))])


def batch(states):
    counts={s:states.count(s) for s in ('not_started','queued','running','cancel_requested','succeeded','failed','cancelled','interrupted')}
    closed=not any(s in ('queued','running','cancel_requested') for s in states)
    return dict(batch_id=uid(7),total=len(states),closed=closed,complete=closed and counts['succeeded']==len(states),
        counts=counts,items=[dict(index=i,audio_asset_id=uid(100+i),job_id=None if s=='not_started' else uid(200+i),state=s)
                             for i,s in enumerate(states)])


@pytest.mark.parametrize('bad',[
    {'max_formant':0.}, {'min_f0':500.,'max_f0':100.},
    {'frameshift_ms':float('nan')}, {'windowsize_ms':float('inf')},
    {'reaper_hilbert':1}, {'num_formants':3.5}, {'native_tool':'anything'},
])
def test_public_settings_reject_invalid(bad):
    with pytest.raises(ValidationError):AcousticSettings.model_validate(bad)


@pytest.mark.parametrize('key,value', [('path','C:/Users/13680'),('native_tool','reaper.exe'),('owner_id',uid(9)),('unknown',1)])
def test_request_has_no_path_owner_or_native_capability(key,value):
    raw=request().model_dump();raw[key]=value
    with pytest.raises(ValidationError):AcousticRequest.model_validate(raw)


@pytest.mark.parametrize('selection', [dict(keys=[]),dict(keys=['pF0','pF0']),dict(keys=['Energy']),
    dict(keys=['SOE_pF0']),dict(mode='legacy_service'),dict(keys=[PARAMETER_MAPPING['pF0']])])
def test_catalog_keys_are_not_labels_or_legacy_extensions(selection):
    with pytest.raises(ValidationError):AcousticSelection.model_validate(selection)


def test_catalog_default_and_explicit_legacy_are_separate():
    assert len(AcousticSelection().keys)==80
    assert len(AcousticSettings.model_fields)==14
    assert AcousticSelection(mode='legacy_service',keys=[]).keys==[]
    assert all(parameter_unit(k) for k in PARAMETER_MAPPING)
    assert parameter_unit('Intensity')=='dB_legacy_amplitude_reference_not_measured_SPL'
    assert parameter_unit('LipArea')=='unknown_without_input_metadata'
    raw=request().model_dump();raw['inputs']['lip']=raw['inputs']['audio'].copy()
    with pytest.raises(ValidationError):AcousticRequest.model_validate(raw)


def test_catalog_labels_units_and_settings_match_independent_audit():
    root=Path(__file__).resolve().parents[2]
    audit=json.loads((root/'docs/modules/evidence/M01-parameter-settings.json').read_text('utf-8'))
    assert list(PARAMETER_MAPPING)==[p['key'] for p in audit['parameters']]
    for row in audit['parameters']:
        assert PARAMETER_MAPPING[row['key']]==row['export_label']
        assert parameter_unit(row['key'])==row['legacy_unit']
    assert AcousticSettings().model_dump()=={s['key']:s['default'] for s in audit['settings']}


@pytest.mark.parametrize('changes,code', [
    ({'owner_id':uid(8)},'asset_unavailable'),({'project_id':uid(8)},'asset_unavailable'),
    ({'state':'deleted'},'asset_unavailable'),({'asset_id':uid(8)},'asset_unavailable'),
    ({'expires_at':100.},'input_expired'),({'expires_at':None},'input_expired'),
    ({'expires_at':float('nan')},'input_expired'),({'sha256':'b'*64},'input_changed'),
    ({'role':'lip'},'input_kind_mismatch'),
])
def test_trusted_resolution_rejects_foreign_stale_or_changed_input(changes,code):
    with pytest.raises(AcousticBoundaryError,match=code):
        resolve_inputs(request(),uid(9),lambda _:replace(asset(),**changes),100.)


def test_same_name_ambiguity_and_missing_inputs():
    first=replace(asset(),role='textgrid',name='汉字.TextGrid')
    options=dict(owner_id=uid(9),project_id=uid(1),name='汉字.textgrid',role='textgrid')
    assert choose_named_association([first,replace(first,owner_id=uid(8))],**options)==first
    with pytest.raises(AcousticBoundaryError,match='ambiguous_association'):
        choose_named_association([first,replace(first,asset_id=uid(8))],**options)
    with pytest.raises(AcousticBoundaryError,match='asset_unavailable'):
        resolve_inputs(request(),uid(9),lambda _:None,100.)


def test_completion_based_lifetime_and_expiry_recheck():
    snapshots=resolve_inputs(request(),uid(9),lambda _:asset(),100.)
    assert output_expiry(mode='server',operation='analysis',completed_at=150.,inputs=snapshots)==150.+RETENTION_SECONDS
    assert output_expiry(mode='server',operation='segment',completed_at=150.,inputs=snapshots)==200.
    with pytest.raises(AcousticBoundaryError,match='input_expired'):
        output_expiry(mode='server',operation='analysis',completed_at=200.,inputs=snapshots)
    with pytest.raises(AcousticBoundaryError,match='input_expired'):
        resolve_inputs(request(),uid(9),lambda _:asset(),200.)
    local=resolve_inputs(request(),uid(9),lambda _:asset(),250.,mode='local')
    assert local[0].expires_at is None
    assert output_expiry(mode='local',operation='analysis',completed_at=250.,inputs=local) is None


def test_json_preserves_zero_nan_signed_infinities_text_and_configuration():
    original=result();wire=original.model_dump_json()
    assert 'NaN' not in wire and 'Infinity' not in wire
    restored=AcousticResult.model_validate_json(wire)
    assert restored==original
    assert restored.numeric[0].values==[0.,None,None,None]
    assert restored.numeric[0].nonfinite==[0,1,2,3]
    assert restored.text[0].values==['阴平','ɑ̃','', '=1+1']
    assert json.loads(wire)['metadata']['decoded']['sample_count']==44100


@pytest.mark.parametrize('changes', [dict(nonfinite=[False,1,2,3]),dict(values=[float('nan'),None,None,None]),
    dict(values=[None,None,None,None]),dict(reason=[None,'unvoiced','legacy_nonfinite_unknown','legacy_nonfinite_unknown']),
    dict(nonfinite=[0,1,2]),dict(unit='ms'),dict(label='fake'),dict(scope='legacy_service_extension')])
def test_invalid_masks_or_semantics_rejected(changes):
    raw=column();raw.update(changes)
    with pytest.raises(ValidationError):AcousticNumericColumn.model_validate(raw)


@pytest.mark.parametrize('change', ['digest','policy','native_hash','backend_stage','duplicate_input','grid','length','order','selection','decoded_budget'])
def test_result_cross_field_invariants(change):
    raw=result().model_dump();meta=raw['metadata']
    if change=='digest':meta['config']['settings']['frameshift_ms']=10.
    elif change=='policy':
        meta['config']['backend_policy']['reaper']='native_required'
        meta['config_sha256']=config_digest(AcousticConfigSnapshot.model_validate(meta['config']))
        meta['backends']=[dict(stage='reaper',actual='reaper_python')]
    elif change=='native_hash':meta['backends']=[dict(stage='reaper',actual='native_reaper')]
    elif change=='backend_stage':meta['backends']=[dict(stage='wm_f0',actual='reaper_python')]
    elif change=='duplicate_input':meta['inputs'].append(meta['inputs'][0].copy())
    elif change=='grid':raw['times_s']=[0.,.004,.01,.015]
    elif change=='length':raw['times_s'].pop()
    elif change=='order':raw['column_order']=['Time_s','pF0','pF0']
    elif change=='selection':
        meta['config']['selection']['keys']=['Intensity']
        meta['config_sha256']=config_digest(AcousticConfigSnapshot.model_validate(meta['config']))
    elif change=='decoded_budget':meta['decoded']['sample_count']=2_000_000
    with pytest.raises(ValidationError):AcousticResult.model_validate(raw)


@pytest.mark.parametrize('change', ['missing','duplicate_format','same_id','replace_input','expiry','long_ttl','expired_input','false_success'])
def test_complete_single_file_means_two_valid_outputs(change):
    raw=manifest().model_dump()
    if change=='missing':raw['files'].pop()
    elif change=='duplicate_format':raw['files'][1]['format']='xlsx'
    elif change=='same_id':raw['files'][1]['asset_id']=raw['files'][0]['asset_id']
    elif change=='replace_input':raw['files'][0]['asset_id']=uid(2)
    elif change=='expiry':raw['files'][0]['expires_at']+=1
    elif change=='long_ttl':
        raw['expires_at']+=1
        for f in raw['files']:f['expires_at']=raw['expires_at']
    elif change=='expired_input':raw['metadata']['inputs'][0]['expires_at']=100.
    elif change=='false_success':raw['complete']=False
    with pytest.raises(ValidationError):AcousticFileManifest.model_validate(raw)


@pytest.mark.parametrize('states', [['succeeded']*17,['succeeded','failed','not_started'],['succeeded','running'],['cancelled','interrupted']])
def test_batch_keeps_every_child_and_truthful_counts(states):
    value=AcousticBatchSummary.model_validate(batch(states))
    assert AcousticBatchSummary.model_validate_json(value.model_dump_json())==value
    assert value.complete==(states==['succeeded']*len(states))


@pytest.mark.parametrize('change', ['count','complete','closed','duplicate_job','job_missing','order'])
def test_partial_batch_cannot_claim_complete_or_hide_children(change):
    raw=batch(['succeeded','running'])
    if change=='count':raw['counts']['succeeded']=2
    elif change=='complete':raw['complete']=True
    elif change=='closed':raw['closed']=True
    elif change=='duplicate_job':raw['items'][1]['job_id']=raw['items'][0]['job_id']
    elif change=='job_missing':raw['items'][1]['job_id']=None
    elif change=='order':raw['items'].reverse()
    with pytest.raises(ValidationError):AcousticBatchSummary.model_validate(raw)


def test_old_manifests_stay_readable_but_batch_is_not_a_file_result():
    old=[dict(kind='pipeline_check_metadata',sha256='a'*64,sample_count=4,core_version='3.0.0a1'),
        dict(kind='managed_files',core_version='3.0.0a1',files=[dict(id=uid(4),name='old.bin',kind='result',size_bytes=4,sha256='b'*64,expires_at=200.)])]
    for value in [*old,manifest().model_dump()]:
        wrapped=ResultManifestEnvelope.model_validate({'manifest':value})
        assert ResultManifestEnvelope.model_validate_json(wrapped.model_dump_json())==wrapped
    with pytest.raises(ValidationError):ResultManifestEnvelope.model_validate({'manifest':batch(['succeeded'])})
    for model in (JobInput,FileJobInput):
        with pytest.raises(ValidationError):model.model_validate(request().model_dump())
