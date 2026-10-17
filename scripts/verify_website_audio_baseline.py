"""Check frozen controls and reject scientific drift before any write occurs."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import website_audio_baseline as baseline
import build_website_gl_comparison as three_way
import build_website_four_comparison as four

policy=baseline.verify()
active_before=next(iter(policy['cases'].values()))['file']
original = baseline.read_json
checks = []
for field, value in [('phase_initialization','pulse'),('momentum',.99),('hop',40),('seed',0),('iterations',256)]:
    def changed(path):
        data=deepcopy(original(path))
        if path.name=='provenance.json' and path.parent.name=='audio':
            row=next(r for r in data['files'] if r['file']==active_before)
            row['parameters']['gl'][field]=value
        return data
    try:
        with patch.object(baseline,'read_json',side_effect=changed):baseline.verify()
    except AssertionError:checks.append(field+' drift rejected')
    else:raise AssertionError('Control drift accepted: '+field)
policy=baseline.verify()
for case in policy['cases'].values():
    path=baseline.SITE/'assets/audio'/case['file'];before=baseline.sha(path.read_bytes())
    try:baseline.protect_pending([(path,b'new replacement')])
    except AssertionError as exc:assert 'cannot be replaced' in str(exc)
    else:raise AssertionError('Replacement was accepted')
    assert baseline.sha(path.read_bytes())==before
try:baseline.generate(Path('unused'))
except AssertionError as exc:assert 'already frozen' in str(exc)
else:raise AssertionError('Automatic regeneration was accepted')
print(json.dumps(dict(success=True,checks=checks,changed_audio_rejected=6,automatic_regeneration_rejected=True)))

# Reject a mismatched neural input and scientific drift in the accelerated track.
three_way.verify()
three_read=four.read
active_after=three_read(baseline.SITE/'assets/spectrogram-examples/comparison.json')['pairs'][0]['after']
for field,value in [('vocoder_input_sha256','0'*64),('repair_model_step',2000),('pulse_initialization',True),('parameter_prediction',True)]:
    def changed_three(path):
        data=deepcopy(three_read(path))
        if path.name=='provenance.json' and path.parent.name=='audio':
            row=next(r for r in data['files'] if r['file']==active_after)
            row['parameters'][field]=value
        return data
    try:
        with patch.object(four,'read',side_effect=changed_three):three_way.verify()
    except AssertionError:pass
    else:raise AssertionError('Three-way drift accepted: '+field)
for case in three_way.verify()['cases'].values():
    for kind in ['accelerated','predicted','after']:
        path=baseline.SITE/'assets/audio'/case[kind]['file']
        try:baseline.protect_pending([(path,b'changed')])
        except AssertionError:pass
        else:raise AssertionError('Frozen comparison replaced')
print(json.dumps(dict(success=True,matched_tracks=24,mismatched_neural_input_rejected=True,scientific_drift_rejected=True)))

# A completed 32-iteration baseline cannot re-enter the 256 -> 32 migration.
try:three_way.verify(expected_settings=dict(baseline.SETTINGS,iterations=256))
except AssertionError:pass
else:raise AssertionError('Completed migration accepted old 256 settings')
history=three_way.read(baseline.SITE/three_way.LOCK)['preserved_audio']
name=next(iter(history));target=baseline.SITE/'assets/audio'/name
try:baseline.protect_pending([(target,b'changed history')])
except AssertionError:pass
else:raise AssertionError('Preserved history replaced')
print(json.dumps(dict(success=True,iterations=32,migration_closed=True,preserved_history_guard=True)))

# Predictions must be applied consistently, and the vocoder must consume route 3.
four.verify()
original_four=four.read
for target,field,value in [('predicted','gl',dict(baseline.SETTINGS,momentum=.99)),('after','parameter_model_sha256','0'*64),('after','vocoder_input_file','route-two.wav')]:
    def corrupt(path):
        data=deepcopy(original_four(path))
        if path.name=='provenance.json' and path.parent.name=='audio':
            pair=original_four(baseline.SITE/'assets/spectrogram-examples/comparison.json')['pairs'][0]
            next(r for r in data['files'] if r['file']==pair[target])['parameters'][field]=value
        return data
    try:
        with patch.object(four,'read',side_effect=corrupt):four.verify()
    except AssertionError:pass
    else:raise AssertionError('Four-route scientific drift accepted: '+field)
print(json.dumps(dict(success=True,prediction_application_and_same_pcm_rejected_on_drift=True)))
try:four.generate(None)
except AssertionError as exc:assert 'frozen' in str(exc)
else:raise AssertionError('Frozen four-way export regenerated')
