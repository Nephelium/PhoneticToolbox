"""Produce a test bundle with the untouched V2 serializer, never edit V2/user data."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import numpy as np


def prepare(saved, destination, v2_root):
    saved,destination,v2_root=map(Path,(saved,destination,v2_root))
    destination.mkdir(parents=True,exist_ok=False)
    frames=[json.loads(line) for line in (saved/'frames.jsonl').read_text('utf8').splitlines()]
    valid=[r for r in frames if r['detected']]
    metrics={k:[r['metrics'][k] for r in valid] for k in valid[0]['metrics']}
    times=[r['time_s'] for r in valid]
    source=v2_root/'phonetic_toolbox/gui/widgets/lip_gui.py';raw=source.read_bytes();tree=ast.parse(raw.decode('utf8'))
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_save_offline_recognition')
    # Explicit manual offset bypasses the unused auto suggestion; the serializer
    # and all metadata/array construction execute from the original V2 source.
    namespace=dict(Path=Path,np=np,pickle=pickle,datetime=datetime)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
    v2=destination/'v2-original';v2.mkdir()
    namespace[method.name](None,v2,metrics,[np.asarray(r['points'],np.float32) for r in valid],times,[1000+t for t in times],
                           np.zeros(48000),48000,1024,[1000],1000.,{},lip_offset_seconds=.125)
    for stem in ('v3','v2'):shutil.copyfile(saved/'audio_recording.wav',destination/(stem+'.wav'))
    shutil.copyfile(saved/'audio_recording.lip.json',destination/'v3.lip.json')
    shutil.copyfile(v2/'audio_recording.pkl',destination/'v2.pkl')
    shutil.copyfile(v2/'audio_recording_timestamps.pkl',destination/'v2_timestamps.pkl')
    result=dict(v2_source_sha256=hashlib.sha256(raw).hexdigest(),v2_serializer=method.name,v2_frames=len(valid),v3_frames=len(frames))
    (destination/'fixture.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf8')
    return result
