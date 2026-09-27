"""Public deterministic /a/-like probe; a pipeline check, not accuracy evidence."""
import math
import struct
import wave
from pathlib import Path


def generate(root, word='a', syllables=4):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rate = 16000
    samples = []
    for index in range(int((.4 + syllables * .6) * rate)):
        t = index / rate - .2
        local = t % .6
        value = 0.
        if 0 <= t < syllables * .6 and local < .48:
            for harmonic in range(1, 31):
                f = harmonic * 180.
                weight = sum(math.exp(-.5*((f-c)/w)**2) for c,w in [(750.,100.),(1200.,140.),(2600.,200.)]) / harmonic
                value += weight * math.sin(2*math.pi*f*t)
            value *= min(1., local/.04, (.48-local)/.04) * .75
        samples.append(max(-32767,min(32767,round(value*32767))))
    with wave.open(str(root/'probe.wav'),'wb') as stream:
        stream.setparams((1,2,rate,len(samples),'NONE','not compressed'))
        stream.writeframes(struct.pack('<'+'h'*len(samples),*samples))
    (root/'probe.lab').write_text(' '.join([word]*syllables),encoding='utf8')


def register(runtime_path, model_path, dictionary_path, *, root=None, publish=True):
    from .runtime import registry_root, run, fingerprint, resolve_runtime, load_registry
    from .components import atomic_json, digest, no_links
    from uuid import uuid4
    import json
    import shutil
    root = Path(root or registry_root())
    no_links(root)
    runtime, _ = resolve_runtime(runtime_path)
    model, dictionary = Path(model_path).absolute(), Path(dictionary_path).absolute()
    for p in (model,dictionary):
        no_links(p)
        if not p.is_file():raise ValueError('m11_model_missing')
    if model.stat().st_size>200_000_000 or dictionary.stat().st_size>16_000_000:
        raise ValueError('m11_model_budget')
    # Model-specific public probe support is explicit. Do not swap the model or
    # dictionary to make an unsupported environment appear validated.
    words=set()
    with dictionary.open(encoding='utf-8-sig') as stream:
        for line in stream:
            fields=line.split()
            if fields and fields[0] in ('a','啊'):words.add(fields[0])
    if not words:raise ValueError('m11_probe_word_unavailable')
    word='a' if 'a' in words else '啊'
    workspace=root/'checks'/uuid4().hex
    workspace.mkdir(parents=True)
    generate(workspace/'corpus',word)
    shutil.copyfile(model,workspace/'model.zip')
    shutil.copyfile(dictionary,workspace/'dictionary.dict')
    before=fingerprint(runtime)
    hashes=dict(model_sha256=digest(model),dictionary_sha256=digest(dictionary))
    evidence={}
    try:
        response=run(runtime,workspace,dict(action='align',model=str(workspace/'model.zip'),
                    dictionary=str(workspace/'dictionary.dict'),config=dict(beam=10,retry_beam=40),expected_files=1),evidence=evidence)
        if before!=fingerprint(runtime) or hashes!=dict(model_sha256=digest(model),dictionary_sha256=digest(dictionary)):
            raise ValueError('m11_component_changed')
    except Exception as exc:
        atomic_json(workspace/'receipt.json',dict(success=False,error=str(exc),resources=evidence))
        raise
    receipt=dict(success=True,versions=response['versions'],resources=evidence,probe='public-formant-a/1',
                 runtime_fingerprint=before,**hashes)
    atomic_json(workspace/'receipt.json',receipt)
    runtime_id='mfa338-'+before[:16]
    model_id='model-'+hashes['model_sha256'][:12]+'-'+hashes['dictionary_sha256'][:8]
    r=dict(id=runtime_id,path=str(runtime),version='3.3.8',fingerprint=before,validated=True,receipt=str(workspace/'receipt.json'),
           receipt_sha256=digest(workspace/'receipt.json'),platform='windows',arch='x86_64',
           versions=response['versions'],installed_bytes=sum(p.stat().st_size for p in runtime.rglob('*') if p.is_file() and '__pycache__' not in p.parts),
           download_bytes=None,source='user-selected existing environment; local content identity only')
    managed={}
    for role,source in (('model',model),('dictionary',dictionary)):
        folder=root/'resources'/role
        no_links(folder);folder.mkdir(parents=True,exist_ok=True)
        destination=folder/(hashes[role+'_sha256']+('.zip' if role=='model' else '.dict'))
        no_links(destination)
        if not destination.exists():
            temporary=folder/(uuid4().hex+'.part')
            shutil.copyfile(source,temporary)
            if digest(temporary)!=hashes[role+'_sha256']:raise ValueError('m11_model_changed')
            import os
            os.replace(temporary,destination)
        if digest(destination)!=hashes[role+'_sha256']:raise ValueError('m11_model_changed')
        managed[role]=str(destination)
    m=dict(id=model_id,name=model.name,**managed,validated_runtime=runtime_id,model_bytes=model.stat().st_size,dictionary_bytes=dictionary.stat().st_size,**hashes)
    prepared=dict(runtime_id=runtime_id,model_id=model_id,receipt=receipt,runtime_record=r,model_record=m)
    if publish: publish_registration(prepared,root)
    return prepared


def publish_registration(prepared,root):
    import json
    from .components import atomic_json
    with _registry_lock:
        state_file=Path(root)/'registry.json'
        registry=json.loads(state_file.read_text(encoding='utf8')) if state_file.exists() else dict(schema='m11-registry/1',runtimes=[],models=[])
        r,m=prepared['runtime_record'],prepared['model_record']
        registry['runtimes']=[old for old in registry['runtimes'] if old['id']!=r['id']]+[r]
        registry['models']=[old for old in registry['models'] if old['id']!=m['id']]+[m]
        atomic_json(state_file,registry)


from threading import Lock
_registry_lock=Lock()
