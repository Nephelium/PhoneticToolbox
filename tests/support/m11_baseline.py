"""Test-only exact V2 pipeline loader in the supplied MFA interpreter.

Run only with a host-owned fresh workspace and -B. It never calls the old service
that copies application files into auto_alignment. No product imports this file.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import types


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
    root = Path(request['workspace'])
    runtime = Path(request['runtime'])
    source = Path(request['source'])
    for key in ('MFA_ROOT_DIR', 'TEMP', 'TMP', 'TMPDIR', 'NUMBA_CACHE_DIR'):
        os.environ[key] = str(root / 'temp')
    (root / 'temp').mkdir(exist_ok=True)
    os.environ['PATH'] = os.pathsep.join(str(p) for p in [runtime,runtime/'Library/bin',runtime/'Scripts',runtime/'DLLs',Path(os.environ['SystemRoot'])/'System32'])
    handles = [os.add_dll_directory(str(p)) for p in (runtime,runtime/'Library/bin',runtime/'DLLs')]
    for name in ('phonetic_toolbox','phonetic_toolbox.core','phonetic_toolbox.core.transcription'):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    codec=load('baseline_codec',source/'phonetic_toolbox/core/transcription/mfa_name_codec.py')
    module=sys.modules['phonetic_toolbox.core.transcription']
    module.encode_fs_name=codec.encode_fs_name
    module.decode_fs_name=codec.decode_fs_name
    pipeline=load('baseline_pipeline',source/'phonetic_toolbox/services/pipelines/mfa_alignment_pipeline.py')
    with (root/'baseline.log').open('w',encoding='utf8') as log:
        sys.stdout=log;sys.stderr=log
        success,message=pipeline.MFAAlignmentPipeline().run(request['corpus'],request['dictionary'],request['model'],str(root/'result'),beam=10,retry_beam=40)
        (root/'response.json').write_text(json.dumps(dict(success=success,message=message),ensure_ascii=False),encoding='utf8')


if __name__=='__main__':main()
