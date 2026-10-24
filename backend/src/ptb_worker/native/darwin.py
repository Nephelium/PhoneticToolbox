"""macOS local child ownership, bounded pipes and sampled process-tree memory.

This is a desktop boundary. It does not claim Linux cgroup enforcement.
Only trusted callers select an entry; no submitted argv or shell is accepted.
"""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
from functools import lru_cache
from ..io.limits import Cancelled,FormatError,LimitError

ENTRIES={'m14':'ptb_worker.m14_child','m07':'ptb_worker.m07_child','m06':'ptb_worker.m06_child',
         'm08':'ptb_worker.m08_child','lpc':'ptb_worker.lpc_child','egg':'ptb_worker.egg_child',
         'acoustic':'ptb_worker.science_child','segment':'ptb_worker.segment_child',
         'fonts':'ptb_worker.font_preflight','spectrogram':'ptb_worker.spectrogram_preview',
         'spec2wav':'ptb_worker.spec2wav_child','spec2wav_preview':'ptb_worker.spec2wav_preview',
         'egg_interactive':'ptb_worker.egg_interactive_child','parameter_preview':'ptb_worker.parameter_preview',
         'lip':'ptb_worker.m05_child','lip_media':'ptb_worker.m05_media_export','vocal_mp4':'ptb_worker.vocal_mp4',
         'acoustic_stream':'ptb_worker.acoustic_stream_child','parameter_bundle':'ptb_worker.parameter_bundle_child',
         'egg_persistent':'ptb_worker.egg_interactive_child'}
VERSIONS={'numpy':'2.2.6','scipy':'1.16.3','praat-parselmouth':'0.4.7','pandas':'2.3.3','matplotlib':'3.10.8'}

def bundle_root():
    return Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[4]))

def load_profile():
    root=bundle_root();cache=Path(os.environ.get('PTB_MACOS_CACHE',Path.home()/'Library/Caches/PhoneticToolbox/v3/science'))
    cache.mkdir(parents=True,exist_ok=True)
    binary=root/'resources/research/reaper'
    return root/'resources/macos-validation.json',{'cache':str(cache),'python':sys.executable,
        'reaper_binary':str(binary),'hashes':{str(binary):hashlib.sha256(binary.read_bytes()).hexdigest()} if binary.is_file() else {}}

def fingerprint():
    if sys.platform!='darwin':raise FormatError('macos_runtime_unavailable')
    actual={name:metadata.version(name) for name in VERSIONS}
    if actual!=VERSIONS:raise FormatError('macos_runtime_version_mismatch')
    import scipy,numpy
    return {'platform':'darwin','versions':actual,'blas':scipy.__config__.CONFIG['Build Dependencies']['blas']['name'],
            'numpy_blas':numpy.__config__.CONFIG['Build Dependencies']['blas']['name'],'equivalence_policy':'macos-numerical-validation/1'}

@lru_cache(maxsize=1)
def _validation():
    root=bundle_root();receipt=json.loads((root/'resources/macos-validation.json').read_text('utf8'))
    if receipt.get('schema')!='ptb-macos-validation/1' or receipt.get('versions')!=VERSIONS:return {}
    hashes=receipt.get('source_sha256',{})
    if not hashes:return {}
    allowed=[root.resolve()]
    if getattr(sys,'frozen',False):allowed.append((root.parent/'Resources').resolve())
    for relative,expected in hashes.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:return {}
        path=(root/relative).resolve()
        if not any(path.is_relative_to(base) for base in allowed) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:return {}
    return receipt

def capability(operation):
    try:
        receipt=_validation()
        if receipt.get('operations',{}).get(operation) is not True:return False
        return all(metadata.version(name)==value for name,value in VERSIONS.items())
    except (OSError,ValueError,metadata.PackageNotFoundError):return False

def command(entry,request):
    if entry not in ENTRIES:raise ValueError('Unsupported fixed macOS entry')
    if getattr(sys,'frozen',False):return [sys.executable,'--darwin-child',entry,str(request)]
    return [sys.executable,'-I','-B',str(bundle_root()/'scripts/macos_science_entry.py'),entry,str(request)]

def _memory(process):
    import psutil
    try:
        parent=psutil.Process(process.pid)
        return sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)] if p.is_running())
    except (psutil.NoSuchProcess,psutil.AccessDenied):return 0

def group_members(group):
    # Darwin can report EPERM for killpg(group, 0) after the last member exits.
    # Inspect group membership directly, without exposing command lines.
    result=subprocess.run(['/bin/ps','-axo','pid=,pgid=,uid=,stat='],capture_output=True,text=True,check=True,timeout=3)
    members=[]
    for line in result.stdout.splitlines():
        parts=line.split()
        if len(parts)==4 and int(parts[1])==group and int(parts[2])==os.getuid() and not parts[3].startswith('Z'):
            members.append(int(parts[0]))
    return members

def _close_group(process):
    # start_new_session makes this exact child the owner of its process group.
    try:os.killpg(process.pid,signal.SIGTERM)
    except (ProcessLookupError,PermissionError):
        if group_members(process.pid):raise
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        process.poll()
        if not group_members(process.pid):break
        time.sleep(.02)
    else:
        try:os.killpg(process.pid,signal.SIGKILL)
        except (ProcessLookupError,PermissionError):
            if group_members(process.pid):raise
    process.wait(timeout=3)
    deadline=time.monotonic()+3
    while group_members(process.pid):
        if time.monotonic()>deadline:raise FormatError('macos_process_group_cleanup_failed')
        time.sleep(.02)
    for stream in (process.stdin,process.stdout):
        if stream and not stream.closed:stream.close()

def run_bounded(argv,payload,cwd,limits,*,stop=lambda:False,on_started=None,evidence=None,on_chunk=None):
    if sys.platform!='darwin':raise FormatError('macos_runtime_unavailable')
    if not isinstance(payload,bytes) or len(payload)>limits.input_bytes:raise LimitError('input_bytes_exceeded')
    if stop():raise Cancelled('cancelled')
    if not argv or not Path(argv[0]).is_absolute():raise ValueError('Explicit trusted executable required')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg',
        PTB_MACOS_CHILD_CPU=str(max(1,int(limits.timeout_seconds)+3)),PYTHONDONTWRITEBYTECODE='1')
    _,profile=load_profile();env['MPLCONFIGDIR']=profile['cache']
    evidence=evidence if evidence is not None else {}
    started=time.monotonic();peak=0;received=0;offset=0;output=bytearray()
    process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
        cwd=cwd,env=env,start_new_session=True)
    selector=selectors.DefaultSelector()
    os.set_blocking(process.stdout.fileno(),False);os.set_blocking(process.stdin.fileno(),False)
    selector.register(process.stdout,selectors.EVENT_READ)
    if payload:selector.register(process.stdin,selectors.EVENT_WRITE)
    else:process.stdin.close()
    if on_started:on_started(process.pid)
    evidence.update(main_pid=process.pid,configured_memory_bytes=limits.process_bytes,memory_enforcement='sampled aggregate RSS; CPU rlimit; owned process group',cleaned=False)
    try:
        while selector.get_map() or process.poll() is None:
            if stop():raise Cancelled('cancelled')
            if time.monotonic()-started>limits.timeout_seconds:raise LimitError('native_timeout')
            peak=max(peak,_memory(process))
            if peak>limits.process_bytes:raise LimitError('native_memory_exceeded')
            for key,mask in selector.select(.05):
                if mask&selectors.EVENT_WRITE:
                    try:count=os.write(key.fd,payload[offset:offset+65536])
                    except BrokenPipeError:count=0;offset=len(payload)
                    offset+=count
                    if offset>=len(payload):selector.unregister(process.stdin);process.stdin.close()
                if mask&selectors.EVENT_READ:
                    chunk=os.read(key.fd,65536)
                    if not chunk:selector.unregister(process.stdout);continue
                    received+=len(chunk)
                    if received>limits.output_bytes:raise LimitError('output_bytes_exceeded')
                    if on_chunk:on_chunk(chunk)
                    else:output.extend(chunk)
        if process.wait()!=0:raise FormatError('native_exit_'+str(process.returncode))
        return bytes(output)
    finally:
        selector.close();_close_group(process)
        evidence.update(cleaned=process.poll() is not None,memory_peak_bytes=peak,elapsed_seconds=time.monotonic()-started)

def run(entry,request,cwd,limits,*,stop=lambda:False,on_started=None,evidence=None,on_chunk=None):
    return run_bounded(command(entry,request),b'',cwd,limits,stop=stop,on_started=on_started,evidence=evidence,on_chunk=on_chunk)

def preview(entry,header,payload,*,timeout=40,output_bytes=64000000,process_bytes=1000000000):
    compute=lambda:_preview_uncached(entry,header,payload,timeout=timeout,output_bytes=output_bytes,process_bytes=process_bytes)
    if entry not in ('spectrogram','spec2wav_preview','parameter_preview'):return compute()
    from .darwin_preview_cache import cache
    identity=json.dumps([entry,header,output_bytes,process_bytes],sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    key=hashlib.sha256(identity+b'\0'+hashlib.sha256(payload).digest()).digest()
    return cache.get(key,compute)


def _preview_uncached(entry,header,payload,*,timeout=40,output_bytes=64000000,process_bytes=1000000000):
    from ..io.limits import Limits
    _,profile=load_profile();raw=json.dumps(header).encode()+b'\n'+payload
    output=run_bounded(command(entry,'--preview'),raw,Path(profile['cache']),
        Limits(input_bytes=max(1,len(raw)),output_bytes=output_bytes,process_bytes=process_bytes,timeout_seconds=timeout))
    if entry in ('spectrogram','spec2wav_preview','parameter_preview','fonts'):
        pid,sep,output=output.partition(b'\n')
        if not sep or not pid.isdigit():raise FormatError('invalid_preview_protocol')
    result=json.loads(output)
    if 'error' in result:
        from ..spectrogram_preview import PreviewError
        raise PreviewError(result['error'])
    return result


class OwnedProcess:
    """File-protocol child using the same ownership boundary as pipe jobs."""
    def __init__(self,argv,cwd,memory_bytes):
        mapping={'m05_child.py':'lip','m05_media_export.py':'lip_media','vocal_mp4.py':'vocal_mp4'}
        modules={'ptb_worker.acoustic_stream_child':'acoustic_stream','ptb_worker.parameter_bundle_child':'parameter_bundle'}
        if len(argv)==5 and argv[1:3]==['-I','-B'] and Path(argv[3]).name in mapping:entry,request=mapping[Path(argv[3]).name],argv[4]
        elif len(argv)==5 and argv[1:3]==['-B','-m'] and argv[3] in modules:entry,request=modules[argv[3]],argv[4]
        elif len(argv)==4 and argv[1]=='--ptb-worker' and argv[2] in modules:entry,request=modules[argv[2]],argv[3]
        else:raise ValueError('Unsupported macOS file child')
        env=os.environ.copy();env.update(PTB_MACOS_CHILD_CPU='21603' if entry=='acoustic_stream' else '1803',PYTHONDONTWRITEBYTECODE='1',
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
        self.process=subprocess.Popen(command(entry,request),cwd=cwd,env=env,
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        self.pid=self.process.pid;self.budget=memory_bytes;self.peak=0;self.group_cleaned=False
        from types import SimpleNamespace
        self.info=SimpleNamespace(pid=self.pid)
    def memory_peak(self):
        self.peak=max(self.peak,_memory(self.process));return self.peak
    def poll(self):
        if self.memory_peak()>self.budget:raise LimitError('native_memory_exceeded')
        return self.process.poll()
    def close(self):
        _close_group(self.process);self.group_cleaned=not group_members(self.pid)
