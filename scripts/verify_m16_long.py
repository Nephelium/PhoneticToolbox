"""Accelerated 60-minute synthetic frame stream; never a hardware-duration claim."""
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4
import numpy as np
from ptb_desktop.recording.capture import Capture
from ptb_desktop.recording.storage import Project,validate_span,iter_audio

ROOT=Path(__file__).resolve().parents[1]
def memory_bytes():
    if os.name!='nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    class Info(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    info=Info();info.cb=ctypes.sizeof(info)
    query=ctypes.windll.psapi.GetProcessMemoryInfo;query.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info),ctypes.c_ulong];query.restype=ctypes.c_int
    if not query(ctypes.c_void_p(-1),ctypes.byref(info),info.cb):raise OSError('Windows working-set probe failed')
    return int(info.WorkingSetSize)

def main():
    out=ROOT/'output/validation/m16'/('long-'+uuid4().hex);project=Project(out/'project',True);total=48000*60*60;block_frames=8192
    cfg={'sample_rate':48000,'channels':2,'roles':['microphone','egg'],'gain_db':18};capture=Capture(project.root,cfg);capture.start(open_stream=False)
    block=np.column_stack((.15*np.sin(np.arange(block_frames)*.031),.6*np.sin(np.arange(block_frames)*.017))).astype(np.float32)
    expected=hashlib.sha256();measurements=[];start=time.monotonic();next_mark=0
    for offset in range(0,total,block_frames):
        # Test source is accelerated and honors a throttle; real callback never waits.
        while capture.queue.qsize()>40:
            if capture.error:raise RuntimeError(capture.error)
            time.sleep(.002)
        part=block[:min(block_frames,total-offset)];expected.update(part.tobytes());capture.submit(part)
        if offset>=next_mark:
            measurements.append({'frame':offset,'working_set':memory_bytes(),'queued':capture.queue.qsize()});next_mark+=48000*60*5
    result=capture.stop();assert not result['error'];assert result['frames']==total
    actual=hashlib.sha256()
    for span in result['spans']:validate_span(project.root,span,hash_check=True)
    for array in iter_audio(project.root,result['spans']):actual.update(array.tobytes())
    assert actual.hexdigest()==expected.hexdigest();project.close()
    report={'success':True,'scope':'accelerated synthetic 60-minute PCM stream, not 60-minute wall clock or physical device','sample_rate':48000,'channels':2,'frames':total,'raw_bytes':total*8,'segments':len(result['spans']),'wall_seconds':time.monotonic()-start,'sha256':actual.hexdigest(),'memory_samples':measurements,'peak_sampled_working_set':max(x['working_set'] for x in measurements),'live_window_frames':capture.live_frames,'queue_bound':capture.queue.maxsize}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({'out':str(out),**{k:v for k,v in report.items() if k!='memory_samples'}},ensure_ascii=False))

if __name__=='__main__':main()
