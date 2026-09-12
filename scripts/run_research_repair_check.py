"""Run and observe only the separately built repair artifact and its children."""
import ctypes
import argparse
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import time
from uuid import uuid4

kernel=ctypes.WinDLL('kernel32',use_last_error=True)
class ProcessEntry(ctypes.Structure):
    _fields_=[('size',wintypes.DWORD),('usage',wintypes.DWORD),('pid',wintypes.DWORD),
              ('heap',ctypes.c_size_t),('module',wintypes.DWORD),('threads',wintypes.DWORD),
              ('parent',wintypes.DWORD),('priority',wintypes.LONG),('flags',wintypes.DWORD),
              ('exe',wintypes.WCHAR*260)]
kernel.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
kernel.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessEntry)]
kernel.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessEntry)]
kernel.CloseHandle.argtypes=[wintypes.HANDLE]
kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
kernel.OpenProcess.restype=wintypes.HANDLE
kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
kernel.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]

def process_parents():
    handle=kernel.CreateToolhelp32Snapshot(2,0)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    result={};entry=ProcessEntry();entry.size=ctypes.sizeof(entry)
    try:
        more=kernel.Process32FirstW(handle,ctypes.byref(entry))
        while more:
            result[entry.pid]=entry.parent
            more=kernel.Process32NextW(handle,ctypes.byref(entry))
        return result
    finally:kernel.CloseHandle(handle)

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--state',type=Path,help='Previously initialized, owned verification state only')
options=parser.parse_args()
exe = ROOT / 'dist/research-repair/PhoneticToolbox-v3-Research-Fix1.exe'
out = ROOT / 'output/validation/desktop-repair' / ('frozen-' + uuid4().hex)
out.mkdir(parents=True)
print(out, flush=True)
user = ctypes.WinDLL('user32', use_last_error=True)
callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user.IsWindowVisible.argtypes = [wintypes.HWND]
user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
maximum = 0
seen = {}
with (out/'stdout.log').open('wb') as stdout, (out/'stderr.log').open('wb') as stderr:
    process = subprocess.Popen([str(exe), '--local-root', str(options.state or out/'state'), '--verify-repair', str(out/'results')],
                               stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
    started = time.monotonic()
    while process.poll() is None:
        parents=process_parents();pids={process.pid}
        while True:
            expanded=pids|{pid for pid,parent in parents.items() if parent in pids}
            if expanded==pids:break
            pids=expanded
        for pid in pids-{process.pid}:
            if pid not in seen:
                handle=kernel.OpenProcess(0x100000|0x1000|1,False,pid)
                if handle:seen[pid]=handle
        visible = []
        @callback_type
        def observe(hwnd, _):
            pid = wintypes.DWORD()
            user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value in pids and user.IsWindowVisible(hwnd):
                title = ctypes.create_unicode_buffer(256)
                user.GetWindowTextW(hwnd, title, len(title))
                if title.value == 'PhoneticToolbox 3.0':
                    visible.append(pid.value)
            return True
        user.EnumWindows(observe, 0)
        maximum = max(maximum, len(visible))
        if time.monotonic()-started > 240:
            for handle in seen.values():
                if kernel.WaitForSingleObject(handle,0)==258:kernel.TerminateProcess(handle,1)
            process.kill();process.wait()
            raise TimeoutError('Owned repair verification timed out: '+str(out))
        time.sleep(.3)
    remaining=[]
    for pid,handle in seen.items():
        if kernel.WaitForSingleObject(handle,1000)==258:remaining.append(pid)
        kernel.CloseHandle(handle)
    result = {'exit_code':process.returncode, 'max_visible_workbench_windows':maximum,'remaining_owned_pids':remaining}
    (out/'process-report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result), flush=True)
assert process.returncode == 0, str(out/'stderr.log')
assert maximum == 1, result
assert not remaining, result
for arguments in (['--ptb-worker','os'], ['--ptb-worker'], ['--unknown-worker']):
    rejected = subprocess.run([str(exe),*arguments],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
    assert rejected.returncode == 2, (arguments,rejected.returncode)
result['unknown_dispatch_rejected']=3
(out/'process-report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
