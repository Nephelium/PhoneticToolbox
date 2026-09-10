"""M01-E native-picker capabilities; no renderer path or output write authority.

Windows read handles deny concurrent write/delete and final paths are checked.
This is preview/target preflight; durable output creation is owned by M01-F.
"""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import os
import secrets
import stat


class FileAccessError(ValueError):pass


def checked_path(path):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        info=part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
            raise FileAccessError('链接或重解析点不作为目录授权。')
    return path.resolve(strict=True)


def identity(info):return info.st_dev,info.st_ino
def fingerprint(info):return (*identity(info),info.st_size,info.st_mtime_ns)


def read_locked(path,root,limit,expected):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        import msvcrt
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        create=kernel.CreateFileW
        create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        create.restype=wintypes.HANDLE
        handle=create(str(path),0x80000000,1,None,3,0x00200000,None)
        if handle==ctypes.c_void_p(-1).value:raise FileAccessError('文件正被修改或无法安全打开。')
        close=kernel.CloseHandle;close.argtypes=[wintypes.HANDLE]
        try:
            final=kernel.GetFinalPathNameByHandleW
            final.argtypes=[wintypes.HANDLE,wintypes.LPWSTR,wintypes.DWORD,wintypes.DWORD]
            final.restype=wintypes.DWORD
            buffer=ctypes.create_unicode_buffer(32768);size=final(handle,buffer,len(buffer),0)
            if not 0<size<len(buffer):raise FileAccessError('无法核对文件位置。')
            name=buffer.value
            if name.startswith('\\\\?\\UNC\\'):name='\\\\'+name[8:]
            elif name.startswith('\\\\?\\'):name=name[4:]
            actual=Path(name)
            if actual.parent!=root or actual!=path:raise FileAccessError('文件位置已变化。')
            fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
        except BaseException:
            close(handle);raise
    else:
        directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
        finally:os.close(directory)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or fingerprint(info)!=expected:
            raise FileAccessError('文件已变化，请刷新列表。')
        if info.st_size>limit:raise FileAccessError('文件超过当前预览大小上限。')
        content=stream.read(limit+1)
        if len(content)>limit or fingerprint(os.fstat(stream.fileno()))!=expected:
            raise FileAccessError('文件读取期间发生变化。')
        return content


@dataclass(frozen=True)
class Directory:
    path: Path
    purpose: str
    identity: tuple


@dataclass(frozen=True)
class Entry:
    directory: str
    name: str
    fingerprint: tuple


class FileProvider:
    def __init__(self,*,max_bytes=64_000_000,max_entries=10000):
        self.max_bytes,self.max_entries=max_bytes,max_entries
        self.directories={};self.entries={};self.entry_ids={};self.closed=False
        self.session=secrets.token_urlsafe(24)

    def choose(self,purpose,picker):
        if self.closed or purpose not in ('input','output','association'):raise FileAccessError('目录能力不可用。')
        chosen=picker()
        if not chosen:return None
        try:
            path=checked_path(chosen);info=path.stat()
            if not stat.S_ISDIR(info.st_mode):raise FileAccessError('请选择目录。')
        except OSError:raise FileAccessError('目录不可访问。') from None
        for key,value in self.directories.items():
            if value.path==path and value.purpose==purpose and value.identity==identity(info):
                return {'id':key,'label':path.name or path.anchor,'purpose':purpose}
        if len(self.directories)>=128:raise FileAccessError('本次窗口选择的目录过多，请重新打开窗口。')
        key=secrets.token_urlsafe(24);self.directories[key]=Directory(path,purpose,identity(info))
        return {'id':key,'label':path.name or path.anchor,'purpose':purpose}

    def directory(self,key):
        value=self.directories.get(key) if not self.closed else None
        if value is None:raise FileAccessError('目录授权已失效，请重新选择。')
        try:
            if checked_path(value.path)!=value.path or identity(value.path.stat())!=value.identity:
                raise FileAccessError('目录已变化，请重新选择。')
        except OSError:raise FileAccessError('目录已不可访问。') from None
        return value

    def list(self,key):
        directory=self.directory(key)
        if directory.purpose=='output':raise FileAccessError('输出目录没有读取授权。')
        result=[]
        with os.scandir(directory.path) as scan:
            for count,entry in enumerate(scan):
                if count>=self.max_entries:raise FileAccessError('目录条目超过10000，请选择较小目录。')
                lower=entry.name.lower()
                kind='audio' if lower.endswith('.wav') else 'textgrid' if lower.endswith('.textgrid') else 'lip' if lower.endswith('.lip.json') else 'lip_pickle' if lower.endswith('.pkl') else 'parameter' if lower.endswith(('.xlsx','.ptb.sqlite','.ptb.sqlite3')) else None
                # Windows scandir caches zero inode/link counts; obtain real identity.
                info=Path(entry.path).lstat()
                if not kind or not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or getattr(info,'st_file_attributes',0)&0x400:continue
                value=Entry(key,entry.name,fingerprint(info))
                physical=(directory.path/entry.name,value.fingerprint)
                file_id=self.entry_ids.get(physical)
                if file_id is None:
                    if len(self.entries)>=30000:raise FileAccessError('本次窗口文件记录过多，请重新打开窗口。')
                    file_id=secrets.token_urlsafe(24);self.entries[file_id]=value;self.entry_ids[physical]=file_id
                result.append({'id':file_id,'name':entry.name,'kind':kind,'size':info.st_size})
        return sorted(result,key=lambda v:(v['name'].casefold(),v['name']))

    def read(self,key):
        entry=self.entries.get(key) if not self.closed else None
        if entry is None:raise FileAccessError('文件授权已失效，请刷新列表。')
        directory=self.directory(entry.directory)
        try:
            path=checked_path(directory.path/entry.name)
            if path.parent!=directory.path:raise FileAccessError('文件超出目录授权。')
            limit=self.max_bytes if entry.name.lower().endswith('.wav') else min(self.max_bytes,16_000_000 if entry.name.lower().endswith(('.pkl','.xlsx','.ptb.sqlite','.ptb.sqlite3')) else 2_000_000)
            raw=read_locked(path,directory.path,limit,entry.fingerprint)
            self.directory(entry.directory)
        except OSError:raise FileAccessError('文件不可读取，请刷新列表。') from None
        return raw,hashlib.sha256(raw).hexdigest()

    def output_target(self,directory_id,audio_id,format):
        directory=self.directory(directory_id);entry=self.entries.get(audio_id)
        if directory.purpose not in ('input','output') or entry is None or not entry.name.lower().endswith('.wav') or format not in ('xlsx','sqlite'):
            raise FileAccessError('输出类型或授权不正确。')
        self.directory(entry.directory)
        name=Path(entry.name).stem+('.xlsx' if format=='xlsx' else '.ptb.sqlite')
        path=directory.path/name
        if path.exists() or path.is_symlink():raise FileAccessError('已有同名结果，请更换输出目录。')
        return path

    def close(self):
        self.closed=True;self.directories.clear();self.entries.clear();self.entry_ids.clear()
