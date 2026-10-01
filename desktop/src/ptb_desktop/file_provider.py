"""M01-E native-picker capabilities; no renderer path or output write authority.

Windows read handles deny concurrent write/delete and final paths are checked.
This is preview/target preflight; durable output creation is owned by M01-F.
"""
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
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


@contextmanager
def open_locked(path,root,expected):
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
        yield stream
        if fingerprint(os.fstat(stream.fileno()))!=expected:
            raise FileAccessError('文件读取期间发生变化。')


def read_locked(path,root,limit,expected):
    with open_locked(path,root,expected) as stream:
        info=os.fstat(stream.fileno())
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
        self.directories={};self.entries={};self.entry_ids={};self.closed=False;self.captured={};self.preview_cache=None
        self.preview_sources=set()
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
                kind='audio' if lower.endswith(('.wav','.mp3','.flac')) else 'textgrid' if lower.endswith('.textgrid') else 'lip' if lower.endswith('.lip.json') else 'lip_pickle' if lower.endswith('.pkl') else 'lab' if lower.endswith('.lab') else 'parameter' if lower.endswith(('.xlsx','.ptb.sqlite','.ptb.sqlite3')) else 'image' if lower.endswith(('.png','.jpg','.jpeg','.bmp')) else None
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

    def validate(self,key):
        """Revalidate a memory preview's grant without decoding the source again."""
        entry=self.entries.get(key) if not self.closed else None
        if entry is None: raise FileAccessError('文件授权已失效，请刷新列表。')
        if key in self.captured: return
        directory=self.directory(entry.directory)
        try:
            path=checked_path(directory.path/entry.name)
            if path.parent!=directory.path or fingerprint(path.stat())!=entry.fingerprint:
                raise FileAccessError('音频已变化，请刷新列表。')
        except OSError: raise FileAccessError('音频已不可访问，请刷新列表。') from None

    def read(self,key):
        entry=self.entries.get(key) if not self.closed else None
        if entry is None:raise FileAccessError('文件授权已失效，请刷新列表。')
        if key in self.captured:
            raw=self.captured[key];return raw,hashlib.sha256(raw).hexdigest()
        directory=self.directory(entry.directory)
        try:
            path=checked_path(directory.path/entry.name)
            if path.parent!=directory.path:raise FileAccessError('文件超出目录授权。')
            limit=self.max_bytes if entry.name.lower().endswith(('.wav','.mp3','.flac')) else min(self.max_bytes,16_000_000 if entry.name.lower().endswith(('.pkl','.xlsx','.ptb.sqlite','.ptb.sqlite3','.png','.jpg','.jpeg','.bmp')) else 2_000_000)
            raw=read_locked(path,directory.path,limit,entry.fingerprint)
            self.directory(entry.directory)
        except OSError:raise FileAccessError('文件不可读取，请刷新列表。') from None
        return raw,hashlib.sha256(raw).hexdigest()

    def scan(self,key):
        """Recursive picker grant, retaining direct-directory file capabilities."""
        root=self.directory(key)
        if root.purpose=='output':raise FileAccessError('输出目录没有读取授权。')
        pending=[(root.path,key)];result=[];visited=0;entries_seen=0
        while pending:
            path,grant=pending.pop();visited+=1
            if visited>512:raise FileAccessError('子目录超过512，请选择较小目录。')
            self.directory(grant)
            result.extend({**item,'name':(path/item['name']).relative_to(root.path).as_posix()}
                          for item in self.list(grant))
            if len(result)>self.max_entries:raise FileAccessError('语料文件超过扫描预算，请选择较小目录。')
            with os.scandir(path) as children:
                for item in children:
                    entries_seen+=1
                    if entries_seen>self.max_entries:raise FileAccessError('目录条目超过扫描预算，请选择较小目录。')
                    info=Path(item.path).lstat()
                    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:continue
                    child=checked_path(item.path);child.relative_to(root.path)
                    existing=next((k for k,d in self.directories.items() if d.path==child and d.purpose==root.purpose and d.identity==identity(info)),None)
                    if existing is None:
                        if len(self.directories)>=640:raise FileAccessError('目录授权数量超出预算，请重新打开窗口。')
                        existing=secrets.token_urlsafe(24)
                        self.directories[existing]=Directory(child,root.purpose,identity(info))
                    pending.append((child,existing))
        self.directory(key)
        return sorted(result,key=lambda item:(item['name'].casefold(),item['name']))

    @contextmanager
    def audio_stream(self,key):
        entry=self.entries.get(key) if not self.closed else None
        if entry is None or not entry.name.lower().endswith('.wav'):raise FileAccessError('WAV授权已失效，请刷新列表。')
        directory=self.directory(entry.directory)
        if directory.purpose=='output':raise FileAccessError('输出目录没有读取授权。')
        try:
            path=checked_path(directory.path/entry.name)
            if path.parent!=directory.path:raise FileAccessError('音频超出目录授权。')
            with open_locked(path,directory.path,entry.fingerprint) as stream:
                yield stream
                self.directory(entry.directory)
        except OSError:raise FileAccessError('音频不可读取，请刷新列表。') from None

    def audio_preview(self,key):
        import base64
        raw,sha,duration,note=self.preview_payload(key)
        return dict(base64=base64.b64encode(raw).decode('ascii'),sha256=sha,
                    sourceDuration=duration,previewNote=note)

    def preview_payload(self,key):
        from .annotation_audio import preview_audio
        with self.audio_stream(key) as stream:
            cached=self.preview_cache
            if cached and cached[0]==key:return cached[1]
            raw,sha,duration,note=preview_audio(stream,self.max_bytes,preserve_channels=True)
        self.preview_cache=(key,(raw,sha,duration,note))
        self.preview_sources.add(key)
        return raw,sha,duration,note

    def spectrogram_payload(self,key):
        if key in self.preview_sources:
            raw,sha,_,_=self.preview_payload(key)
            return raw,sha
        return self.read(key)

    def output_target(self,directory_id,audio_id,format):
        directory=self.directory(directory_id);entry=self.entries.get(audio_id)
        if directory.purpose not in ('input','output') or entry is None or not entry.name.lower().endswith('.wav') or format not in ('xlsx','sqlite'):
            raise FileAccessError('输出类型或授权不正确。')
        self.directory(entry.directory)
        name=Path(entry.name).stem+('.xlsx' if format=='xlsx' else '.ptb.sqlite')
        path=directory.path/name
        if path.exists() or path.is_symlink():raise FileAccessError('已有同名结果，请更换输出目录。')
        return path

    def capture(self,raw):
        if self.closed or not 0<len(raw)<=16_000_000:raise FileAccessError('截图过大或窗口已关闭。')
        for key in self.captured:self.entries.pop(key,None)
        self.captured.clear();key=secrets.token_urlsafe(24);self.captured[key]=raw
        self.entries[key]=Entry('','屏幕截图.png',())
        return dict(id=key,name='屏幕截图.png',kind='image',size=len(raw))

    def close(self):
        self.captured.clear()
        self.preview_cache=None
        self.preview_sources.clear()
        self.closed=True;self.directories.clear();self.entries.clear();self.entry_ids.clear()
