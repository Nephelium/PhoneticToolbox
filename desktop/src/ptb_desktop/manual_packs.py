"""Verified, optional manual resources in the current user's application cache.

This installer accepts JSON and raster/vector images, never code or updates.
The Chinese manual and user research folders are outside its write boundary.
"""
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import threading
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import zipfile
from .native_messages_data import NATIVE_MESSAGES

MAX_ARCHIVE=500_000_000
MAX_EXPANDED=1_000_000_000
MAX_FILE=64_000_000
MAX_FILES=10000
MAX_MANIFEST_BYTES=4_000_000
CHUNK=1_048_576
SHA=re.compile(r'^[a-f0-9]{64}$')
VERSION=re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$')
LOCALES=tuple(code for code in NATIVE_MESSAGES if code!='zh-CN')
# Each locale retains a subset of its manifest plus fixed directory/hash fields.
MAX_INDEX_BYTES=(MAX_MANIFEST_BYTES+4096)*len(LOCALES)


class ManualPackError(ValueError):
    pass


def fail(code):
    raise ManualPackError('manual.pack.'+code)


def safe_path(name):
    if not isinstance(name,str) or len(name)>240 or '\\' in name or '%' in name:
        return False
    parts=PurePosixPath(name).parts
    if not parts or any(part in ('','.','..') for part in parts) or '/'.join(parts)!=name:
        return False
    return name in ('project.json','language-pack.json') or bool(re.fullmatch(r'chapters/[A-Za-z0-9._-]+\.json',name)) or bool(re.fullmatch(r'assets/[A-Za-z0-9._/-]+\.(?:png|jpg|jpeg|webp|svg)',name))


def file_hash(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(CHUNK),b''):result.update(block)
    return result.hexdigest()


def validate_spec(value):
    if not isinstance(value,dict) or value.get('locale') not in LOCALES:
        fail('invalidSpec')
    if not isinstance(value.get('version'),str) or not VERSION.fullmatch(value['version']):fail('invalidSpec')
    if any(not isinstance(value.get(name),str) or not SHA.fullmatch(value[name]) for name in ('sha256','sourceDigest')):fail('invalidSpec')
    return dict(value)


class _CheckedRedirect(HTTPRedirectHandler):
    def __init__(self,check):super().__init__();self.check=check
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        self.check(newurl,redirect=True)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


class ManualPackManager:
    def __init__(self,root=None,*,allow_loopback=False):
        if root is None:
            from .platform_paths import user_data_root
            root=user_data_root()/'manual-packs'
        original=Path(root)
        if original.is_symlink() or getattr(original,'is_junction',lambda:False)():fail('invalidPath')
        self.root=original.resolve()
        self.allow_loopback=allow_loopback
        self._install_lock=threading.Lock()
        self._imports={}
        self._imports_lock=threading.Lock()
        self._closed=threading.Event()
        self._cache=OrderedDict()
        self._cache_bytes=0
        self._cache_lock=threading.Lock()
        self._index_cache=None
        self._manifest_cache=OrderedDict()
        self._healthy={}

    def _ensure_open(self):
        if self._closed.is_set():fail('cancelled')

    def close(self):
        # Only unfinished imports owned by this instance are removed. Network
        # and extraction workers observe the event and clean their own paths.
        with self._imports_lock:
            self._closed.set();pending=list(self._imports.values());self._imports.clear()
        for request in pending:
            try:self._contained(request['path']).unlink(missing_ok=True)
            except OSError:pass
        with self._cache_lock:
            self._cache.clear();self._cache_bytes=0;self._manifest_cache.clear();self._healthy.clear();self._index_cache=None

    def _contained(self,path):
        if not path.resolve().is_relative_to(self.root):fail('invalidPath')
        for part in (path,*path.parents):
            if part==self.root.parent:break
            if part.is_symlink() or getattr(part,'is_junction',lambda:False)():fail('invalidPath')
        return path

    def _index(self):
        path=self._contained(self.root/'index.json')
        if not path.is_file():return {}
        status=path.stat();identity=(status.st_size,status.st_mtime_ns,status.st_ctime_ns,status.st_ino)
        if status.st_size>MAX_INDEX_BYTES:fail('invalidManifest')
        with self._cache_lock:
            if self._index_cache and self._index_cache[0]==identity:return copy.deepcopy(self._index_cache[1])
        try:value=json.loads(path.read_text('utf-8'))
        except (OSError,ValueError):fail('invalidManifest')
        if not isinstance(value,dict) or set(value)-set(LOCALES):fail('invalidManifest')
        with self._cache_lock:self._index_cache=(identity,value)
        return copy.deepcopy(value)

    def list(self):
        result=[]
        for locale,entry in self._index().items():
            if not isinstance(entry,dict) or not isinstance(entry.get('metadata'),dict):fail('invalidManifest')
            meta=entry['metadata']
            if meta.get('locale')!=locale or not isinstance(meta.get('sourceDigest'),str) or not SHA.fullmatch(meta['sourceDigest']):fail('invalidManifest')
            self._pack_path(entry)
            result.append(json.loads(json.dumps(meta)))
        return result

    def _pack_path(self,entry):
        if not isinstance(entry,dict):fail('invalidManifest')
        relative=entry.get('directory')
        if not isinstance(relative,str):fail('invalidPath')
        parts=PurePosixPath(relative).parts
        if len(parts)!=3 or parts[0]!='packs' or parts[1] not in LOCALES or not re.fullmatch(r'[a-f0-9-]+',parts[2]) or '/'.join(parts)!=relative:fail('invalidPath')
        return self._contained(self.root/relative)

    def _selected(self,locale):
        if locale not in LOCALES:fail('invalidSpec')
        entry=self._index().get(locale)
        if not entry:fail('missingFile')
        root=self._pack_path(entry)
        manifest_path=self._contained(root/'language-pack.json')
        if not manifest_path.is_file() or manifest_path.stat().st_size>MAX_MANIFEST_BYTES:fail('invalidManifest')
        status=manifest_path.stat();identity=(str(manifest_path),status.st_size,status.st_mtime_ns,status.st_ctime_ns,status.st_ino,entry.get('manifestSha256'))
        with self._cache_lock:
            manifest=self._manifest_cache.get(identity)
            if manifest is not None:self._manifest_cache.move_to_end(identity)
        if manifest is None:
            try:
                data=manifest_path.read_bytes()
                if hashlib.sha256(data).hexdigest()!=entry.get('manifestSha256'):fail('hashMismatch')
                manifest=json.loads(data)
            except ManualPackError:raise
            except (ValueError,OSError):fail('invalidManifest')
            with self._cache_lock:
                self._manifest_cache[identity]=manifest
                while len(self._manifest_cache)>max(4,len(LOCALES)*4):self._manifest_cache.popitem(last=False)
        if not isinstance(manifest,dict):fail('invalidManifest')
        if any(manifest.get(key)!=entry['metadata'].get(key) for key in ('locale','version','sourceDigest','projectId')):fail('incompatible')
        return root,manifest

    def _read(self,locale,name):
        if not safe_path(name):fail('invalidPath')
        root,manifest=self._selected(locale)
        expected=manifest.get('hashes',{}).get(name)
        if not isinstance(expected,str) or not SHA.fullmatch(expected):fail('missingFile')
        path=self._contained(root/name)
        if not path.is_file() or path.stat().st_size>MAX_FILE:fail('missingFile')
        status=path.stat();key=(str(path),status.st_size,status.st_mtime_ns,status.st_ctime_ns,status.st_ino,expected)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key);return self._cache[key]
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:fail('hashMismatch')
        if len(data)<=16_000_000:
            with self._cache_lock:
                if key in self._cache:return self._cache[key]
                self._cache[key]=data;self._cache_bytes+=len(data)
                while self._cache_bytes>16_000_000:
                    _,old=self._cache.popitem(last=False);self._cache_bytes-=len(old)
        return data

    def read_json(self,locale,name):
        if not isinstance(name,str) or not (name=='project.json' or re.fullmatch(r'chapters/[A-Za-z0-9._-]+\.json',name)):fail('invalidPath')
        try:return json.loads(self._read(locale,name))
        except (UnicodeError,ValueError) as exc:
            if isinstance(exc,ManualPackError):raise
            fail('invalidManifest')

    def read_asset(self,locale,name):
        if not isinstance(name,str) or not name.startswith('assets/') or not safe_path(name):fail('invalidPath')
        return self._read(locale,name)

    def _url(self,value,*,redirect=False):
        if not isinstance(value,str):fail('invalidUrl')
        try:url=urlsplit(value);host=(url.hostname or '').lower();port=url.port
        except ValueError:fail('invalidUrl')
        if url.username or url.password or url.fragment:fail('invalidUrl')
        if self.allow_loopback and url.scheme=='http' and host in ('localhost','127.0.0.1','::1'):return
        if url.scheme!='https' or port not in (None,443):fail('invalidUrl')
        if host in ('www.phonetictoolbox.com','phonetictoolbox.com'):return
        if host=='github.com' and url.path.startswith('/Nephelium/PhoneticToolbox/releases/download/'):return
        if redirect and host in ('objects.githubusercontent.com','release-assets.githubusercontent.com','github-releases.githubusercontent.com'):return
        fail('invalidUrl')

    def _download_path(self):
        folder=self._contained(self.root/'.downloads');folder.mkdir(parents=True,exist_ok=True)
        return folder/(uuid.uuid4().hex+'.zip')

    def _fingerprints(self,root,manifest):
        hashes=manifest.get('hashes')
        if not isinstance(hashes,dict) or len(hashes)>MAX_FILES:fail('invalidManifest')
        result=[]
        for name,expected in sorted(hashes.items()):
            if not safe_path(name) or not isinstance(expected,str) or not SHA.fullmatch(expected):fail('invalidManifest')
            path=self._contained(root/name)
            if not path.is_file():fail('missingFile')
            state=path.stat()
            if state.st_size>MAX_FILE:fail('invalidManifest')
            result.append((str(path),state.st_size,state.st_mtime_ns,state.st_ctime_ns,state.st_ino,expected))
        return tuple(result)

    def _intact(self,locale,entry):
        try:
            root,manifest=self._selected(locale)
            current=(entry['directory'],entry['manifestSha256'],self._fingerprints(root,manifest))
            with self._cache_lock:
                if self._healthy.get(locale)==current:return True
            if any(file_hash(Path(row[0]))!=row[-1] for row in current[2]):return False
            with self._cache_lock:self._healthy[locale]=current
            return True
        except (ManualPackError,OSError,ValueError,KeyError):return False

    @contextmanager
    def _commit_lock(self):
        path=self._contained(self.root/'index.commit.lock')
        path.parent.mkdir(parents=True,exist_ok=True)
        if os.name=='nt':
            from .startup_cache import FileLock
            lock=FileLock(path)
            try:
                try:lock.acquire(timeout=5)
                except TimeoutError:fail('installBusy')
                yield
            finally:lock.close()
        else:
            import fcntl
            with path.open('a+b') as stream:
                deadline=time.monotonic()+5
                while True:
                    try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
                    except BlockingIOError:
                        if time.monotonic()>=deadline:fail('installBusy')
                        time.sleep(.05)
                try:yield
                finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

    def install(self,spec):
        self._ensure_open()
        spec=validate_spec(spec);self._url(spec.get('url'))
        current=self._index().get(spec['locale'])
        if current and current.get('archiveSha256')==spec['sha256'] and current['metadata'].get('sourceDigest')==spec['sourceDigest'] and current['metadata'].get('version')==spec['version'] and self._intact(spec['locale'],current):
            return current['metadata']
        path=self._download_path()
        try:
            opener=build_opener(_CheckedRedirect(self._url))
            with opener.open(Request(spec['url'],headers={'User-Agent':'PhoneticToolbox-Manual/1'}),timeout=30) as response,path.open('xb') as output:
                declared=response.headers.get('Content-Length')
                if declared and (not declared.isdigit() or int(declared)>MAX_ARCHIVE):fail('invalidArchive')
                size=0
                while True:
                    self._ensure_open()
                    data=response.read(CHUNK)
                    if not data:break
                    size+=len(data)
                    if size>MAX_ARCHIVE:fail('invalidArchive')
                    output.write(data)
            return self._install_archive(path,spec)
        except ManualPackError:raise
        except (OSError,ValueError):fail('downloadFailed')
        finally:
            if path.is_file():path.unlink()

    def _install_archive(self,path,spec):
        self._ensure_open()
        spec=validate_spec(spec)
        if path.stat().st_size>MAX_ARCHIVE or file_hash(path)!=spec['sha256']:fail('hashMismatch')
        if not self._install_lock.acquire(False):fail('installBusy')
        stage=self._contained(self.root/'.staging'/uuid.uuid4().hex)
        target=None;committed=False
        try:
            stage.mkdir(parents=True)
            with zipfile.ZipFile(path) as zipped:
                infos=zipped.infolist();names=[info.filename for info in infos]
                if not infos or len(infos)>MAX_FILES or len({name.casefold() for name in names})!=len(names):fail('invalidArchive')
                if any(not safe_path(info.filename) or info.is_dir() or info.flag_bits&1 or stat.S_ISLNK(info.external_attr>>16) or info.file_size>MAX_FILE or info.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) for info in infos):fail('invalidArchive')
                if sum(info.file_size for info in infos)>MAX_EXPANDED or 'language-pack.json' not in names:fail('invalidArchive')
                if zipped.getinfo('language-pack.json').file_size>MAX_MANIFEST_BYTES:fail('invalidManifest')
                try:manifest=json.loads(zipped.read('language-pack.json'))
                except (UnicodeError,ValueError):fail('invalidManifest')
                if not isinstance(manifest,dict) or manifest.get('schemaVersion')!='ptb-manual-language-pack/1':fail('invalidManifest')
                if any(manifest.get(name)!=spec[name] for name in ('locale','version','sourceDigest')):fail('incompatible')
                hashes=manifest.get('hashes');overrides=manifest.get('assetOverrides')
                if not isinstance(hashes,dict) or not isinstance(overrides,dict) or manifest.get('distribution') not in ('public','software'):fail('invalidManifest')
                if set(names)!=set(hashes)|{'language-pack.json'} or 'project.json' not in hashes:fail('invalidManifest')
                if any(not isinstance(key,str) or not isinstance(value,str) or not value.startswith('assets/') or value not in hashes for key,value in overrides.items()):fail('invalidManifest')
                for info in infos:
                    self._ensure_open()
                    destination=self._contained(stage/info.filename);destination.parent.mkdir(parents=True,exist_ok=True)
                    digest=hashlib.sha256();size=0
                    with zipped.open(info) as source,destination.open('xb') as output:
                        while True:
                            self._ensure_open()
                            block=source.read(CHUNK)
                            if not block:break
                            size+=len(block)
                            if size>info.file_size or size>MAX_FILE:fail('invalidArchive')
                            digest.update(block);output.write(block)
                    if size!=info.file_size:fail('invalidArchive')
                    if info.filename!='language-pack.json' and digest.hexdigest()!=hashes[info.filename]:fail('hashMismatch')
                try:project=json.loads((stage/'project.json').read_text('utf-8'))
                except (UnicodeError,ValueError):fail('invalidManifest')
                if not isinstance(project,dict) or project.get('schemaVersion')!='ptb-manual/1' or not isinstance(project.get('id'),str) or not project['id'] or project.get('id')!=manifest.get('projectId') or project.get('language')!=spec['locale'] or not isinstance(project.get('chapters'),list):fail('invalidManifest')
                for chapter in project['chapters']:
                    if not isinstance(chapter,dict) or not isinstance(chapter.get('path'),str) or not chapter['path'].startswith('chapters/') or chapter['path'] not in hashes:fail('invalidManifest')
            metadata={name:manifest[name] for name in ('locale','version','sourceDigest','projectId','manualVersion','softwareVersion','distribution','assetOverrides') if name in manifest}
            metadata['installedAt']=datetime.now(timezone.utc).isoformat()
            target=self._contained(self.root/'packs'/spec['locale']/(spec['sha256'][:16]+'-'+uuid.uuid4().hex))
            target.parent.mkdir(parents=True,exist_ok=True)
            os.replace(stage,target)
            with self._commit_lock():
                index=self._index();entry={'directory':target.relative_to(self.root).as_posix(),'archiveSha256':spec['sha256'],'manifestSha256':file_hash(target/'language-pack.json'),'metadata':metadata};index[spec['locale']]=entry
                encoded=json.dumps(index,ensure_ascii=False,allow_nan=False).encode('utf-8')
                if len(encoded)>MAX_INDEX_BYTES:fail('invalidManifest')
                pending=self._contained(self.root/('index.'+uuid.uuid4().hex+'.tmp'))
                try:
                    pending.write_bytes(encoded)
                    with self._imports_lock:
                        self._ensure_open();os.replace(pending,self.root/'index.json');committed=True
                    with self._cache_lock:self._index_cache=None
                finally:
                    if pending.is_file():pending.unlink()
            # Extraction already verified these exact files. Repeated explicit
            # installs can compare identities without reading every image again.
            try:
                health=(entry['directory'],entry['manifestSha256'],self._fingerprints(target,manifest))
                with self._cache_lock:self._healthy[spec['locale']]=health
            except OSError:pass
            return json.loads(json.dumps(metadata))
        except ManualPackError:raise
        except OSError:fail('storageFailed')
        except (ValueError,zipfile.BadZipFile):fail('invalidArchive')
        finally:
            if stage.exists():self._contained(stage);shutil.rmtree(stage)
            if target is not None and not committed and target.exists():self._contained(target);shutil.rmtree(target)
            self._install_lock.release()

    def invoke(self,action,body):
        self._ensure_open()
        if not isinstance(body,dict):fail('invalidSpec')
        if action=='list':return self.list()
        if action=='install':return self.install(body)
        if action=='readJson':return self.read_json(body.get('locale'),body.get('path'))
        if action=='readAsset':
            data=self.read_asset(body.get('locale'),body.get('path'))
            return {'mime':mimetypes.guess_type(body['path'])[0] or 'application/octet-stream','base64':base64.b64encode(data).decode('ascii')}
        if action=='importBegin':
            spec=validate_spec(body.get('spec'));size=body.get('bytes')
            if type(size) is not int or not 22<=size<=MAX_ARCHIVE:fail('invalidArchive')
            with self._imports_lock:
                self._ensure_open()
                if len(self._imports)>=2:fail('installBusy')
                identifier=uuid.uuid4().hex;path=self._download_path();path.touch(exist_ok=False)
                self._imports[identifier]={'path':path,'spec':spec,'size':size,'written':0}
            return {'id':identifier}
        if action in ('importChunk','importFinish','importCancel'):
            identifier=body.get('id')
            if not isinstance(identifier,str) or not re.fullmatch(r'[a-f0-9]{32}',identifier):fail('invalidSpec')
            with self._imports_lock:
                request=self._imports.get(identifier)
                if not request:fail('invalidSpec')
                if action=='importChunk':
                    encoded=body.get('base64')
                    if type(body.get('offset')) is not int or body['offset']!=request['written'] or not isinstance(encoded,str) or len(encoded)>((CHUNK+2)//3)*4:fail('invalidArchive')
                    try:data=base64.b64decode(encoded,validate=True)
                    except ValueError:fail('invalidArchive')
                    if len(data)>CHUNK or request['written']+len(data)>request['size']:fail('invalidArchive')
                    with request['path'].open('ab') as stream:stream.write(data)
                    request['written']+=len(data);return {'written':request['written']}
                self._imports.pop(identifier)
            try:
                if action=='importCancel':return None
                if request['written']!=request['size']:fail('invalidArchive')
                return self._install_archive(request['path'],request['spec'])
            finally:
                if request['path'].is_file():request['path'].unlink()
        fail('invalidSpec')
