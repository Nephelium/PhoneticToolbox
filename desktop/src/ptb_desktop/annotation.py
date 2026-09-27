"""M12 local annotation capabilities on the unified host, no path API.

Output names are derived from granted WAV/annotation files. Prepared target
versions and source hashes protect edits from stale-window overwrites.
"""
from pathlib import Path
from contextlib import contextmanager
import base64
import hashlib
import os
import secrets
import stat
from .file_provider import Directory, Entry, FileAccessError, checked_path, identity, fingerprint, read_locked, open_locked


class AnnotationFiles:
    def __init__(self, provider):
        self.provider = provider
        self.targets = {}

    @contextmanager
    def _audio_stream(self, file_id):
        entry = self.provider.entries.get(file_id) if not self.provider.closed else None
        if entry is None or not entry.name.lower().endswith('.wav'):
            raise FileAccessError('音频授权已失效，请重新扫描。')
        directory = self.provider.directory(entry.directory)
        if directory.purpose == 'output':
            raise FileAccessError('输出目录没有音频读取授权。')
        try:
            path = checked_path(directory.path / entry.name)
            if path.parent != directory.path:
                raise FileAccessError('音频超出目录授权。')
            with open_locked(path, directory.path, entry.fingerprint) as stream:
                yield stream
                self.provider.directory(entry.directory)
        except OSError:
            raise FileAccessError('音频不可读取，请刷新列表。') from None

    def audio(self, file_id):
        from .annotation_audio import preview_audio
        with self._audio_stream(file_id) as stream:
            raw, sha, duration, note = preview_audio(stream, self.provider.max_bytes)
        return dict(base64=base64.b64encode(raw).decode('ascii'), sha256=sha,
                    sourceDuration=duration, previewNote=note)

    def _source(self, file_id):
        entry = self.provider.entries.get(file_id)
        if entry and entry.name.lower().endswith('.wav'):
            from .annotation_audio import source_hash
            with self._audio_stream(file_id) as stream:
                return b'', source_hash(stream)
        return self.provider.read(file_id)

    def scan(self, key):
        root = self.provider.directory(key)
        if root.purpose == 'output':
            raise FileAccessError('输出目录没有标注读取权限。')
        pending = [(root.path, key)]; result = []; visited = 0
        while pending:
            path, grant = pending.pop()
            visited += 1
            if visited > 512:
                raise FileAccessError('子目录超过 512，请选择更小的语料目录。')
            self.provider.directory(grant)
            for item in self.provider.list(grant):
                if item['kind'] in ('audio', 'textgrid', 'lip', 'lip_pickle', 'lab'):
                    result.append({**item, 'name': (path/item['name']).relative_to(root.path).as_posix()})
            if len(result) > 10000:
                raise FileAccessError('语料文件超过 10000，请缩小目录。')
            with os.scandir(path) as entries:
                for item in entries:
                    info = Path(item.path).lstat()
                    if not stat.S_ISDIR(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400 or stat.S_ISLNK(info.st_mode):
                        continue
                    child = checked_path(item.path)
                    child.relative_to(root.path)
                    existing = next((k for k, d in self.provider.directories.items() if d.path == child and d.purpose == root.purpose and d.identity == identity(info)), None)
                    if not existing:
                        if len(self.provider.directories) >= 640:
                            raise FileAccessError('目录授权数量超出预算，请重开工作台。')
                        existing = secrets.token_urlsafe(24)
                        self.provider.directories[existing] = Directory(child, root.purpose, identity(info))
                    pending.append((child, existing))
        self.provider.directory(key)
        return sorted(result, key=lambda v: (v['name'].casefold(), v['name']))

    def lip(self, file_id):
        from ptb_worker.io.annotation import lip_preview
        raw, sha = self.provider.read(file_id)
        entry = self.provider.entries[file_id]
        if not entry.name.lower().endswith(('.pkl', '.lip.json')) or entry.name.lower().endswith('_timestamps.pkl'):
            raise FileAccessError('请选择唇形记录。')
        companion = None
        if entry.name.lower().endswith('.pkl'):
            companions = [f for f in self.provider.list(entry.directory) if f['name'].casefold() == (entry.name[:-4]+'_timestamps.pkl').casefold()]
            if len(companions) > 1:
                raise FileAccessError('伴随时间文件不明确。')
            if companions:
                companion = self.provider.read(companions[0]['id'])[0]
                if len(companion) > 2_000_000:
                    raise FileAccessError('伴随时间文件超过 2 MB。')
        return {'wire': lip_preview(raw, entry.name, companion), 'sha256': sha}

    def _version(self, path, root):
        if not path.exists() and not path.is_symlink():
            return None, None
        actual = checked_path(path)
        if actual.parent != root:
            raise FileAccessError('保存目标超出授权。')
        info = actual.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise FileAccessError('保存目标不是独立普通文件。')
        raw = read_locked(actual, root, 16_000_000, fingerprint(info))
        return fingerprint(info), hashlib.sha256(raw).hexdigest()

    def target(self, body):
        role = body.get('role')
        entry = self.provider.entries.get(body.get('id'))
        if entry is None:
            raise FileAccessError('来源授权已失效，请重新扫描。')
        self._source(body['id'])
        directory = self.provider.directory(entry.directory)
        if directory.purpose not in ('input', 'association'):
            raise FileAccessError('没有标注写入能力。')
        if role == 'textgrid' and entry.name.lower().endswith('.wav'):
            suffix = body.get('suffix', '_自动保存')
            if type(suffix) != str or len(suffix) > 60 or any(ord(c) < 32 or c in '/\\:*?"<>|' for c in suffix) or suffix.endswith(('.', ' ')):
                raise FileAccessError('文件后缀无效。')
            name = Path(entry.name).stem + suffix + '.TextGrid'
        elif role == 'lip' and entry.name.lower().endswith(('.pkl', '.lip.json')) and not entry.name.lower().endswith('_timestamps.pkl'):
            name = entry.name
        else:
            raise FileAccessError('保存类型与来源不符。')
        if len(name) > 220:
            raise FileAccessError('保存文件名过长。')
        path = directory.path/name
        stamp, sha = self._version(path, directory.path)
        if len(self.targets) >= 2000:
            raise FileAccessError('保存目标准备次数过多，请重开页面。')
        token = secrets.token_urlsafe(24)
        self.targets[token] = (entry.directory, name, stamp, sha, role, body['id'])
        return dict(id=token, name=name, sha256=sha)

    def save(self, body):
        from ptb_worker.io.annotation import validate_textgrid, update_lip_offset
        from .task_bridge import pin_directory
        prepared = self.targets.get(body.get('target'))
        if prepared is None:
            raise FileAccessError('保存目标已失效，请重试。')
        grant, name, stamp, expected, role, origin_id = prepared
        directory = self.provider.directory(grant)
        source = body.get('source', {})
        source_bytes, source_sha = self._source(source.get('id'))
        if source_sha != source.get('sha256'):
            raise FileAccessError('来源文件已被修改，编辑仍保留，请重新读取或另存。')
        source_entry = self.provider.entries[source['id']]
        if source_entry.directory != grant:
            raise FileAccessError('保存来源与目录不符。')
        if role == 'textgrid':
            if not source_entry.name.lower().endswith('.textgrid') and not (source_entry.name.lower().endswith('.wav') and source['id'] == origin_id):
                raise FileAccessError('标注保存需要原 TextGrid 版本，或新建标注所绑定的同一 WAV 版本。')
            validate_textgrid(body.get('text'))
            payload = body['text'].encode('utf-8')
        else:
            if source_entry.name != name:
                raise FileAccessError('唇偏必须写回原记录。')
            payload = update_lip_offset(source_bytes, name, body.get('offset'))
        root = directory.path; target = root/name
        lock = root/('.ptb-annotation-'+hashlib.sha256(name.casefold().encode()).hexdigest()[:20]+'.lock')
        temp = root/('.ptb-annotation-'+secrets.token_hex(16)+'.part')
        with pin_directory(root):
            self.provider.directory(grant)
            try:
                lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                raise FileAccessError('另一个窗口正在保存此文件，请稍后重试。') from None
            try:
                if self._version(target, root) != (stamp, expected):
                    raise FileAccessError('目标已被其他窗口修改，未覆盖；编辑仍保留。')
                if self._source(source['id'])[1] != source_sha:
                    raise FileAccessError('来源已变化，未保存。')
                with temp.open('xb') as stream:
                    stream.write(payload); stream.flush(); os.fsync(stream.fileno())
                if self._version(target, root) != (stamp, expected):
                    raise FileAccessError('目标保存期间发生变化，未覆盖。')
                os.replace(temp, target)
                current = next(f for f in self.provider.list(grant) if f['name'] == name)
                current['sha256'] = hashlib.sha256(payload).hexdigest()
                # Keep a retryable target version after successful write.
                self.targets[body['target']] = (grant, name, fingerprint(target.stat()), current['sha256'], role, origin_id)
                return dict(file=current, name=name, sha256=current['sha256'])
            finally:
                os.close(lock_fd)
                if temp.exists(): temp.unlink()
                lock.unlink()

    def invoke(self, body):
        op = body.get('op')
        if op == 'annotation_audio': return self.audio(body['id'])
        if op == 'annotation_scan': return self.scan(body['directory'])
        if op == 'annotation_lip': return self.lip(body['id'])
        if op == 'annotation_target': return self.target(body)
        if op == 'annotation_save': return self.save(body)
        raise FileAccessError('不支持的标注操作。')
