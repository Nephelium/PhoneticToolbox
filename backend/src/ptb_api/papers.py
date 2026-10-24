"""Managed adapter for the same paper implementation used by the desktop.

Only immutable quota assets persist. Per-request hard links and JSON state are
reconstructed in a private TemporaryDirectory, then retired on every exit path.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import threading
import time
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import StreamingResponse
from .quota import CHUNK_BYTES, StorageError
from .storage import public_asset, regular
from .storage_models import UploadInput

NAME = '论文阅读.m18.json'
READ_ONLY = {'render', 'annotations', 'library', 'trash'}
FIELDS = {
    'status': set(), 'refresh': set(), 'download': {'since'}, 'downloadSelected': {'ids'},
    'autoReceive': {'enabled'}, 'interval': {'hours'}, 'createRecommendationFolder': {'name'},
    'deleteRecommendationFolder': {'folder'}, 'collectRecommendation': {'id', 'folder'},
    'removeDownload': {'id'}, 'library': set(), 'chooseLibrary': {'name', 'files'},
    'importLocal': {'files', 'folder'}, 'createFolder': {'name'}, 'deleteFolder': {'folder'},
    'moveLocal': {'id', 'folder'}, 'removeLocal': {'id'}, 'trash': set(), 'restoreLocal': {'id'},
    'render': {'id', 'language', 'page', 'width'}, 'annotations': {'id', 'language'},
    'saveAnnotation': {'id', 'language', 'revision', 'item', 'remove'},
    'export': {'id', 'language', 'annotated'},
}


class PaperRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    operation: str = Field(max_length=40)
    args: dict = Field(default_factory=dict)


def library_content_versions(value, identity_hashes):
    """Use managed asset identity, independent of per-request hard-link stats."""
    library_root = Path(value['root']) if value.get('root') else None
    for document in value.get('documents', []):
        if library_root is None: continue
        document_stat = (library_root / relative(document['relative'])).stat()
        digest = identity_hashes.get((document_stat.st_dev, document_stat.st_ino))
        if digest: document['contentVersion'] = 'sha256:' + digest


def relative(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value or any(ord(c) < 32 for c in value):
        raise StorageError('invalid_request', 422)
    parts = value.split('/')
    if len(value) > 1000 or any(p in ('', '.', '..') or p.startswith('.') for p in parts):
        raise StorageError('invalid_request', 422)
    return PurePosixPath(value)


class WebPapers:
    def __init__(self, storage, runner=None):
        self.storage = storage
        self.runner = runner or run_child
        self.slots = threading.BoundedSemaphore(1)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ptb-web-papers')

    def cleanup(self):
        """Retire only marked scratch whose owning API and PDF child have exited."""
        if not getattr(self.storage, 'ready', False): return
        marker = json.loads((self.storage.root / '.ptb-storage.json').read_text('utf8'))['instance_id']
        for root in self.storage.root.parent.glob('ptb-papers-*'):
            try:
                if root.is_symlink() or getattr(root.lstat(), 'st_file_attributes', 0) & 0x400: continue
                owner = json.loads((root / 'owner.json').read_text('utf8'))
                if owner.get('schema') != 'ptb-paper-scratch/1' or owner.get('instance') != marker: continue
                if process_alive(owner['pid']): continue
                child = root / 'child.json'
                if child.exists() and process_alive(json.loads(child.read_text('utf8'))['pid']): continue
                if any(p.is_symlink() or getattr(p.lstat(), 'st_file_attributes', 0) & 0x400 for p in root.rglob('*')): continue
                shutil.rmtree(root)
            except (OSError, ValueError, KeyError): continue

    def current(self, owner, project):
        with self.storage._locked() as conn:
            self.storage.project_deadline(conn, owner, project)
            row = conn.execute("SELECT * FROM ptb_storage.assets WHERE owner_id=%s AND project_id=%s AND name=%s AND state='ready' ORDER BY created_at DESC,id DESC LIMIT 1", (owner, project, NAME)).fetchone()
            if row: self.storage._readable(conn, owner, row['id'])
            return public_asset(row) if row else None

    def read(self, owner, asset):
        if asset['size_bytes'] > 32_000_000: raise StorageError('paper_state_too_large', 413)
        raw = bytearray()
        while len(raw) < asset['size_bytes']:
            raw.extend(self.storage.read_block(owner, asset['id'], len(raw), min(CHUNK_BYTES, asset['size_bytes'] - len(raw))))
        if hashlib.sha256(raw).hexdigest() != asset['sha256']: raise StorageError('asset_changed', 409)
        return json.loads(raw)

    def publish(self, owner, project, path, name, stop, reservation=None):
        size = path.stat().st_size
        if size > 200_000_000: raise StorageError('paper_export_too_large', 413)
        body = UploadInput(project_id=project, name=name, expected_bytes=size, idempotency_key=uuid4().hex)
        with self.storage._locked() as conn, conn.transaction():
            if reservation:
                guard = self.storage._row(conn, owner, reservation)
                if guard['reserved_bytes'] < size: raise StorageError('paper_output_budget', 413)
                conn.execute('UPDATE ptb_storage.assets SET reserved_bytes=reserved_bytes-%s WHERE id=%s', (size, reservation))
                conn.execute('UPDATE ptb_storage.quota_accounts SET reserved_bytes=reserved_bytes-%s WHERE owner_id=%s', (size, owner))
            asset = self.storage._create(conn, owner, body, kind='result')
        try:
            with path.open('rb') as source:
                offset = 0
                while block := source.read(CHUNK_BYTES):
                    if stop(): raise StorageError('paper_cancelled', 409)
                    with self.storage._locked() as conn: self.storage._append(conn, owner, asset['id'], offset, block)
                    offset += len(block)
            with self.storage._locked() as conn:
                self.storage.project_deadline(conn, owner, project)
                row = self.storage._sync(conn, self.storage._row(conn, owner, asset['id']))
                with path.open('rb') as source: digest = hashlib.file_digest(source, 'sha256').hexdigest()
                conn.execute("UPDATE ptb_storage.assets SET state='ready',sha256=%s,reserved_bytes=0 WHERE id=%s", (digest, asset['id']))
                return public_asset(self.storage._readable(conn, owner, asset['id']))
        except BaseException:
            self.storage.delete(owner, asset['id'])
            raise

    def call(self, owner, project, operation, args, cancel, progress, check):
        if operation not in FIELDS or set(args) != FIELDS[operation]: raise StorageError('invalid_request', 422)
        created = []; reservation = None; preserve_completed = False
        def stopped():
            if cancel.is_set() and not preserve_completed: return True
            check()
            return False
        old = self.current(owner, project)
        snapshot = self.read(owner, old) if old else {'schema': 'ptb-web-papers/1', 'json': {}, 'files': {}, 'dirs': []}
        if snapshot.get('schema') != 'ptb-web-papers/1': raise StorageError('paper_state_invalid', 409)
        try:
            if operation not in READ_ONLY and not (operation == 'status' and old):
                budget = len(json.dumps(snapshot, ensure_ascii=False).encode('utf8')) + len(json.dumps(args, ensure_ascii=False).encode('utf8')) + 8192
                if operation == 'refresh': budget += 54_000_000  # catalog + at most one 50 MB notice
                if operation == 'export': budget += 200_000_000
                if operation in ('download', 'downloadSelected'):
                    catalog = snapshot['json'].get('cache/catalog.json', {})
                    entries = list(catalog.get('papers', [])) + list(catalog.get('journalPapers', []))
                    if catalog.get('notice'): entries.append({'id': catalog['notice']['id'], 'original': catalog['notice']['pdf']})
                    assets = {}
                    for paper in entries:
                        if operation == 'downloadSelected' and paper['id'] not in args['ids']: continue
                        for kind in ('original', 'translation', 'analysisPdf'):
                            asset = paper.get(kind)
                            if asset and 'cache/files/'+asset['sha256']+'.pdf' not in snapshot['files']: assets[asset['sha256']] = asset['size']
                    budget += sum(assets.values())
                if budget > 1_000_000_000: raise StorageError('paper_output_budget', 413)
                body = UploadInput(project_id=project, name='论文处理预算.m18', expected_bytes=budget, idempotency_key=uuid4().hex)
                with self.storage._locked() as conn:
                    reservation = self.storage._create(conn, owner, body, kind='temporary')['id']
            with self.storage._locked() as conn:
                self.storage._disk_budget(conn, 50_000_000)  # transient raster/text response, never retained
            # Stay on the asset volume so large PDFs use links, without an
            # unreserved second copy in the system temporary directory.
            with tempfile.TemporaryDirectory(prefix='ptb-papers-', dir=self.storage.root.parent) as temporary:
                root = Path(temporary)
                marker = json.loads((self.storage.root / '.ptb-storage.json').read_text('utf8'))['instance_id']
                (root / 'owner.json').write_text(json.dumps({'schema':'ptb-paper-scratch/1','instance':marker,'pid':os.getpid()}), encoding='utf8')
                for directory in snapshot['dirs']: (root / relative(directory)).mkdir(parents=True, exist_ok=True)
                for name, value in snapshot['json'].items():
                    path = root / relative(name); path.parent.mkdir(parents=True, exist_ok=True)
                    if name == 'cache/preferences.json' and value.get('libraryRoot'):
                        value = dict(value, libraryRoot=str(root / relative(value['libraryRoot'])))
                    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf8')
                identities = {}; hashes = {}; identity_hashes = {}
                for name, asset_id in snapshot['files'].items():
                    path = root / relative(name); path.parent.mkdir(parents=True, exist_ok=True)
                    with self.storage._locked() as conn:
                        row = self.storage._readable(conn, owner, asset_id)
                        if str(row['project_id']) != project: raise StorageError('project_not_found', 404)
                        os.link(self.storage._path(asset_id), path)
                        identities[(path.stat().st_dev, path.stat().st_ino)] = asset_id
                        identity_hashes[(path.stat().st_dev, path.stat().st_ino)] = row['sha256']
                        hashes[row['sha256']] = asset_id
                native_args = dict(args)
                if operation in ('chooseLibrary', 'importLocal'):
                    imports = root / 'imports'; imports.mkdir()
                    files = args['files']
                    if not isinstance(files, list) or len(files) > 1000: raise StorageError('invalid_request', 422)
                    names = set()
                    for item in files:
                        name = str(relative(item['relative']))
                        filename = PurePosixPath(name).name
                        # Different subfolders may have equal basenames; stage by an opaque key.
                        staging = uuid4().hex + '.pdf'
                        asset = self.storage.metadata(owner, str(UUID(item['asset_id'])))
                        if asset['project_id'] != project or not asset['name'].lower().endswith('.pdf') or asset['size_bytes'] > 100_000_000:
                            raise StorageError('paper_input_invalid', 422)
                        if name in names: raise StorageError('paper_duplicate_name', 409)
                        names.add(name)
                        target = imports / staging; target.mkdir()
                        os.link(self.storage._path(asset['id']), target / filename)
                        linked = (target / filename).stat()
                        identity_hashes[(linked.st_dev, linked.st_ino)] = asset['sha256']
                        hashes[asset['sha256']] = asset['id']
                        item['name'] = staging + '/' + filename
                    if operation == 'chooseLibrary':
                        name = str(relative(args['name']))
                        if '/' in name: raise StorageError('invalid_request', 422)
                        native_args = {'library': name, 'files': files}
                    else: native_args = {'folder': args['folder'], 'names': [item['name'] for item in files]}
                if operation == 'restoreLocal' and not re.fullmatch('[a-f0-9]{32}', str(args['id'])): raise StorageError('invalid_request', 422)
                try: response = self.runner(root, operation, native_args, stopped, progress)
                except StorageError as exc:
                    if exc.code != 'paper_cancelled' or operation not in ('refresh','download','downloadSelected'): raise
                    preserve_completed = True
                    check()
                    response = self.runner(root, 'status', {}, stopped, progress)
                if stopped(): raise StorageError('paper_cancelled', 409)
                if not response['ok']: return response
                value = response['value']
                if operation == 'export':
                    asset = self.publish(owner, project, root / 'export.pdf', args['id'] + '-' + args['language'] + ('-含批注' if args['annotated'] else '') + '.pdf', stopped, reservation)
                    check()
                    return {'ok': True, 'value': {'cancelled': False, 'name': asset['name'], 'asset_id': asset['id']}}
                if operation not in READ_ONLY:
                    updated = {'schema': 'ptb-web-papers/1', 'json': {}, 'files': {}, 'dirs': []}
                    for base in ('cache', 'libraries', 'trash'):
                        if not (root / base).exists(): continue
                        for path in sorted((root / base).rglob('*')):
                            name = path.relative_to(root).as_posix()
                            if path.is_dir(): updated['dirs'].append(name); continue
                            if not regular(path): raise StorageError('storage_path_rejected', 503)
                            if path.suffix == '.json':
                                item = json.loads(path.read_text('utf8'))
                                if name == 'cache/preferences.json' and item.get('libraryRoot'): item['libraryRoot'] = Path(item['libraryRoot']).relative_to(root).as_posix()
                                updated['json'][name] = item
                            elif path.suffix.lower() == '.pdf':
                                key = (path.stat().st_dev, path.stat().st_ino)
                                asset_id = identities.get(key)
                                if asset_id is None:
                                    with path.open('rb') as source: digest = hashlib.file_digest(source, 'sha256').hexdigest()
                                    asset_id = hashes.get(digest)
                                if asset_id is None:
                                    asset = self.publish(owner, project, path, path.name, stopped, reservation); asset_id = asset['id']; created.append(asset_id)
                                    hashes[asset['sha256']] = asset_id
                                updated['files'][name] = asset_id
                    if updated != snapshot:
                        state_path = root / 'state.json'; state_path.write_text(json.dumps(updated, ensure_ascii=False), encoding='utf8')
                        state = self.publish(owner, project, state_path, '论文阅读.m18-draft.json', stopped, reservation); created.append(state['id'])
                        with self.storage._locked() as conn:
                            self.storage.project_deadline(conn, owner, project)
                            current = conn.execute("SELECT * FROM ptb_storage.assets WHERE owner_id=%s AND project_id=%s AND name=%s AND state='ready' ORDER BY created_at DESC,id DESC LIMIT 1", (owner, project, NAME)).fetchone()
                            if (str(current['id']) if current else None) != (old['id'] if old else None): raise StorageError('paper_state_conflict', 409)
                            if cancel.is_set() and not preserve_completed: raise StorageError('paper_cancelled', 409)
                            conn.execute('UPDATE ptb_storage.assets SET name=%s WHERE id=%s', (NAME, state['id']))
                            if current: self.storage._delete(conn, current)
                            removed = set(snapshot['files'].values()) - set(updated['files'].values())
                            for asset_id in removed: self.storage._delete(conn, self.storage._row(conn, owner, asset_id))
                        created.clear()
                if isinstance(value, dict) and 'root' in value:
                    # Temporary hard links have changing ctime; managed assets
                    # already carry immutable content hashes, without reading PDFs.
                    library_content_versions(value, identity_hashes)
                    value['root'] = Path(value['root']).name if value['root'] else ''
                check()
                return response
        finally:
            for asset_id in created: self.storage.delete(owner, asset_id)
            if reservation: self.storage.delete(owner, reservation)


def process_alive(pid):
    if type(pid) is not int or pid < 1: return False
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle: return False
        try:
            code = wintypes.DWORD()
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]; kernel.CloseHandle(handle)
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True


def run_child(root, operation, args, stop, progress):
    from ptb_worker.io.limits import Limits
    payload = root / 'request.json'; payload.write_text(json.dumps({'operation': operation, 'args': args}, ensure_ascii=False), encoding='utf8')
    until = time.monotonic() + 180
    def stopped():
        report = root / 'progress.json'
        if report.exists():
            try: progress(json.loads(report.read_text('utf8')))
            except (OSError, ValueError): pass
        return stop()
    if os.name == 'nt':
        import sys
        from ptb_worker.native.windows import OwnedProcess
        from ptb_documents import web_child
        process = OwnedProcess([sys.executable, '-E', '-s', '-B', str(Path(web_child.__file__).with_name('web_bootstrap.py')), str(payload)], root, 1_000_000_000)
        (root / 'child.json').write_text(json.dumps({'pid':process.pid}), encoding='utf8')
        try:
            while process.poll() is None:
                if stopped(): raise StorageError('paper_cancelled', 409)
                if time.monotonic() > until: raise StorageError('paper_timeout', 503)
                time.sleep(.08)
            if process.poll() != 0: raise StorageError('paper_runtime_unavailable', 503)
        finally: process.close()
    else:
        from ptb_worker.native.linux_runtime import command
        from ptb_worker.native.posix import run_bounded
        from ptb_worker.io.limits import Cancelled
        try:
            run_bounded(command('papers', str(payload)), b'', root, Limits(output_bytes=1024, process_bytes=1_000_000_000, timeout_seconds=180), stop=stopped,
                on_started=lambda pid:(root/'child.json').write_text(json.dumps({'pid':pid}), encoding='utf8'))
        except Cancelled: raise StorageError('paper_cancelled', 409) from None
    response = root / 'response.json'
    if not response.exists() or response.stat().st_size > 50_000_000: raise StorageError('paper_runtime_unavailable', 503)
    return json.loads(response.read_text('utf8'))


def create_papers_router(ctx, store):
    service = WebPapers(store) if store is not None else None
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def lifespan(app):
        task = None
        if service:
            async def maintain():
                while True:
                    try: await asyncio.to_thread(service.cleanup)
                    except (OSError, StorageError): pass
                    await asyncio.sleep(30)
            task = asyncio.create_task(maintain())
        try: yield
        finally:
            if task:
                task.cancel()
                try: await task
                except asyncio.CancelledError: pass
            if service: service.pool.shutdown(wait=False, cancel_futures=True)
    router = APIRouter(prefix='/api/v1/papers', tags=['papers'], lifespan=lifespan)
    def identity(request: Request):
        if service is None: raise HTTPException(503, 'paper_service_unavailable')
        return ctx.mutation(request)
    @router.post('/{project_id}/request', operation_id='paper_request')
    async def request_paper(project_id: UUID, body: PaperRequest, request: Request, owner=Depends(identity)):
        ctx.project(owner, project_id)
        if body.operation not in FIELDS or set(body.args) != FIELDS[body.operation]: raise HTTPException(422, 'invalid_request')
        if not service.slots.acquire(blocking=False): raise HTTPException(429, 'paper_service_busy', headers={'Retry-After': '2'})
        cancel = threading.Event(); current_progress = {}
        def check():
            if ctx.session(request)['id'] != owner['id']: raise StorageError('account_changed', 409)
        try:
            future = service.pool.submit(service.call, owner['id'], str(project_id), body.operation, body.args, cancel, lambda p: current_progress.update(p), check)
        except BaseException:
            service.slots.release(); raise
        # Completion owns admission release even if ASGI disconnects before the
        # response generator starts. A slow download must not strand the worker.
        future.add_done_callback(lambda _: service.slots.release())
        async def stream():
            previous = None
            try:
                while not future.done():
                    if current_progress and current_progress != previous:
                        previous = dict(current_progress)
                        yield json.dumps({'progress': previous}) + '\n'
                    await asyncio.sleep(.1)
                try: result = future.result()
                except StorageError as exc: result = {'ok': False, 'error': exc.code}
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception('Paper request failed')
                    result = {'ok': False, 'error': '论文服务暂不可用，请稍后重试。'}
                yield json.dumps(result, ensure_ascii=False) + '\n'
            finally:
                cancel.set()
        return StreamingResponse(stream(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
    return router
