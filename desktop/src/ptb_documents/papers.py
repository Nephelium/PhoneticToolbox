"""M18: bounded, content-addressed paper delivery, separate from application assets."""
import hashlib
import json
import os
import re
import threading
import uuid
import math
import shutil
from collections import OrderedDict
from datetime import date
from pathlib import Path
from urllib.request import build_opener, HTTPRedirectHandler, Request

BASE_URL = 'https://www.phonetictoolbox.com/papers/'
LICENSES = {'CC-BY-4.0': 'https://creativecommons.org/licenses/by/4.0/',
            'CC-BY-SA-4.0': 'https://creativecommons.org/licenses/by-sa/4.0/',
            'CC0-1.0': 'https://creativecommons.org/publicdomain/zero/1.0/'}
MAX_FILE = 50_000_000


def asset_kinds(paper):
    if paper.get('notice'): return ('original',)
    return ('original', 'translation', 'analysis') if 'analysisPdf' in paper else ('original', 'translation')


def paper_asset(paper, language):
    return paper['analysisPdf' if language == 'analysis' else language]


def catalog_papers(catalog):
    """Present both sources without rewriting the legacy on-disk list."""
    return catalog['papers'] + catalog.get('journalPapers', [])


class PaperError(ValueError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise PaperError('论文服务器发生重定向，请稍后重试。')


def iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise PaperError('日期格式无效。')
    date.fromisoformat(value)
    return value


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_catalog(value):
    if not isinstance(value, dict) or value.get('schema') != 'ptb-papers/1' or not isinstance(value.get('papers'), list) or len(value['papers']) > 10000:
        raise PaperError('论文目录版本或大小无效。')
    journals = value.get('journalPapers', [])
    if not isinstance(journals, list) or len(value['papers']) + len(journals) > 10000:
        raise PaperError('论文目录版本或大小无效。')
    ids = set()
    for index, p in enumerate(catalog_papers(value)):
        if not isinstance(p, dict) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', p.get('id', '')) or p['id'] in ids:
            raise PaperError('论文编号重复或无效。')
        ids.add(p['id'])
        for field in ('title', 'titleZh', 'authors', 'version', 'translationNote', 'guide'):
            if not isinstance(p.get(field), str) or not 1 <= len(p[field]) <= 12000:
                raise PaperError('论文署名或说明不完整。')
        iso_date(p.get('publishedAt')); iso_date(p.get('submittedAt'))
        if 'updatedAt' in p: iso_date(p['updatedAt'])
        is_journal = index >= len(value['papers'])
        source = p.get('sourceUrl', '')
        pattern = (r'https://doi\.org/10\.\d{4,9}/[a-zA-Z0-9][a-zA-Z0-9._;()/:-]*' if is_journal
                   else r'https://arxiv\.org/abs/\d{4}\.\d{4,5}v\d+')
        if not isinstance(source, str) or len(source) > 2048 or not re.fullmatch(pattern, source):
            raise PaperError('原文版本链接无效。')
        if is_journal:
            journal = p.get('journal')
            if not isinstance(journal, dict) or not {'title', 'publishedAt', 'doi'} <= journal.keys() or set(journal) - {'title', 'publishedAt', 'doi', 'volume', 'issue', 'pages', 'articleNumber'}:
                raise PaperError('期刊出处不完整。')
            for key, text in journal.items():
                if not isinstance(text, str) or not 1 <= len(text) <= 1000 or any(ord(c) < 32 for c in text):
                    raise PaperError('期刊出处无效。')
            try:
                iso_date(journal['publishedAt'])
            except ValueError as exc:
                raise PaperError('期刊发表日期无效。') from exc
            if source != 'https://doi.org/' + journal['doi']:
                raise PaperError('期刊DOI与原文链接不一致。')
        elif 'journal' in p:
            raise PaperError('期刊论文应放在期刊目录中。')
        lic = p.get('license', {})
        if not isinstance(lic, dict) or lic.get('id') not in LICENSES or lic.get('url') != LICENSES[lic['id']]:
            raise PaperError('论文缺少可共享、可改编的已审核许可。')
        for language in asset_kinds(p):
            asset = paper_asset(p, language)
            if isinstance(asset, dict) and set(asset) != {'path', 'size', 'sha256'}:
                raise PaperError('论文文件字段无效。')
            if not isinstance(asset, dict) or not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+\.pdf', asset.get('path', '')):
                raise PaperError('论文文件路径无效。')
            if type(asset.get('size')) is not int or not 1 <= asset['size'] <= MAX_FILE or not re.fullmatch(r'[a-f0-9]{64}', asset.get('sha256', '')):
                raise PaperError('论文文件校验信息无效。')
        for field in ('abstract', 'abstractZh', 'analysis'):
            if field in p and (not isinstance(p[field], str) or len(p[field]) > 40000):
                raise PaperError('论文摘要或分析无效。')
    notice = value.get('notice')
    if notice is not None:
        if not isinstance(notice, dict) or set(notice) != {'id', 'title', 'authors', 'publishedAt', 'updatedAt', 'pdf'}:
            raise PaperError('栏目说明字段无效。')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', notice.get('id', '')) or notice['id'] in ids:
            raise PaperError('栏目说明编号无效。')
        for key in ('title', 'authors'):
            if not isinstance(notice[key], str) or not 1 <= len(notice[key]) <= 1000: raise PaperError('栏目说明署名无效。')
        iso_date(notice['publishedAt']); iso_date(notice['updatedAt'])
        asset = notice['pdf']
        if not isinstance(asset, dict) or set(asset) != {'path', 'size', 'sha256'} or not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+\.pdf', asset.get('path', '')):
            raise PaperError('栏目说明文件路径无效。')
        if type(asset.get('size')) is not int or not 1 <= asset['size'] <= MAX_FILE or not re.fullmatch(r'[a-f0-9]{64}', asset.get('sha256', '')):
            raise PaperError('栏目说明文件校验信息无效。')
    return value


class PaperService:
    def __init__(self, root, *, today=None, opener=None):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.files = self.root / 'files'; self.files.mkdir(exist_ok=True)
        self.opener = opener or build_opener(NoRedirect())
        self.today = today or date.today
        self._verified_cache = {}
        self._local_hashes = {}
        self._library_instance = None
        self._documents = OrderedDict()
        self._pages = OrderedDict()
        self._settings_lock = threading.RLock()
        self._removal_versions = {}
        # The lock covers creation and reading so a second process never sees
        # an incomplete first-launch record. Upgrades never rewrite it.
        from PyQt6.QtCore import QLockFile
        first = self.root / 'first-launch.json'
        first_lock = QLockFile(str(self.root / 'first-launch.lock'))
        if not first_lock.tryLock(5000): raise PaperError('首次启动日期正在写入，请重新打开论文模块。')
        try:
            if not first.exists(): atomic_json(first, {'date': self.today().isoformat()})
            self.first_launch = iso_date(json.loads(first.read_text('utf8'))['date'])
        finally:
            first_lock.unlock()
        self.catalog = {'schema': 'ptb-papers/1', 'papers': []}
        self.cache_warning = ''
        cached = self.root / 'catalog.json'
        if cached.exists():
            try:
                self.catalog = validate_catalog(json.loads(cached.read_text('utf8')))
            except (ValueError, OSError, KeyError, TypeError):
                self.cache_warning = '本机目录无法读取，请刷新论文目录。'

    def _path(self, asset):
        if '_localPath' in asset:
            library = self.library()
            return library.safe(asset['_localPath'])
        return self.files / (asset['sha256'] + '.pdf')

    def _verified(self, asset):
        path = self._path(asset)
        try:
            stat = path.stat()
            if stat.st_size != asset['size']: return False
            identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            key = (str(path), asset['sha256'])
            if self._verified_cache.get(key) == identity: return True
            with path.open('rb') as stream:
                valid = hashlib.file_digest(stream, 'sha256').hexdigest() == asset['sha256']
            if valid:
                if len(self._verified_cache) > 20000: self._verified_cache.clear()
                self._verified_cache[key] = identity
            return valid
        except OSError:
            return False

    def status(self):
        pref = self.preferences()
        return {'firstLaunch': self.first_launch, 'warning': self.cache_warning,
                'autoReceive': pref['autoReceive'],
                'checkIntervalHours': pref.get('checkIntervalHours', 6),
                'folders': pref.get('recommendationFolders', []),
                'collections': pref.get('recommendationAssignments', {}),
                'notice': self.notice(),
                'papers': [dict(p, dismissed=p['id'] in pref.get('removedRecommendations', []), acquired=p['id'] not in pref.get('removedRecommendations', []) and all(self._verified(p[k]) for k in ('original','translation')),
                               analysisDownloaded='analysisPdf' not in p or self._verified(p['analysisPdf']),
                               downloaded=p['id'] not in pref.get('removedRecommendations', []) and all(self._verified(paper_asset(p,k)) for k in asset_kinds(p)))
                           for p in catalog_papers(self.catalog)]}

    def notice(self):
        n = self.catalog.get('notice')
        if not n or max(n['publishedAt'], n['updatedAt']) > self.today().isoformat(): return None
        dismissed = n['id'] in self.preferences().get('removedRecommendations', [])
        return dict(id=n['id'], title=n['title'], titleZh=n['title'], authors=n['authors'],
                    publishedAt=n['publishedAt'], submittedAt=n['publishedAt'], updatedAt=n['updatedAt'],
                    version='栏目说明', license={'id': '', 'url': ''}, sourceUrl='',
                    translationNote='', guide='', original=n['pdf'], translation=n['pdf'],
                    notice=True, dismissed=dismissed, downloaded=not dismissed and self._verified(n['pdf']))

    def download_notice(self, cancel):
        with self._settings_lock:
            n = self.notice()
            if not n or n['downloaded'] or n['dismissed']: return
            version = self._removal_versions.get(n['id'], 0)
        asset = n['original']; data = self._read(asset['path'], asset['size'], cancel)
        if len(data) != asset['size'] or hashlib.sha256(data).hexdigest() != asset['sha256'] or not data.startswith(b'%PDF-'):
            raise PaperError('栏目说明完整性校验失败，请稍后重试。')
        if cancel.is_set(): raise PaperError('获取已取消。')
        path = self._path(asset); temp = path.with_suffix('.' + uuid.uuid4().hex + '.part')
        try:
            with self._settings_lock:
                if self._removal_versions.get(n['id'], 0) != version or n['id'] in self.preferences().get('removedRecommendations', []): return
                with temp.open('xb') as stream:
                    stream.write(data); stream.flush(); os.fsync(stream.fileno())
                os.replace(temp, path)
        finally: temp.unlink(missing_ok=True)

    def preferences(self):
        path = self.root / 'preferences.json'
        if not path.exists(): return {'autoReceive': True, 'libraryRoot': '', 'checkIntervalHours': 6}
        try:
            value = json.loads(path.read_text('utf8'))
            if type(value.get('autoReceive')) is not bool or not isinstance(value.get('libraryRoot', ''), str): raise ValueError()
            removed = value.get('removedRecommendations', [])
            if not isinstance(removed, list) or len(removed) > 10000 or not all(isinstance(i, str) and re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', i) for i in removed): raise ValueError()
            return value
        except (ValueError, OSError, TypeError): raise PaperError('论文偏好无法读取，记录已保留。')

    def set_auto_receive(self, enabled):
        if type(enabled) is not bool: raise PaperError('自动获取设置无效。')
        with self._settings_lock:
            value = self.preferences(); value['autoReceive'] = enabled
            atomic_json(self.root / 'preferences.json', value)
        return self.status()

    def set_interval(self, hours):
        if type(hours) not in (int, float) or not math.isfinite(hours) or not .25 <= hours <= 720:
            raise PaperError('检查间隔应为 0.25 至 720 小时。')
        with self._settings_lock:
            value = self.preferences(); value['checkIntervalHours'] = hours
            atomic_json(self.root / 'preferences.json', value)
        return self.status()

    def remove_download(self, paper_id):
        """Remove this recommendation's cached bytes, retaining its notes.

        Closed PDF objects and same-directory renames keep Windows file locks
        and a simultaneous download from restoring a just-removed paper.
        Other recommendations sharing a content hash retain their asset.
        """
        paper = next((p for p in catalog_papers(self.catalog) if p['id'] == paper_id), None)
        n = self.notice()
        if paper is None and n and n['id'] == paper_id: paper = n
        if paper is None: raise PaperError('推荐论文不存在，请刷新目录。')
        self.close_documents()
        with self._settings_lock:
            value = self.preferences()
            removed = set(value.get('removedRecommendations', []))
            shared = {paper_asset(p, k)['sha256'] for p in catalog_papers(self.catalog)
                      if p['id'] != paper_id and p['id'] not in removed for k in asset_kinds(p)}
            if n and n['id'] != paper_id and n['id'] not in removed: shared.add(n['original']['sha256'])
            paths = {self._path(paper_asset(paper, k)) for k in asset_kinds(paper)
                     if paper_asset(paper, k)['sha256'] not in shared}
            for path in (self.files, *paths):
                if path.exists() and (path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400):
                    raise PaperError('下载缓存包含链接，已停止删除。')
            staged = []
            try:
                for path in paths:
                    if not path.exists(): continue
                    temporary = path.with_name('.removed-' + path.stem + '-' + uuid.uuid4().hex + '.pdf')
                    path.rename(temporary); staged.append((path, temporary))
                value['removedRecommendations'] = sorted(removed | {paper_id})
                atomic_json(self.root / 'preferences.json', value)
            except OSError as exc:
                rollback_failed = False
                for original, temporary in reversed(staged):
                    try: temporary.rename(original)
                    except OSError: rollback_failed = True
                if rollback_failed:
                    raise PaperError('缓存移除未完成，部分文件仍在暂存位置，批注保留。请关闭其他阅读窗口后重试获取。') from exc
                raise PaperError('缓存文件正在使用或无法删除，已恢复原位置。请关闭其他阅读窗口后重试。') from exc
            self._removal_versions[paper_id] = self._removal_versions.get(paper_id, 0) + 1
            for _, temporary in staged:
                try: temporary.unlink()
                except OSError: self.cache_warning = '论文已移除，但部分占用缓存暂未释放。批注仍保留。'
        return self.status()

    def recommendation_folder(self, name):
        if not isinstance(name, str) or not name.strip() or len(name) > 80: raise PaperError('收藏夹名称无效。')
        name = name.strip()
        with self._settings_lock:
            value = self.preferences(); folders = value.setdefault('recommendationFolders', [])
            if name in folders: raise PaperError('同名收藏夹已存在。')
            folders.append(name); atomic_json(self.root / 'preferences.json', value)
        return self.status()

    def collect_recommendation(self, paper_id, folder):
        with self._settings_lock:
            value = self.preferences()
            if folder not in ['', *value.get('recommendationFolders', [])] or paper_id not in {p['id'] for p in catalog_papers(self.catalog)}:
                raise PaperError('论文或收藏夹不存在。')
            value.setdefault('recommendationAssignments', {})[paper_id] = folder
            atomic_json(self.root / 'preferences.json', value)
        return self.status()

    def delete_recommendation_folder(self, name):
        with self._settings_lock:
            value = self.preferences()
            folders = value.get('recommendationFolders', [])
            if not isinstance(name, str) or not name or name not in folders:
                raise PaperError('请选择已有的收藏夹。')
            value['recommendationFolders'] = [folder for folder in folders if folder != name]
            value['recommendationAssignments'] = {key: folder for key, folder in value.get('recommendationAssignments', {}).items() if folder != name}
            atomic_json(self.root / 'preferences.json', value)
        return self.status()

    def library(self):
        from .paper_library import PersonalPaperLibrary
        root = self.preferences().get('libraryRoot', '')
        if not root: raise PaperError('请先选择本地论文文件夹。')
        if self._library_instance is None or str(self._library_instance.root) != root:
            self._library_instance = PersonalPaperLibrary(root)
        return self._library_instance

    def library_status(self):
        if not self.preferences().get('libraryRoot'): return {'root': '', 'folders': [], 'documents': []}
        return self.library().entries()

    def set_library_root(self, root):
        from .paper_library import PersonalPaperLibrary
        library = PersonalPaperLibrary(root)
        # Server storage and the user library must remain physically separate.
        if library.root.is_relative_to(self.root.resolve()) or self.root.resolve().is_relative_to(library.root):
            raise PaperError('请选择独立于应用论文缓存的文件夹。')
        with self._settings_lock:
            value = self.preferences(); value['libraryRoot'] = str(library.root)
            atomic_json(self.root / 'preferences.json', value)
        self.close_documents()
        self._library_instance = library
        return library.entries()

    def _read(self, path, limit, cancel):
        request = Request(BASE_URL + path, headers={'User-Agent': 'PhoneticToolbox-Papers/1', 'Accept-Encoding': 'identity'})
        with self.opener.open(request, timeout=15) as response:
            if response.geturl() != BASE_URL + path:
                raise PaperError('论文下载地址不匹配。')
            result = bytearray()
            while True:
                if cancel.is_set(): raise PaperError('下载已取消，已完成的论文保留。')
                chunk = response.read(min(65536, limit + 1 - len(result)))
                if not chunk: break
                result.extend(chunk)
                if len(result) > limit: raise PaperError('论文文件超过大小限制。')
            return bytes(result)

    def refresh(self, cancel):
        value = validate_catalog(json.loads(self._read('catalog.json', 4_000_000, cancel)))
        atomic_json(self.root / 'catalog.json', value)
        self.catalog = value; self.cache_warning = ''
        self.download_notice(cancel)
        return self.status()

    def download(self, since, cancel, progress, ids=None):
        since = iso_date(since)
        today = self.today().isoformat()
        candidates = catalog_papers(self.catalog)
        n = self.notice()
        if ids is not None and n: candidates.append(n)
        papers = [p for p in candidates if p['publishedAt'] <= today and p.get('updatedAt', p['publishedAt']) <= today and
                  (p['publishedAt'] >= since or p.get('updatedAt', '') >= since or all(self._verified(p[k]) for k in ('original','translation')))]
        if ids is not None:
            if not isinstance(ids, list) or len(ids) > 10000 or not all(isinstance(i, str) for i in ids): raise PaperError('勾选论文无效。')
            if set(ids) - {p['id'] for p in papers}: raise PaperError('勾选论文不在当前可获取目录中。')
            papers = [p for p in papers if p['id'] in ids]
        with self._settings_lock:
            if ids is None:
                removed = set(self.preferences().get('removedRecommendations', []))
                papers = [p for p in papers if p['id'] not in removed]
            versions = {p['id']: self._removal_versions.get(p['id'], 0) for p in papers}
        unique = {paper_asset(p,k)['sha256']:paper_asset(p,k) for p in papers for k in asset_kinds(p)}
        assets = [a for a in unique.values() if not self._verified(a)]
        total = sum(a['size'] for a in assets); received = 0
        for asset in assets:
            progress({'received': received, 'total': total})
            data = self._read(asset['path'], asset['size'], cancel)
            if len(data) != asset['size'] or hashlib.sha256(data).hexdigest() != asset['sha256'] or not data.startswith(b'%PDF-'):
                raise PaperError('论文完整性校验失败，请刷新目录后重试。')
            if cancel.is_set(): raise PaperError('下载已取消，已完成的论文保留。')
            path = self._path(asset); temp = path.with_suffix('.' + uuid.uuid4().hex + '.part')
            try:
                with self._settings_lock:
                    owners = [p for p in papers if any(paper_asset(p, k)['sha256'] == asset['sha256'] for k in asset_kinds(p))]
                    if any(self._removal_versions.get(p['id'], 0) == versions[p['id']] for p in owners):
                        with temp.open('xb') as stream:
                            stream.write(data); stream.flush(); os.fsync(stream.fileno())
                        os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
            received += len(data); progress({'received': received, 'total': total})
        if ids is not None:
            with self._settings_lock:
                value = self.preferences(); removed = set(value.get('removedRecommendations', []))
                restored = {p['id'] for p in papers if self._removal_versions.get(p['id'], 0) == versions[p['id']] and all(self._verified(paper_asset(p, k)) for k in asset_kinds(p))}
                if removed & restored:
                    value['removedRecommendations'] = sorted(removed - restored)
                    atomic_json(self.root / 'preferences.json', value)
        return self.status()

    def asset(self, paper_id, language):
        if language not in ('original', 'translation', 'analysis'): raise PaperError('论文语言无效。')
        if isinstance(paper_id, str) and paper_id.startswith('local_'):
            if language != 'original': raise PaperError('本地 PDF 没有服务器译文。')
            library = self.library(); path = library.find(paper_id)
            stat = path.stat()
            key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            digest = self._local_hashes.get(key)
            if digest is None:
                with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if len(self._local_hashes) > 1000: self._local_hashes.clear()
                self._local_hashes[key] = digest
            return {'sha256': digest, 'size': stat.st_size, '_localPath': path.relative_to(library.root).as_posix()}
        paper = next((p for p in catalog_papers(self.catalog) if p['id'] == paper_id), None)
        if paper is None:
            n = self.notice()
            if n and n['id'] == paper_id: paper = n
        if paper is None or language not in asset_kinds(paper) or not self._verified(paper_asset(paper, language)):
            raise PaperError('论文尚未下载或文件损坏，请重新获取。')
        return paper_asset(paper, language)

    def annotations(self, paper_id, language):
        asset = self.asset(paper_id, language)
        path = self.root / ('annotations-' + asset['sha256'] + '.json')
        if not path.exists(): return {'revision': 0, 'items': []}
        try:
            value = json.loads(path.read_text('utf8'))
            if value.get('schema') != 'ptb-paper-annotations/1' or type(value.get('revision')) is not int or not isinstance(value.get('items'), list): raise ValueError()
            for item in value['items']: self.validate_annotation(item)
            return {'revision': value['revision'], 'items': value['items']}
        except (OSError, ValueError, TypeError, KeyError):
            raise PaperError('批注记录无法读取，原记录已保留。请先备份并检查记录。')

    @staticmethod
    def validate_annotation(item):
        if not isinstance(item, dict) or set(item) != {'id','page','kind','color','text','quote','rects'}: raise PaperError('批注格式无效。')
        if not isinstance(item['id'], str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', item['id']): raise PaperError('批注编号无效。')
        if type(item['page']) is not int or not 1 <= item['page'] <= 1000 or item['kind'] not in ('highlight','note') or item['color'] not in ('yellow','green','pink'): raise PaperError('批注参数无效。')
        if not all(isinstance(item[k], str) and len(item[k]) <= 5000 for k in ('text','quote')): raise PaperError('批注文字超过上限。')
        if not isinstance(item['rects'], list) or not 1 <= len(item['rects']) <= 200: raise PaperError('选区过长，请分段标记。')
        for r in item['rects']:
            if not isinstance(r, list) or len(r) != 4 or not all(type(n) in (int,float) and math.isfinite(n) and 0 <= n <= 1 for n in r) or r[2] <= 0 or r[3] <= 0 or r[0]+r[2] > 1.002 or r[1]+r[3] > 1.002: raise PaperError('批注坐标无效。')

    def save_annotation(self, paper_id, language, revision, item, remove):
        from PyQt6.QtCore import QLockFile
        asset = self.asset(paper_id, language)
        path = self.root / ('annotations-' + asset['sha256'] + '.json')
        lock = QLockFile(str(path) + '.lock')
        if not lock.tryLock(3000): raise PaperError('批注正在由另一窗口保存，请稍后重试。')
        try:
            value = self.annotations(paper_id, language)
            if type(revision) is not int or revision != value['revision']: raise PaperError('批注已在另一窗口更改。请重新载入批注后重试，当前输入仍保留。')
            self.validate_annotation(item)
            if type(remove) is not bool: raise PaperError('批注操作无效。')
            items = [a for a in value['items'] if a['id'] != item['id']]
            if not remove: items.append(item)
            if len(items) > 1000: raise PaperError('此 PDF 已有 1000 条批注，请导出整理后再添加。')
            result = {'revision': revision + 1, 'items': items}
            atomic_json(path, {'schema': 'ptb-paper-annotations/1', **result})
            return result
        finally: lock.unlock()

    def export(self, paper_id, language, annotated, destination):
        asset = self.asset(paper_id, language)
        target = Path(destination)
        if type(annotated) is not bool or target.suffix.lower() != '.pdf': raise PaperError('请选择 PDF 保存位置。')
        if target.resolve() == self._path(asset).resolve(): raise PaperError('请另存副本，不能覆盖下载的原文件。')
        temp = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            if not annotated:
                shutil.copyfile(self._path(asset), temp)
            else:
                from pypdf import PdfWriter
                from pypdf.annotations import Highlight, Text
                from pypdf.generic import ArrayObject, FloatObject, NameObject, TextStringObject
                writer = PdfWriter(clone_from=str(self._path(asset)))
                try:
                    for item in self.annotations(paper_id, language)['items']:
                        if item['page'] > len(writer.pages): raise PaperError('批注页码与 PDF 不匹配。')
                        pdf_page = writer.pages[item['page'] - 1]
                        if pdf_page.rotation: pdf_page.transfer_rotation_to_content()
                        box = pdf_page.cropbox; width,height = float(box.width),float(box.height)
                        rects = [(float(box.left)+x*width,float(box.top)-(y+h)*height,float(box.left)+(x+w)*width,float(box.top)-y*height) for x,y,w,h in item['rects']]
                        rect = (min(r[0] for r in rects),min(r[1] for r in rects),max(r[2] for r in rects),max(r[3] for r in rects))
                        if item['kind'] == 'highlight':
                            quads = ArrayObject([FloatObject(n) for x0,y0,x1,y1 in rects for n in (x0,y1,x1,y1,x0,y0,x1,y0)])
                            annotation = Highlight(rect=rect,quad_points=quads,highlight_color={'yellow':'FFE066','green':'8DE4A7','pink':'FFA6CE'}[item['color']],printing=True)
                            annotation[NameObject('/Contents')] = TextStringObject(item['text'] or item['quote'])
                        else:
                            annotation = Text(rect=(rect[0],rect[3]-18,rect[0]+18,rect[3]),text=item['text']+'\n\n'+item['quote'],flags=4)
                        annotation[NameObject('/T')] = TextStringObject('PhoneticToolbox 个人批注')
                        writer.add_annotation(page_number=item['page']-1,annotation=annotation)
                    with temp.open('xb') as stream: writer.write(stream)
                finally: writer.close()
            os.replace(temp, target)
            return {'cancelled': False, 'name': target.name}
        finally: temp.unlink(missing_ok=True)

    def render(self, paper_id, language, page, width):
        if language not in ('original', 'translation', 'analysis') or type(page) is not int or type(width) is not int or not 400 <= width <= 4096:
            raise PaperError('阅读参数无效。')
        asset = self.asset(paper_id, language)
        # Objects live and are destroyed on the same serialized worker thread.
        from PyQt6.QtPdf import QPdfDocument
        from PyQt6.QtCore import QBuffer, QIODevice, QSize
        import base64
        path = self._path(asset); key = (str(path), asset['sha256'])
        doc = self._documents.get(key)
        if doc is None:
            doc = QPdfDocument(None)
            doc.load(str(path))
            if doc.status() != QPdfDocument.Status.Ready or not 1 <= doc.pageCount() <= 1000:
                doc.close()
                raise PaperError('PDF 无法读取。')
            self._documents[key] = doc
            while len(self._documents) > 2:
                _, previous = self._documents.popitem(last=False); previous.close()
        self._documents.move_to_end(key)
        cache_key = (*key, page, width)
        if cache_key in self._pages:
            self._pages.move_to_end(cache_key)
            return self._pages[cache_key]
        try:
            if not 1 <= page <= doc.pageCount(): raise PaperError('页码超出范围。')
            size = doc.pagePointSize(page - 1)
            if size.width() <= 0 or not .1 <= size.height() / size.width() <= 5:
                raise PaperError('PDF 页面尺寸不受支持。')
            ratio = size.height() / size.width()
            # Bound both dimension and decoded pixels, including unusual tall pages.
            render_width = max(1, int(min(width, 6144 / ratio, (16_777_216 / ratio) ** .5)))
            render_height = max(1, min(6144, 16_777_216 // render_width, round(render_width * ratio)))
            rendered = doc.render(page - 1, QSize(render_width, render_height))
            if rendered.isNull(): raise PaperError('PDF 页面渲染失败。')
            buffer = QBuffer(); buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            if not rendered.save(buffer, 'PNG'): raise PaperError('PDF 页面渲染失败。')
            text = doc.getAllText(page - 1).text()
            lines = []; index = 0
            for line in text.splitlines(keepends=True):
                length = len(line.encode('utf-16-le')) // 2
                if line.strip():
                    selection = doc.getSelectionAtIndex(page - 1,index,length-len(line)+len(line.rstrip('\r\n')))
                    rect = selection.boundingRectangle()
                    if rect.width()>0 and rect.height()>0:
                        lines.append({'text':line.rstrip('\r\n'),'x':rect.x()/size.width(),'y':rect.y()/size.height(),'w':rect.width()/size.width(),'h':rect.height()/size.height()})
                index += length
            result = {'pages': doc.pageCount(), 'page': page, 'imageWidth':rendered.width(), 'imageHeight':rendered.height(), 'image': 'data:image/png;base64,' + base64.b64encode(bytes(buffer.data())).decode('ascii'),
                    'text': text, 'lines':lines, 'ratio': size.height() / size.width(),
                    'ratios':[doc.pagePointSize(i).height()/doc.pagePointSize(i).width() for i in range(doc.pageCount())]}
            self._pages[cache_key] = result
            while len(self._pages) > 12 or sum(len(p['image']) for p in self._pages.values()) > 48_000_000:
                self._pages.popitem(last=False)
            return result
        finally:
            pass

    def close_documents(self):
        for doc in self._documents.values(): doc.close()
        self._documents.clear(); self._pages.clear()
