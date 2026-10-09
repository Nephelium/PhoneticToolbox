"""M18: bounded, content-addressed paper delivery, separate from application assets."""
import hashlib
import json
import os
import re
import threading
import uuid
import math
import shutil
from datetime import date
from pathlib import Path
from urllib.request import build_opener, HTTPRedirectHandler, Request

BASE_URL = 'https://www.phonetictoolbox.com/papers/'
LICENSES = {'CC-BY-4.0': 'https://creativecommons.org/licenses/by/4.0/',
            'CC-BY-SA-4.0': 'https://creativecommons.org/licenses/by-sa/4.0/',
            'CC0-1.0': 'https://creativecommons.org/publicdomain/zero/1.0/'}
MAX_FILE = 50_000_000


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
    ids = set()
    for p in value['papers']:
        if not isinstance(p, dict) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', p.get('id', '')) or p['id'] in ids:
            raise PaperError('论文编号重复或无效。')
        ids.add(p['id'])
        for field in ('title', 'titleZh', 'authors', 'version', 'translationNote', 'guide'):
            if not isinstance(p.get(field), str) or not 1 <= len(p[field]) <= 12000:
                raise PaperError('论文署名或说明不完整。')
        iso_date(p.get('publishedAt')); iso_date(p.get('submittedAt'))
        if not re.fullmatch(r'https://arxiv\.org/abs/\d{4}\.\d{4,5}v\d+', p.get('sourceUrl', '')):
            raise PaperError('原文版本链接无效。')
        lic = p.get('license', {})
        if not isinstance(lic, dict) or lic.get('id') not in LICENSES or lic.get('url') != LICENSES[lic['id']]:
            raise PaperError('论文缺少可共享、可改编的已审核许可。')
        for language in ('original', 'translation'):
            asset = p.get(language, {})
            if not isinstance(asset, dict) or not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+\.pdf', asset.get('path', '')):
                raise PaperError('论文文件路径无效。')
            if type(asset.get('size')) is not int or not 1 <= asset['size'] <= MAX_FILE or not re.fullmatch(r'[a-f0-9]{64}', asset.get('sha256', '')):
                raise PaperError('论文文件校验信息无效。')
    return value


class PaperService:
    def __init__(self, root, *, today=None, opener=None):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.files = self.root / 'files'; self.files.mkdir(exist_ok=True)
        self.opener = opener or build_opener(NoRedirect())
        self.today = today or date.today
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
        return self.files / (asset['sha256'] + '.pdf')

    def _verified(self, asset):
        path = self._path(asset)
        try:
            if path.stat().st_size != asset['size']: return False
            with path.open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest() == asset['sha256']
        except OSError:
            return False

    def status(self):
        return {'firstLaunch': self.first_launch, 'warning': self.cache_warning,
                'papers': [dict(p, downloaded=all(self._verified(p[k]) for k in ('original', 'translation')))
                           for p in self.catalog['papers']]}

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
        return self.status()

    def download(self, since, cancel, progress):
        since = iso_date(since)
        papers = [p for p in self.catalog['papers'] if since <= p['publishedAt'] <= self.today().isoformat()]
        assets = [p[k] for p in papers for k in ('original', 'translation') if not self._verified(p[k])]
        total = sum(a['size'] for a in assets); received = 0
        for asset in assets:
            progress({'received': received, 'total': total})
            data = self._read(asset['path'], asset['size'], cancel)
            if len(data) != asset['size'] or hashlib.sha256(data).hexdigest() != asset['sha256'] or not data.startswith(b'%PDF-'):
                raise PaperError('论文完整性校验失败，请刷新目录后重试。')
            if cancel.is_set(): raise PaperError('下载已取消，已完成的论文保留。')
            path = self._path(asset); temp = path.with_suffix('.' + uuid.uuid4().hex + '.part')
            try:
                with temp.open('xb') as stream:
                    stream.write(data); stream.flush(); os.fsync(stream.fileno())
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
            received += len(data); progress({'received': received, 'total': total})
        return self.status()

    def asset(self, paper_id, language):
        if language not in ('original', 'translation'): raise PaperError('论文语言无效。')
        paper = next((p for p in self.catalog['papers'] if p['id'] == paper_id), None)
        if paper is None or not self._verified(paper[language]):
            raise PaperError('论文尚未下载或文件损坏，请重新获取。')
        return paper[language]

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
        if language not in ('original', 'translation') or type(page) is not int or type(width) is not int or not 400 <= width <= 2200:
            raise PaperError('阅读参数无效。')
        paper = next((p for p in self.catalog['papers'] if p['id'] == paper_id), None)
        if paper is None or not self._verified(paper[language]):
            raise PaperError('论文尚未下载或文件损坏，请重新获取。')
        # Objects live and are destroyed on the same serialized worker thread.
        from PyQt6.QtPdf import QPdfDocument
        from PyQt6.QtCore import QBuffer, QIODevice, QSize
        import base64
        doc = QPdfDocument(None)
        try:
            doc.load(str(self._path(paper[language])))
            if doc.status() != QPdfDocument.Status.Ready or not 1 <= doc.pageCount() <= 1000:
                raise PaperError('PDF 无法读取。')
            if not 1 <= page <= doc.pageCount(): raise PaperError('页码超出范围。')
            size = doc.pagePointSize(page - 1)
            if size.width() <= 0 or not .1 <= size.height() / size.width() <= 5:
                raise PaperError('PDF 页面尺寸不受支持。')
            render_width = min(width, int(4000 * size.width() / size.height()))
            rendered = doc.render(page - 1, QSize(render_width, round(render_width * size.height() / size.width())))
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
            return {'pages': doc.pageCount(), 'page': page, 'image': 'data:image/png;base64,' + base64.b64encode(bytes(buffer.data())).decode('ascii'),
                    'text': text, 'lines':lines, 'ratio': size.height() / size.width(),
                    'ratios':[doc.pagePointSize(i).height()/doc.pagePointSize(i).width() for i in range(doc.pageCount())]}
        finally:
            doc.close()
