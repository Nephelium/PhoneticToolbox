"""User-selected filesystem PDF library. Never follows links or overwrites files."""
import hashlib
import os
from pathlib import Path
import re
import shutil
from datetime import datetime
import uuid

from .papers import PaperError


def linked(path):
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


class PersonalPaperLibrary:
    def __init__(self, root):
        self.root = Path(root).absolute()
        for path in (self.root, *self.root.parents):
            if linked(path): raise PaperError('论文文件夹不能位于符号链接或联接内。')
        if not self.root.is_dir(): raise PaperError('论文文件夹不存在，请重新选择。')
        self.root = self.root.resolve()
        self.index = None

    def safe(self, relative):
        if not isinstance(relative, str): raise PaperError('库内位置无效。')
        part = Path(relative)
        if part.is_absolute() or part.drive or '..' in part.parts or ':' in relative:
            raise PaperError('操作必须位于所选论文文件夹内。')
        path = self.root / part
        if not path.resolve().is_relative_to(self.root): raise PaperError('库内位置越界。')
        for component in (path, *path.parents):
            if component.exists() and linked(component): raise PaperError('不能读取或整理符号链接与联接。')
            if component == self.root: break
        return path

    @staticmethod
    def identity(relative):
        return 'local_' + hashlib.sha256(relative.encode('utf8')).hexdigest()[:32]

    def entries(self):
        folders = []; documents = []
        for base, dirs, names in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not linked(Path(base) / d) and not d.startswith('.'))
            folder = Path(base).relative_to(self.root).as_posix()
            if folder != '.': folders.append(folder)
            for name in sorted(names):
                path = Path(base) / name
                if path.suffix.lower() != '.pdf' or linked(path): continue
                stat = path.stat(); relative = path.relative_to(self.root).as_posix()
                documents.append({'id': self.identity(relative), 'title': path.stem, 'titleZh': path.stem,
                                  'folder': '' if folder == '.' else folder, 'relative': relative,
                                  'size': stat.st_size, 'publishedAt': datetime.fromtimestamp(stat.st_mtime).date().isoformat()})
        self.index = {p['id']: p for p in documents}
        return {'root': str(self.root), 'folders': folders, 'documents': documents}

    def find(self, document_id):
        if not isinstance(document_id, str) or not re.fullmatch(r'local_[a-f0-9]{32}', document_id):
            raise PaperError('本地 PDF 编号无效。')
        # IDs convey no filesystem authority; resolve only through the selected root.
        if self.index is None: self.entries()
        item = self.index.get(document_id)
        if item is None: raise PaperError('PDF 已被移动或不存在，请刷新本地库。')
        path = self.safe(item['relative'])
        if not path.is_file(): raise PaperError('PDF 已被移动或不存在，请刷新本地库。')
        return path

    def create_folder(self, name):
        if not isinstance(name, str) or not name.strip() or len(name) > 80 or re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or name.endswith((' ', '.')) or name in ('.', '..') or re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?', name):
            raise PaperError('收藏夹名称无效，请使用普通文件夹名称。')
        target = self.safe(name)
        try: target.mkdir()
        except FileExistsError: raise PaperError('同名收藏夹已存在。')
        return self.entries()

    def remove(self, document_id, *, trash=None):
        """Recycle only the selected library file; never fall back to unlink."""
        path = self.find(document_id)
        if trash is None:
            from PyQt6.QtCore import QFile
            trash = QFile.moveToTrash
        result = trash(str(path))
        ok = result[0] if isinstance(result, tuple) else result
        if not ok: raise PaperError('无法移入回收站，PDF 保留。请关闭其他阅读窗口后重试。')
        self.index = None
        return self.entries()

    def folder(self, relative):
        target = self.safe(relative)
        if not target.is_dir(): raise PaperError('目标收藏夹不存在。')
        return target

    def delete_folder(self, relative):
        source = self.folder(relative)
        if source == self.root: raise PaperError('不能删除论文根文件夹。')
        parent = source.parent
        # Validate the complete move before changing any file. Non-PDF files and
        # nested folders belong to the user too, so preserve them all.
        children = sorted(source.iterdir(), key=lambda p: p.name)
        for base, dirs, names in os.walk(source, followlinks=False):
            for name in (*dirs, *names):
                if linked(Path(base) / name):
                    raise PaperError('收藏夹包含符号链接或联接，请先在文件管理器中整理。')
        moves = []
        for child in children:
            target = self.safe((parent / child.name).relative_to(self.root).as_posix())
            if target.exists():
                raise PaperError('上一级已有同名文件或文件夹：' + child.name + '。未移动文件，也未删除收藏夹。')
            moves.append((child, target))
        before = self.entries()
        relocated = {}
        for item in before['documents']:
            path = self.safe(item['relative'])
            if path.is_relative_to(source):
                new_relative = (parent / path.relative_to(source)).relative_to(self.root).as_posix()
                relocated[item['id']] = self.identity(new_relative)
        completed = []
        try:
            for old, new in moves:
                if new.exists(): raise PaperError('上一级出现同名项目，已停止整理。')
                os.rename(old, new)
                completed.append((old, new))
            source.rmdir()
        except OSError as exc:
            failure = exc
        except PaperError as exc:
            failure = exc
        else:
            return {**self.entries(), 'relocated': relocated}
        rollback_failed = False
        for old, new in reversed(completed):
            try:
                if old.exists(): raise OSError('Original location occupied')
                os.rename(new, old)
            except OSError: rollback_failed = True
        self.index = None
        if rollback_failed:
            raise PaperError('整理未完成，部分文件已留在上一级。文件未删除，请刷新并检查两个位置。') from failure
        raise PaperError('无法删除收藏夹，文件已保留在原位置。请检查文件占用和文件夹权限。') from failure

    def import_files(self, sources, folder, *, copy_file=shutil.copyfile):
        from pypdf import PdfReader
        target_folder = self.folder(folder); imported = []; failures = []
        for source in sources:
            source = Path(source)
            temporary = target_folder / ('.ptb-import-' + uuid.uuid4().hex + '.tmp')
            target = target_folder / source.name
            try:
                if source.suffix.lower() != '.pdf' or not source.is_file() or not 0 < source.stat().st_size <= 200_000_000:
                    raise PaperError('请选择不超过 200 MB 的 PDF。')
                if target.exists(): raise PaperError('同名文件已存在，未覆盖。')
                copy_file(source, temporary)
                source_stat = source.stat()
                os.utime(temporary, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
                with temporary.open('rb') as stream:
                    if stream.read(5) != b'%PDF-': raise PaperError('文件不是有效 PDF。')
                    stream.seek(0); doc = PdfReader(stream)
                    if doc.is_encrypted or not 1 <= len(doc.pages) <= 1000: raise PaperError('暂不支持加密 PDF 或超过 1000 页的文档。')
                # Windows rename refuses an existing destination, including on FAT/exFAT.
                # Other platforms use a no-replace hard link on this same volume.
                if os.name == 'nt': os.rename(temporary, target)
                else: os.link(temporary, target)
                relative = target.relative_to(self.root).as_posix()
                imported.append(self.identity(relative))
            except Exception as exc:
                failures.append(source.name + '：' + str(exc))
            finally: temporary.unlink(missing_ok=True)
        return {**self.entries(), 'imported': imported, 'failures': failures}

    def move(self, document_id, folder):
        source = self.find(document_id); target = self.folder(folder) / source.name
        if source == target: return {**self.entries(), 'selected': document_id}
        if target.exists(): raise PaperError('目标收藏夹已有同名 PDF，未覆盖。')
        if os.name == 'nt': os.rename(source, target)
        else:
            # Same selected root/volume. Link first; preserve source on failure.
            os.link(source, target)
            try: source.unlink()
            except OSError:
                target.unlink(); raise
        relative = target.relative_to(self.root).as_posix()
        return {**self.entries(), 'selected': self.identity(relative)}
