"""One request per bounded process. No user-supplied commands or server paths."""
import json
import os
from pathlib import Path
import shutil
import sys
import uuid


def execute(root, operation, args):
    from .papers import PaperService, PaperError, atomic_json
    service = PaperService(root / 'cache')
    library_root = root / 'libraries'
    library_root.mkdir(exist_ok=True)
    progress_path = root / 'progress.json'
    def progress(value): atomic_json(progress_path, value)
    try:
        if operation == 'status':
            if not service.preferences().get('libraryRoot'):
                target = library_root / '我的 PDF'; target.mkdir(exist_ok=True)
                service.set_library_root(target)
            return service.status()
        if operation == 'refresh': return service.refresh(_Cancel())
        if operation == 'download': return service.download(args['since'], _Cancel(), progress)
        if operation == 'downloadSelected': return service.download('0001-01-01', _Cancel(), progress, args['ids'])
        if operation == 'autoReceive': return service.set_auto_receive(args['enabled'])
        if operation == 'interval': return service.set_interval(args['hours'])
        if operation == 'createRecommendationFolder': return service.recommendation_folder(args['name'])
        if operation == 'deleteRecommendationFolder': return service.delete_recommendation_folder(args['folder'])
        if operation == 'collectRecommendation': return service.collect_recommendation(args['id'], args['folder'])
        if operation == 'removeDownload': return service.remove_download(args['id'])
        if operation == 'library': return service.library_status()
        if operation == 'chooseLibrary':
            target = library_root / args['library']
            target.mkdir(exist_ok=True)
            result = service.set_library_root(target)
            for item in args['files']:
                relative = item['relative']
                from .paper_library import PersonalPaperLibrary
                library = PersonalPaperLibrary(target)
                destination = library.safe(str(Path(relative).parent))
                destination.mkdir(parents=True, exist_ok=True)
                folder = Path(relative).parent.as_posix()
                result = library.import_files([root / 'imports' / item['name']], '' if folder == '.' else folder, copy_file=os.link)
                if result['failures']: raise PaperError('\n'.join(result['failures']))
            return service.library_status()
        if operation == 'importLocal': return service.library().import_files([root / 'imports' / name for name in args['names']], args['folder'], copy_file=os.link)
        if operation == 'createFolder': return service.library().create_folder(args['name'])
        if operation == 'deleteFolder': return service.library().delete_folder(args['folder'])
        if operation == 'moveLocal': return service.library().move(args['id'], args['folder'])
        if operation == 'removeLocal':
            library = service.library()
            path = library.find(args['id'])
            key = uuid.uuid4().hex
            destination = root / 'trash' / (key + '.pdf')
            destination.parent.mkdir(exist_ok=True)
            relative = path.relative_to(root).as_posix()
            result = library.remove(args['id'], trash=lambda source: (shutil.move(source, destination), True)[1])
            atomic_json(root / 'trash' / (key + '.json'), {'relative': relative, 'title': path.stem})
            return result
        if operation == 'trash':
            return [{'id': p.stem, 'title': json.loads(p.read_text('utf8'))['title']} for p in sorted((root / 'trash').glob('*.json'))] if (root / 'trash').exists() else []
        if operation == 'restoreLocal':
            key = args['id']
            metadata = root / 'trash' / (key + '.json')
            item = json.loads(metadata.read_text('utf8'))
            from .paper_library import PersonalPaperLibrary
            target = root / item['relative']
            if not target.resolve().is_relative_to(library_root.resolve()) or target.exists():
                raise PaperError('原位置已有同名文件，未覆盖。请先整理后重试。')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(root / 'trash' / (key + '.pdf'), target)
            metadata.unlink()
            return service.library_status()
        if operation == 'render': return service.render(args['id'], args['language'], args['page'], args['width'])
        if operation == 'annotations': return service.annotations(args['id'], args['language'])
        if operation == 'saveAnnotation': return service.save_annotation(args['id'], args['language'], args['revision'], args['item'], args['remove'])
        if operation == 'export':
            return service.export(args['id'], args['language'], args['annotated'], root / 'export.pdf')
        raise PaperError('论文操作无效。')
    finally: service.close_documents()


class _Cancel:
    def is_set(self): return False  # Parent owns cancellation and kills its exact process group.


def main():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6.QtGui import QGuiApplication
    app = QGuiApplication([])
    request = Path(sys.argv[1]).resolve()
    payload = json.loads(request.read_text('utf8'))
    from .papers import PaperError
    try: result = {'ok': True, 'value': execute(request.parent, payload['operation'], payload['args'])}
    except (PaperError, ValueError, KeyError, OSError) as exc:
        result = {'ok': False, 'error': str(exc) if isinstance(exc, PaperError) else '论文操作未完成，请刷新目录后重试。'}
    (request.parent / 'response.json').write_text(json.dumps(result, ensure_ascii=False), encoding='utf8')
    del app


if __name__ == '__main__': main()
