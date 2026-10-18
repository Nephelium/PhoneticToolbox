"""Explicit cache cleanup, retaining settings, project stores and recordings."""
import json
from pathlib import Path
from .platform_paths import user_data_root, windows_workbench_storage
from .startup_cache import plain, remove_owned, atomic_json


def result_caches():
    """Only pre-existing, recognized workspaces; never initialize a database."""
    from .local_service import LocalService
    results=[]
    for name in ('local-preview-20260927','research-v1'):
        root=plain(user_data_root()/name,user_data_root())
        if not root.exists():continue
        marker=plain(root/'workspace.json',root)
        if not marker.is_file() or json.loads(marker.read_text('utf8'))!={'kind':'ptb-local-workspace','version':1}:
            raise ValueError('本机任务缓存归属未确认，现有文件已保留。')
        database=plain(root/'jobs.sqlite3',root);files=plain(root/'files',root)
        if not database.is_file() or not files.is_dir():raise ValueError('本机任务目录不完整，已保留。')
        with LocalService(database,local_files_root=files,cache_cleanup=True) as service:
            result=service.request('/api/v1/jobs/local-storage/clear-all','POST')
        results.append(result)
    return {'complete':all(row.get('complete',False) for row in results),'workspaces':results}


def disposable_caches():
    root=user_data_root();removed=[];failures=[]
    # Never remove the WebEngine profile itself: IndexedDB and localStorage
    # hold drafts, experimental sessions and user settings.
    from .cache_location import portable_directory, update_directory
    directory=portable_directory()
    browser=directory if directory else windows_workbench_storage()
    names=('browser-cache',) if directory else ('Cache','http-cache')
    for name in names:
        try:
            target=plain(browser/name,browser)
            if target.is_dir():remove_owned(target,browser);removed.append(name)
        except (OSError,ValueError):failures.append(name)
    from .graphics_cache import NAMES, request_clear, cleanup_requested
    try:
        for name in NAMES:
            if (windows_workbench_storage()/name).exists():request_clear(name)
        graphics=cleanup_requested()
        if not graphics['complete']:failures.append('graphics')
    except (OSError,ValueError):failures.append('graphics')
    updates=update_directory()
    updates=plain(updates,directory if directory else root)
    summary=None
    if updates.is_dir():
        from .update_cache import clean_update_cache
        summary=clean_update_cache(updates,all_cache=True)
        if summary['state']!='complete' or summary['retainedDirectories']:
            failures.append('updates')
    result={'complete':not failures,'removed':removed,'retained':failures,'updates':summary}
    root.mkdir(parents=True,exist_ok=True);atomic_json(root/'cache-cleanup-report.json',result)
    return result
