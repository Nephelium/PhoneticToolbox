"""Read-only workspace byte inventory. Never follow links or delete files."""
import argparse
from collections import defaultdict
from datetime import datetime
import heapq
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if not out.is_relative_to(root):
        raise ValueError('Inventory output must stay within the workspace')
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode('utf-8').split('\0')
    groups = defaultdict(lambda: {'bytes': 0, 'files': 0, 'tracked_files': 0})
    for name in filter(None, tracked):
        parts = name.split('/')
        for depth in range(1, min(len(parts), 3) + 1):
            groups['/'.join(parts[:depth])]['tracked_files'] += 1
    largest, errors, skipped = [], [], []
    pending = [root]
    count = total = 0
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative = Path(entry.path).relative_to(root).as_posix()
                    try:
                        stat = entry.stat(follow_symlinks=False)
                        if getattr(stat, 'st_file_attributes', 0) & 0x400 or entry.is_symlink():
                            skipped.append(relative)
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            size = stat.st_size
                            count += 1
                            total += size
                            parts = relative.split('/')
                            for depth in range(1, min(len(parts), 3) + 1):
                                item = groups['/'.join(parts[:depth])]
                                item['bytes'] += size
                                item['files'] += 1
                            row = (size, relative)
                            if len(largest) < 80:
                                heapq.heappush(largest, row)
                            elif row > largest[0]:
                                heapq.heapreplace(largest, row)
                    except OSError as exc:
                        errors.append({'path': relative, 'error': str(exc)})
        except OSError as exc:
            errors.append({'path': str(directory.relative_to(root)), 'error': str(exc)})
    report = {'root': str(root), 'created_local': datetime.now().isoformat(),
              'scope': 'Logical bytes within v3 only; no link traversal, deletion or sibling V2 scan',
              'bytes': total, 'files': count, 'errors': errors, 'skipped_links': skipped,
              'groups': dict(sorted(groups.items(), key=lambda x: (-x[1]['bytes'], x[0]))),
              'largest_files': [{'bytes': size, 'path': name} for size, name in sorted(largest, reverse=True)]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('bytes', 'files', 'errors', 'skipped_links')}, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report['groups'].items() if '/' not in k}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
