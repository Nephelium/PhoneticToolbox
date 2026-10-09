"""Bounded local inventory: directory totals and largest files, never a full path dump."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import heapq
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inventory(root):
    totals = defaultdict(lambda: dict(bytes=0, files=0))
    largest, links, errors = [], [], []
    pending = [root]
    while pending:
        folder = pending.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    relative = path.relative_to(root).as_posix()
                    try:
                        stat = entry.stat(follow_symlinks=False)
                        if entry.is_symlink() or getattr(stat, 'st_file_attributes', 0) & 0x400:
                            links.append(relative)
                        elif entry.is_dir(follow_symlinks=False):
                            pending.append(path)
                        elif entry.is_file(follow_symlinks=False):
                            parts = Path(relative).parts[:-1]
                            keys = ['/'.join(parts[:i]) for i in range(1, min(3, len(parts)) + 1)] or ['[root files]']
                            for key in keys:
                                totals[key]['bytes'] += stat.st_size
                                totals[key]['files'] += 1
                            heapq.heappush(largest, (stat.st_size, relative))
                            if len(largest) > 80:
                                heapq.heappop(largest)
                    except OSError as exc:
                        errors.append(dict(path=relative, error=type(exc).__name__))
        except OSError as exc:
            errors.append(dict(path=str(folder), error=type(exc).__name__))
    return dict(schema='ptb-workspace-inventory/1', root=str(root),
                created_at=datetime.now(timezone.utc).isoformat(),
                logical_bytes_note='Sum of file lengths; hard links can be counted more than once; links are not followed',
                directories=dict(sorted(totals.items())),
                largest_files=[dict(path=path, bytes=size) for size, path in sorted(largest, reverse=True)],
                reparse_points_skipped=links, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'output/maintenance/inventory.json')
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.is_relative_to(ROOT / 'output/maintenance'):
        raise ValueError('Inventory belongs in output/maintenance')
    result = inventory(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    top = {k: v for k, v in result['directories'].items() if '/' not in k}
    print(json.dumps(dict(logical_bytes=sum(v['bytes'] for v in top.values()), top_level=top,
                          errors=len(result['errors']), report=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
