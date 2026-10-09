from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from core import ManualError, validate


def main() -> int:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='校验说明书工程、引用与素材哈希')
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--distribution', choices=('software', 'public'), default='software')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.project, args.distribution, args.strict)
        summary = {k: result[k] for k in ('warnings', 'errors', 'skippedAssets', 'mediaBytes')}
        summary['chapters'] = len(result['chapters'])
        summary['assets'] = len(result['project']['assets']) - len(result['skippedAssets'])
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if result['errors'] else 0
    except ManualError as exc:
        print(f'校验失败: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
