"""Inventory only the six known temporary copies from public-statistics QA.

No deletion, archive extraction, server access or cleanup authorization is done.
The JSON and Markdown are for a separate reviewer, not an executable manifest.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import stat
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/validation/web-launch'
SOURCES = {
    'website-stat-source.tar.gz': '服务器匿名统计日志及计数状态的本次传输归档副本',
    'public-stats-preview/scratch/ptb-website-stats.log': '本次历史回填使用的匿名当前日志副本',
    'public-stats-preview/scratch/ptb-website-stats.log.1': '本次历史回填使用的匿名轮转日志副本',
    'public-stats-preview/scratch/ptb-website-stats.log.2.gz': '本次历史回填使用的压缩匿名轮转日志副本',
    'public-stats-preview/scratch/ptb-website-stats.log.3.gz': '本次历史回填使用的压缩匿名轮转日志副本',
    'public-stats-preview/scratch/state.json': '本次验证使用的统计计数/采样状态副本，非服务器正在使用的状态文件',
}
PRESERVE = [
    'public-stats-preview/site-stats.json',
    'public-stats-preview/site-stats-history.json',
    'stats-source-cleanup.json',
    'public-stats-ui.txt',
    'public-stats-ui-final.txt',
]


def inspect(relative):
    path = BASE / relative
    row = {'path': str(path), 'relative_path': relative}
    # Inspect each path component without following a junction or symlink.
    for parent in [*reversed(path.parents), path]:
        try:
            info = parent.lstat()
        except FileNotFoundError:
            return dict(row, status='missing')
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            return dict(row, status='blocked_reparse_point', component=str(parent))
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        return dict(row, status='blocked_not_regular_file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    stable = (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    return dict(row, status='review_candidate' if stable else 'changed_during_read',
                bytes=after.st_size, sha256=digest.hexdigest(),
                modified_at=datetime.fromtimestamp(after.st_mtime, ZoneInfo('Asia/Shanghai')).isoformat(),
                hardlink_count=after.st_nlink)


def main():
    candidates = [dict(inspect(name), purpose=purpose) for name, purpose in SOURCES.items()]
    preserved = [inspect(name) for name in PRESERVE]
    for row in preserved:
        if row['status'] == 'review_candidate':
            row['status'] = 'preserve_verified'
    scratch = BASE / 'public-stats-preview/scratch'
    expected = {Path(name).name for name in SOURCES if '/scratch/' in name}
    # No recursive discovery. Anything outside the exact allowlist stays out.
    extras = []
    safe_parents = all(not row['status'].startswith('blocked_') for row in candidates)
    if safe_parents and scratch.is_dir() and not scratch.is_symlink() and not (getattr(scratch.lstat(), 'st_file_attributes', 0) & 0x400):
        extras = [str(p) for p in scratch.iterdir() if p.name not in expected]
    total = sum(row.get('bytes', 0) for row in candidates if row['status'] == 'review_candidate')
    payload = {
        'schema': 'ptb-web-stats-copy-review/1',
        'generated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'mode': 'inventory_only', 'deletion_performed': False,
        'root': str(ROOT), 'scope': 'Six explicitly named local statistics QA copies only',
        'candidate_count': sum(row['status'] == 'review_candidate' for row in candidates),
        'logical_bytes': total, 'freed_bytes': 0,
        'candidates': candidates, 'preserve': preserved, 'unexpected_entries_preserve': extras,
        'review_requirements': [
            '本清单仅供核验，不是删除授权，也不是cleanup_workspace.ps1的可执行清单。',
            '此前清理被自动审批拒绝，返回blocked by policy且无进一步原因；不得绕过执行环境限制。',
            '删除前逐文件重查绝对路径、大小、SHA256、联接/硬链接及当前进程占用，有变化或不明确则保留。',
            '核对公开聚合JSON、验证报告及截图完整保留，确认不再需要重跑原始历史回填。',
            '仅处理明确列出的六个文件，不递归删除web-launch、public-stats-preview或整个scratch。',
            '不涉及服务器原始日志/状态、任何数据库、管理端QA副本、安装包、研究资料或运行环境。',
            '逻辑大小合计不代表磁盘实际释放量；本次释放0字节。',
        ],
    }
    output = ROOT / 'output/maintenance'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'web-stats-copy-review.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 网页统计临时副本核验清单', '', f"生成时间：{payload['generated_at']}。只盘点，未删除。", '',
             f"候选 {payload['candidate_count']} 个文件，逻辑大小 {total:,} 字节（{total / 1024:.1f} KiB）。实际释放 0 字节。", '',
             '## 核验要求', '', *['- ' + text for text in payload['review_requirements']], '', '## 逐文件清单', '']
    for row in candidates:
        lines += [f"### {row['relative_path']}", '', f"- 绝对路径：`{row['path']}`", f"- 用途：{row['purpose']}",
                  f"- 状态：{row['status']}", f"- 大小：{row.get('bytes', '未知')} 字节", f"- SHA256：`{row.get('sha256', '未知')}`", '']
    lines += ['## 必须保留', '', *[f"- `{row['path']}`" for row in preserved],
              '- 相关源码、测试脚本、文档、截图和其他验证报告。', '', '## 清单外文件', '',
              *(['- `' + path + '`' for path in extras] or ['scratch当前没有额外直属项；此项不是递归检查或删除许可。']), '']
    (output / 'web-stats-copy-review.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'candidate_count': payload['candidate_count'], 'logical_bytes': total, 'deleted': False,
                      'json': str(output / 'web-stats-copy-review.json'), 'markdown': str(output / 'web-stats-copy-review.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
