"""Verify generated reader files against their author project, without rebuilding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .build import reader_chapter
from .core import ManualError, dump, reading_project, relative, validate


def safe_path(root: Path, value: str) -> Path:
    relative(value)
    current = root
    for part in value.split('/'):
        current = current / part
        try:
            stat = current.lstat()
        except FileNotFoundError:
            continue
        if current.is_symlink() or getattr(stat, 'st_file_attributes', 0) & 0x400:
            raise ManualError('linked resource path: ' + value)
    return current


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tree_files(root: Path) -> set[str]:
    """Do not traverse junctions while inventorying an output tree."""
    files, pending = set(), [root]
    while pending:
        for child in pending.pop().iterdir():
            relative_path = child.relative_to(root).as_posix()
            safe_path(root, relative_path)
            if child.is_dir():
                pending.append(child)
            elif child.is_file():
                files.add(relative_path)
    return files


def expected_reader(source: Path, distribution: str) -> tuple[dict, dict, dict]:
    index = json.loads(safe_path(source, 'project.json').read_text(encoding='utf-8'))
    for item in index['chapters'] + index['assets']:
        safe_path(source, item['path'])
    result = validate(source, distribution, strict=True)
    if result['errors']:
        raise ManualError('invalid author project: ' + '; '.join(result['errors']))
    project = reading_project(result, distribution)
    hashes = {}
    for descriptor, chapter in zip(project['chapters'], result['chapters']):
        hashes[descriptor['path']] = hashlib.sha256(dump(reader_chapter(chapter))).hexdigest()
    for asset in project['assets']:
        # Hash the author input even when its optional declared digest is absent.
        digest = file_hash(safe_path(source, asset['path']))
        prior = hashes.get(asset['path'])
        if prior is not None and prior != digest:
            raise ManualError('conflicting generated asset identity: ' + asset['path'])
        hashes[asset['path']] = digest
    report = dict(schemaVersion='ptb-manual-build/1', distribution=distribution,
                  chapters=len(project['chapters']), assets=len(project['assets']),
                  mediaBytes=result['mediaBytes'], skippedAssets=result['skippedAssets'],
                  warnings=result['warnings'], hashes=hashes)
    return project, hashes, report


def check_generated_manual(root: Path, declaration: dict) -> tuple[list[str], set[str]]:
    """An absent generated tree is allowed in a clean checkout; partial/stale trees fail."""
    errors, registered = [], set()
    label = declaration.get('path', '<missing>')
    try:
        output = safe_path(root, label)
        source = safe_path(root, declaration['source_project'])
        distribution = declaration['distribution']
        if distribution not in ('software', 'public'):
            raise ManualError('invalid generated distribution')
        if output == source or source in output.parents or output in source.parents:
            raise ManualError('source and generated tree must be separate')
        if not output.exists():
            return [], set()
        actual = tree_files(output)
        # This validator owns this exact tree. Extras still fail below, not ignored.
        registered = {label + '/' + name for name in actual}
        project, hashes, expected_report = expected_reader(source, distribution)
        expected = set(hashes) | {'project.json', 'build-report.json'}
        extras = sorted(actual - expected)
        if extras:
            errors.append(f'generated manual undeclared files: {label}: {len(extras)}; ' + ', '.join(extras[:8]))
        for name in sorted(expected - actual):
            errors.append('generated manual missing file: ' + label + '/' + name)
        for name, digest in hashes.items():
            if name in actual and file_hash(safe_path(output, name)) != digest:
                errors.append('generated manual hash mismatch: ' + label + '/' + name)
        if 'project.json' in actual:
            generated = json.loads(safe_path(output, 'project.json').read_text(encoding='utf-8'))
            if generated != project:
                errors.append('generated manual project drift: ' + label + '/project.json')
        if 'build-report.json' in actual:
            report = json.loads(safe_path(output, 'build-report.json').read_text(encoding='utf-8'))
            if report != expected_report:
                errors.append('generated manual build report drift: ' + label + '/build-report.json')
    except (ManualError, OSError, ValueError, KeyError, TypeError) as exc:
        errors.append('generated manual invalid: ' + str(label) + ': ' + str(exc))
    return errors, registered
