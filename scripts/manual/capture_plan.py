"""Inventory only the screenshots required by the current manual and their recipes.

This command does not start apps, take screenshots, or modify author media.
Recipe matches are source-code hints and require locale-aware verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__:
    from .build import source_fingerprint, write_atomic
    from .core import dump, read_json, safe_file
else:
    from build import source_fingerprint, write_atomic
    from core import dump, read_json, safe_file

ROOT = Path(__file__).resolve().parents[2]
RECIPES = {
    'web-introduction': ['scripts/verify_web_manual.py'],
    'getting-started': ['scripts/manual/capture_settings.py', 'scripts/manual/capture_qt.py'],
    'settings': ['scripts/manual/capture_settings.py'],
    'm01': ['scripts/manual/capture_qt.py'],
    'm02': ['scripts/manual/capture_m02_current.py', 'scripts/manual/capture_m02_states.py'],
    'm03': ['scripts/manual/capture_m03_r8.py', 'scripts/manual/capture_m03.py'],
    'm04': ['scripts/manual/capture_qt.py'],
    'm05': ['scripts/manual/capture_m05_current.py'],
    'm06': ['scripts/manual/capture_qt.py'],
    'm07': ['scripts/manual/capture_m07_chapter.py'],
    'm08': ['scripts/manual/capture_m08_states.py'],
    'm09': ['scripts/manual/capture_qt.py'],
    'm10': ['scripts/verify_m10_r19_qt.py', 'scripts/verify_m10_qt.py'],
    'm11': ['scripts/manual/capture_m11_states.py'],
    'm12': ['scripts/manual/capture_m12_current.py'],
    'm13': ['scripts/manual/capture_m13_result_layout.py'],
    'm14': ['scripts/manual/capture_m14_r2.py', 'scripts/manual/capture_m14_test_table.py', 'scripts/manual/capture_m14_states.py'],
    'm15': ['scripts/manual/capture_qt.py'],
    'm16': ['scripts/manual/capture_m16_update.py', 'scripts/manual/capture_m16_states.py'],
    'm17': ['scripts/manual/capture_m17_states.py'],
    'm18': ['scripts/verify_m18_r3_qt.py', 'scripts/verify_m18_r4_qt.py'],
    'm19': ['scripts/manual/capture_m19_current.py'],
}


def create_plan(project_dir: Path, root: Path = ROOT) -> dict:
    project = read_json(safe_file(project_dir, 'project.json'))
    review_file = safe_file(project_dir, 'locales/scientific-figures.json')
    figure_reviews = read_json(review_file).get('figures', {}) if review_file.exists() else {}
    assets = {a['id']: a for a in project['assets']}
    uses: dict[str, list[dict]] = {}
    for descriptor in project['chapters']:
        chapter = read_json(safe_file(project_dir, descriptor['path']))

        def visit(node: dict, heading: str = ''):
            attrs = node.get('attrs', {})
            if attrs.get('assetId'):
                uses.setdefault(attrs['assetId'], []).append({'chapterId': chapter['id'], 'figureId': attrs.get('id'),
                                                              'headingId': heading or None, 'caption': attrs.get('caption'), 'alt': attrs.get('alt')})
            for child in node.get('content', []):
                if child.get('type') == 'heading':
                    heading = child.get('attrs', {}).get('id', heading)
                visit(child, heading)
        visit(chapter['body'])
    source_files = sorted(set((root / 'scripts/manual').glob('capture_*.py')) | {root / file for files in RECIPES.values() for file in files})
    code = {file.relative_to(root).as_posix(): file.read_text('utf-8') for file in source_files if file.is_file()}
    screenshots, scientific, shared = [], [], []
    capture_states: dict[str, dict] = {}
    for identity in sorted(uses):
        asset = assets[identity]
        path = safe_file(project_dir, asset['path'])
        row = {'assetId': identity, 'sourcePath': asset['path'], 'sourceSha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'sourceBytes': path.stat().st_size, 'distribution': asset['distribution'], 'git': asset.get('git', True),
               'uses': uses[identity]}
        source_type = asset.get('sourceType', '')
        if asset['kind'] != 'image':
            shared.append({**row, 'kind': asset['kind'], 'action': 'reuse-base-media', 'translateMedia': False})
            continue
        if not any(word in source_type for word in ('截图', '屏幕')):
            review = figure_reviews.get(identity, {})
            reviewed = review.get('sourceSha256') == row['sourceSha256'] and review.get('reviewed') is True
            localize = reviewed and review.get('decision') == 'localize-generated-labels'
            scientific.append({**row, 'sourceType': source_type, 'action': 'reexport-localized-generated-labels' if localize else 'reuse-reviewed-scientific-figure' if reviewed else 'review-scientific-labels-preserve-data',
                               'visualReviewRequired': not reviewed, 'review': review if reviewed else None})
            continue
        capture = asset.get('capture', {})
        derivative = '局部' in source_type
        parent_candidates = [a['id'] for a in assets.values() if a['id'] != identity and a.get('sha256') and a['sha256'] == asset.get('originalSha256') and a.get('width', 0) >= 2000 and a.get('height', 0) >= 1200]
        inferred_parent = parent_candidates[0] if len(parent_candidates) == 1 else None
        base_id = asset.get('originalAssetId') or inferred_parent or identity
        base = assets.get(base_id)
        evidence = []
        tokens = [identity, capture.get('originalId'), capture.get('name')]
        for file, text in code.items():
            exact = [token for token in tokens if token and len(token) > 7 and token in text]
            if exact:
                evidence.append({'path': file, 'match': 'literal-capture-name', 'tokens': exact})
        if not evidence:
            for chapter in sorted({use['chapterId'] for use in uses[identity]}):
                for file in RECIPES.get(chapter, []):
                    if file in code and not any(r['path'] == file for r in evidence):
                        evidence.append({'path': file, 'match': 'chapter-recipe-candidate'})
        prefix = 'assets/public' if asset['distribution'] == 'public' else 'assets/software-only'
        row.update({'sourceType': source_type, 'classification': 'metadata-only-needs-visual-review',
                    'action': 'derive-from-localized-full-capture' if derivative else 'recapture-localized-interface',
                    'baseCaptureAssetId': base_id, 'baseCapturePath': base.get('path') if base else None,
                    'parentMappingRequired': derivative and not (asset.get('originalAssetId') or inferred_parent),
                    'parentMappingEvidence': 'declared-originalAssetId' if asset.get('originalAssetId') else 'unique-originalSha256-match' if inferred_parent else None,
                    'existingSize': [asset.get('width'), asset.get('height')],
                    'existingCrop': asset.get('crop') or capture.get('crop'), 'existingAnnotations': asset.get('annotations'),
                    'priorCaptureName': capture.get('name') or capture.get('originalId'), 'priorReport': capture.get('report'),
                    'recipeCandidates': evidence, 'recipesRequireLocaleAdaptation': True,
                    'targets': {locale: f'{prefix}/localized/{locale}/{identity}.png' for locale in ('en', 'ja')}})
        if identity in ('m11-task-results-light-scaled-r20261007', 'm11-saved-results-light-scaled-r20261007'):
            row['sourceCorrection'] = {'topic': 'MFA output filenames', 'required': 'Use the input WAV basename; no leading sequence number; retain unique suffixes only for collisions',
                                       'oldVisibleValue': '001_probe.TextGrid', 'replacementExample': 'probe.TextGrid',
                                       'status': 'corrected' if capture.get('filenameCorrection') == 'input-basename-no-sequence-prefix' else 'pending',
                                       'currentCaptureReport': capture.get('report'),
                                       'evidence': 'output/validation/manual-i18n-modules-b/m11-source-correction.json'}
        screenshots.append(row)
        state = capture_states.setdefault(base_id, {'baseCaptureAssetId': base_id, 'baseCapturePath': base.get('path') if base else None,
                                                    'referencedInBody': base_id in uses, 'outputAssetIds': [], 'chapterIds': []})
        state['outputAssetIds'].append(identity)
        state['chapterIds'] = sorted(set(state['chapterIds']) | {u['chapterId'] for u in uses[identity]})
    return {'schemaVersion': 'ptb-manual-localization-capture-plan/1', 'sourceDigest': source_fingerprint(project_dir),
            'scope': 'current-body-referenced-assets-only', 'locales': ['en', 'ja'],
            'captureRequirements': {'fullImage': [2560, 1600], 'dpr': 1.5, 'pageZoom': 1, 'theme': 'light', 'maximized': True,
                                    'moduleNavigation': 'collapsed-unless-teaching-navigation',
                                    'roundingCropOnly': '2561x1601 may crop one right/bottom pixel; no resizing',
                                    'derivedAnnotations': 'reposition on localized capture; old coordinates are guidance only',
                                    'dataAndPaperText': 'preserve user content, scientific values, original paper text and data labels'},
            'summary': {'screenshots': len(screenshots), 'screenshotBytes': sum(r['sourceBytes'] for r in screenshots),
                        'derivedImages': sum(r['action'].startswith('derive') for r in screenshots),
                        'baseCaptureStates': len(capture_states), 'baseStatesNotDirectlyReferenced': sum(not r['referencedInBody'] for r in capture_states.values()),
                        'missingParentMappings': sum(r['parentMappingRequired'] for r in screenshots),
                        'noRecipeCandidate': sum(not r['recipeCandidates'] for r in screenshots),
                        'scientificFigures': len(scientific), 'scientificFiguresForReview': sum(r['visualReviewRequired'] for r in scientific),
                        'scientificLocalizedExports': sum(r['action'] == 'reexport-localized-generated-labels' for r in scientific), 'sharedAudioVideo': len(shared)},
            'screenshots': screenshots, 'captureStates': list(capture_states.values()),
            'scientificFigures': scientific, 'sharedMedia': shared}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT / 'manual')
    parser.add_argument('--output', type=Path, default=ROOT / 'manual/locales/capture-plan.json')
    args = parser.parse_args()
    plan = create_plan(args.project)
    write_atomic(args.output, dump(plan))
    print(json.dumps({'output': str(args.output), **plan['summary']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
