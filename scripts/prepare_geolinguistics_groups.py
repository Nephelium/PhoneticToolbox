"""Apply explicit, source-bound broad groups; retain raw observations and fine labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def apply_reviews(catalog, reviews, manifest):
    result = json.loads(json.dumps(catalog, ensure_ascii=False))
    features = {f['id']: f for f in result['features']}
    seen, membership, volumes, coverage = set(), set(), set(), {}
    for feature in features.values():
        feature['groups'] = []
        feature['groupExceptions'] = {}
        feature.pop('groupReview', None)
        feature.pop('groupNote', None)
    for review in reviews:
        if review.get('schema') != 'ptb-atlas-broad-review/1' or review.get('bookSha256') != manifest['sourceSha256']:
            raise ValueError('Broad review schema or PDF identity mismatch')
        volume = review['volume']
        if volume not in ('P', 'L', 'S') or volume in volumes:
            raise ValueError('Repeated or unknown volume')
        volumes.add(volume)
        expected = {f['id'] for f in features.values() if f['volume'] == volume}
        rows = review['coverage']
        if len(rows) != len(expected) or {r['featureId'] for r in rows} != expected:
            raise ValueError('Each volume requires a complete feature coverage inventory')
        for row in rows:
            feature = features[row['featureId']]
            if row['pdfPage'] != feature['page']:
                raise ValueError('Coverage page mismatch')
            if row.get('sourceNote'):
                if not isinstance(row['sourceNote'], str):
                    raise ValueError('Invalid source note')
                feature['groupNote'] = row['sourceNote']
            coverage[row['featureId']] = row
        for entry in review['entries']:
            fid, group_id = entry['featureId'], entry['groupId']
            if fid not in expected or not isinstance(group_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', group_id):
                raise ValueError('Invalid broad group identity')
            feature = features[fid]
            if (fid, group_id) in seen or entry['pdfPage'] != feature['page']:
                raise ValueError('Repeated broad group or page mismatch')
            seen.add((fid, group_id))
            codes = entry['codes']
            if not codes or len(set(codes)) != len(codes) or any(code not in feature['categories'] for code in codes):
                raise ValueError('Unknown or repeated category in broad group')
            if entry['status'] != 'accepted':
                continue
            if entry['textStatus'] != 'verified' or entry['mappingStatus'] != 'verified':
                raise ValueError('Accepted broad group lacks complete evidence')
            label, evidence = entry['label'], entry['evidence']
            if not isinstance(label, str) or not label.strip() or '\ufffd' in label or not isinstance(evidence, str) or not evidence.strip():
                raise ValueError('Invalid label or evidence')
            if entry.get('labelKind') not in ('source', 'summary'):
                raise ValueError('Broad label must distinguish source transcription from summary')
            for code in codes:
                if (fid, code) in membership:
                    raise ValueError('A fine category belongs to more than one broad group')
                membership.add((fid, code))
            # Text and code membership are transcribed as reviewed, never normalized or expanded by prefix.
            feature['groups'].append({'id': group_id, 'label': label, 'codes': codes,
                                      'labelKind': entry['labelKind'], 'page': entry['pdfPage']})
        for exception in review.get('exceptions', []):
            fid, site = exception['featureId'], exception['siteId']
            if fid not in expected or not re.fullmatch(r'S(?:[1-9]\d{0,2})', site) or int(site[1:]) > 930:
                raise ValueError('Invalid broad mapping exception')
            if exception['pdfPage'] != features[fid]['page'] or not exception.get('evidence') or not exception.get('reason'):
                raise ValueError('Exception requires a page and evidence')
            if exception.get('code') not in features[fid]['categories']:
                raise ValueError('Exception requires an original category')
            previous = features[fid]['groupExceptions'].get(site)
            items = previous if isinstance(previous, list) else [previous] if previous else []
            if any(item['code'] == exception['code'] for item in items):
                raise ValueError('Repeated site category exception')
            items = items + [{'code': exception['code'], 'reason': exception['reason']}]
            features[fid]['groupExceptions'][site] = items[0] if len(items) == 1 else items
    if volumes != {'P', 'L', 'S'}:
        raise ValueError('Supply all three volume reviews to avoid losing existing mappings')
    for fid, feature in features.items():
        count = sum(len(group['codes']) for group in feature['groups'])
        feature['groupReview'] = {'verifiedCodes': count, 'totalCodes': len(feature['categories']),
            'verifiedGroups': len(feature['groups']), 'totalGroups': coverage[fid]['totalGroups'],
            'complete': count == len(feature['categories']) and count > 0}
    result['groupReview'] = {
        'verifiedGroups': sum(len(f['groups']) for f in features.values()),
        'verifiedCodes': len(membership), 'totalCodes': sum(len(f['categories']) for f in features.values()),
        'completeFeatures': sum(f['groupReview']['complete'] for f in features.values()),
        'partialFeatures': sum(bool(f['groups']) and not f['groupReview']['complete'] for f in features.values()),
        'featuresWithGroups': sum(bool(f['groups']) for f in features.values()),
        'inventoriedFeatures': len(coverage),
        'exceptions': sum(len(f['groupExceptions']) for f in features.values()),
        'exceptionAtoms': sum(len(items) if isinstance(items, list) else 1
                              for f in features.values() for items in f['groupExceptions'].values()),
    }
    return result


def apply_files(catalog, paths, manifest):
    raw = [path.read_bytes() for path in paths]
    result = apply_reviews(catalog, [json.loads(data) for data in raw], manifest)
    result['groupReview']['reviewSha256'] = [hashlib.sha256(data).hexdigest() for data in raw]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, nargs=3, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    args = parser.parse_args()
    target = args.resources/'catalog.json'
    result = apply_files(json.loads(target.read_text('utf-8')), args.review,
                         json.loads((args.resources/'book-pages-manifest.json').read_text('utf-8')))
    staged = target.with_suffix('.json.tmp')
    staged.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    staged.replace(target)
    from package_geolinguistics_data import write_data_manifest
    write_data_manifest(args.resources)
    print(json.dumps(result['groupReview'], ensure_ascii=False))


if __name__ == '__main__':
    main()
