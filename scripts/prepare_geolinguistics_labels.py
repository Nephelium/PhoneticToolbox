"""Apply a reviewed, source-bound category dictionary without changing observations.

The private review JSON is an authoring input, not OCR output. Transcription and
code-to-symbol correspondence must both be verified before a label is published.
Pending entries remain in the review input and cannot become display labels.
"""
import argparse
import hashlib
import json
from pathlib import Path


def apply_review(catalog, review, book_manifest):
    if review.get('schema') != 'ptb-atlas-category-review/1':
        raise ValueError('Unknown category review schema')
    if review.get('bookSha256') != book_manifest['sourceSha256']:
        raise ValueError('Category review belongs to another original PDF')
    features = {f['id']: f for f in catalog['features']}
    verified, seen, pending = {}, set(), 0
    for entry in review['entries']:
        fid, code = entry['feature'], entry['code']
        key = fid, code
        if key in seen:
            raise ValueError('Duplicate category: '+str(key))
        seen.add(key)
        if fid not in features or code not in features[fid]['categories']:
            raise ValueError('Unknown feature/category: '+str(key))
        if entry['page'] != features[fid]['page']:
            raise ValueError('Category source page mismatch: '+str(key))
        text_status, mapping_status = entry['textStatus'], entry['mappingStatus']
        if text_status not in ('verified', 'pending') or mapping_status not in ('verified', 'pending'):
            raise ValueError('Unknown review state: '+str(key))
        if text_status != 'verified' or mapping_status != 'verified':
            pending += 1
            continue
        label = entry['label']
        if not isinstance(label, str) or not label.strip() or '\ufffd' in label:
            raise ValueError('Invalid reviewed label: '+str(key))
        # No Unicode normalization: combining marks and IPA distinctions survive.
        for field in ('textEvidence', 'mappingEvidence'):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError('Missing review evidence: '+str(key))
        verified[key] = entry
    result = json.loads(json.dumps(catalog, ensure_ascii=False))
    for fid, note in review.get('featureNotes', {}).items():
        if fid not in features or not isinstance(note, str) or '\ufffd' in note:
            raise ValueError('Invalid feature note: '+fid)
    for feature in result['features']:
        feature.pop('categoryNote', None)
        if note := review.get('featureNotes', {}).get(feature['id']):
            feature['categoryNote'] = note
        entries = {code: verified[(feature['id'], code)] for code in feature['categories']
                   if (feature['id'], code) in verified}
        # Replace, not update: a review withdrawn to pending cannot leave a stale
        # formerly published label in the generated catalog.
        feature['labels'] = {code: entry['label'] for code, entry in entries.items()}
        feature['categoryReview'] = {
            'verified': len(entries), 'total': len(feature['categories']),
            'evidence': {code: {k: entry[k] for k in ('page', 'textEvidence', 'mappingEvidence')}
                         for code, entry in entries.items()},
        }
    result['categoryReview'] = {
        'verified': len(verified), 'pendingEntries': pending,
        'total': sum(len(f['categories']) for f in result['features']),
        'completeFeatures': sum(len(f['labels']) == len(f['categories']) for f in result['features']),
    }
    return result


def apply_file(catalog, review_path, manifest):
    raw = review_path.read_bytes()
    result = apply_review(catalog, json.loads(raw), manifest)
    result['categoryReview']['reviewSha256'] = hashlib.sha256(raw).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True, type=Path)
    parser.add_argument('--resources', required=True, type=Path)
    args = parser.parse_args()
    def read(name):
        return json.loads((args.resources/name).read_text(encoding='utf-8'))
    result = apply_file(read('catalog.json'), args.review, read('book-pages-manifest.json'))
    target = args.resources/'catalog.json'
    staged = target.with_suffix('.json.tmp')
    staged.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    staged.replace(target)
    from package_geolinguistics_data import write_data_manifest
    write_data_manifest(args.resources)
    print(json.dumps(result['categoryReview'], ensure_ascii=False))


if __name__ == '__main__':
    main()
