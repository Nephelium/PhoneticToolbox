"""Prepare complete authorized atlas pages for local reading and OCR review."""
import argparse
import hashlib
import json
from pathlib import Path
import fitz

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ocr', type=Path, required=True)
    parser.add_argument('--refresh', action='store_true', help='Replace previously generated page images with complete pages')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.ocr.mkdir(parents=True, exist_ok=True)
    entries = json.loads(args.index.read_text(encoding='utf-8'))
    with args.pdf.open('rb') as stream:
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest_path = args.output.parent/'book-pages-manifest.json'
    previous = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            if isinstance(manifest, dict) and manifest.get('sourceSha256') == source_hash:
                previous = {row['feature']: row for row in manifest['pages']}
        except (ValueError, KeyError, TypeError):
            # Unidentified caches are rebuilt; they never acquire a new source identity.
            previous = {}
    def reusable(path, expected):
        return not args.refresh and expected and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    images = []
    with fitz.open(args.pdf) as book:
        for i, entry in enumerate(entries):
            page = book[entry['pdf_page'] - 1]
            w, h = page.rect.width, page.rect.height
            stem = entry['feature_id']
            cached = previous.get(stem, {})
            if cached.get('pdfPage') != entry['pdf_page']:
                cached = {}
            title = args.ocr / f'{stem}.png'
            if not reusable(title, cached.get('ocrSha256')):
                page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=fitz.Rect(0, h*.022, w*.68, h*.065)).save(title)
            # Keep the complete page: long legends and footnotes can extend far
            # below the page header. Prefer the original embedded scan bytes.
            excerpt = args.output / f'{stem}.jpg'
            existing = fitz.Pixmap(excerpt) if reusable(excerpt, cached.get('sha256')) else None
            full_page = existing is not None and abs(existing.height/existing.width-h/w) < .01
            if args.refresh or not full_page:
                scans = page.get_images(full=True)
                scan = scans[0] if len(scans) == 1 else None
                bounds = page.get_image_rects(scan[0]) if scan else []
                complete = len(bounds) == 1 and all(abs(a-b) < 1 for a,b in zip(bounds[0],page.rect))
                raw = book.extract_image(scan[0]) if complete else None
                if raw and raw['ext'] == 'jpeg':
                    excerpt.write_bytes(raw['image'])
                else:
                    page.get_pixmap(matrix=fitz.Matrix(3, 3)).save(excerpt, jpg_quality=94)
            raw = excerpt.read_bytes()
            images.append({'feature':stem,'pdfPage':entry['pdf_page'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                           'ocrSha256':hashlib.sha256(title.read_bytes()).hexdigest()})
            if i % 50 == 0:
                print(f'{i+1}/{len(entries)}', flush=True)
    with args.pdf.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != source_hash:
            raise ValueError('Source PDF changed during preparation; rerun with a stable source')
    staged = manifest_path.with_suffix('.json.tmp')
    staged.write_text(json.dumps({'sourceSha256':source_hash,'scope':'complete original map pages, private local reading','pages':images},ensure_ascii=False,indent=2),encoding='utf-8')
    staged.replace(manifest_path)

if __name__ == '__main__':
    main()
