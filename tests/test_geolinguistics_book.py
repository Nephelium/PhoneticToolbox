"""Real synthetic PDF regression for source-bound atlas and OCR caches."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import fitz


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/prepare_geolinguistics_book.py'


class BookIdentity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ptb-book-identity-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pdf = self.root/'source.pdf'
        self.index = self.root/'index.json'
        self.index.write_text(json.dumps([{'feature_id':'F1_P','pdf_page':1}]),encoding='utf-8')

    def source(self, texts):
        doc = fitz.open()
        for text in texts:
            page = doc.new_page(width=300,height=420)
            page.insert_text((10,22),text,fontsize=12)
            page.insert_text((50,200),text,fontsize=30)
        self.pdf.write_bytes(doc.tobytes());doc.close()

    def run_builder(self, *extra):
        subprocess.run([sys.executable,str(SCRIPT),'--pdf',str(self.pdf),'--index',str(self.index),
                        '--output',str(self.root/'book'),'--ocr',str(self.root/'ocr'),*extra],check=True,capture_output=True)
        manifest=json.loads((self.root/'book-pages-manifest.json').read_text('utf-8'))
        self.assertEqual(manifest['sourceSha256'],hashlib.sha256(self.pdf.read_bytes()).hexdigest())
        return [(self.root/name).read_bytes() for name in ['book/F1_P.jpg','ocr/F1_P.png']]

    def test_new_pdf_rebuilds_both_caches_even_with_identical_page_dimensions(self):
        self.source(['Original A']);before=self.run_builder()
        self.source(['Replacement B']);after=self.run_builder()
        for old,new in zip(before,after):self.assertNotEqual(old,new)

    def test_changed_page_mapping_rebuilds_both_caches(self):
        self.source(['First page','Second page']);before=self.run_builder()
        self.index.write_text(json.dumps([{'feature_id':'F1_P','pdf_page':2}]),encoding='utf-8')
        after=self.run_builder()
        for old,new in zip(before,after):self.assertNotEqual(old,new)

    def test_unchanged_source_reuses_verified_bytes_and_repairs_modified_cache(self):
        self.source(['Stable source']);expected=self.run_builder()
        image=self.root/'book/F1_P.jpg';ocr=self.root/'ocr/F1_P.png'
        times=(image.stat().st_mtime_ns,ocr.stat().st_mtime_ns)
        self.assertEqual(expected,self.run_builder())
        self.assertEqual(times,(image.stat().st_mtime_ns,ocr.stat().st_mtime_ns))
        image.write_bytes(b'broken');ocr.write_bytes(b'broken')
        self.assertEqual(expected,self.run_builder())

    def test_unidentified_existing_caches_are_rebuilt(self):
        self.source(['Source A']);self.run_builder()
        (self.root/'book-pages-manifest.json').write_text('{}',encoding='utf-8')
        self.source(['Source B']);actual=self.run_builder()
        self.assertEqual(actual,self.run_builder('--refresh'))


if __name__=='__main__':unittest.main()
