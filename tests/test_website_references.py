"""Focused bibliography refresh boundaries; no application or website server."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('website_expansion',ROOT/'scripts/build_website_expansion.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

ACADEMIC=b'<details class="reference-catalog" data-reference-group="academic"><summary>Old academic</summary><details class="reference-adaptation"><summary>Nested</summary><p>Keep structure</p></details></details>'
SOFTWARE=b'<details class="reference-catalog" data-reference-group="software"><summary>Old software</summary></details>'
NEW_ACADEMIC=b'<details class="reference-catalog" data-reference-group="academic"><summary>New academic</summary></details>'
NEW_SOFTWARE=b'<details class="reference-catalog" data-reference-group="software"><summary>New software</summary></details>'
PREFIX=b'\xef\xbb\xbf<!doctype html>\r\n<script src="site.js?v=unchanged-hash"></script>\r\n<details data-class="reference-catalog"><summary>Other author</summary></details>\r\n<section>'
GAP='<p>其他作者目录之间的内容。</p>\r\n'.encode('utf-8')
SUFFIX='<p>其他作者目录之后的内容。</p>\r\n</section>\n<!-- mixed newline stays -->'.encode('utf-8')

class ReferenceRefreshTests(unittest.TestCase):
    def test_nested_details_and_every_outside_byte_including_script_hash_stay(self):
        original=PREFIX+ACADEMIC+GAP+SOFTWARE+SUFFIX
        expected=PREFIX+NEW_ACADEMIC+GAP+NEW_SOFTWARE+SUFFIX
        actual=module._replace_reference_catalogs(original,NEW_ACADEMIC+NEW_SOFTWARE)
        self.assertEqual(actual,expected)
        self.assertEqual(module._replace_reference_catalogs(actual,NEW_ACADEMIC+NEW_SOFTWARE),expected)

    def test_malformed_or_ambiguous_catalogues_are_rejected(self):
        for data in [ACADEMIC,ACADEMIC+SOFTWARE+SOFTWARE,ACADEMIC+SOFTWARE.removesuffix(b'</details>')]:
            with self.subTest(data=data),self.assertRaises(ValueError):
                module._replace_reference_catalogs(data,NEW_ACADEMIC+NEW_SOFTWARE)

    def test_explicit_candidate_path_isolated_and_idempotent(self):
        scratch=ROOT/'output/validation/website-reference-unit/scratch'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='owned-',dir=scratch) as directory:
            candidate=Path(directory)/'index.html';candidate.write_bytes(PREFIX+ACADEMIC+GAP+SOFTWARE+SUFFIX)
            local_before=(module.SITE/'index.html').read_bytes()
            with patch.object(module,'verify_basic_gl'),patch.object(module,'verify_three_way'),patch.object(module,'reference_catalog',return_value=(NEW_ACADEMIC+NEW_SOFTWARE).decode('utf-8')):
                module.update_references_only(candidate)
                first=candidate.read_bytes();mtime=candidate.stat().st_mtime_ns
                module.update_references_only(candidate)
                self.assertEqual(candidate.read_bytes(),first)
                self.assertEqual(candidate.stat().st_mtime_ns,mtime)
            self.assertEqual(first,PREFIX+NEW_ACADEMIC+GAP+NEW_SOFTWARE+SUFFIX)
            self.assertEqual((module.SITE/'index.html').read_bytes(),local_before)

    def test_actual_catalogue_uses_public_projection_and_complete_academic_citations(self):
        class Catalogue(HTMLParser):
            def __init__(self):super().__init__();self.ids=[]
            def handle_starttag(self,tag,attrs):
                values=dict(attrs)
                if tag=='article' and 'data-source-id' in values:self.ids.append(values['data-source-id'])
        sources=json.loads((ROOT/'frontend/src/generated/sources.json').read_text('utf-8'))
        rendered=module.reference_catalog();parsed=Catalogue();parsed.feed(rendered)
        self.assertEqual(len(parsed.ids),424)
        self.assertEqual(len(set(parsed.ids)),424)
        self.assertEqual(set(parsed.ids),{source['id'] for source in sources})
        for source_id in ['M19-ZENODO','M19-PAPER','M19-SHAOZI','M19-ATLAS','M19-LANGUAGE-ATLAS','M19-DATAV']:
            self.assertIn(source_id,parsed.ids)
        for source in sources:
            if source['id'] in ['M19-ZENODO','M19-PAPER','M19-SHAOZI','M19-ATLAS','M19-LANGUAGE-ATLAS']:
                self.assertIn(module.esc(source['citation']),rendered)
        self.assertIn('areas_v3（页面标注2021.5）',rendered)
        self.assertIn('DataV GeoAtlas学习交流用途告知',rendered)

if __name__=='__main__':unittest.main()
