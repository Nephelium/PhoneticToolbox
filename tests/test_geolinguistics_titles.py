"""The title import must not silently publish OCR or normalize IPA symbols."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_geolinguistics', ROOT/'scripts/prepare_geolinguistics.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReviewedTitles(unittest.TestCase):
    def parse(self, text, expected):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'review.tsv'
            path.write_text('feature_id\ttitle\ttoc_pdf_page\n'+text, encoding='utf-8')
            return module.reviewed_titles(path, expected)

    def test_ipa_diacritics_and_modifiers_remain_exact(self):
        title = '[ɓ ɗ ɠ tʂʰ ɬ a\u0303 n\u0329 ʔ] 音标'
        result = self.parse('F1_P\t'+title+'\t328\n', {'F1_P'})
        self.assertEqual(result['F1_P']['title'], title)
        self.assertEqual(result['F1_P']['titleReview']['tocPage'], 328)

    def test_duplicate_does_not_hide_missing_title(self):
        with self.assertRaises(ValueError):
            self.parse('F1_P\t调类\t328\nF1_P\t调类\t328\n', {'F1_P','F2_P'})

    def test_incomplete_and_unknown_ids_fail(self):
        with self.assertRaises(ValueError):
            self.parse('F999_P\t调类\t328\n', {'F1_P'})

    def test_ocr_subtitle_and_replacement_character_fail(self):
        for title in ['上声的分化 · DivisionOftheS/7而1gTone', '韵母\ufffd']:
            with self.assertRaises(ValueError):
                self.parse('F1_P\t'+title+'\t328\n', {'F1_P'})


if __name__ == '__main__':
    unittest.main()
