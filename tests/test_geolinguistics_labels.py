import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('labels', Path(__file__).resolve().parents[1]/'scripts/prepare_geolinguistics_labels.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CategoryReview(unittest.TestCase):
    def setUp(self):
        self.catalog = {'features': [{'id': 'F1_P', 'page': 337, 'categories': ['A1', 'A2'], 'labels': {'A2': 'old'}}]}
        self.manifest = {'sourceSha256': 'original-book'}
        self.review = {'schema': 'ptb-atlas-category-review/1', 'bookSha256': 'original-book', 'entries': [{
            'feature': 'F1_P', 'code': 'A1', 'page': 337, 'label': '[a\u0303 n\u0329 ŋ ɲ ʔ tʂʰ]',
            'textStatus': 'verified', 'mappingStatus': 'verified',
            'textEvidence': 'original legend', 'mappingEvidence': 'site and symbol crosscheck'}]}

    def test_exact_ipa_and_source_unchanged(self):
        before = copy.deepcopy(self.catalog)
        result = module.apply_review(self.catalog, self.review, self.manifest)
        self.assertEqual(result['features'][0]['labels'], {'A1': self.review['entries'][0]['label']})
        self.assertEqual(self.catalog, before)
        self.assertEqual(result['categoryReview']['completeFeatures'], 0)

    def test_pending_text_or_mapping_cannot_publish(self):
        for field in ['textStatus', 'mappingStatus']:
            review = copy.deepcopy(self.review); review['entries'][0][field] = 'pending'
            result = module.apply_review(self.catalog, review, self.manifest)
            self.assertEqual(result['features'][0]['labels'], {})

    def test_wrong_source_page_code_and_duplicate_fail(self):
        for field, value in [('page', 338), ('code', 'B'), ('mappingEvidence', ''), ('label', '\ufffd')]:
            review = copy.deepcopy(self.review); review['entries'][0][field] = value
            with self.assertRaises(ValueError): module.apply_review(self.catalog, review, self.manifest)
        review = copy.deepcopy(self.review); review['entries'] *= 2
        with self.assertRaises(ValueError): module.apply_review(self.catalog, review, self.manifest)
        review = copy.deepcopy(self.review); review['bookSha256'] = 'another-book'
        with self.assertRaises(ValueError): module.apply_review(self.catalog, review, self.manifest)


if __name__ == '__main__': unittest.main()
