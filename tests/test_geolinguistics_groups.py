import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('groups', Path(__file__).resolve().parents[1]/'scripts/prepare_geolinguistics_groups.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BroadReview(unittest.TestCase):
    def setUp(self):
        self.catalog = {'features': [{'id': 'F1_'+v, 'volume': v, 'page': page,
            'categories': ['A1', 'A2', 'B1'], 'labels': {'B1': '保留细类'}} for v, page in [('P',337),('L',16),('S',226)]]}
        self.reviews = [{'schema':'ptb-atlas-broad-review/1','volume':f['volume'],'bookSha256':'book',
            'entries':[{'featureId':f['id'],'pdfPage':f['page'],'groupId':'A','label':'[a\u0303 n\u0329 ŋ]',
            'codes':['A1','A2'],'labelKind':'source','status':'accepted','textStatus':'verified','mappingStatus':'verified','evidence':'书图例与调查点对应'}],
            'coverage':[{'featureId':f['id'],'pdfPage':f['page'],'totalGroups':2}]} for f in self.catalog['features']]

    def apply(self):
        return module.apply_reviews(self.catalog,self.reviews,{'sourceSha256':'book'})

    def test_exact_unicode_separate_coverage_and_source_preserved(self):
        self.reviews[0]['coverage'][0]['sourceNote']='所提供 PDF 缺少本图。'
        before=copy.deepcopy(self.catalog);result=self.apply()
        self.assertEqual(result['features'][0]['groupNote'],'所提供 PDF 缺少本图。')
        self.assertEqual(self.catalog,before)
        self.assertEqual(result['features'][0]['labels'],{'B1':'保留细类'})
        self.assertEqual(result['features'][0]['groups'][0]['label'],'[a\u0303 n\u0329 ŋ]')
        self.assertEqual(result['groupReview']['partialFeatures'],3)
        self.assertEqual(result['groupReview']['completeFeatures'],0)
        self.assertEqual(result['groupReview']['verifiedCodes'],6)

    def test_unverified_and_withdrawn_mappings_cannot_linger(self):
        self.catalog=self.apply();self.reviews[0]['entries'][0]['status']='pending'
        self.assertEqual(self.apply()['features'][0]['groups'],[])

    def test_site_conflicts_preserve_multiple_atoms_and_reject_duplicates(self):
        first=dict(featureId='F1_P',siteId='S1',pdfPage=337,code='A1',reason='第一项冲突',evidence='原图与原表')
        self.reviews[0]['exceptions']=[first]
        self.assertEqual(self.apply()['features'][0]['groupExceptions']['S1'],{'code':'A1','reason':'第一项冲突'})
        self.reviews[0]['exceptions'].append({**first,'code':'A2','reason':'第二项冲突'})
        result=self.apply()
        self.assertEqual(result['features'][0]['groupExceptions']['S1'],[{'code':'A1','reason':'第一项冲突'},{'code':'A2','reason':'第二项冲突'}])
        self.assertEqual(result['groupReview']['exceptions'],1)
        self.assertEqual(result['groupReview']['exceptionAtoms'],2)
        self.reviews[0]['exceptions'].append(first)
        with self.assertRaises(ValueError):self.apply()

    def test_inconsistent_evidence_duplicate_membership_and_missing_volume_fail(self):
        original=copy.deepcopy(self.reviews)
        for key,value in [('mappingStatus','pending'),('pdfPage',1),('codes',['A9']),('evidence',''),('label','\ufffd')]:
            self.reviews=copy.deepcopy(original);self.reviews[0]['entries'][0][key]=value
            with self.assertRaises(ValueError):self.apply()
        self.reviews=copy.deepcopy(original);entry=copy.deepcopy(self.reviews[0]['entries'][0]);entry['groupId']='another';self.reviews[0]['entries'].append(entry)
        with self.assertRaises(ValueError):self.apply()
        self.reviews=original[:2]
        with self.assertRaises(ValueError):self.apply()


if __name__=='__main__':unittest.main()
