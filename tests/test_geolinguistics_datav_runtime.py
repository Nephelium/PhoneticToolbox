"""Targeted runtime derivation/association tests, with no hardware or research data."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verification_artifacts import verification_run
from prepare_geolinguistics_datav import atomic, digest, encoded
from prepare_geolinguistics import gcj_inverse
import package_geolinguistics_data as packer

spec = importlib.util.spec_from_file_location('datav_runtime', ROOT / 'scripts/prepare_geolinguistics_datav_runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def polygon(x, y, size=1):
    return {'type': 'Polygon', 'coordinates': [[[x,y],[x+size,y],[x+size,y+size],[x,y+size],[x,y]]]}


def nodes():
    return {n['adcode']:n for n in [
        {'adcode':100000,'name':'中国','level':'country','parent':None},
        {'adcode':440000,'name':'广东省','level':'province','parent':100000},
        {'adcode':440100,'name':'广州市','level':'city','parent':440000},
        {'adcode':440101,'name':'测试区','level':'district','parent':440100},
        {'adcode':420000,'name':'湖北省','level':'province','parent':100000},
        {'adcode':429004,'name':'仙桃市','level':'city','parent':420000},
        {'adcode':810000,'name':'香港特别行政区','level':'province','parent':100000},
        {'adcode':810001,'name':'中西区','level':'district','parent':810000},
        {'adcode':710000,'name':'台湾省','level':'province','parent':100000}]}


class DatavRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact_run=verification_run('geolinguistics-datav-runtime','unit','Synthetic coordinate cache, datum bypass, hierarchy conflicts, classification, fallback and pinned supplement approval')
        cls.out,cls.scratch=cls.artifact_run.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.artifact_run.__exit__(None,None,None)

    def test_existing_inverse_is_used_once_per_distinct_native_coordinate(self):
        adapter=module.CoordinateAdapter();raw=polygon(113,23)
        with patch.object(module,'gcj_inverse',wraps=gcj_inverse) as inverse:
            first=adapter.geometry(raw);second=adapter.geometry(raw)
            self.assertEqual(inverse.call_count,4)
        self.assertEqual(first,second);self.assertEqual(adapter.positions,10)
        self.assertEqual(first['coordinates'][0][0],list(gcj_inverse(113,23)))
        self.assertEqual(raw['coordinates'][0][0],[113,23])

    def test_wgs84_supplement_is_added_after_inverse_and_never_double_transformed(self):
        from shapely.geometry import shape,Point,mapping
        item={'id':'taiping','coordinateSystem':{'basis':'EPSG:4326'},'geometry':polygon(115,10),
              'proposedDisplayTargets':[{'adcode':440000}],'source':{'sha256':'source'},'candidateFileSha256':'candidate',
              'selection':{'rule':'explicit source component'},'secondarySourceId':'nlsc-taiping'}
        adapter=module.CoordinateAdapter();geometries={440000:shape(adapter.geometry(polygon(113,23)))}
        calls=len(adapter.cache);audit=module.add_supplements(geometries,[item],'EPSG:4326','after-inverse-WGS84-bypass')
        self.assertEqual(len(adapter.cache),calls)
        self.assertTrue(geometries[440000].covers(Point(115,10)))
        parts=list(geometries[440000].geoms)
        added=next(p for p in parts if p.covers(Point(115.5,10.5)))
        self.assertEqual(set(map(tuple,mapping(added)['coordinates'][0])),set(map(tuple,item['geometry']['coordinates'][0])))
        self.assertEqual(audit[0]['datum'],'EPSG:4326')

    def test_only_hit_ancestors_are_removed_incomparable_regions_stay_conflicted(self):
        n=nodes()
        self.assertEqual(module.choose_deepest([440000,440100,440101],n),([440101],[440000,440100]))
        self.assertEqual(module.choose_deepest([440000,440101,420000],n),([420000,440101],[440000]))

    def test_spatial_association_uses_real_depth_without_nearest_or_coordinate_changes(self):
        from shapely.geometry import shape
        n=nodes();geometries={440000:shape(polygon(0,0,10)),440100:shape(polygon(0,0,5)),440101:shape(polygon(0,0,2))}
        original=[{'id':'S1','name':'原书点','lon':1,'lat':1,'admin':None},
                  {'id':'S2','name':'父级点','lon':4,'lat':4,'admin':None},
                  {'id':'S3','name':'界外点','lon':20,'lat':20,'admin':None}]
        sites,_=module.associate_sites(original,geometries,n)
        self.assertEqual([s['admin'] for s in sites],['district:440101','city:440100',None])
        self.assertEqual(sites[2]['adminCandidates'],[])
        geometries[420000]=shape(polygon(0,0,10))
        conflicted,_=module.associate_sites(original,geometries,n)
        self.assertIsNone(conflicted[0]['admin']);self.assertEqual(len(conflicted[0]['adminCandidates']),2)
        self.assertEqual([(s['lon'],s['lat'],s['name']) for s in sites],[(s['lon'],s['lat'],s['name']) for s in original])

    def test_classification_requires_same_code_name_and_actual_ancestors(self):
        row={'values':['440000','广东省','440100','广州市','440101','测试区','','','','']}
        self.assertEqual(module.row_identity(row,nodes())[:2],(440101,'same-code-name-real-ancestors'))
        for column,value,reason in [(5,'历史旧名','same-code-name-changed-or-different'),(2,'420000','city-code-or-name-ancestor-conflict'),(1,'湖北省','province-name-ancestor-conflict'),(4,'','missing-explicit-target-code')]:
            bad=copy.deepcopy(row);bad['values'][column]=value
            self.assertEqual(module.row_identity(bad,nodes())[1],reason)
        direct={'values':['420000','湖北省','429000','省直辖县级行政区划','429004','仙桃市','','','','']}
        code,reason,details=module.row_identity(direct,nodes())
        self.assertEqual(code,429004);self.assertIn('noRealIntermediateCity',details)

    def test_old_fake_city_id_is_not_preserved_and_original_classification_values_stay(self):
        row={'id':'r1','values':['810000','香港特别行政区','','','810000','香港特别行政区','','','',''],
             'original':[810000,'原值'],'admin':'city:810000','match':'old'}
        source={'sheets':[{'id':'combined','rows':[row]}]};snapshot=copy.deepcopy(source)
        result,audit=module.associate_classification(source,nodes(),{'city:810000':{'id':'city:810000','level':'city','name':'香港特别行政区'}})
        self.assertEqual(result['sheets'][0]['rows'][0]['admin'],'province:810000')
        self.assertEqual(source,snapshot);self.assertEqual(result['sheets'][0]['rows'][0]['values'],row['values'])
        self.assertEqual(audit[0]['oldLayerChanged']['newLevel'],'province')

    def test_exact_name_fallback_requires_real_ancestors_and_marks_conflicting_original_code(self):
        row={'id':'r2','values':['440000','广东省','440100','广州市','420000','测试区','','','',''],
             'original':['unchanged'],'admin':None,'match':None}
        source={'sheets':[{'id':'combined','rows':[row]}]};snapshot=copy.deepcopy(source)
        result,audit=module.associate_classification(source,nodes(),{})
        self.assertEqual(result['sheets'][0]['rows'][0]['admin'],'district:440101')
        self.assertIn('exact-ancestor-name-fallback',result['sheets'][0]['rows'][0]['match'])
        self.assertIn('explicit-code-name-conflict',result['sheets'][0]['rows'][0]['match'])
        self.assertEqual(audit[0]['explicitCodeConflict']['originalCode'],'420000')
        self.assertEqual(audit[0]['selectedNameEvidence']['exactScopeCandidateCount'],1)
        self.assertEqual(source,snapshot)
        row['values'][3]='不存在的市'
        rejected,_=module.associate_classification(source,nodes(),{})
        self.assertIsNone(rejected['sheets'][0]['rows'][0]['admin'])

    def test_same_name_multiple_candidates_in_one_ancestor_scope_remain_pending(self):
        n=nodes();n[440102]={'adcode':440102,'name':'测试区','level':'district','parent':440100}
        row={'id':'r3','values':['440000','广东省','440100','广州市','','测试区','','','',''],'admin':None}
        result,audit=module.associate_classification({'sheets':[{'id':'combined','rows':[row]}]},n,{})
        self.assertIsNone(result['sheets'][0]['rows'][0]['admin'])
        self.assertEqual(audit[0]['nameFallbackCandidates'],[440101,440102])
        self.assertEqual(audit[0]['nameFallbackOutcome'],'ambiguous')

    def test_known_same_context_code_with_changed_name_is_not_reassigned_by_name(self):
        n=nodes();n[440102]={'adcode':440102,'name':'改名后的区','level':'district','parent':440100}
        row={'id':'r4','values':['440000','广东省','440100','广州市','440102','测试区','','','',''],'admin':None}
        result,audit=module.associate_classification({'sheets':[{'id':'combined','rows':[row]}]},n,{})
        self.assertIsNone(result['sheets'][0]['rows'][0]['admin'])
        self.assertEqual(audit[0]['nameFallbackOutcome'],'known-code-name-change-pending')

    def test_fallbacks_keep_real_id_level_and_do_not_create_index_objects(self):
        n=nodes();features={c:{'type':'Feature','id':module.runtime_id(c,n),'properties':{'level':r['level']},'geometry':polygon(0,0)} for c,r in n.items() if c!=100000}
        original=copy.deepcopy(features);layers,audit=module.drawing_layers(features,n)
        self.assertTrue(any(f['id']=='province:810000' and f['properties']['renderFallback']=='city' for f in layers['city']))
        self.assertTrue(any(f['id']=='city:429004' and f['properties']['renderFallback']=='district' for f in layers['district']))
        self.assertTrue(any(f['id']=='province:710000' and f['properties']['level']=='province' for f in layers['district']))
        self.assertEqual(features,original)
        self.assertEqual(len({f['id'] for f in layers['district']}),len(layers['district']))
        self.assertFalse(any(f['id']=='city:810000' for values in layers.values() for f in values))

    def test_supplement_requires_exact_approval_targets_source_arrays_and_hashes(self):
        directory=self.scratch/'review';directory.mkdir()
        geom=polygon(1,2);source={'type':'FeatureCollection','features':[{'type':'Feature','properties':{'adcode':440000},'geometry':{'type':'MultiPolygon','coordinates':[geom['coordinates']]}}]}
        source_path=directory/'source.json';atomic(source_path,encoded(source))
        candidate=directory/'candidate.json';atomic(candidate,encoded({'type':'FeatureCollection','features':[{'type':'Feature','geometry':geom,'properties':{}}]}))
        item={'id':'one','candidateFile':'candidate.json','candidateFileSha256':digest(candidate.read_bytes()),
              'source':{'path':str(source_path),'sha256':digest(source_path.read_bytes()),'featureIndex':0,'componentIndex':0,'sourceProperties':{'adcode':440000},'geometryPath':'features[0].geometry.coordinates[0]'},
              'selection':{'rule':'explicit component0'},'coordinateSystem':{'basis':'datav-native-inferred'},
              'proposedDisplayTargets':[{'adcode':440000,'level':'province','name':'广东省'}]}
        review=directory/'review.json';atomic(review,encoded({'schema':'ptb-island-supplement-review/1','candidates':[item]}));review_sha=digest(review.read_bytes())
        decision=directory/'decision.json';atomic(decision,encoded({'schema':'ptb-island-supplement-decisions/1','reviewItemsSha256':review_sha,'items':[{'id':'one','decision':'approve','targets':[440000]}]}))
        result=module.review_supplements(review,review_sha,['one'],nodes(),decision,digest(decision.read_bytes()))
        self.assertEqual(result[0]['geometry'],geom)
        with self.assertRaisesRegex(ValueError,'pinned decision'):
            module.review_supplements(review,review_sha,['one'],nodes())
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            module.review_supplements(review,review_sha,['one','one'],nodes(),decision,digest(decision.read_bytes()))
        candidate.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'SHA'):
            module.review_supplements(review,review_sha,['one'],nodes(),decision,digest(decision.read_bytes()))

    def test_independent_review_bundles_keep_each_approval_and_reject_duplicate_ids(self):
        groups=[]
        for identity in ('original','additional'):
            directory=self.scratch/('bundle-'+identity);directory.mkdir()
            geom=polygon(1,2)
            source=directory/'source.json'
            atomic(source,encoded({'features':[{'properties':{'adcode':440000},'geometry':{'type':'MultiPolygon','coordinates':[geom['coordinates']]}}]}))
            candidate=directory/'candidate.json'
            atomic(candidate,encoded({'type':'FeatureCollection','features':[{'geometry':geom}]}))
            item={'id':identity,'candidateFile':candidate.name,'candidateFileSha256':digest(candidate.read_bytes()),
                  'source':{'path':str(source),'sha256':digest(source.read_bytes()),'featureIndex':0,'componentIndex':0,'sourceProperties':{'adcode':440000},'geometryPath':'features[0].geometry.coordinates[0]'},
                  'selection':{'rule':'reviewed component0'},'coordinateSystem':{'basis':'datav-native-inferred'},
                  'proposedDisplayTargets':[{'adcode':440000,'level':'province','name':'广东省'}]}
            review=directory/'review.json';atomic(review,encoded({'schema':'ptb-island-supplement-review/1','candidates':[item]}))
            review_sha=digest(review.read_bytes());decision=directory/'decision.json'
            atomic(decision,encoded({'schema':'ptb-island-supplement-decisions/1','reviewItemsSha256':review_sha,'items':[{'id':identity,'decision':'approve','targets':[440000]}]}))
            groups.append({'review':review,'reviewSha256':review_sha,'approvedIds':[identity],'decisions':decision,'decisionsSha256':digest(decision.read_bytes())})
        selected,reviews=module.collect_supplements(groups,nodes())
        self.assertEqual([r['id'] for r in selected],['original','additional'])
        self.assertEqual([r['reviewManifestSha256'] for r in selected],[g['reviewSha256'] for g in groups])
        self.assertEqual([r['decisionsSha256'] for r in reviews],[g['decisionsSha256'] for g in groups])
        with self.assertRaisesRegex(ValueError,'across review bundles'):
            module.collect_supplements([groups[0],groups[0]],nodes())
        unapproved=dict(groups[1],decisions=None,decisionsSha256=None)
        with self.assertRaisesRegex(ValueError,'pinned decision'):
            module.collect_supplements([groups[0],unapproved],nodes())

    def test_content_bound_authority_refreshes_only_matching_candidate_not_old_diagnostics(self):
        directory=self.scratch/'authority-identity';directory.mkdir()
        schema='ptb-geolinguistics-boundary-provenance/1'
        originals={};paths={}
        for generation in (1,2,3):
            candidate=directory/('candidate'+str(generation));candidate.mkdir();paths[generation]=candidate
            for name in packer.DATA_FILES:
                # Classification is the only runtime difference between old 1/2.
                payload=('classification-'+str(generation) if name=='classification.json' else name).encode('utf8')
                atomic(candidate/name,payload)
            manifest=packer.data_manifest(candidate)
            bindings={r['path']:{'bytes':r['bytes'],'sha256':r['sha256']} for r in manifest['files'] if r['path'] in packer.BOUNDARY_BINDINGS}
            version,sha=module.boundary_identity(bindings)
            self.assertEqual((version,sha),module.boundary_identity(dict(reversed(list(bindings.items())))))
            provenance={'schema':schema,'version':version if generation==3 else module.SOURCE,
                        'bindings':bindings,'bindingsSha256':sha,'status':'isolated-'+str(generation)}
            manifest['boundaries']=provenance
            atomic(candidate/'data-manifest.json',encoded(manifest));originals[generation]=manifest
        self.assertNotEqual(originals[1]['boundaries']['bindingsSha256'],originals[2]['boundaries']['bindingsSha256'])
        authority=copy.deepcopy(originals[3]['boundaries']);authority['status']='reviewed-public-authority'
        authority_file=directory/'repo/resources/research/geolinguistics-public-boundaries.json'
        atomic(authority_file,encoded(authority))
        fake_script=directory/'repo/scripts/package_geolinguistics_data.py'
        before={g:(p/'data-manifest.json').read_bytes() for g,p in paths.items()}
        with patch.object(packer,'__file__',str(fake_script)):
            refreshed=packer.data_manifest(paths[3])
            self.assertEqual(refreshed['boundaries'],authority)
            for generation in (1,2):
                preserved=packer.data_manifest(paths[generation])
                self.assertEqual(preserved['boundaries'],originals[generation]['boundaries'])
                self.assertNotEqual(preserved['boundaries']['version'],authority['version'])
            # Even a different content-bound version cannot inherit this authority.
            old3=copy.deepcopy(originals[3]);old3['boundaries']['version']+='-other'
            atomic(paths[3]/'data-manifest.json',encoded(old3))
            self.assertEqual(packer.data_manifest(paths[3])['boundaries'],old3['boundaries'])
            atomic(paths[3]/'data-manifest.json',before[3])
            # A changed bound file must fail before authority substitution.
            atomic(paths[3]/'sites.json',b'changed-after-review')
            with self.assertRaisesRegex(ValueError,'inputs changed'):
                packer.data_manifest(paths[3])
        self.assertEqual((paths[1]/'data-manifest.json').read_bytes(),before[1])
        self.assertEqual((paths[2]/'data-manifest.json').read_bytes(),before[2])


if __name__=='__main__':
    unittest.main()
