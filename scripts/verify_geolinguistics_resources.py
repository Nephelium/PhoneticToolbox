"""Compare the local display resources to their authorized original tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--classification', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import openpyxl
    import fitz
    inputs = []

    def record(path):
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        inputs.append({'name':path.name,'bytes':path.stat().st_size,'sha256':digest})

    def rows(name, encoding='utf-8-sig'):
        path = args.data/name
        record(path)
        with path.open(encoding=encoding,newline='') as stream:
            return list(csv.DictReader(stream))

    def resource(name):
        return json.loads((args.resources/name).read_text('utf-8'))

    source = rows('all_china.csv','gb18030')
    observations = resource('observations.json')
    catalog = resource('catalog.json')['features']
    sites = resource('sites.json')
    assert len(source) == len(sites) == 930 and len(catalog) == len(observations) == 510
    for index,row in enumerate(source):
        assert sites[index]['name'] == row['cities']
        assert sites[index]['lon'] == float(row['long']) and sites[index]['lat'] == float(row['lat'])
        for feature in catalog:
            column = observations[feature['id']]
            actual = column['raw'].get(str(index),{}).get('raw',column['values'][index])
            assert actual == row[feature['id']], (index,feature['id'])
    paper = resource('paper.json')
    tax = rows('cn_tax.csv')
    for index,row in enumerate(tax):
        point = paper['points'][index]
        assert point['mds'] == [float(row[f'd{d}']) for d in range(1,4)]
        for kind in ['sharp','heat']:
            assert point[kind] == [row[f'{kind}{k}'] for k in range(2,11)]
    for kind,name in [('sharp','fill_type1.csv'),('heat','fill_type2.csv')]:
        palette = rows(name)
        for count in range(2,11):
            assert paper['colors'][kind][str(count)] == {r['cluster_code']:r['color_code'] for r in palette if int(r['cluster_num'])==count}
    matrices = []
    for key,name in zip(['main','type-0-0','type-0-1','type-1-0','type-1-1'],['dist_matrix.csv','Type_1_0_form_1_0.csv','Type_1_0_form_1_1.csv','Type_1_1_form_1_0.csv','Type_1_1_form_1_1.csv']):
        table = rows(name)
        original = np.array([[float(row[f'S{i}']) for i in range(1,931)] for row in table],dtype='<f8')
        actual = np.fromfile(args.resources/(key+'.f64'),dtype='<f8').reshape(930,930)
        assert np.array_equal(original,actual), key
        matrices.append({'name':key,'cells':int(actual.size),'equal':True})
    record(args.classification)
    book = openpyxl.load_workbook(args.classification,read_only=True,data_only=True)
    classification = resource('classification.json')
    for sheet in classification['sheets']:
        original = list(book[sheet['name']].values)
        for row in sheet['rows']:
            assert list(original[row['row']-1][:10]) == row['original'], row['id']
    book.close()
    pages = resource('book-pages-manifest.json')
    assert {p['feature'] for p in pages['pages']} == {f['id'] for f in catalog}
    for page in pages['pages']:
        path = args.resources/'book'/(page['feature']+'.jpg')
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == page['sha256']
        image = fitz.Pixmap(path)
        assert image.height > image.width and image.width >= 1900, path
    report = {'success':True,'scope':'Original cells, coordinates and precomputed results compared exactly; no new linguistic statistics',
              'observationCells':930*510,'sites':930,'features':510,'matrices':matrices,
              'classificationSheets':{s['id']:len(s['rows']) for s in classification['sheets']},
              'completeBookPages':len(pages['pages']),'inputs':inputs}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='inputs'},ensure_ascii=False))


if __name__ == '__main__':
    main()
