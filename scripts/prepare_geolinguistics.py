"""Build local-only M19 resources, retaining source identity and raw observations.

Install pyshp/shapely in a task-local directory and pass --tools when needed.
No source files are edited. Generated resources are deliberately Git-ignored.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile
from collections import Counter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')


def rows(path, encoding='utf-8-sig', delimiter=','):
    with path.open(encoding=encoding, newline='') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def normalized(value):
    if value is None or str(value).strip() in ['#N/A', 'nan']:
        return ''
    return re.sub(r'\s+', '', str(value)).removesuffix('.0')


def reviewed_titles(path, expected):
    """Use a visually checked source index. OCR alone never supplies UI titles.

    IPA is preserved verbatim, including combining characters and superscripts.
    Parenthesized text transcribes the book's small contextual annotations.
    """
    records = rows(path, delimiter='\t')
    result = {}
    for row in records:
        key, title = row['feature_id'], row['title']
        if key in result or not title.strip() or '\ufffd' in title:
            raise ValueError('Duplicate or invalid reviewed title: '+key)
        if re.search(r'\b(?:Division|Initials|Finals|Tone|Merger)\w*', title, re.I):
            raise ValueError('Unreviewed English OCR in title: '+key)
        result[key] = {'title': title, 'titleReviewed': True, 'titleReview': {
            'method':'visual-source-toc', 'tocPage':int(row['toc_pdf_page']),
            'notation':'原书小字说明改排为括注；国际音标保留 Unicode 字符'}}
    if set(result) != set(expected):
        raise ValueError('Reviewed title IDs must exactly match the 510 source features')
    return result


def gcj_forward(lon, lat):
    # Conventional GCJ trigonometric offset; used only to align display geometry.
    # Scientific survey coordinates are never modified by this operation.
    x, y = lon-105, lat-35
    a, ee = 6378245.0, 0.00669342162296594323
    dlat = -100+2*x+3*y+.2*y*y+.1*x*y+.2*math.sqrt(abs(x))
    dlon = 300+x+2*y+.1*x*x+.1*x*y+.1*math.sqrt(abs(x))
    common = (20*math.sin(6*x*math.pi)+20*math.sin(2*x*math.pi))*2/3
    dlat += common+(20*math.sin(y*math.pi)+40*math.sin(y/3*math.pi))*2/3+(160*math.sin(y/12*math.pi)+320*math.sin(y*math.pi/30))*2/3
    dlon += common+(20*math.sin(x*math.pi)+40*math.sin(x/3*math.pi))*2/3+(150*math.sin(x/12*math.pi)+300*math.sin(x/30*math.pi))*2/3
    rad = lat/180*math.pi
    magic = 1-ee*math.sin(rad)**2
    return lon+dlon*180/(a/math.sqrt(magic)*math.cos(rad)*math.pi), lat+dlat*180/(a*(1-ee)/(magic**1.5)*math.pi)


def gcj_inverse(x, y, z=None):
    lon, lat = x, y
    for _ in range(4):
        fx, fy = gcj_forward(lon, lat)
        lon, lat = lon+x-fx, lat+y-fy
    return lon, lat


def main():
    p = argparse.ArgumentParser()
    for key in ['data', 'classification', 'index', 'maps']:
        p.add_argument('--'+key, type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--titles', type=Path, required=True)
    p.add_argument('--titles-only', action='store_true')
    p.add_argument('--category-review', type=Path, help='Private visually reviewed code-to-meaning input')
    p.add_argument('--broad-review', type=Path, nargs=3, help='Three private, reviewed broad-group dictionaries')
    p.add_argument('--tools', type=Path)
    args = p.parse_args()
    expected = {f'F{i}_{volume}' for volume,count in [('P',205),('L',203),('S',102)] for i in range(1,count+1)}
    title_index = reviewed_titles(args.titles, expected)
    title_hash = hashlib.sha256(args.titles.read_bytes()).hexdigest()
    if args.titles_only:
        catalog = json.loads((args.output/'catalog.json').read_text(encoding='utf-8'))
        if {f['id'] for f in catalog['features']} != expected:
            raise ValueError('Existing catalog does not contain the source feature set')
        for feature in catalog['features']:
            feature.update(title_index[feature['id']])
        if args.category_review:
            from prepare_geolinguistics_labels import apply_file
            catalog = apply_file(catalog, args.category_review, json.loads((args.output/'book-pages-manifest.json').read_text(encoding='utf-8')))
        if args.broad_review:
            from prepare_geolinguistics_groups import apply_files
            catalog = apply_files(catalog, args.broad_review, json.loads((args.output/'book-pages-manifest.json').read_text(encoding='utf-8')))
        for source in catalog['sources']:
            if source['id'] == 'shaozi':
                source.update(url='https://zenodo.org/records/15897647', license='CC BY 4.0')
        write(args.output/'catalog.json', catalog)
        audit = json.loads((args.output/'build-audit.json').read_text(encoding='utf-8'))
        audit.update(title_reviewed=510, title_ocr=0, title_index_sha256=title_hash)
        write(args.output/'build-audit.json', audit)
        from package_geolinguistics_data import write_data_manifest
        write_data_manifest(args.output)
        print('510 reviewed titles updated; observation tables and paper results unchanged')
        return
    if any(getattr(args,key) is None for key in ['data','classification','index','maps']):
        p.error('--data, --classification, --index and --maps are required for a full data build')
    if args.tools:
        sys.path.insert(0, str(args.tools.resolve()))
    import numpy as np
    import openpyxl
    import shapefile
    from shapely import make_valid
    from shapely.geometry import shape, mapping, Point
    from shapely.ops import transform
    from shapely.strtree import STRtree

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    audit = {'inputs': [], 'map_repairs': [], 'unmatched_classification': [], 'unmatched_sites': []}
    def fingerprint(path):
        raw = path.read_bytes()
        audit['inputs'].append({'name': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    main_csv = args.data/'all_china.csv'
    fingerprint(main_csv)
    for name in ['cn_tax.csv','fill_type1.csv','fill_type2.csv','dist_matrix.csv','Type_1_0_form_1_0.csv','Type_1_0_form_1_1.csv','Type_1_1_form_1_0.csv','Type_1_1_form_1_1.csv']:
        fingerprint(args.data/name)
    for path in [args.index,args.titles]:
        fingerprint(path)
    data = rows(main_csv, 'gb18030')
    tax = rows(args.data/'cn_tax.csv')
    assert len(data) == len(tax) == 930
    entries = json.loads(args.index.read_text(encoding='utf-8'))
    features, observations, anomalies = [], {}, []
    for e in entries:
        key = e['feature_id']
        vals, raw_overrides = [], {}
        for i, row in enumerate(data):
            raw = row[key]
            value = raw.strip()
            match = re.fullmatch(r'0\s+([A-Z]\d*[a-z]?(?:#[A-Z]\d*[a-z]?)*)\s*Name:\s*'+key+r',\s*dtype:\s*object', value)
            if match:
                value = match.group(1)
            valid = bool(re.fullmatch(r'[A-Z]+(?:\d+[a-z]?)?(?:#[A-Z]+(?:\d+[a-z]?)?)*', value))
            if value != raw or not valid:
                raw_overrides[str(i)] = {'raw': raw, 'rule': 'series-text-unwrapped' if match else 'trimmed' if valid else 'unresolved'}
                anomalies.append({'site': i, 'feature': key, **raw_overrides[str(i)]})
            vals.append(value if valid else None)
        categories = sorted({v for value in vals if value for v in value.split('#')}, key=lambda s:(s[0],int(re.search(r'\d+',s)[0]) if re.search(r'\d+',s) else 0,s))
        observations[key] = {'values': vals, 'raw': raw_overrides}
        features.append({'id': key, 'volume': key[-1], 'number':e['map_number'], 'page':e['pdf_page'], **title_index[key], 'categories':categories, 'labels':{}, 'legend':'book/'+key+'.jpg'})
    assert len(features) == 510 and len(observations) == 510
    catalog = {'version':'m19-local-20261022', 'features':features, 'survey':'2000–2008', 'sources':[
        {'id':'zenodo','title':'Huang 等（2024）数据和代码','url':'https://zenodo.org/records/10697975','license':'CC BY 4.0'},
        {'id':'atlas','title':'《汉语方言地图集》（2008）语音、词汇、语法卷','license':'用户提供的本机阅读资料'},
        {'id':'shaozi','title':'井韶子整理 · 《中国语言地图集（第2版）汉语方言卷》县级分布','url':'https://zenodo.org/records/15897647','license':'CC BY 4.0'}]}
    if args.category_review:
        from prepare_geolinguistics_labels import apply_file
        fingerprint(args.category_review)
        catalog = apply_file(catalog, args.category_review, json.loads((out/'book-pages-manifest.json').read_text(encoding='utf-8')))
    elif (out/'catalog.json').exists() and json.loads((out/'catalog.json').read_text(encoding='utf-8')).get('categoryReview'):
        raise ValueError('This resource set has reviewed category labels; pass --category-review to retain them')
    if args.broad_review:
        from prepare_geolinguistics_groups import apply_files
        for path in args.broad_review:
            fingerprint(path)
        catalog = apply_files(catalog, args.broad_review, json.loads((out/'book-pages-manifest.json').read_text(encoding='utf-8')))
    elif (out/'catalog.json').exists() and json.loads((out/'catalog.json').read_text(encoding='utf-8')).get('groupReview'):
        raise ValueError('This resource set has reviewed broad groups; pass --broad-review to retain them')
    write(out/'catalog.json', catalog)
    write(out/'observations.json', observations)
    write(out/'anomalies.json', anomalies)
    sites = [{'id': f'S{i+1}', 'name': row['cities'], 'lon':float(row['long']), 'lat':float(row['lat'])} for i,row in enumerate(data)]
    colors = {kind:{str(n):{r['cluster_code']:r['color_code'] for r in rows(args.data/filename) if int(r['cluster_num'])==n} for n in range(2,11)} for kind,filename in [('sharp','fill_type1.csv'),('heat','fill_type2.csv')]}
    write(out/'paper.json', {'points':[{'mds':[float(t['d1']),float(t['d2']),float(t['d3'])], 'sharp':[t[f'sharp{k}'] for k in range(2,11)], 'heat':[t[f'heat{k}'] for k in range(2,11)]} for t in tax], 'matrices':['main','type-0-0','type-0-1','type-1-0','type-1-1'], 'colors':colors})
    for name, filename in zip(['main','type-0-0','type-0-1','type-1-0','type-1-1'], ['dist_matrix.csv','Type_1_0_form_1_0.csv','Type_1_0_form_1_1.csv','Type_1_1_form_1_0.csv','Type_1_1_form_1_1.csv']):
        matrix_rows = rows(args.data/filename)
        matrix = np.array([[float(row[f'S{i}']) for i in range(1,931)] for row in matrix_rows], dtype='<f8')
        assert matrix.shape==(930,930) and np.isfinite(matrix).all() and np.array_equal(matrix,matrix.T) and not matrix.diagonal().any()
        matrix.tofile(out/(name+'.f64'))
    print('linguistic resources ready', flush=True)

    all_admin, geometries = [], {}
    def feature(geometry, props, tolerance=0):
        if not geometry.is_valid:
            audit['map_repairs'].append(props['id'])
            geometry = make_valid(geometry)
        if tolerance:
            geometry = geometry.simplify(tolerance, preserve_topology=True)
        center = geometry.representative_point()
        props.update(center=[center.x,center.y], extent=list(geometry.bounds))
        geometries[props['id']] = geometry
        all_admin.append(props)
        return {'type':'Feature', 'id':props['id'], 'properties':props, 'geometry':mapping(geometry)}
    level_features = {}
    for level,prefix,tol in [('province','pr',.008),('city','ct',.003),('district','dt',.001)]:
        reader = shapefile.Reader(str(args.maps/level), encoding='utf8')
        items = []
        for item in reader.iterShapeRecords():
            rec = item.record.as_dict()
            if level!='province' and rec['pr_adcode']=='710000':
                continue
            code = rec[prefix+'_adcode']
            props = {'id':level+':'+code, 'code':code, 'name':rec[prefix+'_name'], 'level':level, 'parent':'100000' if level=='province' else 'province:'+rec['pr_adcode'] if level=='city' else 'city:'+rec['ct_adcode'], 'province':'province:'+rec['pr_adcode'], 'source':'gary-20240206-gcj-iterative'}
            geo = transform(gcj_inverse, shape(item.shape.__geo_interface__).simplify(tol, preserve_topology=True))
            items.append(feature(geo, props))
        level_features[level] = items
        print(level, len(items), flush=True)
    for level,zipname,prefix in [('city','taiwan-county.zip','COUNTY'),('district','taiwan-town.zip','TOWN')]:
        with zipfile.ZipFile(args.maps/zipname) as archive:
            stem = next(n[:-4] for n in archive.namelist() if n.upper().startswith(prefix+'_MOI_') and n.lower().endswith('.shp'))
            reader = shapefile.Reader(shp=io.BytesIO(archive.read(stem+'.shp')), dbf=io.BytesIO(archive.read(stem+'.dbf')), encoding='utf8')
            for item in reader.iterShapeRecords():
                rec = item.record.as_dict()
                props = {'id':level+':TW-'+rec[prefix+'CODE'], 'code':'TW-'+rec[prefix+'CODE'], 'name':rec[prefix+'NAME'], 'level':level, 'parent':'province:710000' if level=='city' else 'city:TW-'+rec['COUNTYCODE'], 'province':'province:710000', 'source':'nlsc-twd97', 'sourceName':rec.get('COUNTYNAME','')}
                level_features[level].append(feature(shape(item.shape.__geo_interface__), props, .0008))
    assert len(level_features['city']) == 393
    assert len(level_features['district']) == 3244
    for level, items in level_features.items():
        write(out/'maps'/(level+'.json'), {'type':'FeatureCollection','features':items})
    territory = []
    for path in sorted(args.data.rglob('*.shp')):
        fingerprint(path)
        reader = shapefile.Reader(str(path))
        for i,item in enumerate(reader.iterShapes()):
            geo = shape(item.__geo_interface__).simplify(.002, preserve_topology=True)
            territory.append({'type':'Feature','properties':{'source':'zenodo','name':path.stem,'line':'dash' if 'dash' in path.stem else 'land'},'geometry':mapping(geo)})
    write(out/'maps'/'territory.json', {'type':'FeatureCollection','features':territory})
    write(out/'maps'/'world.json', json.loads((args.maps/'world.geojson').read_text(encoding='utf-8')))
    write(out/'map-index.json', all_admin)
    # A spatial containment association is geographic context, not dialect identity.
    districts = [p for p in all_admin if p['level']=='district']
    tree = STRtree([geometries[p['id']] for p in districts])
    for i,site in enumerate(sites):
        point = Point(site['lon'],site['lat'])
        ids = [districts[int(index)]['id'] for index in tree.query(point, predicate='intersects')]
        site['admin'] = ids[0] if len(ids)==1 else None
        site['adminCandidates'] = ids
        if len(ids)!=1:
            audit['unmatched_sites'].append(site['id'])
    write(out/'sites.json', sites)

    fingerprint(args.classification)
    wb = openpyxl.load_workbook(args.classification, read_only=True, data_only=True)
    combined, sources = [], []
    lookup = {p['id']:p for p in all_admin}
    def simple(s):
        return normalized(s).replace('臺','台').replace('縣','县').replace('區','区').replace('鄉','乡').replace('鎮','镇')
    for sheet_name, source_id in [('方言区+方言岛','combined'),('方言区','areas'),('方言岛','islands')]:
        sheet = wb[sheet_name]
        source_rows = []
        for row_num, row in enumerate(list(sheet.values)[1:],2):
            if not any(v is not None for v in row):
                continue
            values = [normalized(v) for v in row[:10]]
            code = values[4]
            matched = lookup.get('district:'+code)
            method = 'code-and-name'
            if matched and simple(lookup.get(matched['province'],{}).get('name',''))!=simple(values[1]):
                matched = None
            if not matched or simple(matched['name']) != simple(values[5]):
                matched = next((lookup.get(level+':'+code) for level in ['city','province'] if lookup.get(level+':'+code) and simple(lookup[level+':'+code]['name'])==simple(values[5]) and simple(lookup.get(lookup[level+':'+code]['province'],{}).get('name',''))==simple(values[1])),None)
            if not matched or simple(matched['name']) != simple(values[5]):
                matches = [p for p in districts if simple(p['name'])==simple(values[5]) and (simple(lookup.get(p['province'],{}).get('name',''))==simple(values[1]))]
                matched = matches[0] if len(matches)==1 else None
                method = 'unique-province-name'
            record = {'id':f'{source_id}:{row_num}', 'sheet':sheet_name, 'row':row_num, 'values':values, 'original':list(row[:10]), 'admin':matched['id'] if matched else None, 'match':method if matched else None}
            source_rows.append(record)
            if not matched:
                audit['unmatched_classification'].append(record['id'])
        sources.append({'id':source_id,'name':sheet_name,'rows':source_rows})
        if source_id=='combined': combined=source_rows
    assert len(combined)==3666
    write(out/'classification.json', {'author':'井韶子','book':'《中国语言地图集（第2版）汉语方言卷》','columns':['省编码','省','市编码','市','县编码','县','方言大区','方言区','方言片','方言小片'],'sheets':sources})
    audit.update(sites=len(sites),features=len(features),classification_rows={s['id']:len(s['rows']) for s in sources},map_counts={level:len(items) for level,items in level_features.items()},title_reviewed=len(title_index),title_ocr=0,title_index_sha256=title_hash)
    write(out/'build-audit.json',audit)
    from package_geolinguistics_data import write_data_manifest
    write_data_manifest(out)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['inputs','map_repairs','unmatched_classification','unmatched_sites']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
