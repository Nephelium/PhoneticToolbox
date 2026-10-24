"""Freeze reviewed local map candidates and a separately licensed boundary ZIP.

Copies an explicit runtime surface, full source-specific boundary databases and
the reviewed public provenance. Never overwrites a destination or publishes it.
Linguistic observations and private association audits are not in the boundary ZIP.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from package_geolinguistics_data import DATA_FILES, data_manifest


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_checked(source, target):
    if source.is_symlink() or (hasattr(source, 'is_junction') and source.is_junction()):
        raise ValueError('Linked input')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if digest(source) != digest(target):
        raise ValueError(f'Copy identity changed: {target.name}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--licenses', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args=parser.parse_args()
    destination=args.destination.resolve()
    if destination.exists():
        raise ValueError('Destination exists; choose a new candidate identity')
    for source in [args.runtime,args.candidate,args.licenses]:
        if destination.is_relative_to(source.resolve()):
            raise ValueError('Destination must be outside its input directories')
    provenance=read(args.provenance)
    if provenance['status']!='reviewed-candidate-with-documented-source-differences':
        raise ValueError('Explicitly reviewed provenance required')
    manifest=data_manifest(args.runtime)
    if manifest.get('boundaries',{}).get('version')!=provenance['version']:
        raise ValueError('Runtime and reviewed boundary identities differ')
    if max(f['bytes'] for f in manifest['files'])>100_000_000 or sum(f['bytes'] for f in manifest['files'])>200_000_000:
        raise ValueError('Runtime exceeds the existing size limits')
    for item in manifest['files']:
        copy_checked(args.runtime/item['path'],destination/'runtime'/item['path'])
    manifest['boundaries']=provenance
    manifest['distribution']='reviewed local candidate; public publication pending explicit approval'
    write(destination/'runtime/data-manifest.json',manifest)

    candidate=read(args.candidate/'candidate-audit.json')
    quality=read(args.candidate/'quality-audit.json')
    coast=read(args.candidate/'overlap-coast-audit.json')
    names={}
    package=destination/'boundary-source-data'
    for source in provenance['sources']:
        item=source['database']
        input_path=args.candidate/item['path']
        if input_path.stat().st_size!=item['bytes'] or digest(input_path)!=item['sha256']:
            raise ValueError('Full boundary database differs from reviewed provenance')
        copy_checked(input_path,package/item['path'])
        for feature in read(input_path)['features']:
            names[feature['id']]=feature['properties'].get('name',feature['id'])
        license_path=Path(source['licenseFile'])
        copy_checked(args.licenses/license_path.name,package/license_path)
    parents=[{**r,'name':names.get(r['id'],r['id']),'parentName':names.get(r['parent'],r['parent'])}
             for r in candidate['hierarchyIssues']]
    geometry_audit={
        'schema':'ptb-public-boundary-geometry-audit/1','version':provenance['version'],
        'conclusion':provenance['geometryReview'],
        'coverage':{k:quality[k] for k in ['mainlandExpected','mainlandDirectMatches','crossRegionalCoverage','perProvince','unresolvedOfficialEntities']},
        'invalidUnsimplifiedGeometry':quality['invalidUnsimplifiedGeometry'],
        'duplicateDistrictGeometry':quality['duplicateDistrictGeometry'],
        'districtOverlaps':coast['pairs'],
        'parentGeometryDisagreements':parents,
        'parentChildAreaAudit':quality['hierarchyAreaAudit'],
        'geographicProbes':quality['geographicProbes'],
        'displaySimplification':read(args.candidate/'display-geometry-audit.json'),
        'method':coast['method'],'limitation':coast['limitation'],
    }
    write(package/'boundary-provenance.json',provenance)
    write(package/'geometry-audit.json',geometry_audit)
    (package/'README.md').write_text(
        '# 语言地图边界资料候选\n\n'
        '本包包含三份独立的完整边界派生数据库。OSM 数据适用 ODbL 1.0，'
        '国土测绘中心资料适用政府资料开放授权条款第 1 版，香港资料适用 DATA.GOV.HK 使用条款。'
        '各来源、原始版本、改动及校验值见 boundary-provenance.json，许可全文见 licenses/。\n\n'
        '几何差异清单见 geometry-audit.json。实体覆盖已逐项核对，面几何有效。'
        '海域直线、内陆来源矛盾及父子界线差异均保留，不能据此认定所有边界相互一致或具有法定效力。'
        '显示层按层级简化，完整数据库保留原面、小岛、洞及来源信息。\n\n'
        '本包不包含语言观测、距离矩阵、用户分类原表或个人修改，边界许可不改变这些独立资料的权利归属。'
        '当前为本地审阅候选，尚未公开发布。\n', encoding='utf8')

    audit=destination/'audit'
    for filename in ['candidate-audit.json','quality-audit.json','association-audit.json','mainland-audit.json',
                     'regional-audit.json','derivation-audit.json','display-geometry-audit.json',
                     'overlap-coast-audit.json','overlap-coast-review.png']:
        copy_checked(args.candidate/filename,audit/filename)
    with (audit/'overlap-pairs.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['a','aName','b','bName','overlapKm2','landOverlapKm2','marineOverlapKm2','landFraction'])
        writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in coast['pairs'])
    write(audit/'parent-disagreements.json',parents)
    major='\n'.join(f"|{r['aName']} / {r['bName']}|{r['overlapKm2']:.6f}|{r['landOverlapKm2']:.6f}|{r['marineOverlapKm2']:.6f}|" for r in coast['pairs'][:12])
    (audit/'review.md').write_text(
        '# 语言地图开放底图候选 5 审阅材料\n\n'
        '行政实体覆盖、有效几何、来源之间的边界差异分别验收。本候选实体覆盖齐全，'
        '完整面有效，仍保留明确列出的海域重叠、内陆矛盾和父子范围差异，尚未公开发布。\n\n'
        '- 民政部实时树基线读取于 2026-10-24，未声明代码生效年份。大陆应有 3211 项，'
        '3210 项直接同名同层级对应。金门由 NLSC 的实际县域几何覆盖，另列行政代码口径，未制造重复面。\n'
        '- 运行层 34 省、395 个地级显示单元、3244 个县级显示单元，数量包括各来源所需显示聚合，不能单凭总数认证完整。'
        '逐省应有、缺漏、重复、额外来源项见 quality-audit.json 和 mainland-audit.json。\n'
        '- 无效完整面 0，重复县级几何 0。显示简化丢失面部件 0、洞 0、无效面 0。'
        '完整数据库未简化，点关联使用完整面，未移动地点坐标。\n'
        '- 74 对正面积重叠全部与独立 Natural Earth 1:10m land 5.1.1 相交定位。20 对无陆地相交，'
        '54 对含陆地相交，其中 9 对陆地相交面积大于 1 平方公里。该数值仅用于排列审阅顺序，未作为放行阈值。'
        '广义海岸线不能判定细小岸线差异的法定归属。全部 74 对见 overlap-pairs.csv。\n'
        '- 最大海域重叠为阳西/吴川 2586.900515、苍南/福鼎 804.049939、江城/吴川 709.539817、'
        '徐闻/阳西 159.621802 平方公里，独立陆地相交均为 0。海上直线来自有效源几何，本次未为显示效果裁界。\n'
        '- 最大内陆矛盾为麦盖提/巴楚 117.630896 平方公里。关系 2748426 v16 与 2748427 v18 '
        '均于 2026-05-26 编辑，公开更改集描述为完善图木舒克市区划。未找到足以选定界线的权威矢量依据，原样保留。'
        '这两项源几何互相冲突，不能据此声称县域互斥或边界无误。\n'
        '- NLSC 的瑪家三和补充面按原始同一 TOWNCODE 合入瑪家。屏东县 111 年 03 月官方防灾图'
        '说明部分行政划分尚未完成，三和村涉及盐埔、內埔和长治地段。瑪家与長治 2.564165、內埔 1.927648、'
        '鹽埔 0.841904 平方公里陆地重叠保留。此说明支持保留源范围，不提供替代法定界线。\n'
        '- 父子几何不完全包含 276 项，其中 NLSC 254 项跨县市 2020 与乡镇 2023 版本，OSM 22 项。'
        '版本不同不能自动解释每项差异。雄安三县与保定显示父界范围不同，上海市与城区源范围涉及境外农场，'
        '其管理权不能自动认定法定辖境。全部条目和实际包含比例见 parent-disagreements.json。\n'
        '- 930 个原始地点保留，929 个唯一面关联，无多重强配；呼玛原点仍未定。柳州和九江县的唯一空间关联变化见 association-audit.json。'
        '原坐标、510 个特征、原分类值和五份矩阵未改。6 行分类的地区关联变化涉及金门、重庆撤并旧区，'
        '保留原表值与未定状态；重庆江北行原代码 330205 与省份不符也保留审计。\n\n'
        '|县级来源对|总重叠 km²|陆地相交 km²|海域相交 km²|\n|---|---:|---:|---:|\n'+major+
        '\n\n官方瑪家原件：[屏东县防灾地图](https://pteoc.pthg.gov.tw/PreventionUploads/PreventMap/2022/%E7%91%AA%E5%AE%B6%E9%84%89/7.%E7%91%AA%E5%AE%B6%E9%84%89%E9%98%B2%E7%81%BD%E5%9C%96.pdf)。'
        '独立海岸参考：[Natural Earth](https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip)。\n\n'
        '运行资料、公共来源元数据、三份完整数据库、许可及本地审计文件的大小和 SHA-256 见 freeze-manifest.json。'
        'runtime 仍仅有 16 个运行必需文件及清单。独立边界 ZIP 含原始派生几何和许可，未包含语言研究数据。\n',encoding='utf8')

    archive=destination/'geolinguistics-boundary-source-data-20261022-candidate-5.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(package.rglob('*')):
            if path.is_file():
                info=zipfile.ZipInfo(path.relative_to(package).as_posix(),date_time=(2026,10,24,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                z.writestr(info,path.read_bytes(),compresslevel=9)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Boundary ZIP integrity failed')
        for name in z.namelist():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest(package/name):
                raise ValueError('Boundary ZIP member identity changed')
    files=[{'path':p.relative_to(destination).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)}
           for p in sorted(destination.rglob('*')) if p.is_file()]
    freeze={'schema':'ptb-boundary-candidate-freeze/1','version':provenance['version'],
            'status':'frozen-local-review-candidate','runtimeFiles':len(DATA_FILES),
            'runtimeBytes':sum(f['bytes'] for f in manifest['files']),
            'maximumRuntimeFileBytes':max(f['bytes'] for f in manifest['files']),
            'boundaryArchive':archive.name,'files':files,'publication':'pending explicit review and approval'}
    write(destination/'freeze-manifest.json',freeze)
    print(json.dumps({k:v for k,v in freeze.items() if k!='files'},ensure_ascii=False,indent=2),flush=True)
    print('ZIP',archive.stat().st_size,digest(archive),flush=True)


if __name__=='__main__':
    main()
