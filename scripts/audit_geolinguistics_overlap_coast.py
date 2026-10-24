"""Locate candidate overlap pairs against an independent Natural Earth land mask.

This classifies evidence for review; it never clips or repairs administrative
boundaries. Natural Earth 1:10m is a separate generalized coastal reference,
not cadastral evidence for a disputed administrative line.
"""
import argparse
import io
import json
import math
from pathlib import Path
import sys
import zipfile


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--land',type=Path,required=True)
    parser.add_argument('--tools',type=Path,action='append',default=[])
    args=parser.parse_args()
    for tool in args.tools:sys.path.insert(0,str(tool.resolve()))
    import shapefile
    from shapely.geometry import shape,mapping,box
    from shapely.ops import unary_union,transform
    from pyproj import Transformer
    candidate=args.candidate.resolve()
    features=[]
    for path in sorted((candidate/'boundary-databases').glob('*.geojson')):features.extend(load(path)['features'])
    by_id={f['id']:f for f in features}
    pairs=load(candidate/'quality-audit.json')['districtOverlaps']
    ids={r[k] for r in pairs for k in ['a','b']}
    geometries={key:shape(by_id[key]['geometry']) for key in ids}
    z=zipfile.ZipFile(args.land)
    reader=shapefile.Reader(shp=io.BytesIO(z.read('ne_10m_land.shp')),shx=io.BytesIO(z.read('ne_10m_land.shx')),
                            dbf=io.BytesIO(z.read('ne_10m_land.dbf')))
    frame=box(70,0,140,60)
    coast=[]
    for row in reader.shapes():
        geometry=shape(row.__geo_interface__)
        if geometry.intersects(frame):coast.append(geometry.intersection(frame))
    land=unary_union(coast)
    if not land.is_valid:raise ValueError('Independent land mask invalid; no silent repair')
    area_projection=Transformer.from_crs(4326,6933,always_xy=True).transform
    area=lambda geometry:transform(area_projection,geometry).area/1e6
    reviewed=[]
    intersections={}
    for pair in pairs:
        overlap=geometries[pair['a']].intersection(geometries[pair['b']])
        inland=overlap.intersection(land)
        sea=overlap.difference(land)
        row={**pair,'overlapKm2':area(overlap),'landOverlapKm2':area(inland),'marineOverlapKm2':area(sea),
             'landFraction':inland.area/overlap.area,'overlapBounds':list(overlap.bounds),
             'coastalEvidence':'Natural Earth 1:10m land v5.1.1; generalized independent reference, not statutory boundary proof'}
        reviewed.append(row)
        intersections[(pair['a'],pair['b'])]=overlap
    report={'schema':'ptb-boundary-overlap-coast-audit/1','status':'classified-for-review','landSource':load(args.land.with_suffix(args.land.suffix+'.receipt.json')),
            'landVersion':z.read('ne_10m_land.VERSION.txt').decode('utf8').strip() if 'ne_10m_land.VERSION.txt' in z.namelist() else '5.1.1',
            'pairs':sorted(reviewed,key=lambda r:r['landOverlapKm2'],reverse=True),
            'administrativeGeometryChanges':0,'method':'Exact candidate overlap intersected with independent land mask; EPSG:6933 areas in km2.',
            'limitation':'Generalized coastline cannot resolve small coastal slivers. Inland overlaps are direct evidence of a source disagreement, not an instruction to select a winner.'}
    (candidate/'overlap-coast-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report['pairs'][:12],ensure_ascii=False,indent=2),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from shapely.plotting import plot_polygon,plot_line
    plt.rcParams['font.family']=['Microsoft YaHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    selected=sorted(reviewed,key=lambda r:r['overlapKm2'],reverse=True)[:4]
    for a,b in [('district:653127','district:653130'),('district:TW-10013060','district:TW-10013280')]:
        found=next(r for r in reviewed if {r['a'],r['b']}=={a,b})
        if found not in selected:selected.append(found)
    fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained')
    def draw_polygon(ax,geometry,**kwargs):
        if geometry.is_empty:return
        if geometry.geom_type=='Polygon':plot_polygon(geometry,ax=ax,add_points=False,**kwargs)
        elif geometry.geom_type in {'MultiPolygon','GeometryCollection'}:
            for part in geometry.geoms:draw_polygon(ax,part,**kwargs)
    for ax,row in zip(axes.flat,selected):
        overlap=intersections[(row['a'],row['b'])]
        x0,y0,x1,y1=overlap.bounds
        padding=max(.02,max(x1-x0,y1-y0)*.18)
        viewport=box(x0-padding,y0-padding,x1+padding,y1+padding)
        ax.set_facecolor('#e7f1f5')
        draw_polygon(ax,land.intersection(viewport),facecolor='#e3dfd4',edgecolor='#9d978b',linewidth=.8)
        for key,color in [(row['a'],'#1565a7'),(row['b'],'#22854b')]:
            plot_line(geometries[key].boundary.intersection(viewport),ax=ax,add_points=False,color=color,linewidth=1.3)
        draw_polygon(ax,overlap,facecolor='#c8444455',edgecolor='#c84444',linewidth=.6,hatch='///')
        ax.set_xlim(x0-padding,x1+padding);ax.set_ylim(y0-padding,y1+padding)
        ax.set_aspect(1/math.cos(math.radians((y0+y1)/2)))
        ax.set_title(row['aName']+' / '+row['bName']+'\n'+f"重叠 {row['overlapKm2']:.2f} km²，陆地参考内 {row['landOverlapKm2']:.2f} km²",fontsize=11)
        ax.tick_params(labelsize=8);ax.set_xlabel('经度',fontsize=9);ax.set_ylabel('纬度',fontsize=9)
        ax.grid(alpha=.18)
    fig.suptitle('行政边界候选的海陆重叠定位（原始完整几何未作裁切修正）',fontsize=15)
    fig.legend(handles=[Patch(facecolor='#e3dfd4',label='Natural Earth 陆地'),Patch(facecolor='#c8444455',edgecolor='#c84444',hatch='///',label='两面重叠')],loc='outside lower left',ncols=2,frameon=False)
    fig.text(.99,.005,'© OpenStreetMap contributors (ODbL 1.0) · 内政部国土测绘中心 (政府资料开放授权条款第1版)\nNatural Earth 1:10m v5.1.1 (public domain); generalized coast does not determine statutory administrative boundaries.',ha='right',fontsize=7)
    fig.savefig(candidate/'overlap-coast-review.png',dpi=170)
    plt.close(fig)
    print('Saved overlap-coast-audit.json and overlap-coast-review.png',flush=True)


if __name__=='__main__':main()
