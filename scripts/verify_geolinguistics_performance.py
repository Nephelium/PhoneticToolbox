"""Owned real-data M19 performance observation; test-only OL prototype overlay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
from package_geolinguistics_data import package_data
from verification_artifacts import verification_run

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--frontend-dist',type=Path,help='Reuse a previously frozen observer build without rebuilding changed product sources')
    parser.add_argument('--expect-cached',action='store_true',help='Require one first projection, zero repeated projections and released removed map refs')
    parser.add_argument('--renderer-only',action='store_true',help='Only check live layer counts, rendering policy and exports; no geographic content audit')
    args=parser.parse_args()
    with verification_run('geolinguistics-renderer' if args.renderer_only else 'geolinguistics-performance','chrome',
        'Independent production build + readFeatures prototype timing overlay; real manifest import; no altered result or product source') as (out,scratch):
        print(out,flush=True)
        package_data(args.source,scratch/'geolinguistics-data')
        assert (args.source/'data-manifest.json').read_bytes()==(scratch/'geolinguistics-data/data-manifest.json').read_bytes()
        if args.expect_cached:
            fixture=scratch/'generation-fixture'
            shutil.copytree(scratch/'geolinguistics-data',fixture)
            world=fixture/'maps/world.json';geo=json.loads(world.read_text('utf-8'))
            geo['features'].append({'type':'Feature','id':'qa-generation-marker','properties':{'name':'QA synthetic generation marker'},
                'geometry':{'type':'Polygon','coordinates':[[[130,20],[131,20],[130.5,21],[130,20]]]}})
            world.write_text(json.dumps(geo,separators=(',',':')),encoding='utf-8')
            manifest=json.loads((fixture/'data-manifest.json').read_text('utf-8'))
            entry=next(f for f in manifest['files'] if f['path']=='maps/world.json')
            entry.update(bytes=world.stat().st_size,sha256=hashlib.sha256(world.read_bytes()).hexdigest())
            (fixture/'data-manifest.json').write_text(json.dumps(manifest,separators=(',',':')),encoding='utf-8')
            (out/'generation-fixture.json').write_text(json.dumps({'synthetic':True,'changed':'maps/world.json only; added one triangle, original 16-file source remains untouched',
                'worldSha256':entry['sha256'],'features':len(geo['features'])},indent=2),encoding='utf-8')
        (out/'inputs.json').write_text(json.dumps({'source':str(args.source.resolve()),
            'manifestSha256':hashlib.sha256((args.source/'data-manifest.json').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
        commands=[]
        if args.frontend_dist:
            assert (args.frontend_dist/'index.html').is_file()
            (out/'reused-build.json').write_text(json.dumps({'frontend':str(args.frontend_dist.resolve()),
                'indexSha256':hashlib.sha256((args.frontend_dist/'index.html').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
        else:
            commands.append(('build',['node',str(ROOT/'tests/e2e/geolinguistics-performance-build.mjs'),str(out)]))
        commands.append(('browser',['node',str(ROOT/'tests/e2e'/('geolinguistics-renderer.cjs' if args.renderer_only else 'geolinguistics-performance.cjs')),str(out),str(scratch)]))
        for name,argv in commands:
            env=dict(os.environ)
            env['PTB_GEO_EXPECT_CACHED']='1' if args.expect_cached else '0'
            env['PTB_GEO_RENDERER_ONLY']='1' if args.renderer_only else '0'
            if args.frontend_dist:env['PTB_GEO_PERF_DIST']=str(args.frontend_dist.resolve())
            result=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            (out/(name+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
            if result.returncode:
                raise RuntimeError(name+' failed: '+str(out))
            if name=='build':
                args.frontend_dist=Path(json.loads((out/'build.json').read_text('utf-8'))['frontend'])

if __name__=='__main__':main()
