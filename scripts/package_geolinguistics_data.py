"""Create a private, standalone data folder next to a local trial EXE.

The folder is copied once, with an exhaustive hash manifest. No source deletion
or public upload is performed. Existing destination folders are not overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

DATA_FILES = (
    'catalog.json', 'sites.json', 'observations.json', 'classification.json',
    'map-index.json', 'paper.json',
    'main.f64', 'type-0-0.f64', 'type-0-1.f64', 'type-1-0.f64', 'type-1-1.f64',
    'maps/world.json', 'maps/territory.json', 'maps/province.json',
    'maps/city.json', 'maps/district.json',
)
BOUNDARY_BINDINGS=('sites.json','classification.json','map-index.json',*(n for n in DATA_FILES if n.startswith('maps/')))

def data_manifest(source:Path):
    """Hash only the public-facing runtime surface, never private book/audit files."""
    source=source.resolve()
    files=[]
    for name in DATA_FILES:
        path=source/name
        if any(p.is_symlink() or (hasattr(p,'is_junction') and p.is_junction()) for p in [path,*path.parents] if p!=source and p.is_relative_to(source)) or not path.resolve().is_relative_to(source):raise ValueError('Linked data input')
        raw=path.read_bytes()
        files.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    manifest={'schema':'ptb-geolinguistics-data/1','version':'m19-local-20261022','profile':'interactive-data','distribution':'private local trial authorized by source owner','files':files}
    previous=source/'data-manifest.json'
    if previous.is_file():
        old=json.loads(previous.read_text('utf-8'))
        provenance=old.get('boundaries')
        if provenance is not None:
            if not isinstance(provenance,dict) or provenance.get('schema')!='ptb-geolinguistics-boundary-provenance/1':
                raise ValueError('Invalid boundary provenance')
            prior={item['path']:item for item in old.get('files',[])}
            current={item['path']:item for item in files}
            if any(prior.get(name)!=current[name] for name in BOUNDARY_BINDINGS):
                raise ValueError('Boundary or association inputs changed; rebuild their provenance before refreshing the manifest')
            authority=Path(__file__).resolve().parents[1]/'resources/research/geolinguistics-public-boundaries.json'
            if authority.is_file():
                revised=json.loads(authority.read_text('utf-8'))
                if revised.get('schema')==provenance['schema'] and revised.get('version')==provenance.get('version'):
                    provenance=revised
            manifest['boundaries']=provenance
            manifest['distribution']=old.get('distribution','licensed boundary candidate; publication requires review')
    return manifest

def write_data_manifest(source:Path):
    manifest=data_manifest(source)
    target=source/'data-manifest.json'
    staged=target.with_suffix('.json.tmp')
    staged.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    staged.replace(target)
    return manifest

def package_data(source:Path,destination:Path):
    source=source.resolve();destination=destination.resolve()
    if destination.exists():raise ValueError('Destination already exists; choose a new trial directory')
    if destination.is_relative_to(source):raise ValueError('Destination must be outside the source data')
    manifest=data_manifest(source);files=manifest['files']
    # Explicit runtime inputs only: original book scans and private build audits stay local.
    for item in files:
        target=destination/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/item['path'],target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Copied data identity changed')
    manifest_source=source/'data-manifest.json'
    if manifest_source.is_file() and json.loads(manifest_source.read_text('utf-8'))==manifest:
        # Preserve the reviewed manifest identity, including its original formatting.
        shutil.copyfile(manifest_source,destination/'data-manifest.json')
    else:
        (destination/'data-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'directory':str(destination),'files':len(files),'bytes':sum(f['bytes'] for f in files)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--destination',required=True,type=Path)
    args=p.parse_args()
    print(json.dumps(package_data(args.source,args.destination),ensure_ascii=False))

if __name__=='__main__':main()
