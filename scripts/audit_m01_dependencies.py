"""Register the actual isolated M01 environment and inherited migration evidence."""
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parents[1]
RUNTIME={'numpy','scipy','praat-parselmouth','pandas','python-dateutil','six','pytz','tzdata'}


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,value): p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    registry_path=ROOT/'third_party/source-registry.json'
    registry=json.loads(registry_path.read_text('utf-8'))
    sources={s['id']:s for s in registry['sources']}
    old=json.loads((ROOT/'docs/testing/m01-environment-audit.json').read_text('utf-8'))['reference_packages']
    inventory=[]
    for d in sorted(metadata.distributions(),key=lambda d:d.metadata['Name'].lower()):
        name=d.metadata['Name']; normalized=name.lower().replace('_','-')
        if normalized=='phonetic-core': continue
        sid='M01-PY-'+normalized.upper()
        active=[r for r in d.requires or [] if not Requirement(r).marker or Requirement(r).marker.evaluate({'extra':''})]
        license_value=d.metadata.get('License-Expression') or d.metadata.get('License') or 'unknown'
        license_summary=license_value if len(license_value)<180 else 'See installed distribution license texts; full review pending'
        license_files=[]
        for relative in d.files or []:
            if any(word in str(relative).lower() for word in ['license','copying','notice']) and d.locate_file(relative).is_file():
                license_files.append({'path':str(relative).replace('\\','/'),'sha256':sha(d.locate_file(relative))})
        row={'name':name,'version':d.version,'scope':'science-runtime' if normalized in RUNTIME else 'test-build-only',
             'active_requires':active,'license':license_summary,'license_files':license_files,
             'project_urls':d.metadata.get_all('Project-URL') or [],'source_id':sid}
        if normalized in RUNTIME:
            row['reference_version']=old[normalized]['version']
            assert row['reference_version']==d.version
        inventory.append(row)
        sources[sid]={'id':sid,'title':name,'kind':'dependency' if normalized in RUNTIME else 'test-build-dependency',
            'modules':['M01'],'authors':d.metadata.get('Author') or d.metadata.get('Author-email') or 'Not recorded in installed metadata',
            'actual_included_version':d.version,'license':license_summary,
            'urls':{'release':f'https://pypi.org/project/{name}/{d.version}/'},
            'verification_date':'2026-09-09','distribution_status':'local-development-locked; redistribution review pending',
            'acknowledgement_group':'software','used_at':['requirements-m01-science.lock' if normalized in RUNTIME else 'requirements-m01-test.lock'],
            'relationship':'Unmodified dependency installed from PyPI wheels in project-local M01 environments; no global or v2 changes',
            'evidence':'third_party/m01-dependency-inventory.json'}
    dump(ROOT/'third_party/m01-dependency-inventory.json',{'platform':'Windows x86_64','python':'3.11.14',
        'runtime_package_count':len(RUNTIME),'packages':inventory,'cross_platform_verified':False})
    migration_path=ROOT/'docs/modules/evidence/M01-core-migration.json'
    migration=json.loads(migration_path.read_text('utf-8'))
    for row in migration['files']:
        source=(ROOT/row['source']);target=(ROOT/row['target'])
        assert sha(source)==row['source_sha256']
        row['target_sha256']=sha(target)
        old_tree=ast.parse(source.read_text('utf-8'));new_tree=ast.parse(target.read_text('utf-8'))
        before={n.name:ast.dump(n,include_attributes=False) for n in old_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        row['unchanged_function_or_class_ast']=[n.name for n in new_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and before.get(n.name)==ast.dump(n,include_attributes=False)]
        row['adaptation']='Project-owned migration; unchanged AST list identifies exact function bodies; remaining changes separate I/O, imports, per-call ports and snapshots. Source IDs retain their registered relationship; not a per-line upstream copying claim.'
        for sid in row['source_ids']:
            record=sources[sid]
            paths=record.setdefault('used_at',[])
            if row['target'] not in paths: paths.append(row['target'])
            record['m01_migration_evidence']='docs/modules/evidence/M01-core-migration.json'
    migration['resources']=[{'path':'packages/phonetic_core/src/phonetic_core/acoustic/data/Sinc_hash_1000.mat',
        'sha256':'e3e2fb01d67f722b7f13c1c9a0f4d62559a1ece77167343dfa42de7203f4d860','source_id':'SRC-IRAPT'}]
    dump(migration_path,migration)
    registry['sources']=list(sources.values());registry['total_records']=len(sources)
    dump(registry_path,registry)
    print(json.dumps({'packages':len(inventory),'runtime':len(RUNTIME),'sources':len(sources),'migration_files':len(migration['files'])}))


if __name__=='__main__': main()
