"""Check the single-file artifact against the current registered resources."""
import hashlib
import json
from pathlib import Path
from PyInstaller.archive.readers import CArchiveReader

ROOT=Path(__file__).resolve().parents[1]


def main():
    artifact=ROOT/'dist/m10-recording/PhoneticToolbox-v3-M10-R4.exe'
    archive=CArchiveReader(str(artifact))
    names={name.replace('\\','/'):name for name in archive.toc}
    manifest=json.loads((ROOT/'contracts/resource-manifest.json').read_text('utf-8'))
    count=0
    for item in manifest['resources']:
        path=item['path']
        if not path.startswith(('frontend/public/vocal-tract/','resources/vocal_tract/')):continue
        target=path.replace('frontend/public/','frontend/dist/')
        assert target in names,'Missing packaged resource: '+target
        assert hashlib.sha256(archive.extract(names[target])).hexdigest()==item['sha256'],'Stale packaged resource: '+target
        count+=1
    for path in ['contracts/resource-manifest.json','docs/manual/vocal-tract.md','third_party/source-registry.json']:
        assert archive.extract(names[path])==(ROOT/path).read_bytes(),'Stale package metadata: '+path
    assert not any('phonetic_toolbox/' in name or name.lower().endswith('/icuuc.dll') or name.lower()=='icuuc.dll' for name in names)
    result={'artifact':str(artifact),'bytes':artifact.stat().st_size,'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'm10_resources_checked':count,'archive_entries':len(names),'status':'passed'}
    out=ROOT/'output/validation/m10/bundle.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
