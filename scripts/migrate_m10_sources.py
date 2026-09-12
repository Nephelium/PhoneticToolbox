"""One-time, non-overwriting M10 migration with source hashes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    def copy(source, target, rewrite=None):
        source, target = ROOT/source, ROOT/target
        data = source.read_bytes()
        if rewrite:
            data = rewrite(data.decode('utf-8')).encode('utf-8')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        target.write_bytes(data)
        files.append({'source':source.relative_to(ROOT).as_posix(), 'target':target.relative_to(ROOT).as_posix(),
                      'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'migration_sha256':hashlib.sha256(data).hexdigest()})
    def rewrite(text):
        return text.replace('phonetic_toolbox.core.vocal_tract','phonetic_core.vocal_tract').replace('phonetic_toolbox.models.vocal_source','phonetic_core.vocal_tract.source_models')
    for p in (ROOT/'phonetic_toolbox/core/vocal_tract').glob('*.py'):
        copy(p.relative_to(ROOT), Path('packages/phonetic_core/src/phonetic_core/vocal_tract')/p.name, rewrite)
    copy('phonetic_toolbox/models/vocal_source.py','packages/phonetic_core/src/phonetic_core/vocal_tract/source_models.py')
    copy('phonetic_toolbox/services/vocal_tract/animation.py','packages/phonetic_core/src/phonetic_core/vocal_tract/animation.py',rewrite)
    for name in ['audio_output.py','profile.py','process_guard.py']:
        copy('phonetic_toolbox/services/vocal_tract/'+name,'desktop/src/ptb_desktop/vocal_tract/'+name,rewrite)
    for p in (ROOT/'phonetic_toolbox/gui/resources/vocal_tract').rglob('*'):
        if p.is_file():copy(p.relative_to(ROOT),Path('frontend/public/vocal-tract')/p.relative_to(ROOT/'phonetic_toolbox/gui/resources/vocal_tract'))
    for p in (ROOT/'phonetic_toolbox/resources/vocal_tract').rglob('*'):
        if p.is_file():copy(p.relative_to(ROOT),Path('resources/vocal_tract')/p.relative_to(ROOT/'phonetic_toolbox/resources/vocal_tract'))
    target=ROOT/'docs/modules/evidence/M10-migration.json'
    target.write_text(json.dumps({'version':1,'files':files},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Migrated {len(files)} files without overwriting old sources.')


if __name__ == '__main__':main()
