"""Register selected original artwork and runtime viewports, without redrawing it."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/Users/13680/.codex/generated_images/01a12901-80a6-7413-8040-676c619088ae')
TARGET = ROOT / 'frontend/public/app-icons'
SHEETS = {
    'k': ('f55d91eb-646c-4dea-a76b-da20c914cf5f', [[69, 43, 525, 523]]),
    'm': ('20df6156-4e81-41fd-9eaf-682495066169', [[99, 31, 531, 510]]),
    'r': ('3b1bb6c6-7487-4564-ad1d-2205b14bff12', [[180, 35, 484, 471]]),
    'v': ('f44be893-37a5-40d7-8267-193c680765d3', [[690, 52, 480, 466], [143, 603, 480, 465]]),
    'ipa-ad': ('793bbc69-3bec-48ff-8479-bc6e24e79e48', [[74,40,532,523],[648,40,532,523],[74,627,532,524],[648,627,532,524]]),
    'ipa-eh': ('f0df7683-1735-4032-9809-1af67017e5ef', [[70,20,527,524],[655,20,529,524],[70,623,527,532],[655,623,530,532]]),
    'ipa-il': ('ff7d0700-9044-4c82-87d6-39aa0a1bd476', [[81,42,521,512],[650,42,522,512],[81,633,521,517],[650,631,522,519]]),
    'ipa-mp': ('a11116d9-a8c3-4b81-8803-bbee1e413be4', [[64,53,549,550],[642,53,550,550],[64,632,549,543],[642,632,550,543]]),
    'ipa-qt': ('1714924f-84e1-47da-86a4-931d9a3bfd84', [[83,41,525,519],[645,41,526,519],[83,628,525,517],[645,628,526,517]]),
    'w': ('bc555c24-1325-4a2e-9392-9922c6cd47f9', [[74,53,527,496],[655,52,527,497],[74,631,527,502],[655,631,527,502]]),
    'w1-edited': ('1edf2cd1-0646-41f5-9377-2595ce0fe386', [[53,53,1148,1148]]),
    'x': ('2922fd02-6a57-45d2-b4af-ac810acd7261', [[45,31,559,525],[651,31,559,525],[45,618,559,528],[651,618,559,528]]),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    checking = parser.parse_args().check
    (TARGET / 'sheets').mkdir(parents=True, exist_ok=True)
    provenance = []
    for key, (generation, _) in SHEETS.items():
        source_root = Path('C:/Users/13680/.codex/generated_images/01a129cb-d9b1-75a1-b957-4e7704d80e42') if key == 'w1-edited' else SOURCE
        source = source_root / ('exec-' + generation + '.png')
        target = TARGET / 'sheets' / (key + '.png')
        if not checking:
            shutil.copyfile(source, target)
        provenance.append(dict(sheet=key, generation=generation, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    icons = []

    def add(sheet, index, group, original, identity=None):
        number = len(icons) + 1
        icons.append(dict(id=identity or ('default' if number == 1 else f'builtin-{number:03}'), number=number, group=group, original=original, sheet=sheet + '.png', rect=SHEETS[sheet][1][index], radius=.18))

    add('w1-edited', 0, 'waveform', 'W1（雾紫声波、深蓝末端）')
    add('k', 0, 'waveform', 'K1-A', 'builtin-001')
    for sheet in ('w', 'x'):
        for i in range(1 if sheet == 'w' else 0, 4):
            add(sheet, i, 'waveform', sheet.upper() + str(i + 1))
    add('m', 0, 'spectrogram', 'M3-A')
    add('r', 0, 'mouth', 'R1-A')
    add('v', 0, 'annotation', 'V4-B')
    add('v', 1, 'annotation', 'V4-C')
    for block, sheet in enumerate(('ipa-ad', 'ipa-eh', 'ipa-il', 'ipa-mp', 'ipa-qt')):
        for i in range(4):
            add(sheet, i, 'ipa', 'IPA-' + chr(65 + block * 4 + i))
    # Remove the user-selected second and eighth choices without renaming persisted IDs.
    icons = [icon for icon in icons if icon['id'] not in {'builtin-001', 'builtin-008'}]
    for number, icon in enumerate(icons, 1):
        icon['number'] = number
    catalog = dict(schema='ptb-app-icons/1', default='default', groups=[dict(id=k, name=v) for k,v in [('waveform','波形'),('spectrogram','语谱'),('mouth','口形'),('annotation','标注'),('ipa','IPA')]], icons=icons)
    catalog_text = json.dumps(catalog, ensure_ascii=False, indent=2) + '\n'
    provenance_text = json.dumps(dict(generator='OpenAI built-in image generation', selected_by='User selection in this task', manipulation='Original sheets preserved; runtime viewports exclude printed design labels and clip rounded tile corners; W1 has a separate user-requested built-in image edit recoloring the body muted purple and terminal deep navy, selected as the default; choices K1-A and X3 retired while all surviving preference IDs remain stable', edit=dict(target='W1', source_sheet='w.png', generation='1edf2cd1-0646-41f5-9377-2595ce0fe386', prompt='Preserve W1 silhouette, eyes, smile and textured yellow tile. Recolor body muted purple #8061A5 and far-right terminal deep navy #173A72; cream outer corners clipped at runtime. User approved on 2026-10-12.'), sheets=provenance), ensure_ascii=False, indent=2) + '\n'
    generated = ROOT / 'frontend/src/generated/application-icons.json'
    outputs = [(TARGET / 'catalog.json', catalog_text), (generated, catalog_text), (TARGET / 'provenance.json', provenance_text)]
    for path, content in outputs:
        if checking:
            assert path.read_text(encoding='utf-8') == content, f'Icon catalog/provenance drift: {path}'
        else:
            path.write_text(content, encoding='utf-8')
    manifest_path = ROOT / 'contracts/resource-manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    registered = {entry['path']: entry for entry in manifest['resources']}
    for path in sorted(TARGET.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if checking:
            assert registered.get(relative, {}).get('sha256') == digest, f'Icon resource registration drift: {relative}'
        else:
            if relative not in registered:
                registered[relative] = dict(path=relative, source_id='ASSET-MASCOT')
                manifest['resources'].append(registered[relative])
            registered[relative]['sha256'] = digest
    if not checking:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Registered {len(icons)} choices in contiguous groups; default edited W1')


if __name__ == '__main__':
    main()
