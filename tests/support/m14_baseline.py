"""Independent V2 capture. Run in the original environment, with -B; no V3 imports."""
import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET


def structure(path):
    with ZipFile(path) as z:
        if path.suffix == '.docx':
            root = ET.fromstring(z.read('word/document.xml'))
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            return [''.join(p.itertext()) for p in root.findall('.//w:p', ns)]
    from openpyxl import load_workbook
    ws = load_workbook(path, rich_text=True).active
    return [[str(c.value) if c.value is not None else '' for c in row] for row in ws]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--v2', required=True); p.add_argument('--input', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); root = Path(a.v2); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    from phonetic_toolbox.services.phonology_service import PhonologyInductionService
    service = PhonologyInductionService(); cases = {}
    for source in sorted(Path(a.input).glob('*')):
        if source.suffix not in ('.csv', '.txt', '.tsv', '.xlsx', '.xls'): continue
        for skip in (False, True):
            key = source.name + ':' + str(skip)
            try:
                rows = service.load_rows(str(source), skip_first_row=skip)
                cases[key] = {'rows': [dataclasses.asdict(r) for r in rows], 'policies': {}}
                for zero in (False, True):
                    analysis = service.analyze(rows, zero)
                    folder = out / (source.name + '-' + str(skip) + '-' + str(zero))
                    tone_map = {v: ('合调' if v in ('35', '55') else v) for v in analysis.unique_tones}
                    tones = list(reversed(analysis.unique_tones)); initials = list(reversed(analysis.unique_initials))
                    finals = sorted([v for v in analysis.unique_finals if v], key=lambda x:(x[0],len(x),x)) + ([''] if '' in analysis.unique_finals else [])
                    result = service.export_outputs(analysis,tone_map,tones,initials,finals,str(folder))
                    aliases = service.apply_symbol_aliases(analysis, {'pʰ':'p','p':'m'}, {'ã':'a'})
                    cases[key]['policies'][str(zero)] = dict(analysis=dataclasses.asdict(analysis),aliases=dataclasses.asdict(aliases),tone_map=tone_map,tones=tones,initials=initials,finals=finals,outputs={Path(v).name:structure(Path(v)) for v in dataclasses.asdict(result).values()})
            except Exception as e: cases[key] = {'error': type(e).__name__, 'message': str(e)}
    paths = ['phonetic_toolbox/core/transcription/phonology_induction.py','phonetic_toolbox/core/synthesis/klatt/input_parser.py','phonetic_toolbox/services/phonology_service.py','phonetic_toolbox/models/phonology_models.py','phonetic_toolbox/gui/widgets/phonology_induction_widget.py','Phonetic_Export/index.html']
    data = dict(producer='independent-v2-process',python=sys.executable,sources={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in paths},cases=cases)
    (out/'baseline.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(cases),'errors':{k:v for k,v in cases.items() if 'error' in v}},ensure_ascii=False))


if __name__ == '__main__': main()
