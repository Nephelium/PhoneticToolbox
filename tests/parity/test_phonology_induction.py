"""M14 independent V2 data/content baseline, not self-generated expected values."""
import dataclasses
import io
import json
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import pytest
from openpyxl import load_workbook
from ptb_worker.m14_import import load
from phonetic_core.transcription.phonology import PhonologyRules
from phonetic_core.transcription.phonology.export import export

ROOT=Path(__file__).resolve().parents[2]
FIX=ROOT/'tests/fixtures/m14'
BASE=json.loads((FIX/'v2-baseline.json').read_text(encoding='utf-8'))


def structure(name,raw):
    if name.endswith('.docx'):
        with ZipFile(io.BytesIO(raw)) as z:
            root=ET.fromstring(z.read('word/document.xml'))
            return [''.join(p.itertext()) for p in root.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')]
    return [[str(c.value) if c.value is not None else '' for c in row] for row in load_workbook(io.BytesIO(raw),rich_text=True).active]


@pytest.mark.parametrize('suffix',['xlsx','xls','csv','txt','tsv'])
@pytest.mark.parametrize('skip',[False,True])
@pytest.mark.parametrize('zero',[False,True])
def test_v2_data_and_outputs(suffix,skip,zero):
    name='public.'+suffix; expected=BASE['cases'][name+':'+str(skip)]
    rows,report=load((FIX/name).read_bytes(),name,skip)
    assert [dataclasses.asdict(r) for r in rows]==expected['rows']
    assert report['accepted_rows']==len(rows) and report['duplicate_rows']==1
    r=PhonologyRules(); analysis=r.analyze(rows,zero); e=expected['policies'][str(zero)]
    assert dataclasses.asdict(analysis)==e['analysis']
    assert dataclasses.asdict(r.apply_symbol_aliases(analysis,{'pʰ':'p','p':'m'},{'ã':'a'}))==e['aliases']
    outputs=export(analysis,e['tone_map'],e['tones'],e['initials'],e['finals'])
    assert {n:structure(n,b) for n,b in outputs.items()}==e['outputs']


@pytest.mark.parametrize('name,code',[('empty.txt','m14_empty_input'),('missing.tsv','m14_no_valid_rows'),('broken.xlsx','m14_decode_failed')])
def test_invalid_import(name,code):
    with pytest.raises(ValueError,match=code):load((FIX/name).read_bytes(),name)
