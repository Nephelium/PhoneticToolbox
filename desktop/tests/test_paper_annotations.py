import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from ptb_desktop.papers import PaperError, PaperService


@pytest.fixture
def service(tmp_path):
    s = PaperService(tmp_path / 'profile')
    assets = {}
    for i, language in enumerate(('original', 'translation')):
        path = tmp_path / (language + '.pdf')
        writer = PdfWriter(); writer.add_blank_page(width=200+i*10, height=300)
        with path.open('wb') as f: writer.write(f)
        data = path.read_bytes(); sha = hashlib.sha256(data).hexdigest()
        assets[language] = {'size':len(data),'sha256':sha,'path':language+'.pdf'}
        s._path(assets[language]).write_bytes(data)
    s.catalog = {'papers':[{'id':'test',**assets}]}
    return s


def annotation(**changes):
    return {'id':'one','page':1,'kind':'highlight','color':'yellow','text':'中文批注','quote':'selected words','rects':[[.1,.2,.3,.04]],**changes}


def test_persistent_annotations_language_isolation_and_conflict(service):
    result = service.save_annotation('test','original',0,annotation(),False)
    reopened = PaperService(service.root); reopened.catalog=service.catalog
    assert reopened.annotations('test','original') == result
    assert reopened.annotations('test','translation') == {'revision':0,'items':[]}
    with pytest.raises(PaperError,match='另一窗口'):
        reopened.save_annotation('test','original',0,annotation(text='stale'),False)
    assert reopened.annotations('test','original')['items'][0]['text']=='中文批注'
    service.save_annotation('test','original',1,annotation(text='edited'),False)
    assert service.annotations('test','original')['items'][0]['text']=='edited'
    assert service.save_annotation('test','original',2,annotation(),True)['items']==[]


@pytest.mark.parametrize('change',[{'rects':[[float('nan'),0,.2,.2]]},{'rects':[[.9,.1,.3,.2]]},{'page':True},{'kind':'javascript'},{'text':'x'*5001},{'id':'../bad'}])
def test_invalid_annotation_preserves_existing(service,change):
    service.save_annotation('test','original',0,annotation(),False)
    before = service.annotations('test','original')
    with pytest.raises(PaperError): service.save_annotation('test','original',1,annotation(**change),False)
    assert service.annotations('test','original')==before


def test_corrupt_annotation_not_silently_overwritten(service):
    asset=service.asset('test','original'); path=service.root/('annotations-'+asset['sha256']+'.json')
    path.write_text('broken','utf8')
    with pytest.raises(PaperError,match='原记录已保留'):service.save_annotation('test','original',0,annotation(),False)
    assert path.read_text('utf8')=='broken'


def test_export_original_identity_and_standard_annotations(service,tmp_path):
    asset=service.asset('test','original'); before=service._path(asset).read_bytes()
    service.save_annotation('test','original',0,annotation(),False)
    service.save_annotation('test','original',1,annotation(id='two',kind='note'),False)
    plain=tmp_path/'plain.pdf'; marked=tmp_path/'marked.pdf'
    service.export('test','original',False,plain);service.export('test','original',True,marked)
    assert plain.read_bytes()==before==service._path(asset).read_bytes()
    pdf=PdfReader(marked); page=pdf.pages[0]; entries=[a.get_object() for a in page['/Annots']]
    assert len(pdf.pages)==1
    assert [a['/Subtype'] for a in entries]==['/Highlight','/Text']
    # Top-origin normalized rectangle [0.1,0.2,0.3,0.04] maps to PDF bottom-origin points.
    assert list(entries[0]['/QuadPoints'])==pytest.approx([20,240,80,240,20,228,80,228])
    assert entries[1]['/Contents'].startswith('中文批注')
    with pytest.raises(PaperError,match='不能覆盖'): service.export('test','original',False,service._path(asset))


def test_failed_export_keeps_existing_destination(service,tmp_path):
    target=tmp_path/'keep.pdf';target.write_bytes(b'existing')
    # Corrupt annotation state must stop an annotated export before replacement.
    asset=service.asset('test','original')
    (service.root/('annotations-'+asset['sha256']+'.json')).write_text('{}','utf8')
    with pytest.raises(PaperError):service.export('test','original',True,target)
    assert target.read_bytes()==b'existing'
    assert not list(tmp_path.glob('*.tmp'))
