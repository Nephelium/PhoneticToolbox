"""M18 DOI extension preserves legacy catalogs, downloads and annotation IDs."""
import copy
from datetime import date
import hashlib
import json
import threading
import pytest
from ptb_documents.papers import PaperService, PaperError, validate_catalog, catalog_papers
from test_papers import paper, Opener, BODY


def journal(day='2026-10-21'):
    p=paper(day,'journal-tapa')
    p.update(sourceUrl='https://doi.org/10.3758/s13428-026-03174-y',submittedAt='2025-09-12',
             version='Version of Record, 2026-09-09',
             journal=dict(title='Behavior Research Methods',volume='58',articleNumber='289',
                          publishedAt='2026-09-09',doi='10.3758/s13428-026-03174-y'))
    for index,kind in enumerate(('original','translation','analysisPdf')):
        raw=BODY+bytes([index])
        p[kind]=dict(path=f'journal-tapa/{kind}.pdf',size=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return p


def test_extension_keeps_legacy_list_and_publication_dates():
    legacy=paper();p=journal();value=dict(schema='ptb-papers/1',papers=[legacy],journalPapers=[p])
    assert validate_catalog(value) is value
    assert value['papers']==[legacy]
    assert catalog_papers(value)==[legacy,p]
    assert p['submittedAt']=='2025-09-12' and p['journal']['publishedAt']=='2026-09-09'
    assert p['publishedAt']=='2026-10-21'


@pytest.mark.parametrize('mutate',[
    lambda c:c.update(journalPapers={}),
    lambda c:c.update(journalPapers=[c['papers'][0]]),
    lambda c:c['journalPapers'][0].update(id=c['papers'][0]['id']),
    lambda c:c['journalPapers'][0].update(sourceUrl='http://doi.org/10.3758/a'),
    lambda c:c['journalPapers'][0].update(sourceUrl='https://doi.org.evil.test/10.3758/a'),
    lambda c:c['journalPapers'][0].update(sourceUrl='https://doi.org@evil.test/10.3758/a'),
    lambda c:c['journalPapers'][0].update(sourceUrl='https://doi.org/10.3758/a?target=evil'),
    lambda c:c['journalPapers'][0].update(sourceUrl='https://doi.org/10.3758/a#fragment'),
    lambda c:c['journalPapers'][0].update(sourceUrl='https://doi.org/10.3758/a\nb'),
    lambda c:c['journalPapers'][0].update(sourceUrl='file:///secret'),
    lambda c:c['journalPapers'][0].update(journal={}),
    lambda c:c['journalPapers'][0]['journal'].update(publishedAt='2026-02-30'),
    lambda c:c['journalPapers'][0]['journal'].update(doi='10.3758/different'),
    lambda c:c['journalPapers'][0]['journal'].update(title=''),
    lambda c:c['journalPapers'][0]['journal'].update(volume=58),
    lambda c:c['journalPapers'][0]['journal'].update(title='bad\x00title'),
    lambda c:c['journalPapers'][0]['license'].update(id='CC-BY-ND-4.0'),
    lambda c:c['journalPapers'][0]['analysisPdf'].update(path='../private.pdf'),
    lambda c:c['journalPapers'][0]['translation'].update(size=50_000_001),
    lambda c:c.update(papers=[c['journalPapers'][0]],journalPapers=[]),
])
def test_journal_extension_rejects_bad_sources_metadata_and_assets(mutate):
    catalog=dict(schema='ptb-papers/1',papers=[paper()],journalPapers=[journal()])
    mutate(catalog)
    with pytest.raises(PaperError):validate_catalog(catalog)


def test_journal_addition_downloads_only_three_new_assets_and_preserves_old_notes(tmp_path):
    old=paper('2026-10-20');p=journal()
    catalog=dict(schema='ptb-papers/1',papers=[old])
    opener=Opener({'catalog.json':json.dumps(catalog).encode(),'sample/original.pdf':BODY})
    service=PaperService(tmp_path,today=lambda:date(2026,10,21),opener=opener)
    service.refresh(threading.Event());service.download('2026-10-01',threading.Event(),lambda _:None)
    annotation=tmp_path/('annotations-'+old['original']['sha256']+'.json')
    annotation.write_text(json.dumps(dict(schema='ptb-paper-annotations/1',revision=0,items=[])))
    old_notes=annotation.read_bytes();old_file=service._path(old['original']);old_mtime=old_file.stat().st_mtime_ns
    catalog['journalPapers']=[p];opener.items['catalog.json']=json.dumps(catalog).encode()
    for index,kind in enumerate(('original','translation','analysisPdf')):
        opener.items[p[kind]['path']]=BODY+bytes([index])
    state=service.refresh(threading.Event());assert len(state['papers'])==2
    opener.calls.clear();service.download(service.first_launch,threading.Event(),lambda _:None)
    assert len(opener.calls)==3 and all('journal-tapa/' in url for url in opener.calls)
    assert service.status()['papers'][1]['downloaded']
    assert annotation.read_bytes()==old_notes and old_file.stat().st_mtime_ns==old_mtime
    assert service.asset(p['id'],'analysis')==p['analysisPdf']
    assert json.loads((tmp_path/'catalog.json').read_text())['papers']==[old]
    reopened=PaperService(tmp_path,today=lambda:date(2026,10,21))
    assert len(reopened.status()['papers'])==2 and reopened.status()['papers'][1]['downloaded']
    service.recommendation_folder('期刊');service.collect_recommendation(p['id'],'期刊')
    assert service.status()['collections'][p['id']]=='期刊'
    service.remove_download(p['id']);assert old_file.read_bytes()==BODY
    assert service.status()['papers'][0]['downloaded'] and service.status()['papers'][1]['dismissed']
    opener.calls.clear();service.download(service.first_launch,threading.Event(),lambda _:None)
    assert not opener.calls
    service.download('2026-10-01',threading.Event(),lambda _:None,ids=[p['id']])
    assert service.status()['papers'][1]['downloaded'] and not service.status()['papers'][1]['dismissed']


def test_failed_journal_refresh_keeps_last_good_catalog(tmp_path):
    good=dict(schema='ptb-papers/1',papers=[paper()],journalPapers=[journal()])
    opener=Opener({'catalog.json':json.dumps(good).encode()})
    s=PaperService(tmp_path,today=lambda:date(2026,10,21),opener=opener);s.refresh(threading.Event())
    before=(tmp_path/'catalog.json').read_bytes();bad=copy.deepcopy(good);bad['journalPapers'][0]['journal']['doi']='wrong'
    opener.items['catalog.json']=json.dumps(bad).encode()
    with pytest.raises(PaperError):s.refresh(threading.Event())
    assert (tmp_path/'catalog.json').read_bytes()==before and len(s.status()['papers'])==2
