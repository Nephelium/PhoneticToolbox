"""Real PDF/HTTPS delivery and the captured pre-DOI reader compatibility check."""
import argparse
import base64
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
from urllib.request import urlopen
from io import BytesIO
from verification_artifacts import verification_run
from workbench_source import configure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--legacy-module',type=Path,required=True)
    p.add_argument('--live',action='store_true')
    a=p.parse_args();configure()
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QCoreApplication,QEvent
    from ptb_documents.papers import PaperService,validate_catalog,catalog_papers,BASE_URL
    app=QApplication(['M18-journal-owned-QA'])
    raw=urlopen(BASE_URL+'catalog.json',timeout=25).read() if a.live else a.catalog.read_bytes()
    catalog=validate_catalog(json.loads(raw));baseline={k:v for k,v in catalog.items() if k!='journalPapers'}
    assert len(catalog['papers'])==2 and len(catalog.get('journalPapers',[]))==1
    spec=importlib.util.spec_from_file_location('captured_legacy_papers',a.legacy_module)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    legacy.validate_catalog(catalog)
    with verification_run('m18-journal-native','live' if a.live else 'candidate','Approved CC BY original/translation/analysis; captured old reader; owned cache, same QtPdf rendering, no personal library') as (out,scratch):
        class Response(BytesIO):
            def __init__(self,body,url):super().__init__(body);self.url=url
            def geturl(self):return self.url
        class Opener:
            def __init__(self):self.catalog=baseline;self.calls=[]
            def open(self,request,timeout):
                relative=request.full_url.removeprefix(BASE_URL);self.calls.append(relative)
                if relative=='catalog.json':body=json.dumps(self.catalog).encode('utf8')
                elif not a.live and (a.stage/relative).is_file():body=(a.stage/relative).read_bytes()
                else:
                    with urlopen(request,timeout=25) as response:body=response.read(50_000_001)
                return Response(body,request.full_url)
        opener=Opener();old=legacy.PaperService(scratch/'cache',today=lambda:date(2026,10,21),opener=opener)
        old.refresh(threading.Event());old.download('2026-10-01',threading.Event(),lambda _:None)
        old.recommendation_folder('原有收藏');old.collect_recommendation(catalog['papers'][0]['id'],'原有收藏')
        oldasset=catalog['papers'][0]['original'];notes=old.root/('annotations-'+oldasset['sha256']+'.json')
        notes.write_text(json.dumps(dict(schema='ptb-paper-annotations/1',revision=0,items=[])),'utf8')
        protected={p.name:(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns) for p in old.files.iterdir()}
        prior_notes=notes.read_bytes();prior_preferences=(old.root/'preferences.json').read_bytes()
        opener.catalog=catalog
        assert len(old.refresh(threading.Event())['papers'])==2
        upgraded=PaperService(old.root,today=lambda:date(2026,10,21),opener=opener)
        upgraded.refresh(threading.Event());assert len(upgraded.status()['papers'])==3
        opener.calls.clear();state=upgraded.download(upgraded.first_launch,threading.Event(),lambda _:None)
        target=catalog['journalPapers'][0];expected={target[k]['path'] for k in ('original','translation','analysisPdf')}
        assert set(opener.calls)==expected and len(opener.calls)==3
        assert all((old.files/n).stat().st_mtime_ns==row[1] and hashlib.sha256((old.files/n).read_bytes()).hexdigest()==row[0] for n,row in protected.items())
        assert notes.read_bytes()==prior_notes and (old.root/'preferences.json').read_bytes()==prior_preferences
        assert all(p['downloaded'] for p in state['papers'])
        captures=[]
        try:
            for language,pages in [('original',15),('translation',15),('analysis',11)]:
                for page in [1,pages]:
                    result=upgraded.render(target['id'],language,page,1100)
                    assert result['pages']==pages and result['text'].strip() and result['imageWidth']==1100
                    name=f'{language}-{page}.png';(out/name).write_bytes(base64.b64decode(result['image'].split(',',1)[1]));captures.append(name)
            opener.calls.clear();upgraded.download(upgraded.first_launch,threading.Event(),lambda _:None);assert not opener.calls
            assert len(legacy.PaperService(old.root,today=lambda:date(2026,10,21)).status()['papers'])==2
            report=dict(success=True,live=a.live,legacyPapers=2,currentPapers=3,newDownloads=3,repeatedDownloads=0,pages=dict(original=15,translation=15,analysis=11),originalFilesNotesFoldersPreserved=True,downgradeStillReadsLegacy=True,legacyModuleSha256=hashlib.sha256(a.legacy_module.read_bytes()).hexdigest(),captures=captures)
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n','utf8');print(json.dumps(dict(**report,output=str(out))))
        finally:
            upgraded.close_documents();old.close_documents();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents()


if __name__=='__main__':main()
