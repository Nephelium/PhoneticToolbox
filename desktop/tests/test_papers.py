"""M18 native date, integrity, offline, cancellation and URL boundaries."""
import copy
from datetime import date
import hashlib
from io import BytesIO
import json
import threading
import pytest
from ptb_desktop.papers import PaperService, PaperError, validate_catalog, BASE_URL, NoRedirect

BODY=b'%PDF-1.7\nowned fixture, not a rendered paper'

def paper(day='2026-10-07',id='sample'):
    asset={'path':f'{id}/original.pdf','size':len(BODY),'sha256':hashlib.sha256(BODY).hexdigest()}
    return {'id':id,'title':'Sample','titleZh':'示例','authors':'Example Author','version':'v1','guide':'Guide',
            'translationNote':'Translated; not endorsed','sourceUrl':'https://arxiv.org/abs/2610.00735v1',
            'license':{'id':'CC-BY-4.0','url':'https://creativecommons.org/licenses/by/4.0/'},
            'publishedAt':day,'submittedAt':'2026-09-30','original':asset,'translation':dict(asset)}

class Response(BytesIO):
    def __init__(self,body,url):super().__init__(body);self.url=url
    def geturl(self):return self.url

class Opener:
    def __init__(self,items):self.items=items;self.calls=[]
    def open(self,request,timeout):
        self.calls.append(request.full_url)
        return Response(self.items[request.full_url.removeprefix(BASE_URL)],request.full_url)

def setup(tmp_path, entries=None):
    catalog={'schema':'ptb-papers/1','papers':entries or [paper()]}
    opener=Opener({'catalog.json':json.dumps(catalog).encode(),'sample/original.pdf':BODY,'old/original.pdf':BODY})
    service=PaperService(tmp_path,today=lambda:date(2026,10,7),opener=opener)
    service.refresh(threading.Event())
    return service,opener

def test_first_launch_persists_without_opening_module(tmp_path):
    PaperService(tmp_path,today=lambda:date(2026,10,7))
    newer=PaperService(tmp_path,today=lambda:date(2026,12,1))
    assert newer.status()['firstLaunch']=='2026-10-07'

def test_inclusive_current_subscription_and_optional_history(tmp_path):
    older=paper('2026-10-06','old');oldbytes=BODY+b'old'
    for lang in ('original','translation'):older[lang].update(size=len(oldbytes),sha256=hashlib.sha256(oldbytes).hexdigest())
    future=paper('2026-10-08','future');futurebytes=BODY+b'future'
    for lang in ('original','translation'):future[lang].update(size=len(futurebytes),sha256=hashlib.sha256(futurebytes).hexdigest())
    service,opener=setup(tmp_path,[older,paper(),future])
    opener.items['old/original.pdf']=oldbytes
    service.download(service.first_launch,threading.Event(),lambda _:None)
    values={p['id']:p['downloaded'] for p in service.status()['papers']}
    assert values=={'old':False,'sample':True,'future':False}
    assert BASE_URL+'old/original.pdf' not in opener.calls
    service.download('2026-10-06',threading.Event(),lambda _:None)
    assert service.first_launch=='2026-10-07'
    assert BASE_URL+'old/original.pdf' in opener.calls

@pytest.mark.parametrize('change',[
    lambda p:p['original'].update(path='../secret.pdf'),
    lambda p:p['original'].update(path='https://evil.test/a.pdf'),
    lambda p:p['original'].update(path='a/%2e%2e.pdf'),
    lambda p:p['original'].update(size=50_000_001),
    lambda p:p['original'].update(sha256='x'*64),
    lambda p:p['license'].update(id='arXiv-nonexclusive'),
    lambda p:p['license'].update(url='https://evil.test/'),
    lambda p:p.update(sourceUrl='file:///secret'),
    lambda p:p.update(publishedAt='2026-02-30'),
])
def test_manifest_rejects_invalid_rights_and_paths(change):
    p=paper();change(p)
    with pytest.raises((ValueError,TypeError)):validate_catalog({'schema':'ptb-papers/1','papers':[p]})

def test_failed_refresh_preserves_last_good_directory(tmp_path):
    s,o=setup(tmp_path);o.items['catalog.json']=b'{bad'
    with pytest.raises(ValueError):s.refresh(threading.Event())
    assert PaperService(tmp_path).status()['papers'][0]['id']=='sample'

def test_truncated_corrupt_download_and_retry(tmp_path):
    s,o=setup(tmp_path);o.items['sample/original.pdf']=BODY[:-2]
    with pytest.raises(PaperError):s.download(s.first_launch,threading.Event(),lambda _:None)
    assert not s.status()['papers'][0]['downloaded'];assert not list(s.files.iterdir())
    o.items['sample/original.pdf']=BODY
    s.download(s.first_launch,threading.Event(),lambda _:None)
    assert s.status()['papers'][0]['downloaded']
    s._path(paper()['original']).write_bytes(b'corrupt')
    assert not s.status()['papers'][0]['downloaded']
    s.download(s.first_launch,threading.Event(),lambda _:None)
    assert PaperService(tmp_path).status()['papers'][0]['downloaded']

def test_cancel_and_retry_skip_verified_files(tmp_path):
    s,o=setup(tmp_path);cancel=threading.Event();cancel.set()
    with pytest.raises(PaperError):s.download(s.first_launch,cancel,lambda _:None)
    assert not list(s.files.iterdir())
    cancel.clear();s.download(s.first_launch,cancel,lambda _:None);before=len(o.calls)
    s.download(s.first_launch,cancel,lambda _:None);assert len(o.calls)==before

def test_no_redirect_and_duplicate_ids():
    with pytest.raises(PaperError):NoRedirect().redirect_request(None,None,None,None,None,None)
    with pytest.raises(PaperError):validate_catalog({'schema':'ptb-papers/1','papers':[paper(),paper()]})

def test_corrupt_initial_date_is_never_silently_reset(tmp_path):
    (tmp_path/'first-launch.json').write_text('bad')
    with pytest.raises(ValueError):PaperService(tmp_path)
    assert (tmp_path/'first-launch.json').read_text()=='bad'


def test_concurrent_launches_share_one_complete_first_date(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def launch(day):return PaperService(tmp_path,today=lambda:date(2026,10,day)).first_launch
    with ThreadPoolExecutor(max_workers=4) as pool:
        values=list(pool.map(launch,[7,8,9,10]))
    assert len(set(values))==1
    assert json.loads((tmp_path/'first-launch.json').read_text('utf8'))['date']==values[0]
