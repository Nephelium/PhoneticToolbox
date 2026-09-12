"""M02 actual bounded child decoding, hostile input and auth boundary."""
import hashlib
import io
import pytest
from openpyxl import Workbook
from ptb_worker.parameter_preview import render
from ptb_worker.spectrogram_preview import PreviewError
from fastapi.testclient import TestClient
from ptb_api.main import create_app


def workbook(formula=False):
    wb=Workbook();sheet=wb.active;sheet.append(['Time_s','pF0','TextGrid'])
    sheet.append([0,100,'阴平']);sheet.append([.01,'=1+1' if formula else None,'上声'])
    stream=io.BytesIO();wb.save(stream);return stream.getvalue()


def test_original_cells_and_unicode_from_bounded_child():
    raw=workbook();value=render(raw,'参数.xlsx')
    assert value['sha256']==hashlib.sha256(raw).hexdigest()
    assert value['rows']==[[0,100,'阴平'],[.01,None,'上声']]
    assert value['kinds']==['number','number','text']


def test_legacy_sqlite_exact_numeric_and_annotation_cells():
    import sqlite3
    with sqlite3.connect(':memory:') as conn:
        conn.execute('CREATE TABLE params(Time_s REAL,pF0 REAL,TextGrid TEXT)')
        conn.executemany('INSERT INTO params VALUES(?,?,?)',[(0,100,'阴平'),(.01,None,'上声')]);conn.commit();raw=conn.serialize()
    result=render(raw,'参数.ptb.sqlite')
    assert result['rows']==[[0,100,'阴平'],[.01,None,'上声']]
    assert result['sha256']==hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('raw,name',[(b'invalid','bad.xlsx'),(workbook(True),'formula.xlsx'),(b'SQLite format 3\0','bad.ptb.sqlite')])
def test_reject_invalid_tables(raw,name):
    with pytest.raises(PreviewError):render(raw,name)


def test_local_parameter_auth_before_input():
    token='x'*40;origin='http://127.0.0.1:5177'
    with TestClient(create_app('local',local_token=token,local_origin=origin),base_url=origin) as client:
        path='/api/v1/preview/parameters?name=data.xlsx';headers={'Authorization':'Bearer '+token,'Origin':origin}
        assert client.post(path,content=workbook()).status_code==403
        assert client.post(path,content=workbook(),headers={'Authorization':'Bearer '+token}).status_code==403
        result=client.post(path,content=workbook(),headers=headers)
        assert result.status_code==200,result.text
