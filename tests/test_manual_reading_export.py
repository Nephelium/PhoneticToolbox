"""P19: editor null anchors must not break a frozen reader or mutate authors."""
import importlib.util
from pathlib import Path
import sys

FOLDER=Path(__file__).resolve().parents[1]/'scripts/manual'
sys.path.insert(0,str(FOLDER))
SPEC=importlib.util.spec_from_file_location('ptb_manual_build',FOLDER/'build.py')
BUILD=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def test_null_optional_anchor_is_omitted_without_mutating_author():
    source={'body':{'type':'doc','content':[
        {'type':'paragraph','attrs':{'id':None,'caption':None}},
        {'type':'heading','attrs':{'id':'m05-purpose','level':2}},
        {'type':'crossReference','attrs':{'id':None,'targetId':'m05-purpose'}},
    ]}}
    result=BUILD.reader_chapter(source)
    assert source['body']['content'][0]['attrs']['id'] is None
    assert result['body']['content'][0]['attrs']=={'caption':None}
    assert result['body']['content'][1]==source['body']['content'][1]
    assert result['body']['content'][2]['attrs']=={'targetId':'m05-purpose'}


def test_bad_nonempty_anchor_is_preserved_for_strict_rejection():
    source={'body':{'type':'doc','content':[{'type':'paragraph','attrs':{'id':'bad/id'}}]}}
    assert BUILD.reader_chapter(source)==source
