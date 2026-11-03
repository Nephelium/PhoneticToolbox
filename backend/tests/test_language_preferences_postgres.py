"""Language settings round-trip through the real account API and fresh PG only."""
from ptb_api.preferences import list_preferences
from test_account_preferences_postgres import web
from test_p07_policy_postgres import legacy
from test_web_workspaces_postgres import device
from web_postgres_support import cluster


def test_three_languages_persist_across_sessions_without_overwriting_scientific_settings(web):
    e=web;a,_,_,_=device(e)
    response=a.patch('/api/v1/preferences/M04',json={'expected_revision':0,'changes':{'draft':{'order':'72'}}})
    assert response.status_code==200,response.text
    for revision,code in enumerate(('zh-CN','en','ja')):
        response=a.patch('/api/v1/preferences/APP',json={'expected_revision':revision,'changes':{'uiLocale':code}})
        assert response.status_code==200,response.text
        b,_,_,_=device(e)
        saved={item['module']:item['values'] for item in b.get('/api/v1/preferences').json()['modules']}
        assert saved['APP']['uiLocale']==code
        assert saved['M04']['draft']['order']=='72'
    for value in (None,True,[],{},'fr','../ja'):
        response=a.patch('/api/v1/preferences/APP',json={'expected_revision':3,'changes':{'uiLocale':value}})
        assert response.status_code==422,response.text
    response=a.patch('/api/v1/preferences/APP',json={'expected_revision':3,'changes':{'uiLocale':'en'}},headers={'X-CSRF-Token':'wrong'})
    assert response.status_code==403
    saved={item['module']:item for item in list_preferences(e.accounts,e.owner)}
    assert saved['APP']['revision']==3 and saved['APP']['values']['uiLocale']=='ja'
    bob=e.accounts.create_user('language-other','synthetic-password-only')
    assert list_preferences(e.accounts,str(bob['id']))==[]
