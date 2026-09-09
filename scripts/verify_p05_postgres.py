"""Explicit P05 PostgreSQL validation after migration approval. Creates test rows; never deletes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import getpass
import json
import secrets
from pathlib import Path
from uuid import uuid4
from fastapi.testclient import TestClient
from ptb_api.account_store import PostgresAccountStore, hash_token
from ptb_api.auth import AuthSettings
from ptb_api.main import create_app

ORIGIN='https://p05.test'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-test-data',action='store_true')
    args=parser.parse_args()
    if not args.approved_test_data:parser.error('Requires approved isolated PostgreSQL test data')
    dsn=getpass.getpass('Dedicated migrated test database DSN (hidden): ')
    store=PostgresAccountStore(dsn)
    store.check_schema()
    with store.connection() as conn:
        if not conn.execute('SELECT current_database() AS name').fetchone()['name'].startswith('ptb_p05_test_'):
            parser.error('Expected a dedicated ptb_p05_test_* database')
    suffix=uuid4().hex[:12]
    password=secrets.token_urlsafe(32)
    alice,bob='a_'+suffix,'b_'+suffix
    store.create_user(alice,password);store.create_user(bob,password)
    settings=AuthSettings(origin=ORIGIN,signing_key=secrets.token_urlsafe(48))
    def client():return TestClient(create_app(account_store=PostgresAccountStore(dsn),auth_settings=settings),base_url=ORIGIN)
    def login(c,user):
        challenge=c.get('/api/v1/auth/challenge').json()['csrf_token']
        response=c.post('/api/v1/auth/login',json={'username':user,'password':password},headers={'Origin':ORIGIN,'X-CSRF-Token':challenge})
        assert response.status_code==200
        return {'Origin':ORIGIN,'X-CSRF-Token':response.json()['csrf_token'],'X-PTB-Account':response.json()['user']['id']}
    checks={}
    with client() as a,client() as b:
        ah,bh=login(a,alice),login(b,bob)
        r=a.post('/api/v1/projects',json={'name':'P05 独立数据库验证'},headers=ah)
        assert r.status_code==201
        project=r.json()['id']
        assert b.get('/api/v1/projects/'+project).status_code==404
        assert b.patch('/api/v1/projects/'+project,json={'name':'wrong owner'},headers=bh).status_code==404
        assert b.get('/api/v1/projects').json()['projects']==[]
        checks['two_account_owner_checks']=True
        cookie=a.cookies.get('__Host-ptb-session')
        with client() as rebuilt:
            rebuilt.cookies.set('__Host-ptb-session',cookie)
            assert rebuilt.get('/api/v1/auth/me').status_code==200
            assert rebuilt.get('/api/v1/projects/'+project).json()['name']=='P05 独立数据库验证'
        checks['separate_adapter_and_app_restore']=True
        with store.connection() as conn:
            try:
                with conn.transaction():
                    conn.execute('INSERT INTO ptb_accounts.projects(id,owner_id,name) VALUES (%s,%s,%s)',(uuid4(),ah['X-PTB-Account'],'rollback probe'))
                    raise RuntimeError('intentional rollback')
            except RuntimeError:pass
            assert conn.execute('SELECT count(*) AS n FROM ptb_accounts.projects WHERE owner_id=%s',(ah['X-PTB-Account'],)).fetchone()['n']==1
        checks['transaction_rollback']=True
        assert a.post('/api/v1/auth/logout',headers=ah).status_code==204
        with client() as revoked:
            revoked.cookies.set('__Host-ptb-session',cookie)
            assert revoked.get('/api/v1/auth/me').status_code==401
        checks['revocation_across_connections']=True
    def attempt(i):return PostgresAccountStore(dsn).allow_login('limit_'+suffix,'192.0.2.'+str(i))
    with ThreadPoolExecutor(max_workers=12) as executor:
        results=list(executor.map(attempt,range(12)))
    assert sum(results)==10,results
    checks['concurrent_limit_exactly_10_of_12']=True
    # Attempt rows remain test evidence; no automatic DROP/DELETE or cleanup.
    output=Path(__file__).resolve().parents[1]/'output/validation/p05'
    output.mkdir(parents=True,exist_ok=True)
    report={'time':datetime.now(timezone.utc).isoformat(),'scope':'PostgreSQL connections and reconstructed apps; no process restart or deployment claim','checks':checks}
    (output/('postgres-'+suffix+'.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
