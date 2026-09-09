"""Review-first P05 SQL and controlled account provisioning. No default mutation."""
import argparse
import getpass
from pathlib import Path
import psycopg
from ptb_api.account_store import PostgresAccountStore

ROOT=Path(__file__).resolve().parents[1]
SQL=ROOT/'backend/migrations/001_accounts.sql'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['show','check','apply','create-user'])
    p.add_argument('--approved-empty-test-database',action='store_true')
    args=p.parse_args()
    if args.action=='show':
        print(SQL.read_text('utf-8'))
        return
    dsn=getpass.getpass('Dedicated test database DSN (hidden): ')
    store=PostgresAccountStore(dsn)
    if args.action=='check':
        store.check_schema()
        print('P05 schema version 1 present')
    elif args.action=='apply':
        if not args.approved_empty_test_database:
            p.error('Actual migration requires prior user approval and --approved-empty-test-database')
        with psycopg.connect(dsn,autocommit=True,connect_timeout=5) as conn:
            name=conn.execute('SELECT current_database()').fetchone()[0]
            if not name.startswith('ptb_p05_test_'):
                p.error('Only an explicitly named ptb_p05_test_* database is supported here')
            if conn.execute("SELECT 1 FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog','information_schema') LIMIT 1").fetchone():
                p.error('Refusing to migrate a non-empty database')
            conn.execute(SQL.read_text('utf-8'))
        print('P05 schema applied to approved empty test database')
    else:
        store.check_schema()
        username=input('Researcher login: ')
        password=getpass.getpass('Password (hidden): ')
        if password!=getpass.getpass('Repeat password: '):
            p.error('Passwords differ')
        store.create_user(username,password)
        print('Researcher account created')

if __name__=='__main__':
    main()
