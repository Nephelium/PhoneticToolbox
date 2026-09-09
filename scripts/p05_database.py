"""Review-first P05 SQL and controlled account provisioning. No default mutation."""
import argparse
import getpass
import sys
from pathlib import Path
import psycopg
from ptb_api.account_store import PostgresAccountStore

ROOT=Path(__file__).resolve().parents[1]
SQL=ROOT/'backend/migrations/001_accounts.sql'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['show','check','apply','create-user'])
    p.add_argument('--approved-empty-test-database',action='store_true')
    p.add_argument('--dsn-stdin',action='store_true',help='Read DSN from a private parent pipe, never a command argument')
    args=p.parse_args()
    if args.action=='show':
        print(SQL.read_text('utf-8'))
        return
    dsn=sys.stdin.readline().strip() if args.dsn_stdin else getpass.getpass('Dedicated test database DSN (hidden): ')
    if not dsn:p.error('A private database DSN is required')
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
