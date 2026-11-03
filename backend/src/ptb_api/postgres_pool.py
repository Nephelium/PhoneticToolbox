"""Lazy bounded process-local psycopg connections, shared by account/job/storage.

No background threads, credentials in diagnostics, or schema changes. Borrowers
must finish transactions before returning; session state is discarded on return.
"""
import atexit
from collections import OrderedDict
from contextlib import contextmanager
import threading
import time
import psycopg
from psycopg.rows import dict_row


class Pool:
    def __init__(self, dsn, maximum=8, timeout=5):
        self.dsn=dsn;self.maximum=maximum;self.timeout=timeout
        self.condition=threading.Condition();self.idle=[];self.count=0;self.closed=False
        self.created=0;self.peak=0

    @contextmanager
    def connection(self):
        deadline=time.monotonic()+self.timeout
        with self.condition:
            while True:
                if self.closed:raise psycopg.OperationalError('Database pool closed')
                if self.idle:
                    conn=self.idle.pop();break
                if self.count<self.maximum:
                    self.count+=1;conn=None;break
                remaining=deadline-time.monotonic()
                if remaining<=0:raise psycopg.OperationalError('Database pool busy')
                self.condition.wait(remaining)
        reusable=False
        try:
            if conn is None:
                conn=psycopg.connect(self.dsn,autocommit=True,row_factory=dict_row,
                    connect_timeout=5,options='-c statement_timeout=10000 -c lock_timeout=5000',prepare_threshold=None)
                with self.condition:
                    self.created+=1;self.peak=max(self.peak,self.count)
            # Fail stale idle sockets before any mutating statement. No operation
            # is retried automatically after an ambiguous connection failure.
            conn.execute('SELECT 1')
            yield conn
        finally:
            if conn is not None and not conn.closed:
                try:
                    conn.rollback();conn.autocommit=True
                    conn.execute('DISCARD ALL')
                    reusable=True
                except Exception:pass
            with self.condition:
                if reusable and not self.closed:self.idle.append(conn)
                else:
                    if conn is not None:conn.close()
                    self.count-=1
                self.condition.notify()

    def close(self):
        with self.condition:
            self.closed=True
            for conn in self.idle:conn.close()
            self.count-=len(self.idle);self.idle.clear();self.condition.notify_all()


_guard=threading.Lock()
_pools=OrderedDict()


def pool_for(dsn):
    with _guard:
        pool=_pools.pop(dsn,None)
        if pool is not None and pool.closed:
            with pool.condition:
                if pool.count:
                    _pools[dsn]=pool
                    raise psycopg.OperationalError('Database pool draining')
            pool=None
        if pool is None:
            if len(_pools)>=4:
                for key,old in list(_pools.items()):
                    with old.condition:
                        if old.count!=len(old.idle):continue
                        old.close();del _pools[key];break
                else:raise psycopg.OperationalError('Database pool registry busy')
            pool=Pool(dsn)
        _pools[dsn]=pool
        return pool


@contextmanager
def connection(dsn):
    with pool_for(dsn).connection() as conn:yield conn


def close_pool(dsn):
    with _guard:
        pool=_pools.get(dsn)
        if pool is not None:
            pool.close()
            # Keep active drainers registered, so a concurrent caller cannot
            # open a second pool for the same DSN and bypass the connection cap.
            with pool.condition:
                if not pool.count:_pools.pop(dsn,None)


@atexit.register
def close_all():
    with _guard:
        for pool in _pools.values():pool.close()
        _pools.clear()
