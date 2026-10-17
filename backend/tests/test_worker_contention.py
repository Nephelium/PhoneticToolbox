import threading
import pytest
from ptb_api.quota import StorageError
from ptb_worker.executor import run_worker


def test_worker_survives_temporary_storage_contention():
    stop=threading.Event()
    class Store:
        calls=0
        def claim(self, worker_id):
            assert worker_id=='owned'
            self.calls+=1
            if self.calls<3:raise StorageError('storage_busy',503)
            stop.set()
            return None
    store=Store()
    run_worker(store,stop,'owned')
    assert store.calls==3


def test_worker_does_not_hide_storage_identity_failure():
    class Store:
        def claim(self, worker_id):raise StorageError('storage_instance_mismatch',503)
    with pytest.raises(StorageError,match='storage_instance_mismatch'):
        run_worker(Store(),threading.Event(),'owned')
