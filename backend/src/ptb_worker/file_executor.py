"""Owned worker execution of explicitly labelled storage checks and ZIP tasks."""
import json
import zipfile
import zlib
from ptb_api.quota import CHUNK_BYTES, StorageError
from .archives import archive, extract, OutputWriter


def execute_file_claim(store,claim,worker_id,stop,*,step_delay=0):
    files=store.files
    identity=(claim['id'],worker_id,claim['generation'])
    if files is None:
        store.finish(*identity,error='execution_failed')
        return
    snapshot=json.loads(claim['snapshot'])
    config=snapshot['config']
    try:
        with files.storage._locked() as conn:
            _,inputs=files._fence(conn,identity)
        operation=snapshot['operation']
        if operation=='storage_check':
            # Deterministic engineering fixture; deliberately not an acoustic
            # algorithm, and the UI labels it as a storage pipeline check.
            for index in range(config['probe_files']):
                size=config['probe_bytes']
                asset=files.output(identity,f'存储流程检查-{index+1}.bin','result',size)
                writer=OutputWriter(files,identity,asset['id'],stop)
                pattern=bytes((i+index)%256 for i in range(256))
                for offset in range(0,size,CHUNK_BYTES):
                    if step_delay: stop.wait(step_delay)
                    count=min(CHUNK_BYTES,size-offset)
                    writer.write((pattern*((count+255)//256))[:count])
                files.seal(identity,asset['id'])
        elif operation=='archive_zip': archive(files,identity,inputs,stop)
        elif operation=='extract_zip': extract(files,identity,inputs[0],config['max_output_bytes'],stop)
        else: raise StorageError('unsupported_file_operation',422)
        if stop.is_set(): raise StorageError('cancelled',409)
        files.complete(identity)
    except StorageError as error:
        files.fail(identity,error.code)
    except (zipfile.BadZipFile,zipfile.LargeZipFile,NotImplementedError,EOFError,ValueError,zlib.error):
        files.fail(identity,'archive_rejected')
    except OSError:
        files.fail(identity,'execution_failed')
